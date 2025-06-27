from pathlib import Path
import os
import wandb   
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from dataset import BilingualDataset, causal_mask
from model import build_transformer
from config import get_weights_file_path, get_config

from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

import torchmetrics

wandb.init(project="transformer-from-scratch")

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_trg, max_len, device):
    sos_idx = tokenizer_trg.token_to_id('[SOS]')
    eos_idx = tokenizer_trg.token_to_id('[EOS]')

    # precompute the encoder output and reuse it for every token we get from the decoder
    encoder_output = model.encode(source, source_mask)

    # initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1, device=device).fill_(sos_idx).type_as(source)


    while True:
        if decoder_input.size(1) >= max_len:
            break
        # build mask for the target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        # calculate the output of the decoder
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        # get the next token
        prob = model.project(out[:, -1])
        # get the token with the highest probability
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word.item() == eos_idx:
            break

    return decoder_input


def run_validation(model, validation_ds, tokenizer_src, tokenizer_trg, max_len, device, print_msg, global_step, writer, num_examples=2):
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_trg, max_len, device)

            source_text = batch['src_text'][0]
            target_text = batch['trg_text'][0]
            model_out_text = tokenizer_trg.decode(model_out.detach().cpu().numpy()[0])

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            
            # print to the console
            print_msg('_'*console_width)
            print_msg(f"Source: {source_text}")
            print_msg(f"Target: {target_text}")
            print_msg(f"Predicted: {model_out_text}")
            print_msg('_'*console_width)

            if count == num_examples:
                break

    # Evaluate the character error rate
    # Compute the char error rate 
    metric = torchmetrics.CharErrorRate()
    cer = metric(predicted, expected)
    wandb.log({'validation/cer': cer, 'global_step': global_step})

    # Compute the word error rate
    metric = torchmetrics.WordErrorRate()
    wer = metric(predicted, expected)
    wandb.log({'validation/wer': wer, 'global_step': global_step})

    # Compute the BLEU metric
    metric = torchmetrics.BLEUScore()
    bleu = metric(predicted, expected)
    wandb.log({'validation/BLEU': bleu, 'global_step': global_step})

    model.train()
    return source_texts, expected, predicted

def get_all_sentences(ds, lang):
    for item in ds:
        yield item[lang]


def get_or_build_tokenizer(config, ds, lang):

    tokenizer_path = Path(config['tokenizer_path'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens = ['[UNK]', '[PAD]', '[SOS]','[EOS]'], min_frequency= 2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))

    else: 
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer

def get_ds(config):
    ds_raw = load_dataset('Bhoomi06/english-to-colloquial-hindi-dataset', split="train")

    # build tokenizer
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['src_lang'])
    tokenizer_trg = get_or_build_tokenizer(config, ds_raw, config['trg_lang'])

    # keep 90% as train data and 10% as validation data
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_size, val_ds_size = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_ds_size, tokenizer_src, tokenizer_trg, config['src_lang'], config['trg_lang'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_size, tokenizer_src, tokenizer_trg, config['src_lang'], config['trg_lang'], config['seq_len'])

    max_len_src = 0
    max_len_trg = 0

    for example in ds_raw:
        english_text = example['english']
        hindi_text = example['hindi']
        src_ids = tokenizer_src.encode(english_text).ids
        trg_ids = tokenizer_src.encode(hindi_text).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_trg = max(max_len_trg, len(trg_ids))

    print(f"Max length of source sentence {max_len_src}")
    print(f"Max length of target sentence {max_len_trg}")

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_trg

def get_model(config, vocab_src_size, vocab_trg_size):
    model = build_transformer(vocab_src_size, vocab_trg_size, config['seq_len'], config['seq_len'], config['d_model'])
    return model

def train_model(config):

    # define the device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"using device {device}")

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_trg = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_trg.get_vocab_size()).to(device)

    # tensorboard
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    initial_epoch = 0
    global_step = 0
    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f"preloading model {model_filename}")
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    # define our custom x axis metric
    wandb.define_metric("global_step")
    # define which metrics will be plotted against it
    wandb.define_metric("validation/*", step_metric="global_step")
    wandb.define_metric("train/*", step_metric="global_step")

    for epoch in range(initial_epoch, config['num_epochs']):

        batch_iterator = tqdm(train_dataloader, desc=f"processing epoch {epoch: 02d}")
        for batch in batch_iterator:
            optimizer.zero_grad()
            model.train()
            encoder_input = batch['encoder_input'].to(device) # (B, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (B, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (B, 1, seq_len, seq_len)

            # run the tensors through the transformer
            encoder_output = model.encode(encoder_input, encoder_mask) # (B, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask,decoder_input,decoder_mask) # (B, seq_len, d_model)
            proj_output = model.project(decoder_output) # (B, seq_len, trg_vocab_size)

            label = batch['label'].to(device) # (B, seq_len)

            # (B, seq_len, trg_vocab_size) -> (B * seq_len, trg_vocab_size)
            loss = loss_fn(proj_output.view(-1, tokenizer_trg.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({f"loss": f"{loss.item(): 6.3f}"})

            # log the loss
            writer.add_scalar("train loss", loss.item(), global_step)
            writer.flush()

            # backpropogate the loss
            loss.backward()

            # update the weights 
            optimizer.step()


            global_step += 1

        run_validation(model, val_dataloader, tokenizer_src, tokenizer_trg, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)
        # save the model
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        os.makedirs(os.path.dirname(model_filename), exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)

if __name__ == '__main__':
    config = get_config()
    train_model(config)