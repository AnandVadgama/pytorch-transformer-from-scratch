from pathlib import Path
import os
import wandb   
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# Import inference utilities for validation and decoding
from inference import run_validation, greedy_decode, beam_search_decode
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


def get_device():
    """
    Select the best available device for training (MPS, CUDA, or CPU).
    """
    return torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))


def get_all_sentences(ds, lang):
    """
    Yield all sentences for a given language from the dataset.
    Used for tokenizer training.
    """
    for item in ds:
        yield item[lang]


def get_or_build_tokenizer(config, ds, lang):
    """
    Load a tokenizer from file if it exists, otherwise train a new one on the dataset.
    Tokenizer is saved to disk for future use.
    """
    tokenizer_path = Path(config['tokenizer_path'].format(lang))
    if not Path.exists(tokenizer_path):
        # Train a new tokenizer
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens = ['[UNK]', '[PAD]', '[SOS]','[EOS]'], min_frequency= 2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else: 
        # Load existing tokenizer
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def get_ds(config):
    """
    Prepare train/validation dataloaders and tokenizers.
    Splits the dataset, builds tokenizers, and returns DataLoaders and tokenizers.
    """
    ds_raw = load_dataset('Bhoomi06/english-to-colloquial-hindi-dataset', split="train")
    # Build or load tokenizers for both languages
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['src_lang'])
    tokenizer_trg = get_or_build_tokenizer(config, ds_raw, config['trg_lang'])
    # Split dataset into train and validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_size, val_ds_size = random_split(ds_raw, [train_ds_size, val_ds_size])
    # Create PyTorch datasets
    train_ds = BilingualDataset(train_ds_size, tokenizer_src, tokenizer_trg, config['src_lang'], config['trg_lang'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_size, tokenizer_src, tokenizer_trg, config['src_lang'], config['trg_lang'], config['seq_len'])
    # Print max sequence lengths for info
    max_len_src = max(len(tokenizer_src.encode(ex['english']).ids) for ex in ds_raw)
    max_len_trg = max(len(tokenizer_trg.encode(ex['hindi']).ids) for ex in ds_raw)
    print(f"Max length of source sentence {max_len_src}")
    print(f"Max length of target sentence {max_len_trg}")
    # Create DataLoaders
    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)
    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_trg


def get_model(config, vocab_src_size, vocab_trg_size):
    """
    Build the Transformer model with the given vocabulary sizes and config.
    """
    return build_transformer(vocab_src_size, vocab_trg_size, config['seq_len'], config['seq_len'], config['d_model'])


def get_optimizer(model, config):
    """
    Create the Adam optimizer for model training.
    """
    return torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)


def get_loss_fn(tokenizer_src, config, device):
    """
    Create the cross-entropy loss function with label smoothing and padding ignore.
    """
    return nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)


def save_checkpoint(model, optimizer, epoch, global_step, model_filename):
    """
    Save model and optimizer state to disk for checkpointing.
    """
    os.makedirs(os.path.dirname(model_filename), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'global_step': global_step
    }, model_filename)


def load_checkpoint(model, optimizer, model_filename):
    """
    Load model and optimizer state from a checkpoint file.
    Returns the next epoch and global step.
    """
    state = torch.load(model_filename)
    model.load_state_dict(state['model_state_dict'])
    optimizer.load_state_dict(state['optimizer_state_dict'])
    return state['epoch'] + 1, state['global_step']


def train_model(config):
    """
    Main training loop for the Transformer model.
    Handles training, validation, logging, and checkpointing.
    """
    device = get_device()
    print(f"using device {device}")
    # Ensure model folder exists
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)
    # Prepare data and model
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_trg = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_trg.get_vocab_size()).to(device)
    writer = SummaryWriter(config['experiment_name'])
    optimizer = get_optimizer(model, config)
    loss_fn = get_loss_fn(tokenizer_src, config, device)
    initial_epoch = 0
    global_step = 0
    # Optionally preload weights/checkpoint
    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f"preloading model {model_filename}")
        initial_epoch, global_step = load_checkpoint(model, optimizer, model_filename)
    # Setup wandb metrics
    wandb.define_metric("global_step")
    wandb.define_metric("validation/*", step_metric="global_step")
    wandb.define_metric("train/*", step_metric="global_step")
    # Training epochs
    for epoch in range(initial_epoch, config['num_epochs']):
        batch_iterator = tqdm(train_dataloader, desc=f"processing epoch {epoch: 02d}")
        for batch in batch_iterator:
            optimizer.zero_grad()
            model.train()
            # Move batch to device
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)
            # Forward pass
            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = model.project(decoder_output)
            label = batch['label'].to(device)
            # Compute loss
            loss = loss_fn(proj_output.view(-1, tokenizer_trg.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({f"loss": f"{loss.item(): 6.3f}"})
            # Log loss
            writer.add_scalar("train loss", loss.item(), global_step)
            writer.flush()
            wandb.log({'train/loss': loss.item(), 'global_step': global_step})
            # Backpropagation and optimizer step
            loss.backward()
            optimizer.step()
            global_step += 1
        # Run validation at the end of each epoch
        run_validation(model, val_dataloader, tokenizer_src, tokenizer_trg, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)
        # Save checkpoint
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        save_checkpoint(model, optimizer, epoch, global_step, model_filename)


def main():
    """
    Entry point for training from the command line.
    """
    config = get_config()
    train_model(config)

if __name__ == '__main__':
    main()