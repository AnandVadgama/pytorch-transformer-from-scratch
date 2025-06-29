from pathlib import Path
from tokenizers import Tokenizer
from datasets import load_dataset
import sys
import os
import wandb   
import torch
import torch.nn as nn
from dataset import BilingualDataset, causal_mask
from model import build_transformer
from config import get_weights_file_path, get_config, latest_weights_file_path
import torchmetrics

wandb.init(project="transformer-from-scratch")

# define the device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"using device {device}")

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

def beam_search_decode(model, beam_size, source, source_mask, tokenizer_src, tokenizer_trg, max_len, device):
    sos_idx = tokenizer_trg.token_to_id('[SOS]')
    eos_idx = tokenizer_trg.token_to_id('[EOS]')

    # precompute the encoder output and reuse it every time 
    encoder_output = model.encode(source, source_mask)
    # initialize the decoder input with sos token
    decoder_initial_input = torch.empty(1,1).fill_(sos_idx).type_as(source).to(device)

    # create candidate list
    candidate = ([decoder_initial_input, 1])

    while True: 
        # if a candidate reached at maximum length, it means we have run the decoding at least max_legth interation, so stop the search
        if any([cand.size(1) == max_len for cand, _ in candidate]):
            break

        # new list of candidate
        new_candidates = []

        for candidate, score in candidate:
            # do not expand the candidate that have reached the eos token
            if candidate[0][-1].item() == eos_idx:
                continue

            # build candidate's mask 
            candidate_mask = causal_mask(candidate.size(1)).type_as(source_mask).to(device)
            # calculate the output
            out = model.decode(encoder_output, source_mask, candidate, candidate_mask)
            # get the next token's probabilty
            prob = model.project(out[:,-1])
            # get the top k candidates
            topk_prob, topk_idx = torch.topk(prob, beam_size, dim=1)
            for i in range(beam_size):
                # for each of topk candidate, get the token ids and probability
                token = topk_idx[0][i].unsqueeze(0).unsqueeze(0)
                token_prob = topk_prob[0][i].item()

                # create a new candidate appending token to the current candidate
                new_candidate = torch.cat([candidate, token], dim=1)

                # we sum the log probabilities because the probabilities in log space
                new_candidates.append((new_candidate, score + token_prob))

        # sort the new candidate by their score
        candidates = sorted(new_candidates , lambda x: x[1], reverse=True)
        # keep only the topk candidates
        candidates = candidates[:beam_size]

        # if all the candidates have reached eos token,  stop
        if all([cand[0][-1].item() == eos_idx for cand, _ in candidates]):
            break
    # return the best candidate
    return candidates[0][0].squeeze()

def translate(sentence: str, method='greedy', beam_size=5):
    """
    Translate an input sentence using the trained model.
    Args:
        sentence (str or int): Input sentence or index for test set lookup.
        method (str): 'greedy' or 'beam'.
        beam_size (int): Beam width for beam search.
    Returns:
        str: Translated sentence.
    """
    config = get_config()
    tokenizer_src = Tokenizer.from_file(str(Path(config['tokenizer_path'].format(config['src_lang']))))
    tokenizer_trg = Tokenizer.from_file(str(Path(config['tokenizer_path'].format(config['trg_lang']))))

    model = build_transformer(tokenizer_src.get_vocab_size(), tokenizer_trg.get_vocab_size(), config['seq_len'], config['seq_len']).to(device)

    # load the pre_trained weights
    model_filename = get_weights_file_path(get_config(), f"29")
    state = torch.load(model_filename)
    model.load_state_dict(state['model_state_dict'])

    # if the sentence is a number use it as an index to the test set
    label = ""
    if isinstance(sentence, int) or (isinstance(sentence, str) and sentence.isdigit()):
        id = int(sentence)
        ds = load_dataset(f"{config['datasource']}", f"{config['src_lang']}_{config['trg_lang']}", split='all')
        ds = BilingualDataset(ds, tokenizer_src, tokenizer_trg, config['src_lang'], config['trg_lang'], config['seq_len'])
        sentence = ds[id]['src_text']
        label = ds[id]['trg_text']
    seq_len = config['seq_len']

    # Handle empty input
    if not sentence or not isinstance(sentence, str):
        print("Empty or invalid input.")
        return ""

    # Tokenize and pad
    src_ids = tokenizer_src.encode(sentence).ids
    pad_len = seq_len - len(src_ids) - 2
    if pad_len < 0:
        src_ids = src_ids[:seq_len-2]
        pad_len = 0
    source = torch.cat([
        torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64),
        torch.tensor(src_ids, dtype=torch.int64),
        torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64),
        torch.tensor([tokenizer_src.token_to_id('[PAD]')] * pad_len, dtype=torch.int64)
    ], dim=0).to(device)
    source = source.unsqueeze(0)  # Add batch dimension
    source_mask = (source != tokenizer_src.token_to_id('[PAD]')).unsqueeze(1).unsqueeze(2).int().to(device)

    model.eval()
    with torch.no_grad():
        if label != "": print(f"{f'ID: ':>12}{id}")
        print(f"{f'SOURCE: ':>12}{sentence}")
        if label != "": print(f"{f'TARGET: ':>12}{label}")
        print(f"{f'PREDICTED: ':>12}", end='')

        if method == 'greedy':
            output_ids = greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_trg, seq_len, device)
            out_ids = output_ids[0].tolist() if output_ids.dim() > 1 else output_ids.tolist()
            for token_id in out_ids[1:]:  # skip SOS
                print(f"{tokenizer_trg.decode([token_id])}", end=' ')
                if token_id == tokenizer_trg.token_to_id('[EOS]'):
                    break
        elif method == 'beam':
            output_ids = beam_search_decode(model, beam_size, source, source_mask, tokenizer_src, tokenizer_trg, seq_len, device)
            out_ids = output_ids.tolist()
            for token_id in out_ids[1:]:  # skip SOS
                print(f"{tokenizer_trg.decode([token_id])}", end=' ')
                if token_id == tokenizer_trg.token_to_id('[EOS]'):
                    break
        else:
            raise ValueError(f"Unknown decoding method: {method}")

        print()  # for newline after prediction
        # Return the full decoded sentence (excluding SOS, PAD, EOS)
        out_ids = output_ids[0].tolist() if hasattr(output_ids, 'dim') and output_ids.dim() > 1 else output_ids.tolist()
        out_ids = [i for i in out_ids if i not in [tokenizer_trg.token_to_id('[SOS]'), tokenizer_trg.token_to_id('[PAD]')]]
        if tokenizer_trg.token_to_id('[EOS]') in out_ids:
            out_ids = out_ids[:out_ids.index(tokenizer_trg.token_to_id('[EOS]'))]
        return tokenizer_trg.decode(out_ids)

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

if __name__ == "__main__":
    print("Translate English to Colloquial Hindi using Transformer model.")
    sentence = input("Enter a sentence to translate (or index for test set lookup): ").strip()
    if not sentence:
        print("Usage: Run the script and enter a sentence or index when prompted.")
    else:
        method = input("Decoding method? (greedy/beam) [greedy]: ").strip().lower() or 'greedy'
        if method == 'beam':
            try:
                beam_size = int(input("Beam size? [5]: ").strip() or 5)
            except ValueError:
                beam_size = 5
        else:
            beam_size = 5
        translation = translate(sentence, method=method, beam_size=beam_size)
        print(f"\nFinal Translation: {translation}")
