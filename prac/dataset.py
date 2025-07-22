import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer_src, tokenizer_trg, src_lang, trg_lang, seq_len):
        # Dataset: list of dictionaries like {'en': 'I love you', 'hin': 'main tumse pyar karta hun'}
        self.ds = dataset

        # Tokenizers for source and target language
        self.tokenizer_src = tokenizer_src
        self.tokenizer_trg = tokenizer_trg

        # Language codes like 'en', 'hin' etc.
        self.src_lang = src_lang
        self.trg_lang = trg_lang

        # Maximum allowed sequence length
        self.seq_len = seq_len

        # Convert special tokens to tensor form using tokenizer
        self.sos_token = self.token_id('[SOS]')  # Start of sequence
        self.eos_token = self.token_id('[EOS]')  # End of sequence
        self.pad_token = self.token_id('[PAD]')  # Padding token

    def token_id(self, token):
        # Convert a special token string (e.g., '[SOS]') to a tensor([id])
        return torch.tensor([self.tokenizer_src.token_to_id(token)], dtype=torch.int64)

    def encode_text(self, text, tokenizer):
        # Tokenize a text string and return a list of token IDs
        return tokenizer.encode(text).ids

    def pad_sequence(self, tokens, pad_count, add_sos=False, add_eos=False):
        """
        Create a padded tensor sequence with optional SOS and EOS.
        Example:
        tokens = [5, 9, 12], add_sos=True, add_eos=True, pad_count=2
        → [SOS, 5, 9, 12, EOS, PAD, PAD]
        """
        sequence = []

        if add_sos:
            sequence += [self.sos_token.item()]  # Add SOS token

        sequence += tokens  # Add actual tokenized sentence

        if add_eos:
            sequence += [self.eos_token.item()]  # Add EOS token

        sequence += [self.pad_token.item()] * pad_count  # Add PAD tokens to match seq_len

        return torch.tensor(sequence, dtype=torch.int64)

    def create_mask(self, input_tensor, causal=False):
        """
        Create an attention mask:
        - 1 where input ≠ PAD (real tokens)
        - 0 where input == PAD (ignored)
        If causal=True, also apply a causal mask (prevents decoder from seeing future tokens)
        """
        # Shape: (1, 1, seq_len) — broadcasted for multi-head attention
        pad_mask = (input_tensor != self.pad_token).unsqueeze(0).unsqueeze(0).int()

        if causal:
            size = input_tensor.size(0)
            return pad_mask & causal_mask(size)  # Combine pad + causal mask

        return pad_mask  # Only padding mask for encoder

    def __getitem__(self, index):
        # Get a source-target text pair at index
        pair = self.ds[index]
        src_text = pair[self.src_lang]  # e.g., English sentence
        trg_text = pair[self.trg_lang]  # e.g., hindi sentence

        # Tokenize source and target sentences
        enc_tokens = self.encode_text(src_text, self.tokenizer_src)
        dec_tokens = self.encode_text(trg_text, self.tokenizer_trg)

        # Calculate number of [PAD] tokens to add
        enc_pad = self.seq_len - len(enc_tokens) - 2  # [SOS] + [EOS] → 2 tokens
        dec_pad = self.seq_len - len(dec_tokens) - 1  # [SOS] only

        # If sentence is too long, throw an error
        if enc_pad < 0 or dec_pad < 0:
            raise ValueError("Input sequence too long")

        # Build the final encoder input: [SOS] + tokens + [EOS] + [PADs]
        encoder_input = self.pad_sequence(enc_tokens, enc_pad, add_sos=True, add_eos=True)

        # Build the decoder input: [SOS] + tokens + [PADs]
        decoder_input = self.pad_sequence(dec_tokens, dec_pad, add_sos=True)

        # Build the label (target output): tokens + [EOS] + [PADs]
        label = self.pad_sequence(dec_tokens, dec_pad, add_eos=True)

        # Return dictionary of all needed inputs for the model
        return {
            "encoder_input": encoder_input,  # (seq_len,)
            "decoder_input": decoder_input,  # (seq_len,)
            "encoder_mask": self.create_mask(encoder_input),  # (1, 1, seq_len)
            "decoder_mask": self.create_mask(decoder_input, causal=True),  # (1, seq_len, seq_len)
            "label": label,  # (seq_len,)
            "src_text": src_text,  # original source string (optional for debugging)
            "trg_text": trg_text   # original target string
        }

    def __len__(self):
        # Return number of samples in dataset
        return len(self.ds)


def causal_mask(size):
    """
    Create a causal attention mask of shape (1, size, size),
    where position i can only attend to ≤ i (no future tokens).
    """
    # Create upper-triangle of 1s above diagonal (i < j)
    # example: So mask[i][j] = 1 means block attention, and 0 means allow attention.
    # i\j |  0  1  2  3
    # ----------------
    # 0  |  0  1  1  1
    # 1  |  0  0  1  1
    # 2  |  0  0  0  1
    # 3  |  0  0  0  0

    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0  # Flip 1s to 0s and 0s to 1s (causal mask)
    # [[1, 0, 0, 0],
    #  [1, 1, 0, 0],
    #  [1, 1, 1, 0],
    #  [1, 1, 1, 1]]
