# generator_model.py
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.device)

class GeneratorModel(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=4, num_layers=2, max_length=64, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_length)
        decoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        # input_ids: (batch, seq)
        x = self.token_emb(input_ids) * math.sqrt(self.token_emb.embedding_dim)
        x = self.pos_enc(x)

        # Transformer expects (seq, batch, dim)
        x = x.permute(1,0,2)

        # 1. Padding Mask (Existing logic)
        src_key_padding_mask = None
        if attention_mask is not None:
            # In PyTorch, True means "ignore this token" (padding)
            src_key_padding_mask = attention_mask == 0

        # 2. Causal Mask (CRITICAL FIX)
        # Prevents the model from seeing future tokens.
        seq_len = x.size(0)
        # Create a square matrix where upper triangle is -inf
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device) * float('-inf'), diagonal=1)

        # Pass both masks to the transformer
        out = self.transformer(x, mask=causal_mask, src_key_padding_mask=src_key_padding_mask)

        out = out.permute(1,0,2)  # back to (batch, seq, dim)
        out = self.ln(out)
        logits = self.head(out)
        return logits
