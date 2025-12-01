# generator_model.py
import torch.nn as nn
from transformers import GPT2LMHeadModel

class GeneratorModel(nn.Module):
    def __init__(self, vocab_size=None, d_model=None, nhead=None, num_layers=None, max_length=None, dropout=None):
        super().__init__()
        # Ignore custom params, load pre-trained GPT-2
        print("[LOG] Loading pre-trained GPT-2 weights (Fine-Tuning Mode)...")
        self.gpt = GPT2LMHeadModel.from_pretrained("gpt2")
        self.max_length = max_length

    def forward(self, input_ids, attention_mask=None):
        outputs = self.gpt(input_ids, attention_mask=attention_mask)
        return outputs.logits
