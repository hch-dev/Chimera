ATTACKER AI - GAN Email Generator
=================================

This folder contains the components needed to train and run the
Generator model used for phishing email generation.

FILES
-----
1. tokenizer.json
   - Defines the special tokens used by the GPT-2 tokenizer
   - Actual vocabulary is loaded automatically from the GPT-2 tokenizer

2. train_gan.py
   - Loads dataset RonakAJ/phising_email from Hugging Face
   - Tokenizes the 'Email Text' column using GPT-2 tokenizer
   - Trains a Generator (small GPT-like transformer)
   - Trains a dummy Discriminator for GAN feedback
   - Final generator weights saved as:
        models/generator_final.pt

3. generator_model.py
   - Defines the transformer-based GeneratorModel
   - Supports teacher forcing during training
   - Supports autoregressive sampling during inference

4. inference.py
   - Loads tokenizer.json
   - Loads models/generator_final.pt
   - Generates sample phishing-style emails

DATASET
-------
The dataset must be fetched automatically from HuggingFace:
    RonakAJ/phising_email

The dataset contains:
  - "Unnamed"       : Index column
  - "Email Text"    : Raw email body
  - "Email Type"    : Safe Email / Phishing Email

NOTES
-----
- Maximum model input length = 64 tokens
- Use inference.py to generate emails
- Model will NOT work unless:
      models/generator_final.pt
      tokenizer.json
  are present in this directory.

