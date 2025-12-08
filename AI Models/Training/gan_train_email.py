import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from transformers import AutoTokenizer, RobertaTokenizer
from generator_model import GeneratorModel
from configs import GeneratorConfig
from transformers import RobertaForSequenceClassification
from torch.utils.data import DataLoader
from datetime import datetime
import random

# ------------------------------------------------------------
#               GAN Training Config
# ------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GEN_LR = 2e-5
DISC_LR = 1e-5
BATCH_SIZE = 4
EPOCHS = 1                 # One run = 30–40 min
MAX_STEPS = 400            # Hard cap for quick sessions

# Paths
GEN_CKPT = "Attacker_AI/Email/models_email/generator_interrupted.pt"
DISC_CKPT = "Defender_AI/Version_3/models/v3_phishing_roberta"

# ------------------------------------------------------------
#               Load Models
# ------------------------------------------------------------
print("[LOG] Loading generator...")
gen_tok = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
gen_tok.add_special_tokens({'pad_token': '<pad>'})

gcfg = GeneratorConfig()
generator = GeneratorModel(max_length=gcfg.max_length)
generator.gpt.resize_token_embeddings(len(gen_tok))

if os.path.exists(GEN_CKPT):
    generator.load_state_dict(torch.load(GEN_CKPT, map_location="cpu"))
generator.to(DEVICE)

print("[LOG] Loading discriminator...")
disc_tok = RobertaTokenizer.from_pretrained(DISC_CKPT)
discriminator = RobertaForSequenceClassification.from_pretrained(DISC_CKPT)
discriminator.to(DEVICE)

# ------------------------------------------------------------
#               Optimizers
# ------------------------------------------------------------
gen_opt = Adam(generator.parameters(), lr=GEN_LR)
disc_opt = Adam(discriminator.parameters(), lr=DISC_LR)

# ------------------------------------------------------------
#         Helper: sample + keep logprobs for REINFORCE
# ------------------------------------------------------------

def generate_with_logprobs(prompt="Subject: Urgent Action Required", max_new_tokens=128):
    generator.eval()
    input_ids = gen_tok.encode(prompt, return_tensors="pt").to(DEVICE)

    outputs = generator.gpt.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        return_dict_in_generate=True,
        output_scores=True
    )

    seq = outputs.sequences[0]
    scores = outputs.scores

    # Calculate logprobs of sampled tokens
    logprobs = []
    for i, score in enumerate(scores):
        probs = F.softmax(score[0], dim=-1)
        token_id = seq[i + 1]
        logprobs.append(torch.log(probs[token_id] + 1e-12))

    text = gen_tok.decode(seq, skip_special_tokens=True)
    return text, torch.stack(logprobs).sum()

# ------------------------------------------------------------
#         Helper: discriminator score
# ------------------------------------------------------------

def get_disc_prob(text):
    inputs = disc_tok(text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        logits = discriminator(**inputs).logits
        probs = F.softmax(logits, dim=1)
    return probs[0][1].item()     # probability of phishing

# ------------------------------------------------------------
#               Training Loop
# ------------------------------------------------------------
step = 0
print("\n🚀 Starting GAN training...\n")

for epoch in range(EPOCHS):
    while step < MAX_STEPS:

        # -------------------------------------
        # (1) Generate phishing samples
        # -------------------------------------
        fake_texts = []
        fake_logprobs = []

        for _ in range(BATCH_SIZE):
            txt, lp = generate_with_logprobs()
            fake_texts.append(txt)
            fake_logprobs.append(lp)

        # -------------------------------------
        # (2) Train Discriminator
        # -------------------------------------
        disc_opt.zero_grad()

        # Fake samples
        fake_scores = []
        for txt in fake_texts:
            prob = get_disc_prob(txt)
            fake_scores.append(prob)
        fake_scores = torch.tensor(fake_scores).to(DEVICE)

        # Real samples: reuse old V3 dataset
        # Load from Defender_AI/Version_3/data/
        real_path = "Defender_AI/Version_3/data/real_emails.txt"
        with open(real_path, "r", encoding="utf-8") as f:
            real_samples = random.sample(f.readlines(), BATCH_SIZE)

        real_scores = []
        for txt in real_samples:
            prob = get_disc_prob(txt)
            real_scores.append(prob)
        real_scores = torch.tensor(real_scores).to(DEVICE)

        # Discriminator non-saturating loss
        d_loss = -torch.mean(torch.log(real_scores + 1e-12) +
                             torch.log(1 - fake_scores + 1e-12))
        d_loss.backward()
        disc_opt.step()

        # -------------------------------------
        # (3) Train Generator via REINFORCE
        # -------------------------------------
        gen_opt.zero_grad()

        rewards = -torch.log(1 - fake_scores + 1e-12)
        rewards = rewards.detach()

        g_loss = 0
        for lp, r in zip(fake_logprobs, rewards):
            g_loss += -lp * r
        g_loss /= BATCH_SIZE

        g_loss.backward()
        gen_opt.step()

        step += 1

        print(f"[{step}/{MAX_STEPS}]  D_loss={d_loss.item():.4f}  G_loss={g_loss.item():.4f}")

        # Save every 50 steps
        if step % 50 == 0:
            torch.save(generator.state_dict(), GEN_CKPT)
            discriminator.save_pretrained(DISC_CKPT)
            print("Checkpoint saved.")

print("\n✅ Training finished! Models updated.\n")
