import sys
import os
import random
import torch
import torch.nn.functional as F
from torch.optim import Adam

# Add Chimera root
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

# Add attacker email directory
GEN_PATH = os.path.join(ROOT, "Attacker_AI", "Email")
sys.path.append(GEN_PATH)

from generator_model import GeneratorModel
from configs import GeneratorConfig
from transformers import AutoTokenizer, RobertaTokenizer, RobertaForSequenceClassification

# ------------------------------------------------------------
#               GAN Training Config
# ------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Minimal memory-safe settings (adjust only if you have a GPU with lots of RAM) ---
BATCH_SIZE = 1              # lowered to avoid OOM on CPU
MAX_NEW_TOKENS = 32         # limit generation length to control memory
DETACH_EVERY = 16           # detach prefix every N tokens to prune autograd graph

GEN_LR = 2e-5
DISC_LR = 1e-5
EPOCHS = 1
MAX_STEPS = 400

# Paths
GEN_CKPT = "Attacker_AI/Email/models_email/generator_interrupted.pt"

# FIXED: absolute path for discriminator
DISC_CKPT = os.path.abspath(
    os.path.join(ROOT, "Defender_AI", "Version_3", "models", "v3_phishing_roberta")
)

REAL_DATA_PATH = os.path.join(
    ROOT, "Defender_AI", "Version_3", "data", "phishing_data.csv"
)

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
def generate_with_logprobs(prompt="Subject: Urgent Action Required", max_new_tokens=MAX_NEW_TOKENS,
                           detach_every=DETACH_EVERY, temperature=1.0, top_k=50):
    """
    Memory-safe, differentiable token-by-token sampling using past_key_values caching.
    Periodically detaches the generated prefix to prevent unbounded autograd graph growth.
    Returns: (text, differentiable_scalar_logprob)
    """
    generator.train()   # ensure gradients flow for generator parameters

    # encode prompt
    input_ids = gen_tok.encode(prompt, return_tensors="pt").to(DEVICE)  # shape (1, seq_len)
    generated = input_ids.clone()  # will accumulate tokens for decoding
    batch_logprob = None

    past_key_values = None
    token_input_ids = None

    for step_idx in range(max_new_tokens):
        # Use cached past_key_values when available; only feed last token in that case
        inputs_for_forward = generated if past_key_values is None else token_input_ids

        outputs = generator.gpt(input_ids=inputs_for_forward, past_key_values=past_key_values,
                                use_cache=True, return_dict=True)
        logits = outputs.logits[:, -1, :]  # (1, vocab_size)
        past_key_values = outputs.past_key_values

        # apply temperature
        if temperature != 1.0:
            logits = logits / float(temperature)

        probs = F.softmax(logits, dim=-1)  # (1, vocab_size)

        # top-k filtering implemented on probs (simple approach)
        if top_k is not None and top_k > 0:
            topk_vals, topk_idx = torch.topk(probs, top_k, dim=-1)
            mask = torch.zeros_like(probs, dtype=torch.bool)
            mask.scatter_(1, topk_idx, True)
            filtered = probs * mask.float()
            probs = filtered / (filtered.sum(dim=-1, keepdim=True) + 1e-12)

        # sample token
        token_id = torch.multinomial(probs, num_samples=1)  # (1,1)

        # differentiable logprob of sampled token
        token_prob = probs.gather(1, token_id)  # (1,1)
        token_logprob = torch.log(token_prob + 1e-12).squeeze()  # scalar tensor, requires_grad=True

        if batch_logprob is None:
            batch_logprob = token_logprob
        else:
            batch_logprob = batch_logprob + token_logprob

        # prepare next step: only feed the sampled token when past_key_values present
        token_input_ids = token_id.to(DEVICE)  # (1,1)

        # append to generated for decoding and optional detach
        generated = torch.cat([generated, token_id], dim=1)

        # periodically detach the generated tensor to prune autograd graph
        if (step_idx + 1) % detach_every == 0:
            generated = generated.detach()
            # past_key_values remain as cached tensors inside the model outputs

    # decode the generated sequence (move to CPU)
    text = gen_tok.decode(generated[0].cpu().tolist(), skip_special_tokens=True)

    # batch_logprob is a scalar tensor on DEVICE and is differentiable
    return text, batch_logprob

# ------------------------------------------------------------
#         Helper: discriminator score
# ------------------------------------------------------------
def get_disc_prob(text, require_grad=False):
    inputs = disc_tok(text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)

    if require_grad:
        outputs = discriminator(**inputs)
        logits = outputs.logits
    else:
        with torch.no_grad():
            logits = discriminator(**inputs).logits

    probs = F.softmax(logits, dim=1)
    return probs[0][1]

# ------------------------------------------------------------
#               Load real samples
# ------------------------------------------------------------
with open(REAL_DATA_PATH, "r", encoding="utf-8") as f:
    all_real = f.readlines()

if len(all_real) < BATCH_SIZE:
    raise ValueError("Not enough real samples in dataset.")

# ------------------------------------------------------------
#               Training Loop
# ------------------------------------------------------------
step = 0
print("\n🚀 Starting GAN training...\n")

for epoch in range(EPOCHS):
    while step < MAX_STEPS:

        # ---------------------------
        # (1) Generate phishing samples
        # ---------------------------
        fake_texts = []
        fake_logprobs = []

        for _ in range(BATCH_SIZE):
            txt, lp = generate_with_logprobs()
            fake_texts.append(txt)
            fake_logprobs.append(lp)

        # ---------------------------
        # (2) Train Discriminator
        # ---------------------------
        disc_opt.zero_grad()

        # Fake samples scoring (require grad for discriminator params)
        fake_scores = []
        for txt in fake_texts:
            prob = get_disc_prob(txt, require_grad=True)
            fake_scores.append(prob)
        fake_scores = torch.stack(fake_scores).to(DEVICE)

        # Real samples scoring
        real_batch = random.sample(all_real, BATCH_SIZE)
        real_scores = []
        for txt in real_batch:
            prob = get_disc_prob(txt, require_grad=True)
            real_scores.append(prob)
        real_scores = torch.stack(real_scores).to(DEVICE)

        d_loss = -torch.mean(torch.log(real_scores + 1e-12) +
                             torch.log(1 - fake_scores + 1e-12))
        d_loss.backward()
        disc_opt.step()

        # ---------------------------
        # (3) Train Generator
        # ---------------------------
        gen_opt.zero_grad()

        rewards = -torch.log(1 - fake_scores + 1e-12)
        rewards = rewards.detach()

        g_loss = 0
        for lp, r in zip(fake_logprobs, rewards):
            g_loss = g_loss + (-lp * r)
        g_loss = g_loss / float(BATCH_SIZE)

        g_loss.backward()
        gen_opt.step()

        step += 1

        print(f"[{step}/{MAX_STEPS}]  D_loss={d_loss.item():.4f}  G_loss={g_loss.item():.4f}")

        # Save checkpoints
        if step % 50 == 0:
            torch.save(generator.state_dict(), GEN_CKPT)
            discriminator.save_pretrained(DISC_CKPT)
            print("Checkpoint saved.")

print("\n✅ Training finished! Models updated.\n")
