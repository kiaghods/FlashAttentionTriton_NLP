import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from transformer_vanilla import VanillaTransformer
from transformer_flashattention import FlashTransformerDecoderOnly
import math
import time
import csv

# -------------------------------
# Configuration
# -------------------------------
BATCH_SIZE = 16
SEQ_LEN = 256
EPOCHS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VOCAB_MODEL = "gpt2"

# -------------------------------
# Load Dataset and Tokenizer
# -------------------------------
print("Loading dataset and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(VOCAB_MODEL, local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token


def tokenize(example):
    return tokenizer(example["text"], return_tensors=None)

# Train set
dataset = load_dataset("wikitext", "wikitext-103-v1", split="train[:10%]")
tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
all_ids = [torch.tensor(x["input_ids"]) for x in tokenized]
long_tensor = torch.cat(all_ids)
chunks = long_tensor.unfold(0, SEQ_LEN + 1, SEQ_LEN)
inputs = chunks[:, :-1]
targets = chunks[:, 1:]
data = list(zip(inputs, targets))
loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)

# Validation set
val_dataset = load_dataset("wikitext", "wikitext-103-v1", split="validation")
val_tokenized = val_dataset.map(tokenize, batched=True, remove_columns=["text"])
val_ids = [torch.tensor(x["input_ids"]) for x in val_tokenized]
val_long_tensor = torch.cat(val_ids)
val_chunks = val_long_tensor.unfold(0, SEQ_LEN + 1, SEQ_LEN)
val_inputs = val_chunks[:, :-1]
val_targets = val_chunks[:, 1:]
val_data = list(zip(val_inputs, val_targets))
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)


# -------------------------------
# Shared Training & Eval Logic
# -------------------------------
def train_and_evaluate(model, name):
    print(f"\n===== Running {name} =====")
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    epoch_metrics = []

    print("Starting training...")
    train_start = time.time()
    epoch_times = []
    model.train()

    for epoch in range(EPOCHS):
        epoch_start = time.time()
        epoch_loss = 0
        total_steps = 0

        for step, (x, y) in enumerate(loader):
            x, y = x.to(DEVICE).long(), y.to(DEVICE).long()
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, tokenizer.vocab_size), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            total_steps += 1

            if step % 10 == 0:
                print(f"[{name}] Epoch {epoch} Step {step}: Loss = {loss.item():.4f}")

        epoch_duration = time.time() - epoch_start
        epoch_times.append(epoch_duration)
        avg_epoch_loss = epoch_loss / total_steps
        print(f"[{name}] Epoch {epoch} completed in {epoch_duration:.2f}s with Avg Loss = {avg_epoch_loss:.4f}")

        # Train perplexity
        model.eval()
        total_loss, total_tokens = 0, 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(DEVICE).long(), y.to(DEVICE).long()
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, tokenizer.vocab_size), y.view(-1), reduction="sum")
                total_loss += loss.item()
                total_tokens += y.numel()
        train_perplexity = math.exp(total_loss / total_tokens)
        print(f"[{name}] Epoch {epoch} Train Perplexity: {train_perplexity:.2f}")

        # Validation perplexity
        total_loss, total_tokens = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE).long(), y.to(DEVICE).long()
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, tokenizer.vocab_size), y.view(-1), reduction="sum")
                total_loss += loss.item()
                total_tokens += y.numel()
        val_perplexity = math.exp(total_loss / total_tokens)
        print(f"[{name}] Epoch {epoch} Val Perplexity: {val_perplexity:.2f}")
        model.train()

        epoch_metrics.append({
            "epoch": epoch,
            "loss": avg_epoch_loss,
            "train_perplexity": train_perplexity,
            "val_perplexity": val_perplexity,
            "time": epoch_duration
        })

    total_train_time = time.time() - train_start

    # Save metrics to CSV
    csv_filename = f"{name}_metrics.csv"
    with open(csv_filename, mode="w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["epoch", "loss", "train_perplexity", "val_perplexity", "time"])
        writer.writeheader()
        writer.writerows(epoch_metrics)
    print(f"Saved metrics to {csv_filename}")

    return total_train_time, epoch_times, epoch_metrics[-1]["time"], epoch_metrics[-1]["val_perplexity"]


# -------------------------------
# Run Vanilla Transformer
# -------------------------------
vanilla_model = VanillaTransformer(
    vocab_size=tokenizer.vocab_size,
    embed_dim=512,
    num_heads=8,
    num_layers=12,
    max_len=1024,
    dropout=0.1,
    causal=True,
)

# -------------------------------
# Run Flash Transformer
# -------------------------------
flash_model = FlashTransformerDecoderOnly(
    vocab_size=tokenizer.vocab_size,
    embed_dim=512,
    depth=12,
    n_heads=8,
    max_seq_len=1024,
    dropout=0.1,
)
flash_train_time, flash_epoch_times, flash_eval_time, flash_ppl = train_and_evaluate(flash_model, "FlashTransformer")
vanilla_train_time, vanilla_epoch_times, vanilla_eval_time, vanilla_ppl = train_and_evaluate(vanilla_model, "VanillaTransformer")


# -------------------------------
# Final Summary
# -------------------------------
print("\n========== SUMMARY ==========")
print(f"{'Model':<20} {'Train Time (s)':>15} {'Eval Time (s)':>15} {'Val Perplexity':>18}")
print("-" * 70)
print(f"{'Vanilla':<20} {vanilla_train_time:>15.2f} {vanilla_eval_time:>15.2f} {vanilla_ppl:>18.2f}")
print(f"{'Flash':<20} {flash_train_time:>15.2f} {flash_eval_time:>15.2f} {flash_ppl:>18.2f}")

print("\nEpoch-wise Timing:")
for i in range(EPOCHS):
    print(f"Epoch {i}: Vanilla = {vanilla_epoch_times[i]:.2f}s | Flash = {flash_epoch_times[i]:.2f}s")