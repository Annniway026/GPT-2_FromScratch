"""
Training script for GPT2ForSequenceClassification on 20 Newsgroups dataset.

This script trains a GPT-2 based classifier without using any HuggingFace libraries.
"""

import argparse
import json
import os
import time
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from gpt2 import GPT2Config, GPT2ForSequenceClassification


# =========================
# Dataset
# =========================
class NewsDataset(Dataset):
    def __init__(self, path: str, max_length: int):
        self.samples = []
        self.max_length = max_length
 
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                input_ids = obj["token_ids"]
                label = obj["label"]
 
                # 截断
                input_ids = input_ids[:max_length]
                self.samples.append({
                    "input_ids": input_ids,
                    "label": label
                })
 
    def __len__(self):
        return len(self.samples)
 
    def __getitem__(self, idx):
        return self.samples[idx]
 
 
def collate_fn(batch):
    max_len = max(len(x["input_ids"]) for x in batch)
 
    input_ids = []
    labels = []
 
    for item in batch:
        ids = item["input_ids"]
        pad_len = max_len - len(ids)
        # 左 padding：让最后一个 token 始终是真实 token，
        # 这样 GPT2 用 last-token pooling 时取到的是有意义的位置
        input_ids.append([0] * pad_len + ids)
        labels.append(item["label"])
 
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels":    torch.tensor(labels,    dtype=torch.long),
    }
 
 
# =========================
# Evaluate
# =========================
@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
 
    total = 0
    correct = 0
 
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels    = batch["labels"].to(device)
 
        # forward() 返回 SequenceClassifierOutput，取 .logits
        output = model(input_ids)
        logits = output.logits          # (B, num_classes)
        preds  = logits.argmax(dim=-1)
 
        total   += labels.size(0)
        correct += (preds == labels).sum().item()
 
    return correct / total
 
 
# =========================
# Train Loop
# =========================
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    # ===== 超参数 =====
    batch_size  = 8
    lr          = 3e-4
    epochs      = 5
    max_length  = 256
    num_classes = 20
 
    # ===== 数据 =====
    train_dataset = NewsDataset("data/20_newsgroups_train.jsonl", max_length)
    val_dataset   = NewsDataset("data/20_newsgroups_val.jsonl",   max_length)
 
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
 
    # ===== 模型 =====
    config = GPT2Config(num_labels=num_classes)
    model  = GPT2ForSequenceClassification(config=config).to(device)
 
    # ===== 优化器 =====
    # 对 backbone 和 classifier head 使用不同学习率
    backbone_params   = list(model.transformer.parameters())
    classifier_params = list(model.classifier.parameters())
    optimizer = torch.optim.AdamW([
        {"params": backbone_params,   "lr": lr * 0.1},   # backbone 用更小的 lr
        {"params": classifier_params, "lr": lr},
    ])
 
    criterion = nn.CrossEntropyLoss()
 
    # ===== TensorBoard =====
    writer = SummaryWriter(log_dir="runs/gpt2_classifier")
 
    global_step = 0
    best_acc    = 0.0
 
    # =========================
    # Training Loop
    # =========================
    for epoch in range(epochs):
        model.train()
 
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels    = batch["labels"].to(device)
 
            # forward() 返回 SequenceClassifierOutput
            output = model(input_ids)
            logits = output.logits      # (B, num_classes)
 
            loss = criterion(logits, labels)
 
            optimizer.zero_grad()
            loss.backward()
            # gradient clipping，防止梯度爆炸
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
 
            writer.add_scalar("train/loss", loss.item(), global_step)
 
            if global_step % 100 == 0:
                print(f"Epoch {epoch}  Step {global_step}  Loss {loss.item():.4f}")
 
            global_step += 1
 
        # ===== 验证 =====
        val_acc = evaluate(model, val_loader, device)
        writer.add_scalar("val/accuracy", val_acc, epoch)
        print(f"[Epoch {epoch}] Validation Accuracy: {val_acc:.4f}")
 
        # ===== 保存最优模型 =====
        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs("checkpoints", exist_ok=True)
            save_path = "checkpoints/classifier_model.pth"
            torch.save(model.state_dict(), save_path)
            print(f"  -> Saved best model to {save_path}  (acc={best_acc:.4f})")
 
    writer.close()
    print(f"\nBest Validation Accuracy: {best_acc:.4f}")
 
 
# =========================
# Checkpoint Load Test
# =========================
def load_and_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    config = GPT2Config(num_labels=20)
    model  = GPT2ForSequenceClassification(config=config).to(device)
 
    ckpt_path = "checkpoints/classifier_model.pth"
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    print("Checkpoint loaded successfully.")
 
 
if __name__ == "__main__":
    # TODO: implement the training loop for GPT2ForSequenceClassification on the 20 Newsgroups dataset.
    # You can use any techniques or implementations you like.
    train()
    load_and_test()




