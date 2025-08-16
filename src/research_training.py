#!/usr/bin/env python3
"""
Research Training Script for Custom Bangla-to-Code Model
Trains the novel architecture from scratch for research purposes.
"""

import argparse
import json
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    get_linear_schedule_with_warmup,
    set_seed,
    logging
)
from tqdm import tqdm
import wandb
from typing import List, Dict, Any

from src.research_model import BanglaCodeResearchModel, BanglaCodeResearchConfig
from src.data_utils import load_dataset, BlpItem
from src.prompting import build_prompt


class ResearchDataset(Dataset):
    """Dataset for research model training"""
    
    def __init__(self, data: List[BlpItem], tokenizer, max_length: int = 2048):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Build prompt
        prompt = build_prompt(item.instruction)
        
        # Combine prompt and response
        full_text = prompt + "\n" + item.response
        
        # Tokenize
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Create instruction mask (first 30% of tokens)
        seq_length = encoding["input_ids"].shape[1]
        instruction_length = int(seq_length * 0.3)
        instruction_mask = torch.zeros(seq_length, dtype=torch.bool)
        instruction_mask[:instruction_length] = True
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "instruction_mask": instruction_mask,
            "labels": encoding["input_ids"].squeeze(0).clone()
        }


class ResearchTrainer:
    """Custom trainer for research model"""
    
    def __init__(self, model, tokenizer, args):
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Move model to device
        self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        
        # Setup scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=args.num_train_steps
        )
        
        # Setup wandb
        if args.use_wandb:
            wandb.init(
                project="bangla-code-research",
                name=args.experiment_name,
                config=vars(args)
            )
    
    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc="Training")
        
        for batch in progress_bar:
            try:
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                instruction_mask = batch["instruction_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    instruction_mask=instruction_mask,
                    labels=labels
                )
                
                loss = outputs["loss"]
                
                # Check for NaN loss
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: NaN/Inf loss detected: {loss.item()}")
                    continue
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping (more aggressive)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                # Check for NaN gradients
                has_nan_grad = False
                for param in self.model.parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            has_nan_grad = True
                            break
                
                if has_nan_grad:
                    print("Warning: NaN gradients detected, skipping batch")
                    continue
                
                self.optimizer.step()
                self.scheduler.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
                
                # Log to wandb
                if self.args.use_wandb:
                    wandb.log({
                        "train_loss": loss.item(),
                        "learning_rate": self.scheduler.get_last_lr()[0]
                    })
                    
            except Exception as e:
                print(f"Error in batch: {e}")
                continue
        
        return total_loss / max(num_batches, 1)
    
    def evaluate(self, dataloader):
        """Evaluate the model"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                instruction_mask = batch["instruction_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    instruction_mask=instruction_mask,
                    labels=labels
                )
                
                total_loss += outputs["loss"].item()
        
        return total_loss / len(dataloader)
    
    def save_checkpoint(self, epoch, loss, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "loss": loss,
            "config": self.model.config
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.args.output_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.args.output_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
    
    def train(self, train_dataloader, eval_dataloader=None):
        """Main training loop"""
        best_loss = float('inf')
        
        for epoch in range(self.args.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.args.num_epochs}")
            
            # Training
            train_loss = self.train_epoch(train_dataloader)
            print(f"Training Loss: {train_loss:.4f}")
            
            # Evaluation
            if eval_dataloader:
                eval_loss = self.evaluate(eval_dataloader)
                print(f"Evaluation Loss: {eval_loss:.4f}")
                
                # Log to wandb
                if self.args.use_wandb:
                    wandb.log({
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "eval_loss": eval_loss
                    })
                
                # Save best model
                is_best = eval_loss < best_loss
                if is_best:
                    best_loss = eval_loss
                
                self.save_checkpoint(epoch, eval_loss, is_best)
            else:
                self.save_checkpoint(epoch, train_loss)
            
            # Save latest model
            self.model.save_pretrained(os.path.join(self.args.output_dir, "latest_model"))
            self.tokenizer.save_pretrained(os.path.join(self.args.output_dir, "latest_model"))


def create_tokenizer():
    """Create a custom tokenizer for Bangla and code"""
    from transformers import PreTrainedTokenizerFast
    
    # For research, we'll use a simple tokenizer
    # In practice, you'd train a custom tokenizer on Bangla+code data
    tokenizer = PreTrainedTokenizerFast.from_pretrained("bigcode/tiny_starcoder_py")
    
    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Add Bangla-specific tokens
    bangla_tokens = [
        "নির্দেশনা", "স্ক্রিপ্ট", "ফাংশন", "অ্যারে", "তালিকা", 
        "স্ট্রিং", "সংখ্যা", "গণনা", "যোগফল", "সর্বোচ্চ", "সর্বনিম্ন",
        "প্রদত্ত", "লিখুন", "করে", "একটি", "দুটি", "তিনটি"
    ]
    
    tokenizer.add_tokens(bangla_tokens)
    
    return tokenizer


def main():
    parser = argparse.ArgumentParser(description="Train custom Bangla-to-Code research model")
    
    # Data arguments
    parser.add_argument("--train_json", type=str, required=True, help="Path to training JSON")
    parser.add_argument("--eval_json", type=str, help="Path to evaluation JSON")
    
    # Model arguments
    parser.add_argument("--model_size", type=str, default="small", 
                       choices=["tiny", "small", "medium", "large"], 
                       help="Model size configuration")
    parser.add_argument("--vocab_size", type=int, default=50000, help="Vocabulary size")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum sequence length")
    
    # Training arguments
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm")
    
    # Research-specific arguments
    parser.add_argument("--use_dual_stream", action="store_true", help="Use dual-stream architecture")
    parser.add_argument("--use_code_aware_attention", action="store_true", help="Use code-aware attention")
    parser.add_argument("--experiment_name", type=str, default="bangla-code-research", help="Experiment name")
    parser.add_argument("--use_wandb", action="store_true", help="Use wandb logging")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="outputs/research_model", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    logging.set_verbosity_info()
    
    # Load data
    print("Loading dataset...")
    train_data = load_dataset(args.train_json)
    eval_data = load_dataset(args.eval_json) if args.eval_json else None
    
    print(f"Training samples: {len(train_data)}")
    if eval_data:
        print(f"Evaluation samples: {len(eval_data)}")
    
    # Create tokenizer
    print("Creating tokenizer...")
    tokenizer = create_tokenizer()
    
    # Model size configurations
    model_configs = {
        "tiny": {"hidden_size": 256, "num_hidden_layers": 6, "num_attention_heads": 8},
        "small": {"hidden_size": 512, "num_hidden_layers": 8, "num_attention_heads": 8},
        "medium": {"hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12},
        "large": {"hidden_size": 1024, "num_hidden_layers": 16, "num_attention_heads": 16}
    }
    
    config_params = model_configs[args.model_size]
    
    # Create model configuration
    config = BanglaCodeResearchConfig(
        vocab_size=len(tokenizer),
        hidden_size=config_params["hidden_size"],
        num_hidden_layers=config_params["num_hidden_layers"],
        num_attention_heads=config_params["num_attention_heads"],
        intermediate_size=config_params["hidden_size"] * 4,
        max_position_embeddings=args.max_length,
        use_dual_stream=False,  # Disable dual-stream for stability
        use_code_aware_attention=False  # Disable code-aware attention for stability
    )
    
    # Create model
    print(f"Creating {args.model_size} model...")
    model = BanglaCodeResearchModel(config)
    
    # Calculate number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create datasets
    train_dataset = ResearchDataset(train_data, tokenizer, args.max_length)
    eval_dataset = ResearchDataset(eval_data, tokenizer, args.max_length) if eval_data else None
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2
    ) if eval_dataset else None
    
    # Calculate training steps
    args.num_train_steps = len(train_dataloader) * args.num_epochs
    
    # Create trainer
    trainer = ResearchTrainer(model, tokenizer, args)
    
    # Train
    print("Starting training...")
    trainer.train(train_dataloader, eval_dataloader)
    
    print(f"Training completed! Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
