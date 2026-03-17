"""async trainer for blt model"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import asyncio
from typing import Optional, Dict, List
from tqdm import tqdm
import os

from ..models.blt_model import ByteLatentTransformer
from .data_loader import ByteDataLoader


class AsyncBLTTrainer:
    """
    asynchronous trainer for byte latent transformer.
    supports async data loading and gradient accumulation.
    """
    
    def __init__(
        self,
        model: ByteLatentTransformer,
        train_loader: ByteDataLoader,
        val_loader: Optional[ByteDataLoader] = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        max_steps: int = 100000,
        gradient_accumulation_steps: int = 1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_dir: str = "./checkpoints",
        log_interval: int = 100,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.log_interval = log_interval
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # optimizer and scheduler
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max_steps,
        )
        
        # loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # training state
        self.step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        # create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def compute_loss(
        self,
        byte_sequence: torch.Tensor,
    ) -> torch.Tensor:
        """
        compute next-byte prediction loss.
        
        args:
            byte_sequence: [seq_len] byte sequence
        returns:
            loss: scalar loss value
        """
        # move to device
        byte_sequence = byte_sequence.to(self.device)
        
        # forward pass
        logits, patch_sizes = self.model(byte_sequence)
        
        # create targets - predict next byte for each patch
        # for simplicity, we predict the first byte of next patch
        targets = []
        pos = 0
        for i, patch_size in enumerate(patch_sizes[:-1]):
            pos += patch_size
            if pos < len(byte_sequence):
                targets.append(byte_sequence[pos])
        
        if len(targets) == 0:
            return torch.tensor(0.0, device=self.device)
        
        targets = torch.stack(targets)
        
        # compute loss
        logits = logits[:len(targets)]
        loss = self.criterion(logits, targets)
        
        return loss
    
    async def train_step_async(
        self,
        batch: List[torch.Tensor],
    ) -> float:
        """
        async training step for a batch.
        
        args:
            batch: list of byte sequences
        returns:
            average loss for batch
        """
        self.model.train()
        
        total_loss = 0.0
        num_sequences = len(batch)
        
        for i, byte_sequence in enumerate(batch):
            # compute loss
            loss = self.compute_loss(byte_sequence)
            
            # scale loss for gradient accumulation
            loss = loss / self.gradient_accumulation_steps
            
            # backward pass
            loss.backward()
            
            total_loss += loss.item() * self.gradient_accumulation_steps
            
            # update weights
            if (i + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.step += 1
        
        # final update if needed
        if num_sequences % self.gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            self.step += 1
        
        return total_loss / num_sequences
    
    def train_step_sync(
        self,
        batch: List[torch.Tensor],
    ) -> float:
        """
        synchronous wrapper for async training step.
        
        args:
            batch: list of byte sequences
        returns:
            average loss for batch
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.train_step_async(batch))
    
    @torch.no_grad()
    def validate(self) -> float:
        """
        run validation.
        
        returns:
            average validation loss
        """
        if self.val_loader is None:
            return 0.0
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.val_loader:
            for byte_sequence in batch:
                loss = self.compute_loss(byte_sequence)
                total_loss += loss.item()
            num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def save_checkpoint(self, path: Optional[str] = None):
        """save model checkpoint"""
        if path is None:
            path = os.path.join(
                self.checkpoint_dir,
                f"checkpoint_step_{self.step}.pt"
            )
        
        torch.save({
            'step': self.step,
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
        }, path)
        
        print(f"[PASS] checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.step = checkpoint['step']
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"[PASS] checkpoint loaded from {path}")
    
    def train(self, num_epochs: int):
        """
        main training loop.
        
        args:
            num_epochs: number of epochs to train
        """
        print(f"[PASS] starting training for {num_epochs} epochs")
        print(f"[PASS] device: {self.device}")
        print(f"[PASS] model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # training
            pbar = tqdm(self.train_loader, desc=f"epoch {epoch+1}/{num_epochs}")
            for batch in pbar:
                loss = self.train_step_sync(batch)
                
                # logging
                if self.step % self.log_interval == 0:
                    pbar.set_postfix({'loss': f'{loss:.4f}', 'step': self.step})
            
            # validation
            if self.val_loader is not None:
                val_loss = self.validate()
                print(f"[PASS] epoch {epoch+1} - val_loss: {val_loss:.4f}")
                
                # save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(
                        os.path.join(self.checkpoint_dir, "best_model.pt")
                    )
            
            # save periodic checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint()
        
        print("[PASS] training completed")
