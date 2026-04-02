"""byte latent transformer main model"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List
from .byte_encoder import ByteEncoder
from .patch_processor import AsyncAttentionLayer


class ByteLatentTransformer(nn.Module):
    """
    byte latent transformer (blt) - processes raw bytes through
    dynamically-sized patches with entropy-based segmentation.
    """
    
    def __init__(
        self,
        d_model: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        d_ff: int = 2048,
        max_patch_size: int = 16,
        min_patch_size: int = 4,
        entropy_threshold: float = 0.5,
        dropout: float = 0.1,
        vocab_size: int = 256,
        use_async: bool = True,
        max_workers: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.use_async = use_async
        
        # byte encoder with entropy-based patching
        self.byte_encoder = ByteEncoder(
            d_model=d_model,
            max_patch_size=max_patch_size,
            min_patch_size=min_patch_size,
            entropy_threshold=entropy_threshold,
            vocab_size=vocab_size,
        )
        
        # positional encoding for patches
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        
        # transformer layers
        self.layers = nn.ModuleList([
            BLTTransformerLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout,
                use_async=use_async,
                max_workers=max_workers,
            )
            for _ in range(num_layers)
        ])
        
        # output projection to byte vocabulary
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # layer norm
        self.norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        byte_sequence: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        forward pass through blt.
        
        args:
            byte_sequence: [seq_len] tensor of byte values (0-255)
            mask: optional attention mask
        returns:
            logits: [num_patches, vocab_size] next byte predictions
            patch_sizes: list of patch sizes
        """
        # encode bytes into patches
        patch_embeddings, patch_sizes = self.byte_encoder(byte_sequence)
        
        # add batch dimension
        x = patch_embeddings.unsqueeze(0)  # [1, num_patches, d_model]
        
        # add positional encoding
        x = self.pos_encoding(x)
        
        # pass through transformer layers
        for layer in self.layers:
            x = layer(x, mask)
        
        # normalize
        x = self.norm(x)
        
        # project to vocabulary
        logits = self.output_projection(x)  # [1, num_patches, vocab_size]
        
        # remove batch dimension
        logits = logits.squeeze(0)  # [num_patches, vocab_size]
        
        return logits, patch_sizes
    
    def forward_batch(
        self,
        byte_sequences: List[torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[List[int]]]:
        """
        forward pass for batch of sequences.
        
        args:
            byte_sequences: list of [seq_len] tensors
            mask: optional attention mask
        returns:
            logits: [batch_size, max_patches, vocab_size]
            patch_sizes: list of list of patch sizes
        """
        # encode batch
        patch_embeddings, patch_sizes = self.byte_encoder.encode_batch(byte_sequences)
        
        # add positional encoding
        x = self.pos_encoding(patch_embeddings)
        
        # pass through transformer layers
        for layer in self.layers:
            x = layer(x, mask)
        
        # normalize
        x = self.norm(x)
        
        # project to vocabulary
        logits = self.output_projection(x)
        
        return logits, patch_sizes
    
    def generate(
        self,
        prompt: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        generate bytes autoregressively.
        
        args:
            prompt: [prompt_len] initial byte sequence
            max_length: maximum number of bytes to generate
            temperature: sampling temperature
        returns:
            generated: [prompt_len + max_length] generated sequence
        """
        generated = prompt.clone()
        
        for _ in range(max_length):
            # get predictions
            logits, _ = self.forward(generated)
            
            # take last patch prediction
            next_byte_logits = logits[-1] / temperature
            
            # sample next byte
            probs = torch.softmax(next_byte_logits, dim=-1)
            next_byte = torch.multinomial(probs, num_samples=1)
            
            # append to sequence
            generated = torch.cat([generated, next_byte], dim=0)
        
        return generated


class BLTTransformerLayer(nn.Module):
    """single transformer layer for blt"""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        use_async: bool = True,
        max_workers: int = 4,
    ):
        super().__init__()
        
        # multi-head attention
        self.attention = AsyncAttentionLayer(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            use_async=use_async,
            max_workers=max_workers,
        )
        
        # feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        
        # layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        forward pass with residual connections.
        
        args:
            x: [batch_size, seq_len, d_model]
            mask: optional attention mask
        returns:
            [batch_size, seq_len, d_model]
        """
        # attention with residual
        attn_output = self.attention(self.norm1(x), mask)
        x = x + self.dropout(attn_output)
        
        # feed-forward with residual
        ffn_output = self.ffn(self.norm2(x))
        x = x + ffn_output
        
        return x


class PositionalEncoding(nn.Module):
    """positional encoding for patch sequences"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # register as buffer (not a parameter)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        add positional encoding to input.
        
        args:
            x: [batch_size, seq_len, d_model]
        returns:
            [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        x = x + self.pe[:seq_len].unsqueeze(0)
        return self.dropout(x)
