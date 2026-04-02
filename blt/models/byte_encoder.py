"""byte encoder with entropy-based patching for blt"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional


class ByteEncoder(nn.Module):
    """
    encodes raw bytes into patches based on entropy of next byte prediction.
    patches are dynamically sized - more complex regions get more compute.
    """
    
    def __init__(
        self,
        d_model: int = 512,
        max_patch_size: int = 16,
        min_patch_size: int = 4,
        entropy_threshold: float = 0.5,
        vocab_size: int = 256,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_patch_size = max_patch_size
        self.min_patch_size = min_patch_size
        self.entropy_threshold = entropy_threshold
        self.vocab_size = vocab_size
        
        # byte embedding layer
        self.byte_embedding = nn.Embedding(vocab_size, d_model)
        
        # entropy predictor - predicts next byte distribution
        self.entropy_predictor = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, vocab_size),
        )
        
        # patch encoder - encodes variable-length patches
        self.patch_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=8,
                dim_feedforward=d_model * 4,
                batch_first=True,
            ),
            num_layers=2,
        )
        
    def compute_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """
        compute entropy of next byte prediction.
        higher entropy = more uncertainty = need more compute.
        
        args:
            logits: [batch_size, vocab_size]
        returns:
            entropy: [batch_size]
        """
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        # normalize to [0, 1]
        max_entropy = np.log(self.vocab_size)
        return entropy / max_entropy
    
    def segment_into_patches(
        self, 
        byte_sequence: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[int]]:
        """
        segment byte sequence into variable-length patches based on entropy.
        
        args:
            byte_sequence: [seq_len] tensor of byte values (0-255)
        returns:
            patches: list of patch tensors
            patch_sizes: list of patch sizes
        """
        patches = []
        patch_sizes = []
        
        i = 0
        seq_len = byte_sequence.size(0)
        
        while i < seq_len:
            # start new patch
            patch_start = i
            patch_bytes = []
            
            # grow patch until entropy threshold or max size
            while i < seq_len and len(patch_bytes) < self.max_patch_size:
                # embed current byte
                byte_val = byte_sequence[i:i+1]
                byte_emb = self.byte_embedding(byte_val)
                
                # predict next byte distribution
                logits = self.entropy_predictor(byte_emb)
                entropy = self.compute_entropy(logits)
                
                patch_bytes.append(byte_val)
                i += 1
                
                # check if we should end patch
                if len(patch_bytes) >= self.min_patch_size:
                    if entropy.item() < self.entropy_threshold:
                        # low entropy - simple region, can end patch
                        break
            
            # create patch tensor
            patch = torch.cat(patch_bytes, dim=0)
            patches.append(patch)
            patch_sizes.append(len(patch_bytes))
        
        return patches, patch_sizes
    
    def encode_patch(self, patch: torch.Tensor) -> torch.Tensor:
        """
        encode a single patch into a fixed-size representation.
        
        args:
            patch: [patch_len] tensor of byte values
        returns:
            patch_embedding: [d_model] tensor
        """
        # embed bytes
        byte_embeddings = self.byte_embedding(patch)  # [patch_len, d_model]
        
        # add batch dimension
        byte_embeddings = byte_embeddings.unsqueeze(0)  # [1, patch_len, d_model]
        
        # encode with transformer
        encoded = self.patch_encoder(byte_embeddings)  # [1, patch_len, d_model]
        
        # pool to single vector (mean pooling)
        patch_embedding = encoded.mean(dim=1).squeeze(0)  # [d_model]
        
        return patch_embedding
    
    def forward(self, byte_sequence: torch.Tensor) -> Tuple[torch.Tensor, List[int]]:
        """
        encode byte sequence into patch embeddings.
        
        args:
            byte_sequence: [seq_len] tensor of byte values (0-255)
        returns:
            patch_embeddings: [num_patches, d_model]
            patch_sizes: list of patch sizes
        """
        # segment into patches
        patches, patch_sizes = self.segment_into_patches(byte_sequence)
        
        # encode each patch
        patch_embeddings = []
        for patch in patches:
            patch_emb = self.encode_patch(patch)
            patch_embeddings.append(patch_emb)
        
        # stack into tensor
        patch_embeddings = torch.stack(patch_embeddings, dim=0)
        
        return patch_embeddings, patch_sizes
    
    def encode_batch(
        self, 
        byte_sequences: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, List[List[int]]]:
        """
        encode a batch of byte sequences.
        
        args:
            byte_sequences: list of [seq_len] tensors
        returns:
            patch_embeddings: [batch_size, max_patches, d_model] (padded)
            patch_sizes: list of list of patch sizes
        """
        all_embeddings = []
        all_patch_sizes = []
        
        for byte_seq in byte_sequences:
            embs, sizes = self.forward(byte_seq)
            all_embeddings.append(embs)
            all_patch_sizes.append(sizes)
        
        # pad to same number of patches
        max_patches = max(embs.size(0) for embs in all_embeddings)
        
        padded_embeddings = []
        for embs in all_embeddings:
            num_patches = embs.size(0)
            if num_patches < max_patches:
                padding = torch.zeros(
                    max_patches - num_patches, 
                    self.d_model,
                    device=embs.device,
                    dtype=embs.dtype,
                )
                embs = torch.cat([embs, padding], dim=0)
            padded_embeddings.append(embs)
        
        batch_embeddings = torch.stack(padded_embeddings, dim=0)
        
        return batch_embeddings, all_patch_sizes
