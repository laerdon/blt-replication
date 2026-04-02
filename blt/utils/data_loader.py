"""data loader for byte sequences"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional
import os


class ByteDataset(Dataset):
    """dataset for byte sequences"""
    
    def __init__(
        self,
        data_path: Optional[str] = None,
        byte_sequences: Optional[List[torch.Tensor]] = None,
        max_seq_len: int = 512,
    ):
        """
        args:
            data_path: path to text file to load
            byte_sequences: pre-loaded byte sequences
            max_seq_len: maximum sequence length
        """
        self.max_seq_len = max_seq_len
        
        if byte_sequences is not None:
            self.sequences = byte_sequences
        elif data_path is not None:
            self.sequences = self._load_from_file(data_path)
        else:
            raise ValueError("[ERROR] must provide either data_path or byte_sequences")
    
    def _load_from_file(self, path: str) -> List[torch.Tensor]:
        """load byte sequences from text file"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"[ERROR] file not found: {path}")
        
        with open(path, 'rb') as f:
            data = f.read()
        
        # convert to byte tensor
        byte_tensor = torch.tensor(list(data), dtype=torch.long)
        
        # split into chunks
        sequences = []
        for i in range(0, len(byte_tensor), self.max_seq_len):
            chunk = byte_tensor[i:i + self.max_seq_len]
            if len(chunk) > 0:
                sequences.append(chunk)
        
        return sequences
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.sequences[idx]


class ByteDataLoader:
    """data loader wrapper for byte sequences"""
    
    def __init__(
        self,
        data_path: Optional[str] = None,
        byte_sequences: Optional[List[torch.Tensor]] = None,
        batch_size: int = 8,
        max_seq_len: int = 512,
        shuffle: bool = True,
        num_workers: int = 0,
    ):
        self.dataset = ByteDataset(
            data_path=data_path,
            byte_sequences=byte_sequences,
            max_seq_len=max_seq_len,
        )
        
        self.loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self._collate_fn,
        )
    
    def _collate_fn(self, batch: List[torch.Tensor]) -> List[torch.Tensor]:
        """custom collate function - returns list of tensors"""
        return batch
    
    def __iter__(self):
        return iter(self.loader)
    
    def __len__(self):
        return len(self.loader)
