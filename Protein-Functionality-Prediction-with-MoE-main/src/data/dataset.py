"""
Dataset classes for protein functionality prediction.
"""

import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Union


class DMSDataset(Dataset):
    """
    Dataset for Deep Mutational Scanning (DMS) data.

    Args:
        df (pd.DataFrame): DataFrame with columns: mutant, mutated_sequence, DMS_score, DMS_score_bin
        set_type (str): Type of dataset ('support', 'query', 'train', 'test')
        transform (callable, optional): Optional transform to apply to sequences
    """

    def __init__(self, df: pd.DataFrame, set_type: str = "train", transform=None):
        self.df = df.copy()
        self.df["set_type"] = set_type
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Dict[str, Union[str, float, int]]:
        row = self.df.iloc[index]

        item = {
            "sequence": row["mutated_sequence"],
            "score": float(row["DMS_score"]),
            "score_bin": int(row["DMS_score_bin"]) if "DMS_score_bin" in row else 0,
            "set_type": row["set_type"],
            "mutant": row.get("mutant", ""),
            "index": index
        }

        if self.transform:
            item = self.transform(item)

        return item


class EmbeddingDataset(Dataset):
    """
    Dataset for pre-computed embeddings.

    Args:
        embedding_files (List[str]): List of .pt files containing embeddings
        max_files (int, optional): Maximum number of files to load
    """

    def __init__(self, embedding_files: List[str], max_files: Optional[int] = None):
        self.embedding_files = embedding_files[:max_files] if max_files else embedding_files
        self.data = []
        self._load_embeddings()

    def _load_embeddings(self):
        """Load all embeddings into memory."""
        for file_path in self.embedding_files:
            data = torch.load(file_path, map_location='cpu')
            if isinstance(data, list):
                self.data.extend(data)
            elif isinstance(data, dict):
                self.data.append(data)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        item = self.data[index]
        return {
            "embedding": item.get("embedding", item.get("embeddings")),
            "score": item.get("score", item.get("scores")),
            "set_type": item.get("set_type", "unknown")
        }


class ProteinDataModule:
    """
    Data module for organizing protein datasets and dataloaders.

    Args:
        data_dir (str): Directory containing data files
        embedding_dir (str): Directory containing pre-computed embeddings
        batch_size (int): Batch size for dataloaders
        num_workers (int): Number of workers for data loading
        support_size (int): Size of support set for meta-learning
        query_size (int): Size of query set for meta-learning
    """

    def __init__(
        self,
        data_dir: str,
        embedding_dir: Optional[str] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        support_size: int = 128,
        query_size: int = 72,
        min_samples: int = 200
    ):
        self.data_dir = data_dir
        self.embedding_dir = embedding_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.support_size = support_size
        self.query_size = query_size
        self.min_samples = min_samples

        self.train_files = []
        self.test_files = []
        self._discover_files()

    def _discover_files(self):
        """Discover available data files."""
        train_dir = os.path.join(self.data_dir, "Train_split")
        test_dir = os.path.join(self.data_dir, "Test_split")

        if os.path.exists(train_dir):
            self.train_files = [
                os.path.join(train_dir, f) for f in os.listdir(train_dir)
                if f.endswith('.csv')
            ]

        if os.path.exists(test_dir):
            self.test_files = [
                os.path.join(test_dir, f) for f in os.listdir(test_dir)
                if f.endswith('.csv')
            ]

    def create_meta_learning_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create support/query split for meta-learning.

        Args:
            df (pd.DataFrame): Input dataframe

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Support and query dataframes
        """
        if len(df) < self.min_samples:
            return None, None

        # Shuffle the data
        df_shuffled = df.sample(frac=1.0).reset_index(drop=True)

        # Create splits
        support_df = df_shuffled.iloc[:self.support_size].reset_index(drop=True)
        query_df = df_shuffled.iloc[self.support_size:self.support_size + self.query_size].reset_index(drop=True)

        return support_df, query_df

    def get_task_dataloaders(self, csv_file: str) -> Tuple[DataLoader, DataLoader]:
        """
        Create support and query dataloaders for a single task.

        Args:
            csv_file (str): Path to CSV file

        Returns:
            Tuple[DataLoader, DataLoader]: Support and query dataloaders
        """
        df = pd.read_csv(csv_file)
        support_df, query_df = self.create_meta_learning_split(df)

        if support_df is None or query_df is None:
            return None, None

        support_dataset = DMSDataset(support_df, set_type="support")
        query_dataset = DMSDataset(query_df, set_type="query")

        support_loader = DataLoader(
            support_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

        query_loader = DataLoader(
            query_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

        return support_loader, query_loader

    def get_embedding_dataloaders(self, split: str = "train") -> DataLoader:
        """
        Create dataloader for pre-computed embeddings.

        Args:
            split (str): Data split ('train' or 'test')

        Returns:
            DataLoader: Embedding dataloader
        """
        if self.embedding_dir is None:
            raise ValueError("embedding_dir must be specified to use embeddings")

        embedding_path = os.path.join(self.embedding_dir, split)
        if not os.path.exists(embedding_path):
            raise ValueError(f"Embedding path does not exist: {embedding_path}")

        embedding_files = [
            os.path.join(embedding_path, f) for f in os.listdir(embedding_path)
            if f.endswith('.pt')
        ]

        dataset = EmbeddingDataset(embedding_files)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def get_all_task_files(self, split: str = "train") -> List[str]:
        """
        Get all task files for a given split.

        Args:
            split (str): Data split ('train' or 'test')

        Returns:
            List[str]: List of file paths
        """
        if split == "train":
            return self.train_files
        elif split == "test":
            return self.test_files
        else:
            raise ValueError(f"Unknown split: {split}")


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for batching protein data.

    Args:
        batch (List[Dict]): List of data items

    Returns:
        Dict[str, torch.Tensor]: Batched data
    """
    keys = batch[0].keys()
    batched = {}

    for key in keys:
        items = [item[key] for item in batch]

        if key in ['score', 'score_bin', 'index']:
            batched[key] = torch.tensor(items)
        elif key == 'embedding':
            batched[key] = torch.stack(items)
        else:
            batched[key] = items  # Keep as list for strings

    return batched