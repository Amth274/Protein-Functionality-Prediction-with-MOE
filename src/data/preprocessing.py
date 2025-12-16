"""
Data preprocessing utilities for protein functionality prediction.
"""

import os
import torch
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.encoders import ESMEncoder
from .dataset import DMSDataset


class DataPreprocessor:
    """
    Handles preprocessing of DMS data including score normalization and binning.

    Args:
        score_bins (int): Number of bins for score discretization
        score_percentiles (List[float]): Percentiles for score binning
    """

    def __init__(self, score_bins: int = 3, score_percentiles: Optional[List[float]] = None):
        self.score_bins = score_bins
        self.score_percentiles = score_percentiles or [33.3, 66.7]

    def process_dms_file(self, file_path: str) -> pd.DataFrame:
        """
        Process a single DMS CSV file.

        Args:
            file_path (str): Path to CSV file

        Returns:
            pd.DataFrame: Processed dataframe
        """
        df = pd.read_csv(file_path)

        # Ensure required columns exist
        required_cols = ['mutated_sequence', 'DMS_score']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in {file_path}")

        # Remove rows with missing values
        df = df.dropna(subset=required_cols)

        # Normalize scores to [0, 1] range
        df['DMS_score_normalized'] = self._normalize_scores(df['DMS_score'])

        # Create score bins
        df['DMS_score_bin'] = self._create_score_bins(df['DMS_score_normalized'])

        # Add metadata
        df['file_name'] = os.path.basename(file_path)
        df['protein_id'] = os.path.basename(file_path).split('_')[0]

        return df

    def _normalize_scores(self, scores: pd.Series) -> pd.Series:
        """Normalize scores to [0, 1] range using min-max scaling."""
        min_score = scores.min()
        max_score = scores.max()
        if max_score == min_score:
            return pd.Series([0.5] * len(scores), index=scores.index)
        return (scores - min_score) / (max_score - min_score)

    def _create_score_bins(self, normalized_scores: pd.Series) -> pd.Series:
        """Create discrete bins from normalized scores."""
        if self.score_percentiles:
            bins = np.percentile(normalized_scores, [0] + self.score_percentiles + [100])
        else:
            bins = np.linspace(0, 1, self.score_bins + 1)

        return pd.cut(normalized_scores, bins=bins, labels=False, include_lowest=True)

    def process_directory(self, input_dir: str, output_dir: str, min_samples: int = 100):
        """
        Process all CSV files in a directory.

        Args:
            input_dir (str): Input directory containing CSV files
            output_dir (str): Output directory for processed files
            min_samples (int): Minimum number of samples required per file
        """
        os.makedirs(output_dir, exist_ok=True)

        csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]

        for csv_file in tqdm(csv_files, desc="Processing files"):
            input_path = os.path.join(input_dir, csv_file)
            output_path = os.path.join(output_dir, csv_file)

            try:
                df = self.process_dms_file(input_path)

                if len(df) >= min_samples:
                    df.to_csv(output_path, index=False)
                else:
                    print(f"Skipping {csv_file}: insufficient samples ({len(df)} < {min_samples})")

            except Exception as e:
                print(f"Error processing {csv_file}: {e}")


class EmbeddingGenerator:
    """
    Generates protein embeddings using ESM models.

    Args:
        model_name (str): ESM model name
        device (str): Device for computation ('cuda' or 'cpu')
        batch_size (int): Batch size for embedding generation
        max_length (int): Maximum sequence length
    """

    def __init__(
        self,
        model_name: str = 'facebook/esm2_t6_8M_UR50D',
        device: str = 'cuda',
        batch_size: int = 4,
        max_length: int = 1024
    ):
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length

        # Initialize ESM encoder
        self.encoder = ESMEncoder(model_name=model_name, pooling='mean')
        self.encoder.to(device)
        self.encoder.eval()

    def generate_embeddings_from_csv(
        self,
        csv_file: str,
        output_dir: str,
        support_size: int = 128,
        query_size: int = 72
    ) -> Tuple[str, str]:
        """
        Generate embeddings for a CSV file and save support/query splits.

        Args:
            csv_file (str): Path to CSV file
            output_dir (str): Output directory for embeddings
            support_size (int): Size of support set
            query_size (int): Size of query set

        Returns:
            Tuple[str, str]: Paths to support and query embedding files
        """
        df = pd.read_csv(csv_file)

        if len(df) < support_size + query_size:
            return None, None

        # Create splits
        df_support = df.iloc[:support_size].reset_index(drop=True)
        df_query = df.iloc[support_size:support_size + query_size].reset_index(drop=True)

        # Generate embeddings for both splits
        base_name = os.path.basename(csv_file).replace('.csv', '')

        support_path = self._generate_split_embeddings(
            df_support, output_dir, f"{base_name}_support.pt", "support"
        )
        query_path = self._generate_split_embeddings(
            df_query, output_dir, f"{base_name}_query.pt", "query"
        )

        return support_path, query_path

    def _generate_split_embeddings(
        self,
        df: pd.DataFrame,
        output_dir: str,
        filename: str,
        set_type: str
    ) -> str:
        """Generate embeddings for a single split."""
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)

        dataset = DMSDataset(df, set_type=set_type)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        all_batches = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Generating {set_type} embeddings", leave=False):
                sequences = batch["sequence"]
                scores = batch["score"]

                # Tokenize sequences
                tokens = self.encoder.tokenize(sequences, max_length=self.max_length)
                tokens = {k: v.to(self.device) for k, v in tokens.items()}

                # Generate embeddings
                embeddings = self.encoder(tokens=tokens)

                batch_data = {
                    "embedding": embeddings.cpu(),
                    "score": scores,
                    "set_type": [set_type] * len(sequences)
                }

                all_batches.append(batch_data)

                # Clear GPU memory
                del tokens, embeddings
                torch.cuda.empty_cache()

        # Save embeddings
        torch.save(all_batches, output_path)
        return output_path

    def generate_directory_embeddings(
        self,
        input_dir: str,
        output_dir: str,
        split_name: str = "train"
    ):
        """
        Generate embeddings for all CSV files in a directory.

        Args:
            input_dir (str): Directory containing CSV files
            output_dir (str): Output directory for embeddings
            split_name (str): Name of the split (train/test)
        """
        csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]

        support_dir = os.path.join(output_dir, split_name, "support")
        query_dir = os.path.join(output_dir, split_name, "query")

        for csv_file in tqdm(csv_files, desc=f"Processing {split_name} files"):
            csv_path = os.path.join(input_dir, csv_file)

            try:
                support_path, query_path = self.generate_embeddings_from_csv(
                    csv_path, support_dir if "support" in csv_file else query_dir
                )

                if support_path and query_path:
                    print(f"Generated embeddings for {csv_file}")
                else:
                    print(f"Skipped {csv_file}: insufficient data")

            except Exception as e:
                print(f"Error processing {csv_file}: {e}")


def create_amino_acid_tokenizer() -> Dict[str, int]:
    """
    Create a simple amino acid tokenizer.

    Returns:
        Dict[str, int]: Mapping from amino acids to token IDs
    """
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    tokenizer = {aa: i + 1 for i, aa in enumerate(amino_acids)}  # 0 reserved for padding
    tokenizer['<PAD>'] = 0
    tokenizer['<UNK>'] = len(amino_acids) + 1
    return tokenizer


def tokenize_sequence(sequence: str, tokenizer: Dict[str, int], max_length: int = 512) -> torch.Tensor:
    """
    Tokenize a protein sequence.

    Args:
        sequence (str): Amino acid sequence
        tokenizer (Dict[str, int]): Amino acid to token mapping
        max_length (int): Maximum sequence length

    Returns:
        torch.Tensor: Tokenized sequence
    """
    tokens = [tokenizer.get(aa, tokenizer['<UNK>']) for aa in sequence]

    # Pad or truncate
    if len(tokens) < max_length:
        tokens = tokens + [tokenizer['<PAD>']] * (max_length - len(tokens))
    else:
        tokens = tokens[:max_length]

    return torch.tensor(tokens, dtype=torch.long)