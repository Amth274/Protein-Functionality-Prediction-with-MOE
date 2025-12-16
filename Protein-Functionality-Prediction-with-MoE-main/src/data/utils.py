"""
Data utilities for protein functionality prediction.
"""

import os
import torch
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.model_selection import train_test_split


def load_dms_data(file_path: str) -> pd.DataFrame:
    """
    Load and validate DMS data from CSV file.

    Args:
        file_path (str): Path to CSV file

    Returns:
        pd.DataFrame: Validated dataframe

    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If required columns are missing
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path)

    # Check required columns
    required_columns = ['mutated_sequence', 'DMS_score']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Remove rows with missing values in required columns
    df = df.dropna(subset=required_columns)

    return df


def create_support_query_split(
    df: pd.DataFrame,
    support_size: int = 128,
    query_size: int = 72,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create support and query splits for meta-learning.

    Args:
        df (pd.DataFrame): Input dataframe
        support_size (int): Size of support set
        query_size (int): Size of query set
        random_state (int): Random seed

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Support and query dataframes
    """
    if len(df) < support_size + query_size:
        raise ValueError(f"Dataset too small: {len(df)} < {support_size + query_size}")

    # Shuffle the dataframe
    df_shuffled = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    # Create splits
    support_df = df_shuffled.iloc[:support_size].copy()
    query_df = df_shuffled.iloc[support_size:support_size + query_size].copy()

    return support_df, query_df


def stratified_split(
    df: pd.DataFrame,
    split_column: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create stratified train/test split based on a column.

    Args:
        df (pd.DataFrame): Input dataframe
        split_column (str): Column to stratify on
        test_size (float): Fraction for test set
        random_state (int): Random seed

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Train and test dataframes
    """
    if split_column not in df.columns:
        raise ValueError(f"Split column '{split_column}' not found in dataframe")

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[split_column],
        random_state=random_state
    )

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def get_protein_statistics(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate basic statistics for a protein dataset.

    Args:
        df (pd.DataFrame): Protein dataframe

    Returns:
        Dict[str, float]: Statistics dictionary
    """
    stats = {
        'num_sequences': len(df),
        'avg_sequence_length': df['mutated_sequence'].str.len().mean(),
        'min_sequence_length': df['mutated_sequence'].str.len().min(),
        'max_sequence_length': df['mutated_sequence'].str.len().max(),
        'avg_score': df['DMS_score'].mean(),
        'std_score': df['DMS_score'].std(),
        'min_score': df['DMS_score'].min(),
        'max_score': df['DMS_score'].max(),
    }

    if 'DMS_score_bin' in df.columns:
        stats['unique_bins'] = df['DMS_score_bin'].nunique()

    return stats


def filter_by_sequence_length(
    df: pd.DataFrame,
    min_length: int = 10,
    max_length: int = 1024
) -> pd.DataFrame:
    """
    Filter sequences by length.

    Args:
        df (pd.DataFrame): Input dataframe
        min_length (int): Minimum sequence length
        max_length (int): Maximum sequence length

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    seq_lengths = df['mutated_sequence'].str.len()
    mask = (seq_lengths >= min_length) & (seq_lengths <= max_length)
    return df[mask].reset_index(drop=True)


def balance_dataset(
    df: pd.DataFrame,
    target_column: str = 'DMS_score_bin',
    strategy: str = 'undersample'
) -> pd.DataFrame:
    """
    Balance dataset by target column.

    Args:
        df (pd.DataFrame): Input dataframe
        target_column (str): Column to balance on
        strategy (str): Balancing strategy ('undersample' or 'oversample')

    Returns:
        pd.DataFrame: Balanced dataframe
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found")

    value_counts = df[target_column].value_counts()

    if strategy == 'undersample':
        # Sample min count from each class
        min_count = value_counts.min()
        balanced_dfs = []

        for value in value_counts.index:
            class_df = df[df[target_column] == value]
            sampled_df = class_df.sample(n=min_count, random_state=42)
            balanced_dfs.append(sampled_df)

        return pd.concat(balanced_dfs, ignore_index=True).sample(frac=1.0, random_state=42)

    elif strategy == 'oversample':
        # Sample max count for each class (with replacement for smaller classes)
        max_count = value_counts.max()
        balanced_dfs = []

        for value in value_counts.index:
            class_df = df[df[target_column] == value]
            if len(class_df) < max_count:
                sampled_df = class_df.sample(n=max_count, replace=True, random_state=42)
            else:
                sampled_df = class_df
            balanced_dfs.append(sampled_df)

        return pd.concat(balanced_dfs, ignore_index=True).sample(frac=1.0, random_state=42)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def validate_amino_acid_sequence(sequence: str) -> bool:
    """
    Validate that a sequence contains only valid amino acids.

    Args:
        sequence (str): Amino acid sequence

    Returns:
        bool: True if valid, False otherwise
    """
    valid_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
    return all(aa in valid_amino_acids for aa in sequence.upper())


def clean_sequences(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean protein sequences by removing invalid characters.

    Args:
        df (pd.DataFrame): Input dataframe

    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    # Remove sequences with invalid amino acids
    valid_mask = df['mutated_sequence'].apply(validate_amino_acid_sequence)
    cleaned_df = df[valid_mask].copy()

    # Convert to uppercase
    cleaned_df['mutated_sequence'] = cleaned_df['mutated_sequence'].str.upper()

    return cleaned_df.reset_index(drop=True)


def load_embedding_file(file_path: str) -> Dict[str, torch.Tensor]:
    """
    Load pre-computed embeddings from file.

    Args:
        file_path (str): Path to embedding file (.pt)

    Returns:
        Dict[str, torch.Tensor]: Loaded embeddings
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Embedding file not found: {file_path}")

    return torch.load(file_path, map_location='cpu')


def get_dataset_summary(data_dir: str) -> pd.DataFrame:
    """
    Create a summary of all datasets in a directory.

    Args:
        data_dir (str): Directory containing CSV files

    Returns:
        pd.DataFrame: Summary dataframe
    """
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    summaries = []

    for csv_file in csv_files:
        file_path = os.path.join(data_dir, csv_file)
        try:
            df = load_dms_data(file_path)
            stats = get_protein_statistics(df)
            stats['file_name'] = csv_file
            stats['protein_id'] = csv_file.split('_')[0]
            summaries.append(stats)
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")

    return pd.DataFrame(summaries)


def create_task_metadata(data_dir: str, output_file: str):
    """
    Create metadata file for all protein tasks.

    Args:
        data_dir (str): Directory containing CSV files
        output_file (str): Output CSV file path
    """
    summary_df = get_dataset_summary(data_dir)
    summary_df.to_csv(output_file, index=False)
    print(f"Task metadata saved to {output_file}")
    return summary_df