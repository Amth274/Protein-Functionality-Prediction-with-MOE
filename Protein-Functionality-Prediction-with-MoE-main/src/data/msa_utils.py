"""
Utilities for MSA (Multiple Sequence Alignment) retrieval and processing.

Provides functions to:
- Retrieve homologous sequences using BLAST or MMseqs2
- Compute MSA statistics (PSSM, conservation, coevolution)
- Process and encode MSAs for neural network input
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
from Bio import SeqIO, AlignIO
from Bio.Blast import NCBIWWW, NCBIXML


class MSARetriever:
    """
    Retrieve multiple sequence alignments for proteins.

    Supports both BLAST (online) and MMseqs2 (local) for homology search.
    """

    def __init__(
        self,
        method='blast',  # 'blast' or 'mmseqs2'
        database='nr',
        max_sequences=256,
        evalue_threshold=1e-5,
        mmseqs_db_path=None
    ):
        self.method = method
        self.database = database
        self.max_sequences = max_sequences
        self.evalue_threshold = evalue_threshold
        self.mmseqs_db_path = mmseqs_db_path

    def retrieve_msa_blast(self, sequence: str) -> List[str]:
        """
        Retrieve MSA using NCBI BLAST (online).

        Args:
            sequence: Query protein sequence

        Returns:
            List of homologous sequences
        """
        try:
            # Run BLAST search
            result_handle = NCBIWWW.qblast(
                "blastp",
                self.database,
                sequence,
                expect=self.evalue_threshold,
                hitlist_size=self.max_sequences
            )

            # Parse results
            blast_record = NCBIXML.read(result_handle)
            sequences = [sequence]  # Include query

            for alignment in blast_record.alignments[:self.max_sequences - 1]:
                for hsp in alignment.hsps:
                    if hsp.expect < self.evalue_threshold:
                        # Extract aligned sequence
                        sequences.append(str(hsp.sbjct).replace('-', ''))

            return sequences[:self.max_sequences]

        except Exception as e:
            print(f"BLAST search failed: {e}")
            return [sequence]  # Return just query sequence

    def retrieve_msa_mmseqs2(self, sequence: str) -> List[str]:
        """
        Retrieve MSA using MMseqs2 (local, faster).

        Requires MMseqs2 to be installed and database to be set up.

        Args:
            sequence: Query protein sequence

        Returns:
            List of homologous sequences
        """
        if self.mmseqs_db_path is None:
            raise ValueError("MMseqs2 database path not specified")

        try:
            # Create temporary files
            with tempfile.TemporaryDirectory() as tmpdir:
                query_file = Path(tmpdir) / "query.fasta"
                result_file = Path(tmpdir) / "result"
                tmp_db = Path(tmpdir) / "tmp"

                # Write query sequence
                with open(query_file, 'w') as f:
                    f.write(f">query\n{sequence}\n")

                # Run MMseqs2 search
                cmd = [
                    'mmseqs', 'easy-search',
                    str(query_file),
                    self.mmseqs_db_path,
                    str(result_file),
                    str(tmp_db),
                    '--max-seqs', str(self.max_sequences),
                    '-e', str(self.evalue_threshold),
                    '--format-mode', '2'
                ]

                subprocess.run(cmd, check=True, capture_output=True)

                # Parse results
                sequences = [sequence]
                if result_file.exists():
                    with open(result_file) as f:
                        for line in f:
                            parts = line.strip().split('\t')
                            if len(parts) > 1:
                                sequences.append(parts[1])

                return sequences[:self.max_sequences]

        except Exception as e:
            print(f"MMseqs2 search failed: {e}")
            return [sequence]

    def retrieve_msa(self, sequence: str) -> List[str]:
        """
        Retrieve MSA using configured method.

        Args:
            sequence: Query protein sequence

        Returns:
            List of homologous sequences
        """
        if self.method == 'blast':
            return self.retrieve_msa_blast(sequence)
        elif self.method == 'mmseqs2':
            return self.retrieve_msa_mmseqs2(sequence)
        else:
            raise ValueError(f"Unknown method: {self.method}")


class MSAFeatureExtractor:
    """
    Extract features from Multiple Sequence Alignments.

    Computes various evolutionary and conservation features from MSAs.
    """

    def __init__(self, amino_acids='ACDEFGHIKLMNPQRSTVWY'):
        self.amino_acids = amino_acids
        self.aa_to_idx = {aa: i for i, aa in enumerate(amino_acids)}

    def compute_pssm(self, sequences: List[str]) -> np.ndarray:
        """
        Compute Position-Specific Scoring Matrix (PSSM).

        Args:
            sequences: List of aligned sequences (same length)

        Returns:
            PSSM matrix of shape (seq_len, 20) with amino acid frequencies
        """
        if not sequences:
            return np.zeros((0, 20))

        # Find alignment length (use longest sequence)
        max_len = max(len(s) for s in sequences)

        # Initialize count matrix
        counts = np.zeros((max_len, len(self.amino_acids)))

        # Count amino acids at each position
        for seq in sequences:
            for pos, aa in enumerate(seq):
                if aa in self.aa_to_idx:
                    counts[pos, self.aa_to_idx[aa]] += 1

        # Normalize to frequencies (add pseudocount)
        pssm = (counts + 1) / (len(sequences) + 20)

        return pssm

    def compute_conservation(self, sequences: List[str]) -> np.ndarray:
        """
        Compute conservation score for each position.

        Uses Shannon entropy: H = -sum(p * log(p))
        Conservation = 1 - H/H_max

        Args:
            sequences: List of aligned sequences

        Returns:
            Conservation scores of shape (seq_len,)
        """
        pssm = self.compute_pssm(sequences)

        # Compute Shannon entropy
        entropy = -np.sum(pssm * np.log(pssm + 1e-10), axis=1)

        # Normalize (max entropy = log(20) for uniform distribution)
        max_entropy = np.log(len(self.amino_acids))
        conservation = 1 - (entropy / max_entropy)

        return conservation

    def compute_coevolution(self, sequences: List[str]) -> np.ndarray:
        """
        Compute simple coevolution scores using mutual information.

        Args:
            sequences: List of aligned sequences

        Returns:
            Coevolution matrix of shape (seq_len, seq_len)
        """
        pssm = self.compute_pssm(sequences)
        seq_len = pssm.shape[0]

        # Simple coevolution: normalized dot product of PSSM vectors
        coevo = np.dot(pssm, pssm.T)

        # Normalize to [0, 1]
        coevo = coevo / (np.linalg.norm(pssm, axis=1, keepdims=True) *
                         np.linalg.norm(pssm, axis=1, keepdims=True).T + 1e-10)

        return coevo

    def extract_all_features(self, sequences: List[str]) -> Dict[str, np.ndarray]:
        """
        Extract all MSA features.

        Args:
            sequences: List of sequences in MSA

        Returns:
            Dictionary with feature arrays:
                - pssm: (L, 20) amino acid frequencies
                - conservation: (L,) conservation scores
                - coevolution: (L, L) coevolution matrix
        """
        return {
            'pssm': self.compute_pssm(sequences),
            'conservation': self.compute_conservation(sequences),
            'coevolution': self.compute_coevolution(sequences)
        }


def simple_msa_features(sequence: str, num_homologs: int = 50) -> torch.Tensor:
    """
    Simple MSA feature extraction without actual homology search.

    For testing purposes - returns dummy MSA features based on sequence alone.

    Args:
        sequence: Protein sequence
        num_homologs: Number of simulated homologs

    Returns:
        MSA feature tensor of shape (seq_len, 20)
    """
    # Create simple PSSM-like features
    aa_to_idx = {aa: i for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}

    seq_len = len(sequence)
    pssm = np.zeros((seq_len, 20))

    # One-hot encode with noise
    for pos, aa in enumerate(sequence):
        if aa in aa_to_idx:
            idx = aa_to_idx[aa]
            # High probability for actual amino acid
            pssm[pos, idx] = 0.7
            # Small probabilities for others (simulate variability)
            pssm[pos, :] += np.random.dirichlet(np.ones(20) * 0.1)
            # Renormalize
            pssm[pos, :] /= pssm[pos, :].sum()

    return torch.from_numpy(pssm).float()
