############################################################################
# Copyright (c) 2022-2026 University of Helsinki
# Copyright (c) 2019-2022 Saint Petersburg State University
# # All Rights Reserved
# See file LICENSE for details.
############################################################################

"""
Abstraction layer for FASTA file parsing in IsoQuant.

This module provides a unified interface for FASTA file access using pyfaidx.
"""

import logging
from abc import ABC, abstractmethod

logger = logging.getLogger('IsoQuant')


class FASTAReaderInterface(ABC):
    """Abstract interface for FASTA file access."""

    @abstractmethod
    def load(self, fasta_path: str, index_path: str = None) -> None:
        """Load a FASTA file."""
        pass

    @abstractmethod
    def get_sequence(self, chr_id: str, start: int = None, end: int = None) -> str:
        """
        Get sequence for a chromosome or region.

        Args:
            chr_id: Chromosome/contig identifier
            start: Optional start position (0-based)
            end: Optional end position (exclusive)

        Returns:
            Sequence string
        """
        pass

    @abstractmethod
    def get_chromosome_record(self, chr_id: str):
        """
        Get the chromosome record object for direct access.

        This is needed for compatibility with existing IsoQuant code
        that accesses chromosome records directly.
        """
        pass

    @abstractmethod
    def get_chromosome_ids(self) -> list:
        """Get list of all chromosome/contig IDs."""
        pass

    @abstractmethod
    def get_chromosome_length(self, chr_id: str) -> int:
        """Get the length of a chromosome."""
        pass

    @abstractmethod
    def keys(self):
        """Return chromosome IDs (dict-like interface)."""
        pass

    def __getitem__(self, chr_id: str):
        """Allow dict-like access: reader[chr_id]."""
        return self.get_chromosome_record(chr_id)


class PyfaidxReader(FASTAReaderInterface):
    """FASTA reader using pyfaidx."""

    def __init__(self):
        from pyfaidx import Fasta
        self._Fasta = Fasta
        self.fasta = None

    def load(self, fasta_path: str, index_path: str = None) -> None:
        """Load FASTA file using pyfaidx."""
        if index_path:
            self.fasta = self._Fasta(fasta_path, indexname=index_path)
        else:
            self.fasta = self._Fasta(fasta_path)

    def get_sequence(self, chr_id: str, start: int = None, end: int = None) -> str:
        """Get sequence for a region."""
        if start is None and end is None:
            return str(self.fasta[chr_id][:])
        elif start is not None and end is not None:
            return str(self.fasta[chr_id][start:end])
        elif start is not None:
            return str(self.fasta[chr_id][start:])
        else:
            return str(self.fasta[chr_id][:end])

    def get_chromosome_record(self, chr_id: str):
        """Get pyfaidx chromosome record for direct access."""
        return self.fasta[chr_id]

    def get_chromosome_ids(self) -> list:
        """Get list of chromosome IDs."""
        return list(self.fasta.keys())

    def get_chromosome_length(self, chr_id: str) -> int:
        """Get chromosome length."""
        return len(self.fasta[chr_id])

    def keys(self):
        """Return chromosome IDs."""
        return self.fasta.keys()


def create_fasta_reader(fasta_path: str, index_path: str = None) -> FASTAReaderInterface:
    """
    Create and load a FASTA reader.

    Args:
        fasta_path: Path to FASTA file
        index_path: Optional path to index file

    Returns:
        Loaded FASTAReaderInterface
    """
    reader = PyfaidxReader()
    reader.load(fasta_path, index_path)
    return reader
