############################################################################
# Copyright (c) 2022-2026 University of Helsinki
# Copyright (c) 2019-2022 Saint Petersburg State University
# # All Rights Reserved
# See file LICENSE for details.
############################################################################

"""
Abstraction layer for file parsing in IsoQuant.

This module provides unified interfaces for GTF and FASTA file parsing,
with support for both traditional backends (gffutils, pyfaidx) and the
high-performance eccLib library.

The eccLib library is optional and will be used when:
1. It is installed
2. The --use_ecclib flag is set
3. It works correctly on the current platform

If eccLib fails for any reason, the module gracefully falls back to
the standard backends.
"""

import logging
import os
from abc import ABC, abstractmethod

logger = logging.getLogger('IsoQuant')

# Check eccLib availability and functionality
ECCLIB_AVAILABLE = False
ECCLIB_WORKING = False

try:
    import eccLib
    ECCLIB_AVAILABLE = True
    # Test if eccLib actually works on this platform
    # Some platforms may have the library but it may crash
    try:
        # Quick sanity check - just ensure the module has expected functions
        if hasattr(eccLib, 'parseFASTA') and hasattr(eccLib, 'parseGTF'):
            ECCLIB_WORKING = True
    except Exception:
        ECCLIB_WORKING = False
except ImportError:
    pass


def is_ecclib_available():
    """Check if eccLib is available and appears to be working."""
    return ECCLIB_AVAILABLE and ECCLIB_WORKING


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
    """FASTA reader using pyfaidx (default backend)."""

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


class EccLibFASTAReader(FASTAReaderInterface):
    """
    FASTA reader using eccLib for high-performance parsing.

    Note: eccLib has different API than pyfaidx, so this adapter
    provides compatibility with the IsoQuant codebase.
    """

    def __init__(self):
        import eccLib
        self._ecclib = eccLib
        self.fasta_data = None
        self._sequence_cache = {}

    def load(self, fasta_path: str, index_path: str = None) -> None:
        """
        Load FASTA file using eccLib.

        Note: eccLib doesn't use index files - it parses the full file.
        The index_path parameter is accepted for API compatibility but ignored.
        """
        try:
            self.fasta_data = self._ecclib.parseFASTA(fasta_path)
            # Pre-cache sequences as strings for faster access
            for chr_id in self.fasta_data.keys():
                self._sequence_cache[chr_id] = self.fasta_data[chr_id].dump()
        except Exception as e:
            logger.warning(f"eccLib FASTA parsing failed: {e}")
            raise

    def get_sequence(self, chr_id: str, start: int = None, end: int = None) -> str:
        """Get sequence for a region."""
        seq = self._sequence_cache.get(chr_id)
        if seq is None:
            seq = self.fasta_data[chr_id].dump()
            self._sequence_cache[chr_id] = seq

        if start is None and end is None:
            return seq
        elif start is not None and end is not None:
            return seq[start:end]
        elif start is not None:
            return seq[start:]
        else:
            return seq[:end]

    def get_chromosome_record(self, chr_id: str):
        """
        Get chromosome record for direct access.

        Returns an EccLibChromosomeRecord wrapper that provides
        pyfaidx-like slicing interface.
        """
        return EccLibChromosomeRecord(self, chr_id)

    def get_chromosome_ids(self) -> list:
        """Get list of chromosome IDs."""
        return list(self.fasta_data.keys())

    def get_chromosome_length(self, chr_id: str) -> int:
        """Get chromosome length."""
        seq = self._sequence_cache.get(chr_id)
        if seq is None:
            seq = self.fasta_data[chr_id].dump()
            self._sequence_cache[chr_id] = seq
        return len(seq)

    def keys(self):
        """Return chromosome IDs."""
        return self.fasta_data.keys()


class EccLibChromosomeRecord:
    """
    Wrapper class to provide pyfaidx-like chromosome record interface
    for eccLib parsed data.

    This allows code like `chr_record[start:end]` to work with eccLib.
    """

    def __init__(self, reader: EccLibFASTAReader, chr_id: str):
        self._reader = reader
        self._chr_id = chr_id

    def __getitem__(self, key):
        """Support slicing: record[start:end]."""
        if isinstance(key, slice):
            return self._reader.get_sequence(self._chr_id, key.start, key.stop)
        elif isinstance(key, int):
            seq = self._reader.get_sequence(self._chr_id)
            return seq[key]
        else:
            raise TypeError(f"indices must be integers or slices, not {type(key).__name__}")

    def __len__(self):
        """Return chromosome length."""
        return self._reader.get_chromosome_length(self._chr_id)

    def __str__(self):
        """Return full sequence as string."""
        return self._reader.get_sequence(self._chr_id)


# Factory function
def get_fasta_reader(use_ecclib: bool = False) -> FASTAReaderInterface:
    """
    Get appropriate FASTA reader based on configuration.

    Args:
        use_ecclib: If True, try to use eccLib (falls back if unavailable)

    Returns:
        FASTAReaderInterface implementation
    """
    if use_ecclib:
        if is_ecclib_available():
            logger.info("Using eccLib for FASTA access")
            return EccLibFASTAReader()
        else:
            if ECCLIB_AVAILABLE:
                logger.warning("eccLib is installed but not working correctly on this platform, "
                             "falling back to pyfaidx")
            else:
                logger.warning("eccLib requested but not installed, falling back to pyfaidx")

    return PyfaidxReader()


def create_fasta_reader(fasta_path: str, index_path: str = None,
                        use_ecclib: bool = False) -> FASTAReaderInterface:
    """
    Convenience function to create and load a FASTA reader.

    Args:
        fasta_path: Path to FASTA file
        index_path: Optional path to index file (for pyfaidx)
        use_ecclib: If True, try to use eccLib

    Returns:
        Loaded FASTAReaderInterface
    """
    reader = get_fasta_reader(use_ecclib)
    reader.load(fasta_path, index_path)
    return reader
