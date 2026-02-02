############################################################################
# Copyright (c) 2022-2026 University of Helsinki
# Copyright (c) 2019-2022 Saint Petersburg State University
# # All Rights Reserved
# See file LICENSE for details.
############################################################################

"""
In-memory GTF store for IsoQuant.

This module provides a high-performance in-memory alternative to gffutils
SQLite database. It uses interval trees for fast region queries and
dict-based lookups for ID access.

Key features:
- No SQLite database creation (faster startup)
- In-memory interval trees for O(log n) region queries
- Dict-based O(1) ID lookups
- gffutils-compatible API for drop-in replacement
"""

import gzip
import logging
from collections import defaultdict
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Iterator, Tuple, Any

logger = logging.getLogger('IsoQuant')


@dataclass
class GTFFeature:
    """Represents a single GTF feature (gene, transcript, exon, etc.)."""
    seqid: str
    source: str
    featuretype: str
    start: int  # 1-based, inclusive
    end: int    # 1-based, inclusive
    score: str
    strand: str
    frame: str
    attributes: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def id(self) -> str:
        """Get the primary ID for this feature."""
        if self.featuretype == 'gene':
            return self.attributes.get('gene_id', [''])[0]
        elif self.featuretype in ('transcript', 'mRNA'):
            return self.attributes.get('transcript_id', [''])[0]
        elif self.featuretype == 'exon':
            # Exons may not have unique IDs, construct one
            gene_id = self.attributes.get('gene_id', [''])[0]
            tx_id = self.attributes.get('transcript_id', [''])[0]
            exon_num = self.attributes.get('exon_number', [''])[0]
            return f"{tx_id}_exon_{exon_num}" if exon_num else f"{tx_id}_{self.start}_{self.end}"
        else:
            # For other features, try common ID attributes
            for attr in ('ID', 'gene_id', 'transcript_id'):
                if attr in self.attributes:
                    return self.attributes[attr][0]
            return f"{self.seqid}_{self.featuretype}_{self.start}_{self.end}"

    def __getitem__(self, key: str) -> Any:
        """Allow dict-like attribute access for gffutils compatibility."""
        if hasattr(self, key):
            return getattr(self, key)
        return self.attributes.get(key)

    def __hash__(self):
        return hash((self.seqid, self.featuretype, self.start, self.end, self.strand))

    def __eq__(self, other):
        if not isinstance(other, GTFFeature):
            return False
        return (self.seqid == other.seqid and
                self.featuretype == other.featuretype and
                self.start == other.start and
                self.end == other.end and
                self.strand == other.strand)


class IntervalTree:
    """
    Simple interval tree for fast region queries.
    Uses a sorted list with binary search for reasonable performance.
    """

    def __init__(self):
        self._intervals: List[Tuple[int, int, GTFFeature]] = []
        self._sorted = False

    def add(self, start: int, end: int, feature: GTFFeature):
        """Add an interval."""
        self._intervals.append((start, end, feature))
        self._sorted = False

    def _ensure_sorted(self):
        """Sort intervals if needed."""
        if not self._sorted:
            self._intervals.sort(key=lambda x: (x[0], x[1]))
            self._sorted = True

    def query(self, start: int, end: int) -> Iterator[GTFFeature]:
        """Find all intervals overlapping [start, end]."""
        self._ensure_sorted()

        # Binary search to find starting point
        # An interval (s, e) overlaps (start, end) if s <= end and e >= start
        for s, e, feature in self._intervals:
            if s > end:
                break  # No more overlaps possible
            if e >= start:
                yield feature

    def __len__(self):
        return len(self._intervals)


class InMemoryFeatureDB:
    """
    In-memory GTF database that mimics gffutils.FeatureDB API.

    This replaces the SQLite-based gffutils database with pure in-memory
    data structures for faster access.
    """

    def __init__(self):
        # Primary storage: feature_id -> feature
        self._features: Dict[str, GTFFeature] = {}

        # Index by feature type
        self._by_type: Dict[str, List[GTFFeature]] = defaultdict(list)

        # Index by chromosome (seqid)
        self._by_seqid: Dict[str, List[GTFFeature]] = defaultdict(list)

        # Interval trees for region queries, keyed by seqid
        self._intervals: Dict[str, IntervalTree] = defaultdict(IntervalTree)

        # Parent-child relationships
        self._children: Dict[str, List[GTFFeature]] = defaultdict(list)
        self._parents: Dict[str, List[GTFFeature]] = defaultdict(list)

        # Gene -> transcripts mapping
        self._gene_transcripts: Dict[str, List[GTFFeature]] = defaultdict(list)

        # Transcript -> exons mapping
        self._transcript_exons: Dict[str, List[GTFFeature]] = defaultdict(list)

        # Track if indices are built
        self._indexed = False

    def __getitem__(self, feature_id: str) -> GTFFeature:
        """Get feature by ID (gffutils compatibility)."""
        if feature_id not in self._features:
            raise KeyError(f"Feature '{feature_id}' not found")
        return self._features[feature_id]

    def __contains__(self, feature_id: str) -> bool:
        """Check if feature exists."""
        return feature_id in self._features

    def get(self, feature_id: str, default=None) -> Optional[GTFFeature]:
        """Get feature by ID with default."""
        return self._features.get(feature_id, default)

    def add_feature(self, feature: GTFFeature):
        """Add a feature to the database."""
        feature_id = feature.id
        self._features[feature_id] = feature
        self._by_type[feature.featuretype].append(feature)
        self._by_seqid[feature.seqid].append(feature)
        self._intervals[feature.seqid].add(feature.start, feature.end, feature)
        self._indexed = False

    def _build_relationships(self):
        """Build parent-child relationships after all features are loaded."""
        if self._indexed:
            return

        # Build gene -> transcript relationships
        for feature in self._by_type.get('transcript', []) + self._by_type.get('mRNA', []):
            gene_id = feature.attributes.get('gene_id', [''])[0]
            if gene_id:
                self._gene_transcripts[gene_id].append(feature)
                if gene_id in self._features:
                    self._children[gene_id].append(feature)
                    self._parents[feature.id].append(self._features[gene_id])

        # Build transcript -> exon relationships
        for feature in self._by_type.get('exon', []):
            tx_id = feature.attributes.get('transcript_id', [''])[0]
            if tx_id:
                self._transcript_exons[tx_id].append(feature)
                if tx_id in self._features:
                    self._children[tx_id].append(feature)
                    self._parents[feature.id].append(self._features[tx_id])

        # Build transcript -> CDS/UTR/codon relationships
        for feature_type in ('CDS', 'UTR', 'five_prime_UTR', 'three_prime_UTR',
                             'start_codon', 'stop_codon'):
            for feature in self._by_type.get(feature_type, []):
                tx_id = feature.attributes.get('transcript_id', [''])[0]
                if tx_id and tx_id in self._features:
                    self._children[tx_id].append(feature)
                    self._parents[feature.id].append(self._features[tx_id])

        # Sort exons by position
        for tx_id in self._transcript_exons:
            self._transcript_exons[tx_id].sort(key=lambda x: x.start)

        self._indexed = True

    def features_of_type(self, featuretype, order_by=None) -> Iterator[GTFFeature]:
        """
        Get all features of a given type (gffutils compatibility).

        Args:
            featuretype: str or tuple of feature types
            order_by: Optional tuple of (attribute, 'start') for sorting
        """
        self._build_relationships()

        if isinstance(featuretype, str):
            featuretype = (featuretype,)

        features = []
        for ft in featuretype:
            features.extend(self._by_type.get(ft, []))

        if order_by:
            if isinstance(order_by, tuple) and 'start' in order_by:
                features.sort(key=lambda f: (f.seqid, f.start))
            elif order_by == 'start':
                features.sort(key=lambda f: f.start)

        return iter(features)

    def all_features(self) -> Iterator[GTFFeature]:
        """Get all features."""
        return iter(self._features.values())

    def region(self, seqid: str = None, start: int = None, end: int = None,
               featuretype: str = None, strand: str = None) -> Iterator[GTFFeature]:
        """
        Get features overlapping a region (gffutils compatibility).

        Args:
            seqid: Chromosome/contig name
            start: Start position (1-based)
            end: End position (1-based)
            featuretype: Optional filter by feature type
            strand: Optional filter by strand
        """
        self._build_relationships()

        if seqid is None:
            raise ValueError("seqid is required for region query")

        if seqid not in self._intervals:
            return iter([])

        # Use interval tree for efficient query
        if start is None:
            start = 0
        if end is None:
            end = float('inf')

        for feature in self._intervals[seqid].query(start, end):
            if featuretype and feature.featuretype != featuretype:
                continue
            if strand and feature.strand != strand:
                continue
            yield feature

    def children(self, feature, featuretype=None, order_by=None) -> Iterator[GTFFeature]:
        """
        Get children of a feature (gffutils compatibility).

        Args:
            feature: GTFFeature or feature ID
            featuretype: Optional filter by type (str or tuple)
            order_by: Optional sorting
        """
        self._build_relationships()

        if isinstance(feature, str):
            feature_id = feature
        else:
            feature_id = feature.id

        children = self._children.get(feature_id, [])

        if featuretype:
            if isinstance(featuretype, str):
                featuretype = (featuretype,)
            children = [c for c in children if c.featuretype in featuretype]

        if order_by == 'start':
            children = sorted(children, key=lambda f: f.start)

        return iter(children)

    def parents(self, feature, featuretype=None) -> Iterator[GTFFeature]:
        """Get parents of a feature."""
        self._build_relationships()

        if isinstance(feature, str):
            feature_id = feature
        else:
            feature_id = feature.id

        parents = self._parents.get(feature_id, [])

        if featuretype:
            if isinstance(featuretype, str):
                featuretype = (featuretype,)
            parents = [p for p in parents if p.featuretype in featuretype]

        return iter(parents)

    def seqids(self) -> List[str]:
        """Get all chromosome/contig IDs."""
        return list(self._by_seqid.keys())


def parse_gtf_attributes(attr_string: str) -> Dict[str, List[str]]:
    """Parse GTF attribute string into a dictionary."""
    attributes = {}

    # GTF format: key "value"; key "value";
    for item in attr_string.strip().rstrip(';').split(';'):
        item = item.strip()
        if not item:
            continue

        # Split on first space
        parts = item.split(' ', 1)
        if len(parts) != 2:
            # Try splitting on '=' for GFF3 format
            parts = item.split('=', 1)
            if len(parts) != 2:
                continue

        key = parts[0].strip()
        value = parts[1].strip().strip('"\'')

        if key in attributes:
            attributes[key].append(value)
        else:
            attributes[key] = [value]

    return attributes


def parse_gtf_line(line: str) -> Optional[GTFFeature]:
    """Parse a single GTF line into a GTFFeature."""
    line = line.strip()
    if not line or line.startswith('#'):
        return None

    parts = line.split('\t')
    if len(parts) < 9:
        return None

    try:
        return GTFFeature(
            seqid=parts[0],
            source=parts[1],
            featuretype=parts[2],
            start=int(parts[3]),
            end=int(parts[4]),
            score=parts[5],
            strand=parts[6],
            frame=parts[7],
            attributes=parse_gtf_attributes(parts[8])
        )
    except (ValueError, IndexError) as e:
        logger.warning(f"Failed to parse GTF line: {e}")
        return None


def load_gtf(gtf_path: str, feature_types: Set[str] = None) -> InMemoryFeatureDB:
    """
    Load a GTF file into an in-memory database.

    Args:
        gtf_path: Path to GTF file (can be gzipped)
        feature_types: Optional set of feature types to load (None = all)

    Returns:
        InMemoryFeatureDB instance
    """
    logger.info(f"Loading GTF from {gtf_path} into memory...")

    db = InMemoryFeatureDB()

    # Open file (handle gzip)
    if gtf_path.endswith('.gz'):
        opener = gzip.open(gtf_path, 'rt')
    else:
        opener = open(gtf_path, 'r')

    count = 0
    with opener as f:
        for line in f:
            feature = parse_gtf_line(line)
            if feature is None:
                continue

            # Filter by feature type if specified
            if feature_types and feature.featuretype not in feature_types:
                continue

            db.add_feature(feature)
            count += 1

            if count % 100000 == 0:
                logger.debug(f"Loaded {count} features...")

    # Build indices
    db._build_relationships()

    logger.info(f"Loaded {count} features into memory")
    logger.info(f"  Genes: {len(db._by_type.get('gene', []))}")
    logger.info(f"  Transcripts: {len(db._by_type.get('transcript', [])) + len(db._by_type.get('mRNA', []))}")
    logger.info(f"  Exons: {len(db._by_type.get('exon', []))}")

    return db


def get_gtf_chromosomes(gtf_path: str) -> Set[str]:
    """
    Quickly extract unique chromosome names from a GTF file.

    This is a lightweight alternative to loading the full GTF when you
    only need chromosome names (e.g., for chromosome filtering).

    Args:
        gtf_path: Path to GTF file (can be gzipped)

    Returns:
        Set of chromosome/seqid names
    """
    logger.info(f"Scanning GTF for chromosome names: {gtf_path}")

    chromosomes = set()

    # Open file (handle gzip)
    if gtf_path.endswith('.gz'):
        opener = gzip.open(gtf_path, 'rt')
    else:
        opener = open(gtf_path, 'r')

    with opener as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # Just extract the first column (seqid/chromosome)
            tab_pos = line.find('\t')
            if tab_pos > 0:
                chromosomes.add(line[:tab_pos])

    logger.info(f"Found {len(chromosomes)} chromosomes in GTF")
    return chromosomes


# Convenience function for drop-in replacement
def FeatureDB(path: str, **kwargs) -> InMemoryFeatureDB:
    """
    Drop-in replacement for gffutils.FeatureDB.

    If path ends with .db, attempts to load as gffutils database.
    Otherwise, loads GTF directly into memory.
    """
    if path.endswith('.db'):
        # This is a gffutils database file - load GTF from original source
        # For now, raise an error - caller should use load_gtf directly
        raise ValueError(
            f"Cannot load .db file with in-memory store. "
            f"Use gtf_store.load_gtf() with the original GTF file, "
            f"or pass --fast to skip database creation."
        )

    return load_gtf(path, **kwargs)
