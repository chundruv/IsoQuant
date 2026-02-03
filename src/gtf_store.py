############################################################################
# Copyright (c) 2022-2026 University of Helsinki
# Copyright (c) 2019-2022 Saint Petersburg State University
# # All Rights Reserved
# See file LICENSE for details.
############################################################################

"""
In-memory GTF store for IsoQuant.
High-performance replacement for gffutils SQLite database.
"""

import gzip
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Set

logger = logging.getLogger('IsoQuant')

# -----------------------------------------------------------------------------
# 1. Optimized Feature Class
# -----------------------------------------------------------------------------
@dataclass
class GTFFeature:
    # __slots__ drastically reduces memory usage by removing __dict__
    __slots__ = ('seqid', 'source', 'featuretype', 'start', 'end', 
                 'score', 'strand', 'frame', 'attributes', 'id', 
                 'children', 'extra', 'file_order')

    def __init__(self, seqid, source, featuretype, start, end, 
                 score, strand, frame, attributes, feature_id, file_order):
        self.seqid = seqid
        self.source = source
        self.featuretype = featuretype
        self.start = start
        self.end = end
        self.score = score
        self.strand = strand
        self.frame = frame
        self.attributes = attributes
        self.id = feature_id         # Pre-calculated ID
        self.file_order = file_order # File line number for stable sorting
        self.children = []           # Pre-linked children
        self.extra = []              # gffutils compatibility

    def __getitem__(self, key):
        """Allow dictionary-style access to attributes (db['gene_id'])."""
        return self.attributes.get(key)

    def __str__(self):
        """Reconstruct GTF line from memory."""
        # Reconstruct attributes string: key "value";
        attr_parts = []
        for k, v in self.attributes.items():
            # v is a list of values
            val_str = " ".join([f'"{x}"' for x in v])
            attr_parts.append(f'{k} {val_str}')
        attr_str = "; ".join(attr_parts)
        if attr_str:
            attr_str += ";"
            
        return f"{self.seqid}\t{self.source}\t{self.featuretype}\t{self.start}\t{self.end}\t{self.score}\t{self.strand}\t{self.frame}\t{attr_str}"


# -----------------------------------------------------------------------------
# 2. Standalone Helper Functions (No 'self' required)
# -----------------------------------------------------------------------------
def parse_attributes_fast(attr_str):
    """
    Fast parsing of GTF attributes string.
    Returns dict: {key: [value, ...]}
    """
    res = {}
    # Split by semicolon
    for pair in attr_str.split(';'):
        pair = pair.strip()
        if not pair: 
            continue
            
        # Split by first space to separate Key and Value
        # e.g. 'gene_id "ENSG000001"' -> key='gene_id', val='"ENSG000001"'
        parts = pair.split(' ', 1)
        if len(parts) < 2:
            continue
            
        key = parts[0]
        val = parts[1].strip('"') # Remove quotes
        
        # Store as list (gffutils standard)
        if key in res:
            res[key].append(val)
        else:
            res[key] = [val]
    return res


# -----------------------------------------------------------------------------
# 3. The Database Class
# -----------------------------------------------------------------------------
class InMemoryFeatureDB:
    def __init__(self, features_map, genes_list):
        self.features = features_map
        self.genes = genes_list

    def __getitem__(self, key):
        return self.features[key]

    def children(self, feature, featuretype=None, order_by=None, reverse=False, limit=None):
        """
        Retrieve children of a feature.
        Matches gffutils signature but uses pre-linked lists and sorting fixes.
        """
        # Resolve feature ID
        if isinstance(feature, str):
            feature_obj = self.features.get(feature)
        else:
            feature_obj = feature
        
        if not feature_obj:
            return []

        # Get pre-linked children
        # Filter by featuretype if requested
        if featuretype:
            results = [c for c in feature_obj.children if c.featuretype == featuretype]
        else:
            results = list(feature_obj.children) # Copy list

        # -------------------------------------------------------
        # SORTING FIX for Deterministic Output
        # -------------------------------------------------------
        if order_by is None:
            # Default behavior based on parent type
            if feature_obj.featuretype in ['gene', 'mRNA', 'transcript']:
                # Transcripts: Sort by ID (Alphanumeric), then file order
                results.sort(key=lambda x: (x.id, x.file_order))
            else:
                # Exons/CDS: Sort by Genomic Coordinate, then file order
                results.sort(key=lambda x: (x.start, x.end, x.file_order))
        
        elif order_by == 'start':
            results.sort(key=lambda x: (x.start, x.end, x.file_order))
        elif order_by == 'id':
            results.sort(key=lambda x: (x.id, x.file_order))
            
        if reverse:
            results.reverse()
            
        if limit:
            results = results[:limit]
            
        return results
    
    # Required for gffutils compatibility
    def parents(self, feature, featuretype=None):
        # Not strictly needed for IsoQuant's core loop, but good for safety
        # Since we didn't store parent pointers to save RAM, this might be empty
        # If IsoQuant needs this, we would need to add 'parent' slot to GTFFeature
        return []

    def region(self, seqid=None, start=None, end=None, strand=None, featuretype=None):
        # Used for fetching genes in a range
        # Simple linear scan implementation (optimized with ncls in full version if needed)
        # For IsoQuant gene iteration, it usually just iterates 'self.genes'
        results = []
        
        # Optimization: if just iterating all genes
        if featuretype == 'gene' and seqid is None:
            return self.genes
            
        # Basic implementation if needed (slow for random access, fine for one-pass)
        for g in self.genes:
            if seqid and g.seqid != seqid: continue
            if start and g.end < start: continue
            if end and g.start > end: continue
            if strand and g.strand != strand: continue
            results.append(g)
        return results
        
    def all_features(self):
        return self.features.values()


# -----------------------------------------------------------------------------
# 4. The Loader Function (Factory)
# -----------------------------------------------------------------------------
def load_gtf(gtf_path, chromosomes=None):
    """
    Parses GTF and returns an InMemoryFeatureDB.
    """
    logger.info(f"Loading in-memory GTF from {gtf_path}")
    
    features_map = {}
    genes_list = []
    
    # Temporary lookups for linking
    genes_by_id = {}
    transcripts_by_id = {}
    
    # For unique ID generation
    exon_counter = 0
    
    # Handle GZIP
    open_func = gzip.open if gtf_path.endswith('.gz') else open
    
    with open_func(gtf_path, 'rt') as f:
        for i, line in enumerate(f):
            if line.startswith('#'): continue
            
            parts = line.strip().split('\t')
            if len(parts) < 9: continue

            # Filter by chromosome if requested (Speeds up loading)
            seqid = parts[0]
            if chromosomes and seqid not in chromosomes:
                continue

            # 1. Parse Attributes (Using standalone function, NO self.)
            attr_map = parse_attributes_fast(parts[8])
            
            feature_type = sys.intern(parts[2])

            # 2. Determine Feature ID
            # Logic: Use gene_id/transcript_id if available, else generate safe ID
            if feature_type == 'gene':
                fid = attr_map.get('gene_id', [None])[0]
            elif feature_type == 'transcript':
                fid = attr_map.get('transcript_id', [None])[0]
            else:
                # Exons/CDS often lack specific IDs in GTF
                fid = attr_map.get('exon_id', [None])[0]
                if not fid:
                    # Create synthetic unique ID
                    fid = f"{feature_type}:{seqid}:{parts[3]}-{parts[4]}:{parts[6]}:{exon_counter}"
                    exon_counter += 1

            if not fid:
                # Fallback if gene/transcript missing ID
                fid = f"unknown_{i}"

            # 3. Create Feature Object
            # Using sys.intern to save RAM on repeated strings
            feature = GTFFeature(
                sys.intern(seqid),          # seqid
                sys.intern(parts[1]),       # source
                feature_type,               # featuretype
                int(parts[3]),              # start
                int(parts[4]),              # end
                parts[5],                   # score
                sys.intern(parts[6]),       # strand
                parts[7],                   # frame
                attr_map,                   # attributes
                fid,                        # id
                i                           # file_order (Line Number)
            )

            # 4. Store and Link
            features_map[fid] = feature
            
            if feature_type == 'gene':
                genes_list.append(feature)
                genes_by_id[fid] = feature
            
            elif feature_type == 'transcript':
                # Link to Gene
                gid = attr_map.get('gene_id', [None])[0]
                if gid and gid in genes_by_id:
                    genes_by_id[gid].children.append(feature)
                transcripts_by_id[fid] = feature
                
            elif feature_type in ['exon', 'CDS', 'UTR']:
                # Link to Transcript
                tid = attr_map.get('transcript_id', [None])[0]
                if tid and tid in transcripts_by_id:
                    transcripts_by_id[tid].children.append(feature)

    # 5. Return the DB object
    logger.info(f"Loaded {len(features_map)} features.")
    return InMemoryFeatureDB(features_map, genes_list)

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

