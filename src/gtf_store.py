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
        attr_parts = []
        for k, v in self.attributes.items():
            val_str = " ".join([f'"{x}"' for x in v])
            attr_parts.append(f'{k} {val_str}')
        attr_str = "; ".join(attr_parts)
        if attr_str:
            attr_str += ";"
            
        return f"{self.seqid}\t{self.source}\t{self.featuretype}\t{self.start}\t{self.end}\t{self.score}\t{self.strand}\t{self.frame}\t{attr_str}"


# -----------------------------------------------------------------------------
# 2. Standalone Helper Functions
# -----------------------------------------------------------------------------
def parse_attributes_fast(attr_str):
    """
    Fast parsing of GTF attributes string.
    Returns dict: {key: [value, ...]}
    """
    res = {}
    for pair in attr_str.split(';'):
        pair = pair.strip()
        if not pair: 
            continue
        parts = pair.split(' ', 1)
        if len(parts) < 2:
            continue
        key = parts[0]
        val = parts[1].strip('"')
        if key in res:
            res[key].append(val)
        else:
            res[key] = [val]
    return res

def get_gtf_chromosomes(gtf_path):
    """
    Scans GTF file to find all used chromosome names (seqids).
    Returns a set of strings.
    """
    logger.info(f"Scanning GTF for chromosome names: {gtf_path}")
    chromosomes = set()
    
    open_func = gzip.open if gtf_path.endswith('.gz') else open
    
    try:
        with open_func(gtf_path, 'rt') as f:
            for line in f:
                if line.startswith('#'): continue
                
                # We only need the first column (seqid)
                # Optimization: find the first tab instead of splitting the whole line
                tab_index = line.find('\t')
                if tab_index != -1:
                    chromosomes.add(line[:tab_index])
    except Exception as e:
        logger.error(f"Failed to scan chromosomes from {gtf_path}: {e}")
        raise

    logger.info(f"Found {len(chromosomes)} chromosomes in GTF.")
    return chromosomes


# -----------------------------------------------------------------------------
# 3. The Database Class
# -----------------------------------------------------------------------------
class InMemoryFeatureDB:
    def __init__(self, features_map, features_by_type):
        self.features = features_map
        self.features_by_type = features_by_type # Dict[str, List[GTFFeature]]

    def __getitem__(self, key):
        return self.features[key]

    def features_of_type(self, featuretype, order_by=None, reverse=False, limit=None):
        """
        Returns features of a specific type (e.g., 'gene', 'transcript').
        """
        if featuretype not in self.features_by_type:
            return []
        
        # Get the pre-indexed list
        results = list(self.features_by_type[featuretype]) # Copy to avoid side effects
        
        # Apply Sorting
        if order_by:
            if order_by == 'start':
                results.sort(key=lambda x: (x.start, x.end, x.file_order))
            elif order_by == 'id':
                results.sort(key=lambda x: (x.id, x.file_order))
        
        if reverse:
            results.reverse()
            
        if limit:
            results = results[:limit]
            
        return results

    def children(self, feature, featuretype=None, order_by=None, reverse=False, limit=None):
        """
        Retrieve children of a feature.
        """
        if isinstance(feature, str):
            feature_obj = self.features.get(feature)
        else:
            feature_obj = feature
        
        if not feature_obj:
            return []

        if featuretype:
            results = [c for c in feature_obj.children if c.featuretype == featuretype]
        else:
            results = list(feature_obj.children)

        # SORTING for Deterministic Output (Fixes 1-read diff)
        if order_by is None:
            if feature_obj.featuretype in ['gene', 'mRNA', 'transcript']:
                results.sort(key=lambda x: (x.id, x.file_order))
            else:
                results.sort(key=lambda x: (x.start, x.end, x.file_order))
        elif order_by == 'start':
            results.sort(key=lambda x: (x.start, x.end, x.file_order))
        elif order_by == 'id':
            results.sort(key=lambda x: (x.id, x.file_order))
            
        if reverse: results.reverse()
        if limit: results = results[:limit]
            
        return results

    def iter_by_parent(self, parent, featuretype=None, order_by=None, reverse=False, limit=None):
        """Alias for children() used by some parts of gffutils API."""
        return self.children(parent, featuretype, order_by, reverse, limit)

    def parents(self, feature, featuretype=None):
        """Parents not strictly tracked to save RAM, return empty."""
        return []

    def region(self, seqid=None, start=None, end=None, strand=None, featuretype=None):
        """
        Simple region query. 
        For full IsoQuant, gene iteration usually just scans the gene list.
        """
        results = []
        
        # Optimize 'gene' iteration
        source_list = self.features_by_type.get('gene', []) if featuretype == 'gene' else self.features.values()

        for f in source_list:
            if seqid and f.seqid != seqid: continue
            if start and f.end < start: continue
            if end and f.start > end: continue
            if strand and f.strand != strand: continue
            if featuretype and f.featuretype != featuretype: continue
            results.append(f)
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
    features_by_type = defaultdict(list)
    
    # Temporary lookups for linking
    genes_by_id = {}
    transcripts_by_id = {}
    exon_counter = 0
    
    open_func = gzip.open if gtf_path.endswith('.gz') else open
    
    with open_func(gtf_path, 'rt') as f:
        for i, line in enumerate(f):
            if line.startswith('#'): continue
            parts = line.strip().split('\t')
            if len(parts) < 9: continue

            seqid = sys.intern(parts[0])
            if chromosomes and seqid not in chromosomes:
                continue

            # Parse Attributes
            attr_map = parse_attributes_fast(parts[8])
            feature_type = sys.intern(parts[2])

            # Determine ID
            if feature_type == 'gene':
                fid = attr_map.get('gene_id', [None])[0]
            elif feature_type == 'transcript':
                fid = attr_map.get('transcript_id', [None])[0]
            else:
                fid = attr_map.get('exon_id', [None])[0]
                if not fid:
                    # Synthetic ID
                    fid = f"{feature_type}:{seqid}:{parts[3]}-{parts[4]}:{parts[6]}:{exon_counter}"
                    exon_counter += 1

            if not fid: fid = f"unknown_{i}"

            # Create Feature
            feature = GTFFeature(
                seqid,
                sys.intern(parts[1]),
                feature_type,
                int(parts[3]),
                int(parts[4]),
                parts[5],
                sys.intern(parts[6]),
                parts[7],
                attr_map,
                fid,
                i # file_order
            )

            # Store
            features_map[fid] = feature
            features_by_type[feature_type].append(feature)
            
            # Link Hierarchy
            if feature_type == 'gene':
                genes_by_id[fid] = feature
            
            elif feature_type == 'transcript':
                gid = attr_map.get('gene_id', [None])[0]
                if gid and gid in genes_by_id:
                    genes_by_id[gid].children.append(feature)
                transcripts_by_id[fid] = feature
                
            elif feature_type in ['exon', 'CDS', 'UTR']:
                tid = attr_map.get('transcript_id', [None])[0]
                if tid and tid in transcripts_by_id:
                    transcripts_by_id[tid].children.append(feature)

    logger.info(f"Loaded {len(features_map)} features.")
    return InMemoryFeatureDB(features_map, features_by_type)