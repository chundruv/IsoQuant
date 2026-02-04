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
        self.id = feature_id
        self.file_order = file_order 
        self.children = []
        self.extra = []

    def __getitem__(self, key):
        return self.attributes.get(key)

    def __str__(self):
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
    res = {}
    if not attr_str:
        return res
    
    parts = attr_str.split(';')
    idx = 0
    while idx < len(parts):
        part = parts[idx]
        # If odd number of quotes, we likely split inside a string; append next part
        while part.count('"') % 2 == 1 and idx + 1 < len(parts):
            idx += 1
            part += ";" + parts[idx]
        
        part = part.strip()
        if part:
            split_part = part.split(None, 1)
            if len(split_part) == 2:
                key, val = split_part
                val = val.strip()
                if len(val) > 1 and val[0] == '"' and val[-1] == '"':
                    val = val[1:-1]
                if key in res:
                    res[key].append(val)
                else:
                    res[key] = [val]
        idx += 1
    return res

def get_gtf_chromosomes(gtf_path):
    logger.info(f"Scanning GTF for chromosome names: {gtf_path}")
    chromosomes = set()
    open_func = gzip.open if gtf_path.endswith('.gz') else open
    try:
        with open_func(gtf_path, 'rt') as f:
            for line in f:
                if line.startswith('#'): continue
                tab_index = line.find('\t')
                if tab_index != -1:
                    chromosomes.add(line[:tab_index])
    except Exception as e:
        logger.error(f"Failed to scan chromosomes from {gtf_path}: {e}")
        return set()
    logger.info(f"Found {len(chromosomes)} chromosomes in GTF.")
    return chromosomes


# -----------------------------------------------------------------------------
# 3. The Database Class
# -----------------------------------------------------------------------------
class InMemoryFeatureDB:
    def __init__(self, features_map, features_by_type):
        self.features = features_map
        self.features_by_type = features_by_type

    def __getitem__(self, key):
        return self.features[key]

    def features_of_type(self, featuretype, order_by=None, reverse=False, limit=None):
        results = []
        if isinstance(featuretype, str):
            if featuretype in self.features_by_type:
                results = list(self.features_by_type[featuretype])
        else:
            for ft in featuretype:
                if ft in self.features_by_type:
                    results.extend(self.features_by_type[ft])

        if order_by:
            if order_by == 'start':
                results.sort(key=lambda x: (x.start, x.end, x.file_order))
            elif order_by == 'id':
                results.sort(key=lambda x: (x.id, x.file_order))
        
        if reverse: results.reverse()
        if limit: results = results[:limit]
        return results

    def children(self, feature, featuretype=None, order_by=None, reverse=False, limit=None):
        if isinstance(feature, str):
            feature_obj = self.features.get(feature)
        else:
            feature_obj = feature
        
        if not feature_obj:
            return []

        if featuretype:
            if isinstance(featuretype, str):
                results = [c for c in feature_obj.children if c.featuretype == featuretype]
            else:
                types_set = set(featuretype)
                results = [c for c in feature_obj.children if c.featuretype in types_set]
        else:
            results = list(feature_obj.children)

        # -------------------------------------------------------
        # SORTING LOGIC (Matches gffutils)
        # -------------------------------------------------------
        if order_by is None:
            # Gene children (Transcripts) -> File Order
            if feature_obj.featuretype == 'gene':
                results.sort(key=lambda x: x.file_order)
            else:
                # Transcript children (Exons) -> Coordinate Order
                results.sort(key=lambda x: (x.start, x.end, x.file_order))
        elif order_by == 'start':
            results.sort(key=lambda x: (x.start, x.end, x.file_order))
        elif order_by == 'id':
            results.sort(key=lambda x: (x.id, x.file_order))
            
        if reverse: results.reverse()
        if limit: results = results[:limit]
        return results

    def iter_by_parent(self, parent, featuretype=None, order_by=None, reverse=False, limit=None):
        return self.children(parent, featuretype, order_by, reverse, limit)

    def parents(self, feature, featuretype=None):
        return []

    def region(self, seqid=None, start=None, end=None, strand=None, featuretype=None):
        results = []
        if featuretype == 'gene':
            source_list = self.features_by_type.get('gene', [])
        else:
            source_list = self.features.values()

        for f in source_list:
            if seqid and f.seqid != seqid: continue
            if start and f.end < start: continue
            if end and f.start > end: continue
            if strand and f.strand != strand: continue
            
            if featuretype:
                if isinstance(featuretype, str):
                    if f.featuretype != featuretype: continue
                else:
                    if f.featuretype not in featuretype: continue
            
            results.append(f)
        return results
        
    def all_features(self):
        return self.features.values()


# -----------------------------------------------------------------------------
# 4. The Loader Function (Factory)
# -----------------------------------------------------------------------------
def load_gtf(gtf_path, chromosomes=None):
    logger.info(f"Loading in-memory GTF from {gtf_path}")
    
    features_map = {}
    features_by_type = defaultdict(list)
    pending_children = defaultdict(list)
    
    open_func = gzip.open if gtf_path.endswith('.gz') else open
    
    with open_func(gtf_path, 'rt') as f:
        for i, line in enumerate(f):
            if line.startswith('#'): continue
            parts = line.strip().split('\t')
            if len(parts) < 9: continue

            seqid = sys.intern(parts[0])
            if chromosomes and seqid not in chromosomes:
                continue

            attr_map = parse_attributes_fast(parts[8])
            feature_type = sys.intern(parts[2])

            fid = None
            parent_id = None
            
            if feature_type == 'gene':
                fid = attr_map.get('gene_id', [None])[0]
            elif feature_type == 'transcript':
                fid = attr_map.get('transcript_id', [None])[0]
            else:
                fid = attr_map.get('exon_id', [None])[0]
                if not fid:
                    fid = f"{feature_type}:{seqid}:{parts[3]}-{parts[4]}:{parts[6]}:{parts[7]}"

            if not fid: fid = f"unknown_{i}"

            tid = attr_map.get('transcript_id', [None])[0]
            gid = attr_map.get('gene_id', [None])[0]

            if tid and tid != fid:
                parent_id = tid
            elif gid and gid != fid:
                parent_id = gid

            # -----------------------------------------------------------
            # FIX: Upsert Logic (Merge duplicates instead of creating new)
            # -----------------------------------------------------------
            if fid in features_map:
                # Feature exists: Merge data
                existing = features_map[fid]
                
                # Update boundaries (Envelope)
                existing.start = min(existing.start, int(parts[3]))
                existing.end = max(existing.end, int(parts[4]))
                
                # Merge attributes
                for k, v in attr_map.items():
                    if k in existing.attributes:
                        # Append unique new values
                        existing_values = set(existing.attributes[k])
                        for val in v:
                            if val not in existing_values:
                                existing.attributes[k].append(val)
                    else:
                        existing.attributes[k] = v
                
                feature = existing
                # Note: We do NOT append to features_by_type again, preventing double counting
            else:
                # New Feature
                feature = GTFFeature(
                    seqid, sys.intern(parts[1]), feature_type,
                    int(parts[3]), int(parts[4]), parts[5],
                    sys.intern(parts[6]), parts[7], attr_map,
                    fid, i
                )
                features_map[fid] = feature
                features_by_type[feature_type].append(feature)
            
            # Always track hierarchy (even for merged features)
            if parent_id:
                pending_children[parent_id].append(feature)

    logger.info("Linking features...")
    for parent_id, children in pending_children.items():
        parent = features_map.get(parent_id)
        if parent:
            # Avoid duplicate children if multiple lines referred to same child
            # (Simple set check or just extend if we trust uniqueness of children in file)
            # For speed, we just extend, assuming file structure is reasonable.
            # But if we just merged the child 'feature' above, it might be in 'children' list twice?
            # 'pending_children' stores references to the OBJECT. 
            # If we merged, we added the SAME object reference multiple times to pending_children.
            # We should de-duplicate the children list before attaching.
            
            unique_children = []
            seen_ids = set()
            for child in children:
                if child.id not in seen_ids:
                    unique_children.append(child)
                    seen_ids.add(child.id)
            
            parent.children.extend(unique_children)

    # -------------------------------------------------------
    # Sorting Phase
    # -------------------------------------------------------
    logger.info("Sorting feature children...")
    
    for feature in features_map.values():
        if not feature.children:
            continue
            
        if feature.featuretype == 'gene':
             # Genes -> Transcripts: Sort by File Order
             feature.children.sort(key=lambda x: x.file_order)
        else:
             # Transcripts -> Exons: Sort by Coordinate
             feature.children.sort(key=lambda x: (x.start, x.end, x.file_order))

    logger.info(f"Loaded {len(features_map)} features.")
    return InMemoryFeatureDB(features_map, features_by_type)

get_gtf_chromosomes = get_gtf_chromosomes