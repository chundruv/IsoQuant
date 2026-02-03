#!/usr/bin/env python3
"""Test script to compare region query ordering between gffutils and in-memory GTF store."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.gtf_store import load_gtf
import gffutils

# Paths
GTF_PATH = "/slade/home/vc362/paradigm/resources/gencode/v49/gencode.v49.annotation.gtf.gz"
DB_PATH = "comparison/orig/OUT/aux/gffutils.db"  # The original gffutils database

# Region of interest - where the problematic transcript is
CHR = "chr22"
START = 42079000
END = 42080000

def main():
    print("Loading in-memory GTF store...")
    inmem_db = load_gtf(GTF_PATH, chromosomes={CHR})

    print("Loading gffutils database...")
    gff_db = gffutils.FeatureDB(DB_PATH)

    print(f"\n=== Region query: {CHR}:{START}-{END} ===\n")

    # Query both databases for genes
    print("--- GENES ---")
    print("\nIn-memory order:")
    inmem_genes = list(inmem_db.region(seqid=CHR, start=START, end=END, featuretype='gene'))
    for i, g in enumerate(inmem_genes):
        print(f"  {i+1}. {g.id} ({g.start}-{g.end}, {g.strand})")

    print("\ngffutils order:")
    gff_genes = list(gff_db.region(seqid=CHR, start=START, end=END, featuretype='gene'))
    for i, g in enumerate(gff_genes):
        print(f"  {i+1}. {g.id} ({g.start}-{g.end}, {g.strand})")

    # Query both databases for transcripts
    print("\n--- TRANSCRIPTS ---")
    print("\nIn-memory order:")
    inmem_txs = list(inmem_db.region(seqid=CHR, start=START, end=END, featuretype='transcript'))
    for i, t in enumerate(inmem_txs):
        gene_id = t.attributes.get('gene_id', [''])[0]
        print(f"  {i+1}. {t.id} (gene={gene_id}, {t.start}-{t.end}, {t.strand})")

    print("\ngffutils order:")
    gff_txs = list(gff_db.region(seqid=CHR, start=START, end=END, featuretype='transcript'))
    for i, t in enumerate(gff_txs):
        gene_id = t.attributes.get('gene_id', [''])[0]
        print(f"  {i+1}. {t.id} (gene={gene_id}, {t.start}-{t.end}, {t.strand})")

    # Check if orders differ
    print("\n=== Order comparison ===")
    inmem_gene_ids = [g.id for g in inmem_genes]
    gff_gene_ids = [g.id for g in gff_genes]
    print(f"Gene order same: {inmem_gene_ids == gff_gene_ids}")
    if inmem_gene_ids != gff_gene_ids:
        print(f"  In-mem: {inmem_gene_ids}")
        print(f"  gffutils: {gff_gene_ids}")

    inmem_tx_ids = [t.id for t in inmem_txs]
    gff_tx_ids = [t.id for t in gff_txs]
    print(f"Transcript order same: {inmem_tx_ids == gff_tx_ids}")
    if inmem_tx_ids != gff_tx_ids:
        print(f"  In-mem: {inmem_tx_ids}")
        print(f"  gffutils: {gff_tx_ids}")

    # Check children order for the problematic gene
    print("\n=== Children of ENSG00000309045.1 (lncRNA gene) ===")
    gene_id = "ENSG00000309045.1"

    print("\nIn-memory transcripts:")
    inmem_children = list(inmem_db.children(gene_id, featuretype='transcript'))
    for i, t in enumerate(inmem_children):
        print(f"  {i+1}. {t.id} ({t.start}-{t.end}, {t.strand})")

    print("\ngffutils transcripts:")
    gff_children = list(gff_db.children(gene_id, featuretype='transcript'))
    for i, t in enumerate(gff_children):
        print(f"  {i+1}. {t.id} ({t.start}-{t.end}, {t.strand})")

    # Check exons for the problematic transcript
    print("\n=== Children of ENST00000838012.1 (exons) ===")
    tx_id = "ENST00000838012.1"

    print("\nIn-memory exons:")
    inmem_exons = list(inmem_db.children(tx_id, featuretype='exon'))
    for i, e in enumerate(inmem_exons):
        print(f"  {i+1}. {e.start}-{e.end} ({e.strand})")

    print("\ngffutils exons:")
    gff_exons = list(gff_db.children(tx_id, featuretype='exon'))
    for i, e in enumerate(gff_exons):
        print(f"  {i+1}. {e.start}-{e.end} ({e.strand})")

    # Also check the overlapping gene ENSG00000177096.10
    print("\n=== Children of ENSG00000177096.10 (overlapping gene) ===")
    gene_id2 = "ENSG00000177096.10"

    print("\nIn-memory transcripts:")
    inmem_children2 = list(inmem_db.children(gene_id2, featuretype='transcript'))
    for i, t in enumerate(inmem_children2):
        print(f"  {i+1}. {t.id} ({t.start}-{t.end}, {t.strand})")

    print("\ngffutils transcripts:")
    gff_children2 = list(gff_db.children(gene_id2, featuretype='transcript'))
    for i, t in enumerate(gff_children2):
        print(f"  {i+1}. {t.id} ({t.start}-{t.end}, {t.strand})")

if __name__ == "__main__":
    main()
