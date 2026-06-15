# AGREL
## Implementation of "A Unified Framework for Code Vulnerability Detection via Adaptive Pooling and Symbolic Relational Graph Representation"

This repository contains the implementation of **AGREL**, a graph neural network framework for code vulnerability detection that combines structure-aware graph symbolization with dynamic self-attention pooling.

## Requirements

- Python 3.9+
- PyTorch 2.0+
- DGL 2.0+
- CUDA 11.8 
- Joern (see "Joern Setup" below)

## Joern Setup

AGREL uses Joern for code property graph (CPG) extraction. We follow
the **same Joern version used by AMPLE ** to
ensure consistent graph construction with prior work. Specifically:

- Joern version: as released with the AMPLE artifact
  
- Download and install Joern according to AMPLE's instructions.

This choice ensures that any performance differences reported in the
paper stem from model design rather than from variations in graph
construction tooling.

## Datasets

We evaluate AGREL on four public C/C++ vulnerability datasets:

| Dataset  | Source |
|----------|--------|
| Devign   |  |
| Reveal   |  |
| Big-Vul  |  |
| PrimeVul |  |



## Reproducing the Experiments

The following steps reproduce the results reported in the paper.

### Step 1: Preprocess and build code property graphs


This step invokes Joern to extract CPGs and applies the
Structure-Aware Graph Symbolization.

### Step 2: Train AGREL

Choose either the global-pooling or hierarchical-pooling variant.



