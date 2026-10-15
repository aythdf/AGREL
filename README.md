
## Requirements

- Python 3.9+
- PyTorch 2.0+
- DGL 2.0+
- CUDA 11.8 
- Joern (see "Joern Setup" below)

## Joern Setup

AGREL uses Joern for code property graph (CPG) extraction. We follow
the same Joern version used by AMPLE to
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
| Devign   |https://drive.google.com/file/d/1x6hoF7G-tSYxg8AFybggypLZgMGDNHfF/edit  |
| Reveal   | https://drive.google.com/drive/folders/1KuIYgFcvWUXheDhT--cBALsfy1I4utOy |
| Big-Vul  | https://drive.google.com/file/d/1-0VhnHBp9IGh90s2wCNjeCMuy70HPl8X/view |
| PrimeVul | https://drive.google.com/drive/folders/1cznxGme5o6A_9tT8T47JUh3MPEpRYiKK?usp=drive_link |



## Reproducing the Experiments

The following steps reproduce the results reported in the paper.

### Step 1: Preprocess and build code property graphs


This step invokes Joern to extract CPGs and applies the
Structure-Aware Graph Symbolization.

### Step 2: Train AGREL

Choose either the global-pooling or hierarchical-pooling variant.



