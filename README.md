# WConvKB - A Wide Convolutional Neural Network for Knowledge Graph Completion

This repository introduces WConvKB, a Wide Convolutional neural network for Knowledge Graph completion.

Our ongoing research is an extension to ConvKB architecture with two major differences:
1. we have added non-linear fully connected layers motivated by VGG architectures, and
2. we have utilized multiple wide convolutions modules inspired by Inception-based computer vision models.

## Description

The `src` directory contains the code of the model, based directly on ConvKB implementation. 
The RezoJDM16k dataset can be found in `src/benchmarks/RezoJDM16k`.
Finally `notebooks` provides an easy way to produce the initializing embeddings using OpenKE.

## Results

| **Model** | **MRR** | **MR** | **Hits@10** | **Hits@3** | **Hits@1** |
|---|---|---|---|---|---|
| TransE | 0.179 | 203.31 | 0.432 | 0.242 | 0.041 |
| TransH | 0.218 | 177.12 | 0.498 | 0.291 | 0.069 |
| TransD | 0.208 | 186.19 | 0.474 | 0.278 | 0.064 |
| DistMult | 0.220 | 194.47 | 0.445 | 0.252 | 0.109 |
| ComplEx | 0.253 | 201.58 | 0.533 | 0.304 | 0.117 |
| ConvKB | 0.218 | 186.65 | 0.493 | 0.275 | 0.078 |
| WConvKB | 0.337 | 158.27 | 0.590 | 0.384 | 0.218 |
