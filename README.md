# SER Replication Study

<div align="center" style="display: flex; justify-content: center; margin-top: 10px;">
  <a href="https://arxiv.org/abs/2508.02448"><img src="https://img.shields.io/badge/arXiv-2508.02448-AD1C18" style="margin-right: 5px;"></a>
  <a href="https://huggingface.co/autrainer/msp-podcast-emo-class-big4-w2v2-l-emo">
    <img src="https://img.shields.io/badge/🤗-Checkpoints-ED5A22.svg">
  </a>
</div>

## Overview

This repository contains the code to reproduce the results of (Triantafyllopoulos et al., 2025).
It is a large-scale study of deep learning models trained for speech emotion recognition (SER).
Our goal was to chart the progress made since 2009 -- the year the first INTERSPEECH Emotion Challenge was ran.
This repo contains the code to train all models presented in that study using [autrainer](https://autrainer.github.io/autrainer/index.html) or [transformers](https://huggingface.co/docs/transformers/en/index).
It also contains code to reproduce our analysis.

Feel free to read our [preprint](https://arxiv.org/abs/2508.02448) and use the best SER [model](https://huggingface.co/autrainer/msp-podcast-emo-class-big4-w2v2-l-emo) that came out of our study.

## Status

**WARNING:** This repository is still under construction.
The code used for the paper was originally split in multiple scripts/repositories.
We are now in the process of cleaning up.
In particular, we are still lacking most of the code for the analyses presented in the paper.
However, we include the necessary code to reproduce our models.


## Using the repository

We will add detailed instructions on how to use each part of the repository shortly.

### Code structure

Our code is split in three main folders:

1. Code to reproduce audio experiments using [autrainer](https://autrainer.github.io/autrainer/index.html)
2. Code to reproduce text experiments using [transformers](https://huggingface.co/docs/transformers/en/index)
3. Code to run custom analysis (heavily under construction)

### Datasets

This code relies on two standard SER Datasets: MSP-Podcast (v1.11) and FAU-AIBO.
Both need 

### Dataset structure

We assume all data is stored under `data`.

|- data
  |- FAU
  |- MSP-Podcast

## Citation
```
@article{Triantafyllopoulos25-C1P,
  title={Charting 15 years of progress in deep learning for speech emotion recognition: A replication study},
  author={Triantafyllopoulos, Andreas and Batliner, Anton and Schuller, Björn},
  journal={Under review (preprint: https://arxiv.org/abs/2508.02448)},
  year={2025}
}
```