# You Are What You Prompt: Prompt Quality, Domain Shift, and Uncertainty in Agrifood Vision-Language Models

## Overview

Vision-language models enable zero-shot classification through natural language prompts, but performance is sensitive to prompt formulation, especially in specialized domains. This paper evaluates **Zero-shot Prompt Ensembling (ZPE)** in the agrifood domain using CLIP and SigLIP across four datasets and four prompt pools, spanning in-distribution food and out-of-distribution agricultural benchmarks.

We further introduce **PID (Prompt-based Inconsistency Detection)**, which repurposes prompt disagreement as epistemic uncertainty, improving failure detection under severe domain shift where standard confidence measures collapse.

## Authors

Andrea Morales-Garzón, Salvador Lopez-Joya, Miguel López-Pérez, Maria J. Martin-Bautista
*Dept. Computer Science and Artificial Intelligence, University of Granada*

## Repository Structure

```
pid_agrifood/
├── models/
│   ├── zpe.py                                      # Zero-shot Prompt Ensembling (ZPE)
│   └── zpe_full.py                                 # ZPE with full pipeline and PID scoring
├── prompts/
│   ├── pool.py                                     # Prompt pool definitions
│   └── domain_pools.py                             # Domain-specific prompt pools
├── data/
│   └── loaders.py                                  # Dataset loaders
├── analysis/
│   ├── pid.py                                      # PID uncertainty analysis
│   ├── calibration.py                              # Calibration evaluation
│   ├── lexical.py                                  # Lexical diversity metrics
│   ├── results_table.py                            # Results aggregation and tables
│   ├── fig_zpe_vs_accuracy.py                      # ZPE vs accuracy figures
│   └── fig_prompt_diversity_response_reviewers.py  # Prompt diversity figures
└── requirements.txt
```

## Citation

Coming soon.
