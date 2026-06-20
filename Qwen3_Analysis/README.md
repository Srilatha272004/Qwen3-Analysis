# Qwen 0.8B Dense Model Analysis

This directory contains the codebase for configuring, compiling, and training a standard dense 0.8B parameter Transformer model using JAX and Flax.

## Architecture Overview
- Model Type: Standard Dense Transformer
- Total Parameters: ~800 Million (100% Active)
- Layer Depth: 24 Layers
- Global Hidden Size: 1536
- Attention Type: Multi-Head Attention (MHA)

## Operational Guide
This model utilizes an exhaustive full-matrix processing strategy where 100% of the parameters fire for every single token. 

### Critical Hardware Warning
Due to full-matrix operations and AdamW optimizer state inflation, compiling this model via XLA on a **standard CPU will trigger an Out-of-Memory (OOM) crash**. Training is highly resource-intensive and requires dedicated hardware execution.

- **Recommended Environment:** Colab T4 GPU (15GB VRAM) or higher.
- **Execution Script:** `python train.py`
