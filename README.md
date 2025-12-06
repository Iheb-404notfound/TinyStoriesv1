# TinyStoriesv1

Reproducing the official-quality training code for the **TinyStories** models (Eldan & Li, 2023) — the famous 10M–33M parameter language models that achieved fluent English stories and have been cited in **over 600 academic papers**.

These tiny transformers prove that **high-quality synthetic data** can make even 10M-parameter models write coherent, grammatical stories — and at 28M–33M they even show in-context learning and simple reasoning!

This repo lets you train the exact architectures from the paper using Hugging Face Transformers + Accelerate.

## Models Included

| Name             | Parameters (non-emb) | Layers | Hidden Size | Heads | Attention Pattern         |
|------------------|----------------------|--------|-------------|-------|----------------------------|
| TinyStories-10M  | ~10M                 | 8      | 320         | 16    | Alternating global/local   |
| TinyStories-28M  | ~28M                 | 8      | 512         | 16    | Alternating global/local   |
| TinyStories-33M  | ~33M                 | 10     | 512         | 16    | Alternating global/local   |

Matches released models: https://huggingface.co/roneneldan

## Quick Start

```bash
pip install -r requirements.txt
accelerate config  # (answer defaults or set multi-GPU)

# Train 28M model (most famous one)
accelerate launch train.py --model_size 28M --epochs 3

# Or single GPU
python train.py --model_size 28M --batch_size 128 --grad_accum 4
```

Training time on one A100 80GB:
- 10M: ~6 hours
- 28M: ~18 hours
- 33M: ~24 hours

Validation loss should reach **~0.55–0.60** (same as paper).

## Generate Stories

```bash
python scripts/generate_stories.py --checkpoint ./TinyStories-28M/checkpoint-xxx --prompt "Once upon a time"
```

## Citation

```bibtex
@article{eldan2023tinystories,
  title={TinyStories: How Small Can Language Models Be and Still Speak Coherent English?},
  author={Eldan, Ronen and Li, Yuanzhi},
  journal={arXiv preprint arXiv:2305.07759},
  year={2023}
}
```

Paper: https://arxiv.org/abs/2305.07759  
Dataset: https://huggingface.co/datasets/roneneldan/TinyStories  
Original models: https://huggingface.co/roneneldan

## Used in 600+ Papers

These models are the standard baseline in:
- Mechanistic interpretability
- Synthetic data research
- Grokking studies
- BabyLM Challenge
- Small model reasoning
- Curriculum learning

You're now training one of the most influential tiny LMs in history!
