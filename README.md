# Does CLIP Understand "No"? — Reproducing CLIP's Negation Blindness

A mini research project that reproduces and quantifies how vision-language models like
**CLIP fail to handle negation**, and shows that this weakness persists consistently
across different models and datasets.

> **Motivation**: After reading the ICCV 2025 paper from Korea University's Vision & AI Lab,
> *"Know 'No' Better: A Data-Driven Approach for Enhancing Negation Awareness in CLIP"* (Park et al.),
> I wanted to verify by hand how severe CLIP's negation weakness actually is, and whether it
> survives changes in model size and pretraining data.

---

## TL;DR

- CLIP can barely distinguish `"a photo of a cat"` from `"a photo of no cat"`.
- Even when the correct class is **negated**, CLIP still ranks it #1 most of the time
  (**negation-ignored rate** of 71–93%).
- Scaling the model (ViT-B/32 → ViT-L/14) or using more pretraining data
  (OpenAI → LAION-2B) does **not** fix it → supporting the paper's hypothesis that
  the root cause lies in the **training data, not model capacity**.

---

## 1. Core Experiment: Single-Model Reproduction (CIFAR-10)

`clip_negation.ipynb` — Using CLIP ViT-B/32 (OpenAI) on CIFAR-10, I measure three things:

1. **Standard zero-shot baseline**: classification accuracy with `"a photo of a {class}"`
2. **Negation-ignored rate**: how often the true class is still ranked #1 even when the
   prompt is negated to `"a photo of no {class}"` (higher = the model ignored `no`)
3. **Similarity distribution**: cosine similarity of positive vs. negated prompts on the
   same image

### Results

| Metric | Value |
|--------|-------|
| Positive prompt accuracy | **86.4%** |
| Avg. negation-ignored rate | **88.5%** |
| Mean positive–negative similarity gap | **0.0045** |

The negation-ignored rate (88.5%) is even **higher** than the positive accuracy (86.4%) —
CLIP essentially reads `no cat` as `cat`. The similarity gap is near zero, meaning the
negation word has almost no effect on the embedding.

![negation ignored bar](result_bar.png)
![similarity distribution](result_simdist.png)

---

## 2. Generalization: Model × Dataset Grid

`clip_negation_level1.ipynb` — To show this is not a quirk of one setup, I measure the
negation-ignored rate across **3 models × 2 datasets**.

- Models: ViT-B/32 (OpenAI), ViT-L/14 (OpenAI), ViT-B/32 (LAION-2B)
- Datasets: CIFAR-10, Oxford-IIIT Pets

### Negation-ignored rate (%) — higher is worse

| Model \ Dataset | CIFAR-10 | Oxford Pets |
|---|---|---|
| ViT-B/32 (OpenAI) | 87.6 | 71.0 |
| ViT-L/14 (OpenAI) | 87.9 | 82.9 |
| ViT-B/32 (LAION-2B) | 93.2 | 83.9 |

### Baseline zero-shot accuracy (%) — for reference

| Model \ Dataset | CIFAR-10 | Oxford Pets |
|---|---|---|
| ViT-B/32 (OpenAI) | 85.7 | 78.2 |
| ViT-L/14 (OpenAI) | 92.3 | 89.7 |
| ViT-B/32 (LAION-2B) | 94.1 | 88.3 |

![negation heatmap](level1_heatmap_negation.png)
![paired bars](level1_bars.png)

### Findings

- **All 6 combinations** show negation-ignored rates far above chance
  (10% for CIFAR-10, ~2.7% for Pets) → ignoring negation is a **systematic behavior**
  of CLIP-family models.
- Scaling up the model (B/32 → L/14) does not help (Pets even worsens, 71.0 → 82.9).
- The model trained on more data (LAION-2B) is actually **worse** (93.2 on CIFAR-10).
- This matches the paper's claim: the issue is **not model capacity or data volume**,
  but the **lack of negation-containing examples in the training data**.

---

## How to Run

Open the notebooks in **Google Colab with a GPU runtime** and run the cells top to bottom.
The first cell installs the required package:

```bash
pip install open_clip_torch
```

Datasets and CLIP weights are downloaded automatically on first run.

---

## Next Steps (Planned)

- **Cleaner metric**: measure negation awareness only on images the baseline got correct,
  for a fairer estimate.
- **Compositional negation**: extend to `"A and no B"` prompts (multi-object, COCO-based).
- **Training-free mitigation**: explore how far simple score correction can go without
  fine-tuning (the paper solves it via data generation + fine-tuning).

---

## References

- Park et al. *Know "No" Better: A Data-Driven Approach for Enhancing Negation Awareness
  in CLIP.* ICCV 2025. (Vision & AI Lab, Korea University)
- Radford et al. *Learning Transferable Visual Models From Natural Language Supervision* (CLIP). 2021.
