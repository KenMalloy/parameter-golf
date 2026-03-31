# Parameter Golf — NFT-Inspired Submission

## Project Context
Kenneth Malloy (NFT/Navigational Faculty Theory author) is exploring this competition through the lens of NFT concepts. The goal is to train the best language model fitting in 16MB, scored by val_bpb (bits per byte) on FineWeb validation set.

## Scoring System
- **Metric**: val_bpb = (cross_entropy / ln(2)) × (tokens / bytes) — lower is better
- **Artifact size**: code bytes + compressed model bytes ≤ 16,000,000 decimal bytes
- **Pipeline**: train → quantize (int5/int6/int8 per-row) → torch.save → zstd-22 compress → verify size → decompress → dequantize → evaluate
- **Current SOTA**: 1.1428 (10L, mixed int5/int6, BigramHash, SmearGate, SWA, sliding window eval)
- **Baseline**: 1.2244 (9L, 512dim, 1024vocab, tied embeddings)

## NFT → Strategy Mappings

### PRIMARY STRATEGY: Quantization Field (Soft Sigmoid QAT)

Simulates a quantum walk speedup over the quantization lattice. Instead of naive independent rounding (which finds the nearest lattice point coordinate-by-coordinate), the quantization field exploits full-dimensional structure to find better discrete representations.

**Core mechanism:** A temperature-controlled sigmoid holds each weight as a differentiable blend between its two nearest grid neighbors. Backprop computes which grid assignments reduce loss — this is the NFT "back-action." Temperature anneals from exploration (blend ≈ 0.5, superposition) to commitment (blend → 0 or 1, collapsed).

**Two-phase training:**
1. **Parent (float):** Standard baseline training → save checkpoint
2. **Child (NFT field active):** Load parent, activate soft quantization on all weight matrices, train with temperature annealing. At end, T → 0 and weights are hard-quantized with less damage than naive rounding.

**Why it should work:** The GPU processes a 512×1024 weight matrix as ONE object in 524,288-dimensional space. Naive quantization snaps each coordinate independently, but the nearest lattice point coordinate-by-coordinate is NOT the nearest lattice point measured by val_bpb. The NFT loop navigates the full space.

**Design doc:** `docs/plans/2026-03-22-nft-quantization-field-design.md`
**Implementation plan:** `docs/plans/2026-03-22-nft-quantization-field-plan.md` (9 tasks)

### SECONDARY STRATEGY: Branch-Walk Search (Montanaro-inspired)

Tree search over model design space — hyperparameters, architecture choices, training schedules. Built and working in `search/` directory. Has had initial probe runs. Provides systematic exploration of the design space to complement the quantization field work.

### Future Explorations

These NFT mappings are valid but not the current focus:

- **Probability Sculpting → QAT**: Sculpt weight distributions so discretization lands on optimal values. Mixed precision (int5 MLP, int6 attn) as adaptive measurement basis.
- **ENAQT → Noise-calibrated training**: Quantization noise as structured regularizer during training.
- **Criticality → Edge of capacity**: Fill the 16MB budget exactly. Phase boundary between precision and parameter count.
- **Adaptive Measurement → Test-time adaptation**: Sliding window eval, test-time training with LoRA.
- **Substrate Reuse → Depth Recurrence**: 3 unique blocks × 3 passes to free parameter budget for wider layers.
- **Multi-scale Transduction → BigramHash**: Character-pair statistics bridging sub-token to token representations.

## Development Approach
- Develop locally on Mac with MLX (`train_gpt_mlx.py`)
- Architecture changes are fully portable to H100s — relative val_bpb comparisons hold
- Quantization/compression math is hardware-independent
- Only hyperparameters (LR, batch size) need retuning for H100

## Architecture (Baseline MLX)
- Embedding: 1024 vocab × 512 dim, tied
- U-Net style: encoder blocks [0-3] accumulate skips, decoder blocks [4-8] consume reversed skips
- Each block: RMSNorm → GQA Attention (8 heads, 4 KV heads, RoPE) → RMSNorm → relu² MLP (2x expansion)
- Per-block learned scalars: attn_scale, mlp_scale, resid_mix
- Logit softcap (30.0)

## Current State & Next Steps

**Completed:**
- Baseline MLX training script (`train_gpt_mlx.py`) is working
- Branch-walk search harness built in `search/` with initial probe runs
- NFT quantization field design doc and 9-task implementation plan written

**Immediate next step:**
- Create `train_nft_mlx.py` (Task 1 of the implementation plan) — copy baseline, then incrementally add the soft quantization field per Tasks 2–8

**Not yet done:**
- `train_nft_mlx.py` does NOT exist yet
- No overnight NFT runs have completed (baseline overnight runs have been attempted)
- Two-phase parent→child workflow is planned but not implemented
