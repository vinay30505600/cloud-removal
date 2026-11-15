# SAR-Guided Cloud Removal for LISS-IV Optical Imagery

A lightweight deep learning model that removes clouds from optical satellite imagery by fusing the cloudy optical image with a co-registered Sentinel-1 SAR image. SAR sees through clouds, so it provides the ground structure that the optical image has lost.

Built for the North Eastern Region (NER) of India, where persistent cloud cover limits the usable optical data. The whole model was designed, trained and run on RTX 3050 laptop GPUs (4 GB VRAM), so it is small on purpose.

---

## Overview

```
Cloudy optical [3,256,256] ──► CNN encoder ──┐
                                             ├─► cross-attention fusion ─► bottleneck ─► decoder ─► cloud-free optical [3,256,256]
SAR (VV, VH)   [2,256,256] ──► CNN encoder ──┘                                                ▲
                                  optical skip connections ───────────────────────────────────┘
```

| Item | Value |
|---|---|
| Input 1 | Cloudy optical image, 3 bands, 256×256, normalized to [0, 1] |
| Input 2 | Sentinel-1 SAR, 2 bands (VV, VH, in dB), 256×256, normalized to [-1, 1] |
| Output | Cloud-free optical image, 3 bands, 256×256, in [0, 1] |
| Parameters | about 18M |
| Training | 25 epochs, 40,000 optimizer steps |

---

## Model architecture

All tensor shapes are per image (add a batch dimension `B` in front).

### 1. Two CNN encoders (one for optical, one for SAR)

Each encoder has 3 levels. A level is a **ConvBlock** followed by a 2×2 max-pool. The two encoders have separate weights and differ only in their input channels (3 for optical, 2 for SAR).

| Level | Output of ConvBlock | After max-pool |
|---|---|---|
| 1 | `[64, 256, 256]` (skip 1) | `[64, 128, 128]` |
| 2 | `[128, 128, 128]` (skip 2) | `[128, 64, 64]` |
| 3 | `[256, 64, 64]` (skip 3) | `[256, 32, 32]` |

**ConvBlock:** Conv3×3 → InstanceNorm → LeakyReLU(0.2), twice. InstanceNorm is used because batch sizes are tiny (2 per forward pass). Skip maps are saved before pooling.

### 2. Cross-attention fusion

Each encoder output `[256, 32, 32]` is flattened into 1024 tokens of dimension 256, and a learnable positional embedding is added once to each stream.

The optical tokens then pass through **3 identical blocks**. Each block contains:

1. **Cross-attention** (optical queries, SAR keys and values)
2. **Self-attention** on the optical tokens
3. **Feed-forward network** (256 → 1024 → 256, GELU)

Every sub-layer uses pre-LayerNorm and a residual connection. The attention uses 8 heads of dimension 32, with no masking. The SAR tokens are never modified: they are a fixed source of information for every block.

### 3. Bottleneck and decoder

The fused tokens are reshaped back to `[256, 32, 32]`, then:

```
Bottleneck: ConvBlock(256→512), ConvBlock(512→512)         [512,32,32]
Up 1: upsample ×2, concat optical skip 3, ConvBlock → 256   [256,64,64]
Up 2: upsample ×2, concat optical skip 2, ConvBlock → 128   [128,128,128]
Up 3: upsample ×2, concat optical skip 1, ConvBlock → 64    [64,256,256]
Head: Conv 1×1 (64→3) → Sigmoid                             [3,256,256]
```

Upsampling is bilinear (`align_corners=False`). Skip connections are concatenated, not added.

Source: [`cloud_removal_architecture.py`](cloud_removal_architecture.py)

---

## Data

### Source

- **Optical:** 3-band scenes of about 2228×2228 pixels (raw values 0 to about 10,000)
- **SAR:** Sentinel-1, 2 bands (VV, VH), already in dB, aligned to the optical grid

### Preprocessing

1. **Normalization limits** are computed once from the training scenes only: the 1st and 99th percentile of each band, using a random pixel sample, with invalid and very bright pixels excluded. They are saved to `norm_limits.npz` and reused unchanged for validation, test and inference.
   - Optical: `clip((x - lo) / (hi - lo), 0, 1)` → [0, 1]
   - SAR: `clip(x, lo, hi)`, then scaled to [-1, 1]
2. **Patches:** 256×256, stride 128 (50% overlap) for training. Scenes are split into train, val and test **before** cutting patches, so no patch overlaps across splits.
3. **Filtering:** patches with missing data, flat patches, and patches with visible real cloud (more than 1% of pixels with mean brightness above 0.9) are dropped. The remaining patches are used as cloud-free targets.
4. **Synthetic clouds:** each patch gets a random soft-edged cloud covering 10% to 70% of the patch, with opacity 0.5 to 1.0, blended toward white:

   ```
   cloudy = clear × (1 − w) + w          w = alpha × cloud_mask
   ```

### Stored format

Each scene is saved as `image_k.npy` with shape `[N, 6, 256, 256]` (float16, 256 patches per file):

| Channels | Content |
|---|---|
| `0:3` | clear (target) optical |
| `3:5` | SAR |
| `5:6` | cloud weight `w` (also used as the mask in the losses) |

The model input is rebuilt as `cloudy = clear × (1 − w) + w`. Training used 12,800 patches (50 files).

---

## Loss function

```
Loss = 1.0 · L1_weighted + 0.5 · (1 − SSIM) + 0.05 · Spectral
```

| Term | Purpose |
|---|---|
| **Weighted L1** | mean absolute error where cloudy pixels count 5× (`weight = 1 + 4·mask`), so the model cannot score well by copying clear areas |
| **1 − SSIM** | Gaussian-window structural similarity (11×11, σ = 1.5), keeps edges and texture sharp and avoids blur |
| **Spectral** | per-band mean and standard deviation match inside the cloudy region, prevents colour drift |

The cloud mask is used only inside the loss, not as a model input. All terms are means over the batch.

---

## Training

| Setting | Value |
|---|---|
| Hardware | RTX 3050 laptop GPU (4 GB VRAM) |
| Training patches | 12,800 |
| Effective batch size | 8 (micro-batch 2 × gradient accumulation 4) |
| Steps per epoch | 1,600 |
| **Total** | **25 epochs = 40,000 steps** |
| Optimizer | AdamW, weight decay 1e-4 |
| Peak LR | 3e-4 |
| Schedule | 1,000-step linear warmup (from 3e-5), then cosine decay to 1e-6 (`LambdaLR`, stepped once per optimizer step) |
| Gradient clipping | max norm 1.0 |
| Precision | fp16 autocast, loss computed in fp32 |
| Augmentation | random 90° rotations and flips, applied identically to optical, SAR and mask |
| Checkpoint | best validation loss |

The loss and gradient norm of every step are in `training_log.csv`.

### Training behaviour

- The loss started at about 1.2 to 1.6 and fell steadily during warmup.
- Training loss was around 0.10 near epoch 21, while validation loss was still falling (0.56 at epoch 1, 0.48 at epoch 10, 0.33 at epoch 20, 0.30 at epoch 22).
- Gradient norms stayed stable at about 1.0, with only isolated spikes handled by clipping.

---

## Results

> Fill in after final evaluation on the held-out test scenes. Metrics are computed on **cloudy pixels only** with fixed (seeded) clouds, so they are reproducible.

| Method | PSNR (dB) | SSIM |
|---|---|---|
| Cloudy input as-is (baseline) | TBD | TBD |
| Model without SAR (ablation) | TBD | TBD |
| **Ours (optical + SAR)** | TBD | TBD |

Final validation loss: TBD. Sample outputs: add a few `(cloudy, prediction, clear)` triplets here.

---

## Usage

### Inference

```python
import numpy as np, torch

lim = np.load("norm_limits.npz")
OPT_LO, OPT_HI = lim["OPT_LO"].reshape(3,1,1), lim["OPT_HI"].reshape(3,1,1)
SAR_LO, SAR_HI = lim["SAR_LO"].reshape(2,1,1), lim["SAR_HI"].reshape(2,1,1)

# optical_raw [3,256,256], sar_raw [2,256,256]: same units as the training data
opt = np.clip((optical_raw - OPT_LO) / (OPT_HI - OPT_LO), 0, 1)
sar = (np.clip(sar_raw, SAR_LO, SAR_HI) - SAR_LO) / (SAR_HI - SAR_LO) * 2 - 1

model = Model(ConvBlock, Encoder, Encodings, CrossAttention, SelfAttention, FNN, BLOCK, Decoder)
model.load_state_dict(torch.load("best.pt", map_location="cpu"))
model.eval()

with torch.no_grad():
    pred = model(torch.from_numpy(opt)[None].float(),
                 torch.from_numpy(sar)[None].float())[0].numpy()    # [3,256,256] in [0,1]

restored = pred * (OPT_HI - OPT_LO) + OPT_LO                          # back to original units
```

Important: the **same `norm_limits.npz`** must be used for training and inference. Without it, the model's outputs are meaningless. Inputs must be 256×256 and spatially aligned (same pixel grid for optical and SAR). For larger scenes, tile into 256×256 patches (with overlap) and stitch.

---

## Repository layout

```
-Architecture.py   #contains the actual architecture code and losses and  code
-data_collection.ipynb  #contains the code used to collect all training and validation data
-data_preprocessing.ipynb #contains the code for normalization and preprocessing of raw images(includes breaking into patches and masking the clean images)
-get_batches.py -  code for training data pipeline the function inside provides the corresponding input and target given the training step
-training.ipynb -  final training code
-validation.ipynb - validation code
-readme

## Possible improvements

- Generate clouds on the fly each time a patch is loaded
- Add SAR skip connections to the decoder
- Add a perceptual (VGG) loss for sharper textures
- Simulate cloud shadows and thin haze
- Evaluate on real cloudy scenes, and add more NER terrain types (forest, urban, river valley, hill)

Training and validation loss summary after each epoch-
Training loss-
0.35807297903687496
0.21523877793375001
0.2172048061190299
0.20263159567024558
0.19101758587406947
0.17835968116763978
0.17091006034985184
0.16405776858562604
0.1551909752585925
0.15033996433019636
0.14493710288545114
0.14133122948464005
0.1354554377822205
0.13129058737074953
0.12659957720199597
0.12357279919320714
0.10803011278156188
0.11632363012526181
0.11292218406917524
0.1098046997748315
0.10721941997995596
0.10480011333711442
0.1024431874370202
0.0998884525382891
Validation loss-
validation loss before training-1.250
validation loss after 200 steps-0.7958
validation Loss after 500 steps-0.6175
validation Loss after 1000 steos-0.5598
validation Loss after 1 epoch - 0.5597
validation Loss after 2 epoch-0.5707
validation loss after 3 epoch-0.5718
validation Loss after 4 epoch-0.5484
validation Loss after 5 epoch-0.5231
validation Loss after 6 epoch-0.5138
validation Loss after 7 epoch-0.5074
validation Loss after 10 epoch-0.4772
validation Loss after 15 epoch-0.3910
validation Loss after 16 epoch-0.3814
validation Loss after 17 epoch-0.4502
validation Loss after 18 epoch-0.3532
validation Loss after 20 epoch-0.3278
validation Loss after 21 epoch-0.3120
validation Loss after 22 epoch-0.2953
validation Loss after 23-epoch-0.2835
validation Loss after 24-epoch-0.2704
validation Loss after 25-epoch-0.2557
0.09799771366408092
