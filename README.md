# TSViT-DINOv3: Axial 2D RoPE Spatial Backbone & Masked Loss Optimization for Multi-Temporal Crop Segmentation

An enterprise-grade, publication-ready implementation integrating Meta AI's **DINO-v3** spatial representation mechanics into the **Temporal-Spatial Vision Transformer (TSViT)** for high-resolution, multi-temporal satellite crop classification on Sentinel-2 time series (PASTIS-24 and France agricultural benchmarks).

---

## 1. Architectural Motivation & System Overview

Agricultural crop phenology requires modeling both long-range seasonal temporal dynamics and localized spatial plot boundaries. Standard TSViT decouples processing into a Temporal Transformer followed by a Spatial Transformer. However, the original spatial encoder suffered from three architectural bottlenecks:

1. **Static 1D Positional Embeddings**: Flattens a 2D spatial grid $(H_p, W_p)$ into a 1D sequence and adds static learned vectors $P \in \mathbb{R}^{1 \times N_p \times D}$. This breaks spatial rotation equivariance and induces distortion along agricultural field boundaries.
2. **Standard MHA & MLP Saturation**: Standard multi-head attention without adaptive depth scaling exhibits gradient degradation when scaling depth beyond 4 layers on multi-spectral inputs.
3. **Loss Vulnerability under Extreme Class Imbalance**: Uniform Cross-Entropy produces biased gradients toward dominant background/cereal classes, neglecting critical minority crop boundaries.

```
Input Satellite Tensor: (B, T, C=11, H=24, W=24)
              │
              ▼  Patch Partitioning (P=2x2)
         (B, T, N_p=144, D=128)
              │
              ▼  Temporal Transformer (Self-Attention across T dimension)
         (B, T, N_p=144, D=128)
              │
              ▼  Temporal Aggregation / Pooling
         (B, N_p=144, D=128)
              │
  ┌───────────┴─────────────────────────────────────────────┐
  │         DINO-v3 Spatial Backbone (Replaces Vanilla)      │
  │  - 2D Axial RoPE: Q, K rotation via spatial coordinates  │
  │  - LayerScale: Diagonal gamma initialization (1e-5)      │
  │  - SwiGLU: SiLU(xW_gate) * xW_up -> W_down               │
  └───────────┬─────────────────────────────────────────────┘
              │
              ▼  Linear Segmentation Head + Reshape
        Spatial Logits: (B, H=24, W=24, NumClasses=19)
              │
              ▼
  ┌─────────────────────────────────────────────────────────┐
  │       Masked Adaptive Optimization Objective            │
  │  L_total = lambda_CE * L_MCE + lambda_Dice * L_MDice   │
  └─────────────────────────────────────────────────────────┘
```

---

## 2. Core Architectural Upgrades: TSViT with DINO-v3

### 2.1. 2D Axial Rotary Position Embedding (2D RoPE)
Instead of adding static position vectors to patch embeddings, DINO-v3 dynamically rotates the Query and Key vectors in the complex plane along the height and width spatial axes:

$$\mathbf{q}_{m, n} = \mathbf{R}_{\Theta, m, n}^{2\text{D}} \mathbf{W}_q \mathbf{x}_{m, n}, \quad \mathbf{k}_{m, n} = \mathbf{R}_{\Theta, m, n}^{2\text{D}} \mathbf{W}_k \mathbf{x}_{m, n}$$

Where $\mathbf{R}_{\Theta, m, n}^{2\text{D}} = \text{diag}(\mathbf{R}_{\Theta_h, m}, \mathbf{R}_{\Theta_w, n})$ decomposes into independent axial rotations corresponding to row $m \in [0, H_p-1]$ and column $n \in [0, W_p-1]$:

$$\mathbf{R}_{\Theta, p} = \begin{pmatrix} \cos(p \theta_i) & -\sin(p \theta_i) \\ \sin(p \theta_i) & \cos(p \theta_i) \end{pmatrix}$$

**Mathematical Property**: The attention inner product between patch $(m_1, n_1)$ and $(m_2, n_2)$ depends strictly on the relative spatial displacement $(\Delta m, \Delta n) = (m_1 - m_2, n_1 - n_2)$:

$$\langle \mathbf{q}_{m_1, n_1}, \mathbf{k}_{m_2, n_2} \rangle = g(\mathbf{x}_{m_1, n_1}, \mathbf{x}_{m_2, n_2}, m_1 - m_2, n_1 - n_2)$$

This guarantees strict shift-invariance across arbitrary satellite scene coordinates.

### 2.2. SwiGLU Gated Feed-Forward Network
Replacing standard two-layer GELU MLPs, the DINO-v3 spatial blocks implement the Swish Gated Linear Unit:

$$\text{SwiGLU}(\mathbf{x}) = \left( \text{SiLU}(\mathbf{x} \mathbf{W}_{\text{gate}}) \odot (\mathbf{x} \mathbf{W}_{\text{up}}) \right) \mathbf{W}_{\text{down}}$$

Where $\mathbf{W}_{\text{gate}}, \mathbf{W}_{\text{up}} \in \mathbb{R}^{D \times D_{\text{mlp}}}$ and $\mathbf{W}_{\text{down}} \in \mathbb{R}^{D_{\text{mlp}} \times D}$. The multiplicative gating suppresses noisy spectral channels while amplifying salient parcel edge responses.

### 2.3. LayerScale Convergence Stabilization
To prevent early representation collapse during spatial feature refinement, diagonal scaling vectors $\gamma_1, \gamma_2 \in \mathbb{R}^D$ are initialized to $\varepsilon = 10^{-5}$:

$$\mathbf{x}'_l = \mathbf{x}_l + \text{diag}(\boldsymbol{\gamma}_{1, l}) \cdot \text{Attention}(\text{LN}(\mathbf{x}_l))$$
$$\mathbf{x}_{l+1} = \mathbf{x}'_l + \text{diag}(\boldsymbol{\gamma}_{2, l}) \cdot \text{SwiGLU}(\text{LN}(\mathbf{x}'_l))$$

---

## 3. Loss Function Formulations & Optimization Dynamics

Earth Observation datasets feature irregular parcel geometries, unannotated non-agricultural regions (void/clouds/roads), and heavy long-tailed class distributions (e.g., Meadow vs. Sorghum). The loss suite in `metrics/loss_functions.py` was systematically updated to support masked execution and compound multi-objective balancing.

### 3.1. Masked Cross-Entropy Loss ($\mathcal{L}_{\text{MCE}}$)
Vanilla Cross-Entropy evaluates across all pixels indiscriminately. In agricultural remote sensing, ground truth masks $M \in \{0, 1\}^{B \times H \times W}$ specify valid parcel labels. $\mathcal{L}_{\text{MCE}}$ isolates valid pixels through vectorized tensor slicing:

$$\mathcal{L}_{\text{MCE}}(\mathbf{Z}, \mathbf{Y}, \mathbf{M}) = - \frac{1}{\sum_{i=1}^{N} m_i} \sum_{i=1}^{N} m_i \cdot \log \left( \frac{\exp(z_{i, y_i})}{\sum_{c=1}^{C} \exp(z_{i, c})} \right)$$

Where $N = B \cdot H \cdot W$, $z_{i, c}$ denotes the predicted logit for class $c$ at pixel $i$, and $y_i$ is the target class.

**Vectorized PyTorch Implementation** (`metrics/loss_functions.py` lines 141–189):
```python
class MaskedCrossEntropyLoss(torch.nn.Module):
    def __init__(self, mean=True):
        super(MaskedCrossEntropyLoss, self).__init__()
        self.mean = mean

    def forward(self, logits, ground_truth):
        target, mask = ground_truth if len(ground_truth) == 2 else (ground_truth[0], None)
        
        if mask is not None:
            mask_flat = mask.reshape(-1, 1).bool()
            nclasses = logits.shape[-1]
            logits_flat = logits.reshape(-1, nclasses)
            masked_logits = logits_flat[mask_flat.squeeze(-1)]
            masked_target = target.reshape(-1, 1)[mask_flat.squeeze(-1)].to(torch.int64)
        else:
            masked_logits = logits.reshape(-1, logits.shape[-1])
            masked_target = target.reshape(-1, 1).to(torch.int64)

        log_probs = F.log_softmax(masked_logits, dim=-1)
        loss = -torch.gather(log_probs, dim=1, index=masked_target)
        return loss.mean() if self.mean else loss
```

### 3.2. Masked Focal Loss ($\mathcal{L}_{\text{MFL}}$) for Long-Tailed Crop Classes
To prevent dominant vegetative classes (e.g., soft winter wheat) from overwhelming gradient updates, Masked Focal Loss applies a dynamic modulating factor $(1 - p_t)^\gamma$:

$$\mathcal{L}_{\text{MFL}} = - \frac{1}{\sum_i m_i} \sum_{i=1}^{N} m_i \cdot \alpha_{y_i} (1 - p_{t, i})^\gamma \log(p_{t, i})$$

Where $p_{t, i} = \text{Softmax}(\mathbf{z}_i)_{y_i}$, $\gamma \in [1.0, 2.5]$ focuses optimization onto hard parcel transitions, and $\alpha \in \mathbb{R}^C$ provides class-balanced inverse frequency weighting.

### 3.3. Multi-Class Masked Dice Loss ($\mathcal{L}_{\text{MDice}}$)
To optimize boundary alignment directly in accordance with Intersection-over-Union (mIoU) metrics, Masked Dice loss computes generalized soft spatial overlap over valid masked pixels:

$$\mathcal{L}_{\text{MDice}} = 1 - \frac{1}{C} \sum_{c=1}^{C} \frac{2 \sum_{i \in \mathcal{M}} p_{i, c} \cdot y_{i, c} + \epsilon}{\sum_{i \in \mathcal{M}} p_{i, c}^2 + \sum_{i \in \mathcal{M}} y_{i, c}^2 + \epsilon}$$

Where $\mathcal{M} = \{i \mid m_i = 1\}$ is the set of annotated pixel coordinates, and $\mathbf{y}_i \in \{0, 1\}^C$ is the one-hot target tensor.

### 3.4. Compound Hybrid Objective Function
For state-of-the-art segmentation, the configuration engine supports compound loss blending:

$$\mathcal{L}_{\text{Total}} = \lambda_{\text{CE}} \mathcal{L}_{\text{MCE}} + \lambda_{\text{Focal}} \mathcal{L}_{\text{MFL}} + \lambda_{\text{Dice}} \mathcal{L}_{\text{MDice}}$$

* **Default solver recipe**: $\lambda_{\text{CE}} = 1.0$, $\lambda_{\text{Dice}} = 0.5$, $\gamma = 1.5$.
* **Benefits with DINO-v3**: Sharpens parcel contours while preventing boundary over-smoothing typical of standalone cross-entropy.

---

## 4. Codebase Architecture & File Mapping

| Module Path | Role in Architecture | Key Technical Upgrades |
|---|---|---|
| [`models/TSViT/dinov3.py`](models/TSViT/dinov3.py) | **DINO-v3 Core Engine** | Implements `DINOv3RoPE`, `DINOv3LayerScale`, `DINOv3GatedMLP`, `DINOv3Attention`, `DINOv3SpatialEncoder`. |
| [`models/TSViT/TSViTdense.py`](models/TSViT/TSViTdense.py) | **Dense Segmentation Model** | Replaces `space_transformer` with `DINOv3SpatialEncoder`. Passes 2D grid dimensions $(H_p, W_p)$ to RoPE during forward pass. |
| [`models/TSViT/TSViTcls.py`](models/TSViT/TSViTcls.py) | **Crop Classification Model** | Incorporates DINO-v3 spatial encoding with dedicated `[CLS]` token preservation. |
| [`metrics/loss_functions.py`](metrics/loss_functions.py) | **Optimization Loss Suite** | Implements `MaskedCrossEntropyLoss`, `MaskedFocalLoss`, `MaskedDiceLoss`, `MaskedContrastiveLoss`, and per-class loss diagnostics. |
| [`models/__init__.py`](models/__init__.py) | **Factory Dispatcher** | Streamlined entry point routing for `TSViT_DINOv3` and `TSViTcls_DINOv3`. |
| [`configs/PASTIS24/TSViT_DINOv3_fold1.yaml`](configs/PASTIS24/TSViT_DINOv3_fold1.yaml) | **Experiment Configuration** | Full YAML recipe specifying hyperparameters, solver schedules, and loss parameters. |

---

## 5. Experiment Setup & Training Workflow

### 5.1. Solver Configuration (`configs/PASTIS24/TSViT_DINOv3_fold1.yaml`)

```yaml
MODEL:
  architecture: "TSViT_DINOv3"
  spatial_backbone: "dinov3"
  img_res: 24
  patch_size: 2
  max_seq_len: 60
  num_channels: 11
  num_classes: 19
  dim: 128
  temporal_depth: 4
  spatial_depth: 4
  heads: 4
  dim_head: 32
  scale_dim: 4

SOLVER:
  num_epochs: 100
  num_warmup_epochs: 10
  steps: (0, 80000)
  loss_function: masked_cross_entropy   # options: [masked_cross_entropy, masked_focal_loss, masked_dice_loss]
  class_weights: {}
  lr_scheduler: 'cosine'
  lr_base: 1.0e-3
  lr_min: 5.0e-6
  lr_start: 1.0e-8
  weight_decay: 0.01

CHECKPOINT:
  load_from_checkpoint: 'Checkpoint/checkpoint/best.pth'
  partial_restore: True
  save_path: 'models/saved_models/PASTIS24/TSViT_DINOv3_fold1'
```

### 5.2. Training & Evaluation Execution

To train TSViT with the DINO-v3 spatial backbone:

```bash
# Multi-GPU Semantic Segmentation (PASTIS-24 benchmark)
python train_and_eval/segmentation_training_transf.py \
  --config configs/PASTIS24/TSViT_DINOv3_fold1.yaml \
  --device 0,1

# Single-GPU Field-Level Classification
python train_and_eval/classification_train_transf.py \
  --config configs/PASTIS24/TSViT_cls.yaml \
  --device 0
```

