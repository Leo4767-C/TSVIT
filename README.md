# TSViT-DINOv3: Integrating DINO-v3 Spatial Backbone into Temporal-Spatial Vision Transformer

---

## Mục lục

| # | Mục | Nội dung tóm tắt |
|:---:|---|---|
| **1** | [Project Overview](#1-project-overview) | Dự án làm gì, cho ai, giải quyết vấn đề gì. |
| **2** | [System Overview](#2-system-overview) | Kiến trúc ngắn gọn, luồng xử lý chính. |
| **3** | [Environment](#3-environment) | Yêu cầu môi trường, version, file cần có. |
| **4** | [Setup](#4-setup) | Các bước cài đặt theo thứ tự. |
| **5** | [Run & Verify](#5-run--verify) | Cách chạy và cách biết chạy đúng. |
| **6** | [Usage / API](#6-usage--api) | Cách dùng giao diện hoặc gọi API. |
| **7** | [AI Model](#7-ai-model) | Model version, input/output, giới hạn. |
| **8** | [Troubleshooting](#8-troubleshooting) | Lỗi thường gặp và cách xử lý. |

---

## 1. Project Overview

### Tổng quan dự án

Dự án này xây dựng **TSViT-DINOv3**, một kiến trúc nâng cấp của **Temporal-Spatial Vision Transformer (TSViT)** nhằm phân đoạn ngữ nghĩa ảnh viễn thám đa thời gian (Satellite Image Time Series — SITS) cho bài toán phân loại cây trồng nông nghiệp.

Trong kiến trúc TSViT gốc, bộ mã hóa không gian dùng ViT cơ bản với mã hóa vị trí 1D tĩnh, MLP dùng GELU, và kết nối tàn dư thông thường. TSViT-DINOv3 thay thế bộ mã hóa không gian này bằng **DINO-v3 Spatial Backbone**, mang đến ba cải tiến chính:

1. **2D Axial Rotary Position Embedding (2D RoPE)** — mã hóa vị trí không gian 2 chiều liên tục, bất biến tịnh tiến.
2. **SwiGLU Gated Feed-Forward Networks** — mạng truyền thẳng có cổng cấp, tránh hiện tượng neuron chết.
3. **LayerScale Diagonal Stabilization** (γ = 10⁻⁵) — ổn định gradient khi huấn luyện sâu.

### Thông số dự án

| Chiều | Giá trị | Ghi chú |
|---|---|---|
| **Nhiệm vụ chính** | Phân đoạn ngữ nghĩa cấp pixel & phân loại cây trồng cấp thửa ruộng | Bản đồ loại cây trồng |
| **Tập dữ liệu** | **PASTIS-24** (Fold 1–5) | Chuỗi thời gian ảnh Sentinel-2 nông nghiệp thực tế |
| **Độ phân giải không gian** | 24 × 24 pixel (10 m/pixel → 240 × 240 m mặt đất) | Giữ nguyên độ phân giải gốc |
| **Chiều dài chuỗi thời gian** | T = 60 ngày chụp | Khoảng thời gian không đều, bao phủ toàn bộ chu kỳ sinh trưởng |
| **Kênh phổ** | 10 băng tần Sentinel-2 + 1 chỉ số DOY = 11 kênh | Phản xạ đa phổ |
| **Số lớp mục tiêu** | 19 loại cây trồng (nhãn 0–18) + nhãn Void (19) | Mặt nạ hiệu lực động M |
| **Đóng góp kiến trúc** | Đưa DINO-v3 ViT Engine vào tầng không gian | 2D RoPE + SwiGLU + LayerScale |
| **Khả năng hạ nguồn** | Phân đoạn pixel (`TSViT_DINOv3`) & Phân loại tile (`TSViTcls_DINOv3`) | Hỗ trợ trong cùng mã nguồn |

### Câu hỏi nghiên cứu

1. **Làm sao bảo toàn hình học thửa ruộng chi tiết trong Vision Transformer?**
   Vanilla ViT duỗi phẳng lưới không gian thành chuỗi 1D và cộng vector vị trí 1D học được, mất đi hình học tọa độ Euclid. 2D Axial RoPE của DINO-v3 nhúng các phép quay tần số liên tục trực tiếp vào phép chiếu query và key, đảm bảo tính bất biến tịnh tiến toán học qua các ranh giới thửa ruộng.

2. **Làm sao ổn định transformer không gian sâu đa thời gian khi chuyển giao học?**
   Thay thế khối transformer tiêu chuẩn thường làm mất ổn định trọng số đã huấn luyện. Bằng LayerScale (γ = 10⁻⁵) và SwiGLU, mạng bắt đầu tối ưu gần như là toán tử đồng nhất (x ≈ x), cho phép khôi phục trọng số từng phần (`partial_restore: True`) suôn sẻ.

3. **Làm sao giữ đầy đủ 10 băng phổ cảm biến mà không chuyển đổi phá hủy sang RGB?**
   Khác với các phương pháp thay thế đầu vào bằng mô hình DINOv3 RGB (buộc bỏ 7 băng Sentinel-2 và nội suy 24×24 → 192×192), thiết kế của chúng tôi đưa DINO-v3 vào **tầng attention không gian**, giữ nguyên 10 băng phổ vật lý và hiệu chuẩn cảm biến gốc.

---

## 2. System Overview

### Kiến trúc tổng thể

TSViT-DINOv3 phân rã bài toán attention trên cả không gian và thời gian thành hai cơ chế attention tuần tự: attention tự thời gian theo ngày chụp, rồi attention tự không gian theo các patch.

<p align="center">
  <img src="figures/tsvit-baseline.png" alt="Kiến trúc TSViT gốc" width="850"/>
  <br>
  <em>Hình 1: Kiến trúc Temporal-Spatial Vision Transformer (TSViT) gốc từ Tarasiou et al. (CVPR 2023), minh họa bộ mã hóa thời gian tách rời, trích xuất class-token, bộ mã hóa không gian vanilla với positional embedding 1D, và đầu tái tạo pixel dày đặc.</em>
</p>

Luồng xử lý chính:

```
Ảnh vệ tinh (10 băng + DOY)
    → Patch Embedding
    → Temporal Transformer (T=60)
    → DINO-v3 Spatial Backbone
    → Segmentation Head
    → Bản đồ phân đoạn pixel
```

<p align="center">
  <img src="figures/tsvit-dinov3-spatial.png" alt="Kiến trúc TSViT-DINOv3 tích hợp" width="850"/>
  <br>
  <em>Hình 2: Kiến trúc TSViT-DINOv3 sau khi thay thế Spatial Transformer vanilla bằng DINO-v3 Vision Backbone. Đặc trưng từ Temporal Transformer trải qua phép chiếu tuyến tính (d → D), nối token [CLS] và register động, các ViT block với 2D Axial RoPE self-attention, và lọc token trước khi đến segmentation head dày đặc.</em>
</p>

### Dòng dữ liệu qua hệ thống (Tensor Lifecycle)

| Giai đoạn | Module | Kích thước đầu vào | Kích thước đầu ra | Phép toán & ghi chú |
|:---:|:---:|:---:|:---:|---|
| **1** | DataLoader | — | `(B, 60, 11, 24, 24)` | B batch, T=60 ngày chụp, 10 băng S2 + 1 DOY, 24×24 pixel. |
| **2** | DOY Extraction | `(B, 60, 11, 24, 24)` | Reflectance: `(B, 60, 10, 24, 24)`, DOY: `(B, 60)` | Tách kênh cuối. DOY chuẩn hóa về [0, 1] rồi ánh xạ sang ngày 0–365. |
| **3** | DOY Embedding | `(B, 60)` | `(B, 60, 128)` | `F.one_hot(doy, 366)` rồi `nn.Linear(366, 128)`. |
| **4** | Patch Embedding | `(B, 60, 10, 24, 24)` | `(B*144, 60, 128)` | Patch hóa 2×2 không chồng lấp (P=2 → 12×12 = 144 patch). Chiếu tuyến tính 40 → 128. |
| **5** | DOY Addition | `(B*144, 60, 128)` | `(B*144, 60, 128)` | Cộng broadcast positional embedding thời gian cho mỗi chuỗi patch. |
| **6** | Temporal Tokens | `(1, 19, 128)` | `(B*144, 79, 128)` | Nối K=19 class token học được vào đầu chuỗi thời gian: T' = 79. |
| **7** | Temporal ViT | `temporal_transformer` | `(B*144, 19, 128)` | 4 Transformer block với attention thời gian; trích đầu ra class token `[:, :19]`. |
| **8** | Spatial Rearrange | Permute & Rearrange | `(B*19, 144, 128)` | Nhóm lại sao cho mỗi lớp k ∈ {0…18}, toàn bộ 144 patch tạo thành lưới. |
| **9** | **DINO-v3 Backbone** | [`DINOv3SpatialEncoder`](models/TSViT/dinov3.py) | `(B*19, 144, 128)` | 4 block DINO-v3: quay 2D RoPE (lưới 12×12), LayerScale (γ=10⁻⁵), SwiGLU MLP. |
| **10** | Segmentation Head | `mlp_head` | `(B*19, 144, 4)` | `nn.LayerNorm(128)` → `nn.Linear(128, 4)` chiếu mỗi embedding patch xuống P² = 4 pixel. |
| **11** | Dense Logit Fold | Rearrange | `(B, 19, 24, 24)` | Tái cấu trúc patch 12×12 và sub-pixel 2×2 thành logit dày đặc toàn bộ độ phân giải. |

### Biến thể kiến trúc: Phân đoạn và Phân loại

Mã nguồn hỗ trợ cả phân đoạn dày đặc và phân loại toàn cục:

```python
# Phân đoạn ngữ nghĩa (models/TSViT/TSViTdense.py)
class TSViT_DINOv3(TSViT):
    def __init__(self, model_config):
        model_config = dict(model_config)
        model_config['spatial_backbone'] = 'dinov3'
        super().__init__(model_config)
```

Trong [`models/TSViT/TSViTcls.py`](models/TSViT/TSViTcls.py), phân loại nối thêm token [CLS] không gian chuyên biệt:

```python
# Phân loại cây trồng với bảo toàn token [CLS] (models/TSViT/TSViTcls.py)
cls_space_tokens = repeat(self.space_token, '() N d -> b N d', b=B * self.num_classes)
x = torch.cat((cls_space_tokens, x), dim=1)  # Shape: (B * 19, 1 + 144, 128)

# Trong dinov3.py: has_cls_token=True bảo vệ token [CLS] khỏi phép quay tọa độ 2D
x = self.space_transformer(x, h=self.num_patches_1d, w=self.num_patches_1d, has_cls_token=True)[:, 0]
out = self.mlp_head(x)  # Output Shape: (B, 19)
```

### So sánh hướng tiếp cận kiến trúc

* **Hướng 1 — Thay thế Patch-Embedding đầu vào**:
  `Ảnh vệ tinh (10 băng)` → *(Bỏ 7 băng)* → `RGB` → *(Nội suy 24×24 lên 192×192)* → `DINOv3 đông cứng` → `Chiếu` → `TSViT`

* **Hướng 2 — Thay thế toàn bộ Spatial Backbone (hướng của chúng tôi)**:
  `Ảnh vệ tinh (10 băng + DOY)` → `Patch Embedding` → `Temporal ViT (T=60)` → `DINO-v3 Spatial Backbone` → `Segmentation Head`

Lý do hướng 2 phù hợp hơn cho SITS:
1. **Giữ nguyên hiệu chuẩn đa phổ gốc**: Sentinel-2 cung cấp 10 băng phản xạ vật lý cần thiết cho chỉ số thực vật (Red Edge B05–B07, NIR B08, SWIR B11–B12). Bỏ 7 băng để dùng backbone RGB sẽ mất tín hiệu sinh lý cây trồng quan trọng.
2. **Tránh nội suy song tuyến phá hủy**: Ép tile 24×24 lên 192×192 bằng nội suy 8× tạo ra nhiễu mờ mà không thêm độ phân giải mặt đất thực.
3. **Phân rã thời gian trước**: Áp dụng DINO-v3 sau bộ mã hóa thời gian cho phép attention không gian hoạt động trực tiếp trên class token đã tổng hợp sinh trưởng, tối đa hóa khả năng phân định ngữ nghĩa.

---

## 3. Environment

### Yêu cầu hệ thống đã xác minh

| Thành phần | Yêu cầu |
|---|---|
| **Hệ điều hành** | Ubuntu 22.04 / 24.04 LTS (x86_64) |
| **Python** | 3.10, 3.11, 3.12, 3.13 |
| **PyTorch / Torchvision** | PyTorch 2.0+ với CUDA 11.8 / CUDA 12.1+ |
| **GPU** | NVIDIA GPU (RTX 3090, RTX 4090, A100, V100, RTX Laptop) |
| **CUDA Drivers** | CUDA Driver Version ≥ 525.60 |

### Thư viện phụ thuộc

| Thư viện | Mục đích |
|---|---|
| `torch`, `torchvision` | Framework deep learning, xử lý ảnh |
| `numpy` | Tính toán mảng |
| `einops` | Thao tác tensor (rearrange, repeat) |
| `pyyaml` | Đọc cấu hình YAML |
| `tensorboard` | Ghi log huấn luyện |
| `scikit-learn` | Đánh giá metrics |
| `scipy` | Hỗ trợ tính toán khoa học |
| `matplotlib` | Trực quan hóa |

### Cấu trúc thư mục dự án

```
TSVIT/
├── configs/                  # File cấu hình YAML cho từng thí nghiệm
│   └── PASTIS24/
│       └── TSViT_DINOv3_fold1.yaml
├── data/                     # Dữ liệu PASTIS-24
├── figures/                  # Hình minh họa kiến trúc
├── metrics/                  # Hàm loss và đánh giá
├── models/                   # Định nghĩa mô hình
│   └── TSViT/
│       ├── TSViTdense.py     # TSViT phân đoạn
│       ├── TSViTcls.py       # TSViT phân loại
│       └── dinov3.py         # DINO-v3 Spatial Backbone
├── train_and_eval/           # Script huấn luyện và đánh giá
├── utils/                    # Tiện ích (checkpoint, logging, ...)
├── deepsatmodels_env.yml     # File cấu hình Conda environment
├── LICENSE.txt               # Giấy phép Apache 2.0
└── README.md                 # Tài liệu này
```

---

## 4. Setup

### Bước 1 — Clone mã nguồn

```bash
git clone https://github.com/Leo4767-C/TSVIT.git
cd TSVIT
```

### Bước 2 — Cài đặt môi trường

**Cách A: Dùng Conda (khuyến nghị)**

```bash
conda env create -f deepsatmodels_env.yml
conda activate deepsatmodels
```

**Cách B: Dùng pip trong virtualenv Python 3.10+**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy einops pyyaml tensorboard scikit-learn scipy matplotlib
```

### Bước 3 — Xác minh cài đặt

```bash
python -c "import torch; print('PyTorch:', torch.__version__, '| CUDA available:', torch.cuda.is_available())"
```

Kết quả mong đợi:

```
PyTorch: 2.x.x | CUDA available: True
```

### Bước 4 — Chuẩn bị dữ liệu PASTIS-24

Đặt dữ liệu PASTIS-24 vào thư mục `data/` theo đường dẫn đã khai báo trong file cấu hình YAML tương ứng.

### Bước 5 — Cấu hình checkpoint (nếu dùng pretrained)

Đảm bảo checkpoint pretrained TSViT được đặt đúng đường dẫn và bật khôi phục từng phần:

```yaml
CHECKPOINT:
  load_from_checkpoint: 'Checkpoint/checkpoint/best.pth'
  partial_restore: True
  save_path: 'models/saved_models/PASTIS24/TSViT_DINOv3_fold1'
  train_metrics_steps: 250
  eval_steps: 2000
  save_steps: 1000000000
```

Khi `partial_restore: True`, module [`utils/torch_utils.py:load_from_checkpoint`](utils/torch_utils.py) tự động lọc bỏ các key spatial transformer không khớp kích thước, tránh hoàn toàn lỗi shape mismatch.

---

## 5. Run & Verify

### Huấn luyện phân đoạn ngữ nghĩa (End-to-End)

```bash
# Chạy trên 1 GPU (GPU 0)
python train_and_eval/segmentation_training_transf.py \
    --config configs/PASTIS24/TSViT_DINOv3_fold1.yaml \
    --device 0

# Chạy trên nhiều GPU (GPU 0 và 1)
python train_and_eval/segmentation_training_transf.py \
    --config configs/PASTIS24/TSViT_DINOv3_fold1.yaml \
    --device 0,1
```

Trong quá trình huấn luyện:
- Metrics in ra mỗi `train_metrics_steps` (250 step).
- Đánh giá trên tập validation mỗi `eval_steps` (2000 step).
- TensorBoard log (Loss, mIoU, Accuracy, F1 theo lớp) được ghi vào `CHECKPOINT.save_path`.

### Đánh giá Linear Probing

Đóng băng toàn bộ đặc trưng, chỉ huấn luyện đầu phân loại tuyến tính để đánh giá khả năng tách rời tuyến tính của đặc trưng DINO-v3:

```bash
python train_and_eval/segmentation_training_transf.py \
    --config configs/PASTIS24/TSViT_DINOv3_fold1.yaml \
    --device 0 \
    --lin
```

### Phân loại cây trồng cấp thửa

Huấn luyện phân loại tile-level dùng `TSViTcls_DINOv3` (bảo toàn token [CLS]):

```bash
python train_and_eval/classification_train_transf.py \
    --config configs/PASTIS24/TSViT_cls.yaml \
    --device 0
```

### Cách biết chạy đúng

| Kiểm tra | Dấu hiệu hoạt động bình thường |
|---|---|
| Loss giảm dần | Loss ban đầu ~3.0, giảm xuống dưới 1.0 trong ~6.500 step đầu |
| Không có lỗi CUDA | Không xuất hiện `CUDA out of memory` hay `RuntimeError` |
| Metrics tăng dần | mIoU và Accuracy tăng qua các lần đánh giá validation |
| TensorBoard có dữ liệu | Mở TensorBoard và thấy đồ thị loss, mIoU, F1 được cập nhật |

---

## 6. Usage / API

### Cấu hình qua file YAML

Toàn bộ cấu hình thí nghiệm được quản lý qua file [`configs/PASTIS24/TSViT_DINOv3_fold1.yaml`](configs/PASTIS24/TSViT_DINOv3_fold1.yaml):

| Trường YAML | Giá trị mặc định | Mô tả |
|---|:---:|---|
| `MODEL.architecture` | `"TSViT_DINOv3"` | Kiến trúc mô hình (`TSViT`, `TSViT_DINOv3`, `TSViTcls`). |
| `MODEL.spatial_backbone` | `"dinov3"` | Kích hoạt `DINOv3SpatialEncoder` (2D RoPE, SwiGLU, LayerScale). |
| `MODEL.img_res` | `24` | Kích thước tile (24 × 24 pixel). |
| `MODEL.patch_size` | `2` | Kích thước patch (2 × 2 → 144 patch). |
| `MODEL.dim` | `128` | Chiều embedding token D. |
| `MODEL.temporal_depth` | `4` | Số block Temporal Transformer. |
| `MODEL.spatial_depth` | `4` | Số block DINO-v3 Spatial Transformer. |
| `MODEL.heads` | `4` | Số attention head. |
| `MODEL.dim_head` | `32` | Chiều mỗi attention head (4 × 32 = 128). |
| `SOLVER.loss_function` | `masked_cross_entropy` | Masked Cross-Entropy Loss với mặt nạ hiệu lực M. |
| `SOLVER.lr_base` | `1.0e-3` | Learning rate đỉnh. |
| `SOLVER.lr_min` | `5.0e-6` | Learning rate tối thiểu sau cosine decay. |
| `SOLVER.num_warmup_epochs` | `10` | Số epoch warmup tuyến tính. |

### Giao diện dòng lệnh

```bash
# Cú pháp chung cho phân đoạn
python train_and_eval/segmentation_training_transf.py \
    --config <đường_dẫn_yaml> \
    --device <gpu_id> \
    [--lin]                   # Tùy chọn: chỉ huấn luyện linear head

# Cú pháp chung cho phân loại
python train_and_eval/classification_train_transf.py \
    --config <đường_dẫn_yaml> \
    --device <gpu_id>
```

| Tham số | Bắt buộc | Mô tả |
|---|:---:|---|
| `--config` | Có | Đường dẫn đến file cấu hình YAML |
| `--device` | Có | ID GPU (ví dụ: `0` hoặc `0,1` cho multi-GPU) |
| `--lin` | Không | Bật chế độ linear probing (đóng băng backbone) |

### Vòng lặp huấn luyện (Training Loop)

Mỗi bước huấn luyện trong [`train_and_eval/segmentation_training_transf.py`](train_and_eval/segmentation_training_transf.py):

```python
def train_step(net, sample, loss_fn, optimizer, device, loss_input_fn):
    optimizer.zero_grad()                               # 1. Xóa gradient
    outputs = net(sample['inputs'].to(device))          # 2. Forward pass: (B, K=19, H=24, W=24)
    outputs = outputs.permute(0, 2, 3, 1)               # 3. Transpose: (B, H=24, W=24, K=19)
    ground_truth = loss_input_fn(sample, device)        # 4. Lấy nhãn và mặt nạ hiệu lực
    loss = loss_fn['mean'](outputs, ground_truth)       # 5. Masked Cross-Entropy Loss
    loss.backward()                                     # 6. Backward pass: tính dL/dθ
    optimizer.step()                                    # 7. Cập nhật tham số (AdamW)
    return outputs, ground_truth, loss
```

---

## 7. AI Model

### Thành phần DINO-v3

Module DINO-v3 được triển khai trong [`models/TSViT/dinov3.py`](models/TSViT/dinov3.py). Mỗi block không gian gồm: pre-LayerNorm, 2D Axial RoPE multi-head attention, LayerScale (γ = 10⁻⁵), và SwiGLU Gated Feed-Forward.

#### 1) 2D Axial Rotary Position Embedding (`DINOv3RoPE`)

Với mỗi attention head có chiều 32, kênh đặc trưng được chia đều cho chiều dọc và ngang (16 hàng, 16 cột):

$$
d_h = \frac{d_{\text{head}}}{2}, \qquad d_w = \frac{d_{\text{head}}}{2}
$$

Tần số dọc mỗi trục với cơ sở θ = 100.0:

$$
\omega_{h, i} = \theta^{-2i / d_h}, \quad 0 \le i < \frac{d_h}{2}
$$

$$
\omega_{w, j} = \theta^{-2j / d_w}, \quad 0 \le j < \frac{d_w}{2}
$$

Với token patch tại tọa độ lưới (y, x) trên lưới 12 × 12, biểu diễn quay 2D R(y, x) quay vector query q:

$$
q_{\text{rot}}(y, x) =
\begin{pmatrix}
q_{[0:d_h]} \odot \cos(y \cdot \omega_h) + \text{RotHalf}(q_{[0:d_h]}) \odot \sin(y \cdot \omega_h) \\
q_{[d_h:d_{\text{head}}]} \odot \cos(x \cdot \omega_w) + \text{RotHalf}(q_{[d_h:d_{\text{head}}]}) \odot \sin(x \cdot \omega_w)
\end{pmatrix}
$$

trong đó toán tử quay nửa vector ([`DINOv3RoPE.rotate_half`](models/TSViT/dinov3.py)):

$$
\text{RotHalf}(z) = \left[ -z_{\frac{d}{2}:d}, \; z_{0:\frac{d}{2}} \right]
$$

**Chứng minh bất biến tịnh tiến tương đối**:
Logit attention giữa patch 1 và patch 2:

$$
\langle q_{\text{rot}}(y_1, x_1), k_{\text{rot}}(y_2, x_2) \rangle = q^T \mathbf{R}_{2D}^T(y_1, x_1) \mathbf{R}_{2D}(y_2, x_2) k = q^T \mathbf{R}_{2D}(y_1 - y_2, x_1 - x_2) k
$$

Attention chỉ phụ thuộc vào vector dịch chuyển không gian (Δy, Δx), mang lại tính đẳng biến tịnh tiến 2D thực sự qua các ranh giới thửa ruộng nông nghiệp.

#### 2) SwiGLU Gated Feed-Forward Network (`DINOv3GatedMLP`)

FFN tiêu chuẩn dùng FFN(x) = GELU(xW⁽¹⁾)W⁽²⁾. Trong DINO-v3, kiến trúc SwiGLU với ma trận cổng và chiếu học được:

$$
\text{SwiGLU}(x) = \left( \text{SiLU}(x W_{\text{gate}}) \odot (x W_{\text{up}}) \right) W_{\text{down}}
$$

trong đó SiLU(z) = z · σ(z) = z / (1 + e⁻ᶻ). Cơ chế cổng điều khiển luồng thông tin động qua tương tác nhân, tránh hiện tượng neuron chết.

#### 3) LayerScale Stabilization (`DINOv3LayerScale`)

Cập nhật tàn dư ViT sâu được điều biến bởi tensor đường chéo γ ∈ ℝᴰ:

$$
x_{l+1} = x_l + \text{diag}(\gamma_l) \cdot f(x_l), \quad \gamma_l \in \mathbb{R}^D, \quad \text{khởi tạo } \gamma_l = 10^{-5}
$$

Vì γ ≈ 0 khi khởi tạo, mỗi block DINO-v3 hoạt động gần như toán tử đồng nhất (x ≈ x), loại bỏ sốc biểu diễn khi khôi phục trọng số đã huấn luyện (`partial_restore: True`).

### Hàm Loss: Masked Cross-Entropy

Ảnh vệ tinh nông nghiệp bao gồm mây, nhiễu khí quyển, và nền phi nông nghiệp (đường, công trình, mặt nước) mã hóa là void label 19. Mặt nạ hiệu lực nhị phân M:

$$
M_{b, i, j} =
\begin{cases}
1 & \text{nếu } Y_{b, i, j} \in [0, 18] \\
0 & \text{nếu } Y_{b, i, j} = 19 \text{ (void/nền)}
\end{cases}
$$

Masked Cross-Entropy Loss chỉ đánh giá trên pixel cây trồng hợp lệ (M = 1):

$$
\mathcal{L}_{\text{MCE}}(\hat{Y}, Y, M) = - \frac{1}{\sum_{b, i, j} M_{b, i, j}} \sum_{b=1}^B \sum_{i=1}^H \sum_{j=1}^W M_{b, i, j} \sum_{c=0}^{K-1} I(Y_{b, i, j} = c) \log \left( \frac{\exp(\hat{Y}_{b, i, j, c})}{\sum_{c'=0}^{K-1} \exp(\hat{Y}_{b, i, j, c'})} \right)
$$

Triển khai trong [`metrics/loss_functions.py:MaskedCrossEntropyLoss`](metrics/loss_functions.py):

```python
mask_flat = mask.reshape(-1, 1)
nclasses = logits.shape[-1]
logits_flat = logits.reshape(-1, nclasses)
masked_logits = logits_flat[mask_flat.repeat(1, nclasses)].view(-1, nclasses)
masked_target = target.reshape(-1, 1)[mask_flat].unsqueeze(dim=-1).to(torch.int64)

log_probs = F.log_softmax(masked_logits, dim=-1)
loss = -torch.gather(log_probs, dim=1, index=masked_target).mean()
```

Tính chất gradient quan trọng:

$$
M_{b, i, j} = 0 \implies \frac{\partial \mathcal{L}_{\text{MCE}}}{\partial \hat{Y}_{b, i, j, k}} \equiv 0.0, \quad \forall k \in [0, K-1]
$$

Pixel nền và pixel không hợp lệ phát ra gradient bằng 0, ngăn nhiễu nền làm hỏng trọng số mô hình.

### Dòng gradient qua DINO-v3

<p align="center">
  <img src="figures/gradient-flow-dinov3.png" alt="Dòng gradient qua DINO-v3 Spatial Backbone" width="850"/>
  <br>
  <em>Hình 3: Dòng gradient giải tích qua DINO-v3 Spatial Backbone trong quá trình lan truyền ngược. Sơ đồ minh họa cách gradient phi tuyến liên tục lan truyền qua SwiGLU, LayerScale duy trì ổn định tàn dư qua nhân tỉ lệ đường chéo (γ = 10⁻⁵), và 2D Axial RoPE bảo toàn chuẩn Frobenius của gradient attention.</em>
</p>

Các cơ chế ổn định gradient:

1. **Gradient phi tuyến liên tục (SwiGLU)**: Đạo hàm SiLU'(z) = σ(z)·(1 + z(1 − σ(z))) luôn khác 0 trong vùng hoạt động, hoàn toàn ngăn hiện tượng "dying ReLU".

2. **Giảm chấn gradient qua LayerScale** (γ = 10⁻⁵): Tham số đường chéo nhỏ γ = 10⁻⁵ hoạt động như bộ giảm chấn gradient quy nạp, bảo vệ biểu diễn transformer không gian sâu khỏi bùng nổ gradient.

3. **Dòng gradient không mất mát qua 2D RoPE**: Vì ma trận quay 2D R là đơn nhất (RᵀR = I), phép quay là phép đẳng cự bảo toàn chuẩn Frobenius: ‖∇_q L‖_F = ‖∇_{q_rot} L‖_F, đảm bảo không mất gradient khi biểu diễn không gian lan truyền ngược vào các block attention thời gian.

### Tối ưu hóa và lịch learning rate

* **Optimizer AdamW**:

$$
\theta_t = \theta_{t-1} - \eta_t \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_{t-1} \right)
$$

với λ = 0.0 (hoặc 0.01), β = (0.9, 0.999), ε = 10⁻⁸.

* **Cosine Annealing với Linear Warmup**:
  * **Pha warmup (Epoch 1–10)**: Tăng tuyến tính từ 10⁻⁸ đến learning rate đỉnh 10⁻³.
  * **Pha Cosine Annealing (Epoch 11–100)**: Giảm cosine mượt từ 10⁻³ xuống learning rate tối thiểu 5 × 10⁻⁶ trong 80.000 step.

### Kết quả benchmark

Đánh giá định lượng so sánh TSViT baseline với **TSViT-DINOv3** trên tập kiểm tra PASTIS-24 (19 lớp):

| Kiến trúc | Spatial Backbone | Vị trí không gian | Val Loss ↓ | mIoU (%) ↑ | Macro F1 (%) ↑ | OA (%) ↑ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| TSViT Baseline | Vanilla ViT (GELU) | 1D tĩnh học được | 0.4128 | 61.40 | 73.20 | 81.65 |
| TSViT-Single Token | Vanilla ViT (GELU) | 1D tĩnh học được | 0.4410 | 59.80 | 71.50 | 80.12 |
| **TSViT-DINOv3** | **DINO-v3 Backbone** | **2D Axial RoPE** | **0.3152** | **65.65** | **77.84** | **84.12** |
| *Cải thiện ròng* | — | — | **−0.0976** | **+4.25%** | **+4.64%** | **+2.47%** |

Nhận xét:
1. **Hội tụ warmup nhanh hơn 35%**: TSViT-DINOv3 giảm loss từ ~3.02 xuống dưới 1.0 trong 6.500 step (so với 10.000 step ở baseline), nhờ 2D RoPE đưa hình học tọa độ giải tích vào ngay lập tức.
2. **Loại bỏ đột biến phương sai gradient**: LayerScale (γ = 10⁻⁵) ngăn các đột biến tối ưu thường thấy quanh step 8.000–15.000 ở vanilla transformer.
3. **Loss hội tụ thấp hơn**: Masked Cross-Entropy đạt loss validation ổn định thấp hơn (0.3152 so với 0.4128).
4. **Cải thiện ranh giới thửa**: Attention bất biến tịnh tiến qua dịch chuyển tương đối (Δy, Δx) giúp TSViT-DINOv3 phân định rõ ràng ranh giới thửa ruộng sắc nét, trực tiếp đóng góp vào mức tăng +4.25% mIoU.

---

## 8. Troubleshooting

| Triệu chứng / Lỗi | Nguyên nhân gốc | Cách xử lý |
|---|---|---|
| `CUDA out of memory` khi huấn luyện | Batch size lớn với chuỗi thời gian dài (T=60). | Giảm `DATASETS.train.batch_size` trong YAML (ví dụ từ 48 xuống 16 hoặc 24), hoặc dùng multi-GPU `--device 0,1`. |
| `Missing key(s) in state_dict: space_transformer...` | Key checkpoint baseline không khớp module DINO-v3. | Đảm bảo `CHECKPOINT.partial_restore: True` trong file cấu hình. |
| `RuntimeError: The size of tensor a must match size of tensor b` | Kích thước lưới không khớp trong bộ nhớ đệm 2D RoPE. | Kiểm tra `img_res` (24) chia hết cho `patch_size` (2), tạo ra lưới 12 × 12. |
| `ValueError: no valid target pixels for mean reduction` | Batch chỉ chứa pixel nền hoặc mây (mặt nạ M = 0 toàn bộ). | Hành vi bình thường cho tile hoàn toàn void; đảm bảo tập dữ liệu chứa chú thích thửa nông nghiệp hợp lệ. |
| DataLoader chạy chậm | `num_workers` thấp gây GPU đói dữ liệu. | Đặt `DATASETS.train.num_workers: 4` hoặc `8` để tải tile bất đồng bộ đa tiến trình. |
| Loss không giảm sau vài nghìn step | Learning rate quá cao hoặc checkpoint không tương thích. | Kiểm tra `SOLVER.lr_base`, thử giảm xuống 5e-4. Xác nhận checkpoint đúng phiên bản. |
| Import lỗi `einops` | Chưa cài thư viện einops. | Chạy `pip install einops`. |

---

## Tài liệu tham khảo

- **TSViT / DeepSatModels**: Michail Tarasiou, Erik Chavez, Stefanos Zafeiriou. *"ViTs for SITS: Vision Transformers for Satellite Image Time Series"*, CVPR 2023. [[Paper](https://arxiv.org/abs/2301.04944)] [[Code](https://github.com/michaeltrs/DeepSatModels)].
- **DINO-v3**: Meta AI Research. *"DINO-v3: Vision Transformers for Self-Supervised Visual Representation Learning"*, 2024. [[Paper](https://arxiv.org/abs/2508.10104)] [[Meta Repository](https://github.com/facebookresearch/dinov3)].
- **PASTIS Benchmark**: Vivien Sainte Fare Garnot, Loic Landrieu. *"Panoptic Segmentation of Satellite Image Time Series with Field-Level Crop Labels"*, ICCV 2021. [[Dataset](https://github.com/VSainteuf/pastis-benchmark)].

## Giấy phép

Dự án này được cấp phép theo **Apache License 2.0** — xem file [LICENSE.txt](LICENSE.txt) để biết chi tiết.
