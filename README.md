# TSViT-DINOv3: Tích hợp DINO-v3 Spatial Backbone vào Temporal-Spatial Vision Transformer

Dự án tập trung vào việc **thay thế khối Spatial Encoder (bộ mã hóa không gian) của kiến trúc TSViT bằng DINO-v3 Backbone**, nâng cao khả năng biểu diễn không gian và đường biên thửa ruộng trên chuỗi ảnh vệ tinh đa thời gian (PASTIS-24, France).

---

## 1. Sơ đồ kiến trúc TSViT với DINO-v3 Backbone

Kiến trúc TSViT phân tách việc xử lý chuỗi ảnh viễn thám thành hai giai đoạn độc lập:
1. **Temporal Transformer**: Trích xuất quan hệ tương quan chuỗi thời gian giữa các ngày chụp.
2. **DINO-v3 Spatial Encoder**: Thay thế Transformer không gian nguyên bản, trích xuất đặc trưng vị trí và biên dạng ô thửa trong không gian 2D.

```
Ảnh vệ tinh đầu vào: (Batch, Thời gian, Kênh phổ, H=24, W=24)
              │
              ▼  Chia Patch không gian (Patch Size = 2x2)
         (B, T, N_patches=144, Dim=128)
              │
              ▼  Temporal Transformer (Self-Attention theo trục thời gian T)
         (B, T, N_patches=144, Dim=128)
              │
              ▼  Gom tụ đặc trưng thời gian (Temporal Token / Pooling)
         (B, N_patches=144, Dim=128)
              │
  ┌───────────┴───────────────────────────────────────────────────────┐
  │              DINO-v3 Spatial Backbone (Thay thế Spatial cũ)        │
  │  1. 2D Axial RoPE: Mã hóa tọa độ không gian 2D cho Query và Key   │
  │  2. LayerScale: Khởi tạo tham số gamma = 1e-5 giữ gradient ổn định│
  │  3. SwiGLU Gated-MLP: SiLU(xW_gate) * (xW_up) -> W_down           │
  └───────────┬───────────────────────────────────────────────────────┘
              │
              ▼  Linear Head & Tái tạo không gian
         Output: (B, NumClasses, H=24, W=24)
```

---

## 2. Các cải tiến cốt lõi của DINO-v3 Spatial Encoder

Khối mã hóa không gian mới được xây dựng tại [`models/TSViT/dinov3.py`](models/TSViT/dinov3.py) với 3 thành phần chính:

### 2.1. 2D Axial Rotary Position Embedding (2D RoPE)
* **Cơ chế cũ (TSViT)**: Sử dụng 1D learnable position embedding tĩnh dạng cộng vector (`x = x + pos_embedding`). Khi làm phẳng lưới `(H x W)` thành chuỗi 1D, thông tin hình học 2 chiều và khoảng cách tương đối giữa các ô thửa ruộng bị mất đi.
* **Cơ chế mới (DINO-v3)**:
  * Phân tách tần số quay theo 2 trục không gian độc lập: trục hàng (`H`) và trục cột (`W`).
  * Xoay trực tiếp vector Query (`Q`) và Key (`K`) trong không gian phức:
    ```python
    # Phép xoay Query và Key theo tọa độ 2D (H, W)
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    ```
  * **Ưu điểm**: Tích vô hướng chú ý giữa hai patch chỉ phụ thuộc vào khoảng cách tương đối `(ΔH, ΔW)`, đảm bảo tính bất biến tịnh tiến không gian, giúp nhận diện biên thửa ruộng chính xác hơn.

### 2.2. SwiGLU Gated Feed-Forward Network
* **Cơ chế cũ (TSViT)**: MLP 2 lớp với hàm kích hoạt GELU thông thường (`Linear -> GELU -> Dropout -> Linear`).
* **Cơ chế mới (DINO-v3)**: Sử dụng khối Swish Gated Linear Unit (SwiGLU):
  ```python
  # Cơ chế cổng nhân SwiGLU
  output = (F.silu(x @ W_gate) * (x @ W_up)) @ W_down
  ```
  Nhánh cổng nhân (`multiplicative gating`) giúp mô hình chủ động lọc bớt nhiễu phổ vệ tinh và làm sắc nét các đường ranh giới ô đất.

### 2.3. LayerScale Stabilization
* **Cơ chế**: Khởi tạo vector tỉ lệ đường chéo `gamma` với giá trị nhỏ (`1e-5`) cho mỗi khối Attention và MLP trước khi cộng residual:
  ```python
  x = x + gamma_1 * Attention(LayerNorm(x))
  x = x + gamma_2 * SwiGLU(LayerNorm(x))
  ```
  Giúp kiểm soát biên độ kích hoạt và ngăn chặn hiện tượng bão hòa tín hiệu khi xếp chồng nhiều khối spatial transformer.

---

## 3. Cấu trúc Source Code & Các vị trí thay thế

| File | Vai trò | Chi tiết thay đổi |
|---|---|---|
| [`models/TSViT/dinov3.py`](models/TSViT/dinov3.py) | **DINO-v3 Engine** | Hiện thực `DINOv3RoPE`, `DINOv3LayerScale`, `DINOv3GatedMLP`, `DINOv3Attention`, `DINOv3SpatialEncoder`. |
| [`models/TSViT/TSViTdense.py`](models/TSViT/TSViTdense.py) | **Mô hình Phân đoạn (Segmentation)** | Thay `self.space_transformer` thành `DINOv3SpatialEncoder`. Truyền kích thước lưới `(Hp, Wp)` vào RoPE trong hàm `forward`. Thêm class `TSViT_DINOv3`. |
| [`models/TSViT/TSViTcls.py`](models/TSViT/TSViTcls.py) | **Mô hình Phân loại (Classification)** | Tích hợp `DINOv3SpatialEncoder` với cơ chế bảo toàn `[CLS]` space token (`has_cls_token=True`). Thêm class `TSViTcls_DINOv3`. |
| [`models/__init__.py`](models/__init__.py) | **Model Dispatcher** | Định tuyến khởi tạo mô hình khi cấu hình yêu cầu `TSViT_DINOv3` hoặc `spatial_backbone: "dinov3"`. |
| [`configs/PASTIS24/TSViT_DINOv3_fold1.yaml`](configs/PASTIS24/TSViT_DINOv3_fold1.yaml) | **Cấu hình huấn luyện** | Cấu hình mẫu cho bài toán phân đoạn PASTIS-24 sử dụng backbone DINO-v3. |

---

## 4. Hướng dẫn cấu hình & Chạy mô hình

### 4.1. Thiết lập trong file cấu hình (`configs/PASTIS24/TSViT_DINOv3_fold1.yaml`)

Để kích hoạt DINO-v3 spatial backbone, chỉ định tham số `spatial_backbone: "dinov3"`:

```yaml
MODEL:
  architecture: "TSViT_DINOv3"
  spatial_backbone: "dinov3"   # Kích hoạt DINO-v3 Spatial Backbone
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
  loss_function: masked_cross_entropy
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

### 4.2. Lệnh chạy thực nghiệm

* **Huấn luyện Phân đoạn ô thửa (Semantic Segmentation):**
  ```bash
  python train_and_eval/segmentation_training_transf.py --config configs/PASTIS24/TSViT_DINOv3_fold1.yaml --device 0
  ```

* **Huấn luyện Phân loại cây trồng (Classification):**
  ```bash
  python train_and_eval/classification_train_transf.py --config configs/PASTIS24/TSViT_cls.yaml --device 0
  ```
