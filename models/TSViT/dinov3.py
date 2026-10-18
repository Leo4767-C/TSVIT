import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.config_files_utils import get_params_values


class DINOv3RoPE(nn.Module):
    """
    2D Axial Rotary Position Embedding (RoPE) for Vision Transformers (DINO-v3).
    Computes 2D spatial frequencies and rotates query and key representations
    based on their (row, col) grid coordinates.
    """
    def __init__(self, dim_head, theta=100.0):
        super().__init__()
        self.dim_head = dim_head
        self.theta = theta
        self.cached_freqs = None
        self.cached_grid_size = None

    def _get_freqs(self, h, w, device, dtype):
        if self.cached_grid_size == (h, w) and self.cached_freqs is not None:
            cos, sin = self.cached_freqs
            if cos.device == device and cos.dtype == dtype:
                return cos, sin

        dim_h = self.dim_head // 2
        dim_w = self.dim_head - dim_h

        inv_freq_h = 1.0 / (self.theta ** (torch.arange(0, dim_h, 2, device=device).float() / dim_h))
        inv_freq_w = 1.0 / (self.theta ** (torch.arange(0, dim_w, 2, device=device).float() / dim_w))

        pos_h = torch.arange(h, device=device).float()
        pos_w = torch.arange(w, device=device).float()

        sinusoid_h = torch.einsum('i,j->ij', pos_h, inv_freq_h)
        sinusoid_w = torch.einsum('i,j->ij', pos_w, inv_freq_w)

        cos_h = torch.cos(sinusoid_h).repeat_interleave(2, dim=-1)
        sin_h = torch.sin(sinusoid_h).repeat_interleave(2, dim=-1)
        cos_w = torch.cos(sinusoid_w).repeat_interleave(2, dim=-1)
        sin_w = torch.sin(sinusoid_w).repeat_interleave(2, dim=-1)

        # 2D grid of frequencies for spatial patches
        cos_grid = torch.cat([
            cos_h.unsqueeze(1).expand(-1, w, -1),
            cos_w.unsqueeze(0).expand(h, -1, -1)
        ], dim=-1).reshape(h * w, self.dim_head).to(dtype=dtype)

        sin_grid = torch.cat([
            sin_h.unsqueeze(1).expand(-1, w, -1),
            sin_w.unsqueeze(0).expand(h, -1, -1)
        ], dim=-1).reshape(h * w, self.dim_head).to(dtype=dtype)

        self.cached_grid_size = (h, w)
        self.cached_freqs = (cos_grid, sin_grid)
        return cos_grid, sin_grid

    @staticmethod
    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., :x.shape[-1] // 2]), dim=-1)

    def forward(self, q, k, h, w, has_cls_token=False):
        """
        q, k shape: (Batch, Heads, Num_Tokens, Dim_Head)
        """
        cos, sin = self._get_freqs(h, w, q.device, q.dtype)

        if not has_cls_token:
            cos = cos.unsqueeze(0).unsqueeze(1)
            sin = sin.unsqueeze(0).unsqueeze(1)
            q_rot = (q * cos) + (self.rotate_half(q) * sin)
            k_rot = (k * cos) + (self.rotate_half(k) * sin)
            return q_rot, k_rot
        else:
            cls_q, patch_q = q[:, :, :1, :], q[:, :, 1:, :]
            cls_k, patch_k = k[:, :, :1, :], k[:, :, 1:, :]
            cos = cos.unsqueeze(0).unsqueeze(1)
            sin = sin.unsqueeze(0).unsqueeze(1)
            patch_q_rot = (patch_q * cos) + (self.rotate_half(patch_q) * sin)
            patch_k_rot = (patch_k * cos) + (self.rotate_half(patch_k) * sin)
            q_out = torch.cat([cls_q, patch_q_rot], dim=2)
            k_out = torch.cat([cls_k, patch_k_rot], dim=2)
            return q_out, k_out


class DINOv3LayerScale(nn.Module):
    """
    LayerScale layer as used in DINO-v3 to stabilize deep ViT representations.
    """
    def __init__(self, dim, init_values=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim) * init_values)

    def forward(self, x):
        return x * self.gamma


class DINOv3GatedMLP(nn.Module):
    """
    SwiGLU / Gated-MLP feed-forward block featured in DINO-v3.
    """
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.w_gate = nn.Linear(dim, hidden_dim, bias=True)
        self.w_up = nn.Linear(dim, hidden_dim, bias=True)
        self.w_down = nn.Linear(hidden_dim, dim, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class DINOv3Attention(nn.Module):
    """
    DINO-v3 Multi-Head Attention layer with 2D Rotary Position Embeddings (RoPE).
    """
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., qkv_bias=True, proj_bias=True):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5

        self.qkv = nn.Linear(dim, inner_dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(inner_dim, dim, bias=proj_bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, rope=None, h=None, w=None, has_cls_token=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.dim_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if rope is not None and h is not None and w is not None:
            q, k = rope(q, k, h, w, has_cls_token=has_cls_token)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v).permute(0, 2, 1, 3).reshape(B, N, self.heads * self.dim_head)
        return self.proj(out)


class DINOv3Block(nn.Module):
    """
    Single DINO-v3 Vision Transformer Block:
    Pre-LayerNorm -> Attention with 2D RoPE -> LayerScale 1 -> Residual ->
    Pre-LayerNorm -> Gated-MLP (SwiGLU) -> LayerScale 2 -> Residual
    """
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0., init_values=1e-5):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = DINOv3Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.ls1 = DINOv3LayerScale(dim, init_values=init_values)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = DINOv3GatedMLP(dim, mlp_dim, dropout=dropout)
        self.ls2 = DINOv3LayerScale(dim, init_values=init_values)

    def forward(self, x, rope=None, h=None, w=None, has_cls_token=False):
        x = x + self.ls1(self.attn(self.norm1(x), rope=rope, h=h, w=w, has_cls_token=has_cls_token))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class DINOv3SpatialEncoder(nn.Module):
    """
    DINO-v3 Spatial Encoder for TSViT.
    Replaces the standard TSViT spatial transformer with Meta's DINO-v3 ViT architecture.
    Features:
    - 2D Axial RoPE encoding for 2D spatial patch grids
    - LayerScale per block
    - SwiGLU Gated-MLP feed-forward networks
    - Optional loading of pretrained weights from HuggingFace / checkpoint file.
    """
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.,
                 init_values=1e-5, theta=100.0, model_config=None):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.rope = DINOv3RoPE(dim_head, theta=theta)
        self.layers = nn.ModuleList([
            DINOv3Block(dim, heads, dim_head, mlp_dim, dropout=dropout, init_values=init_values)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

        # Check for pretrained weight loading
        if model_config is not None:
            weights_path = get_params_values(model_config, 'dinov3_weights_path', None)
            if weights_path and os.path.exists(weights_path):
                self.load_pretrained_weights(weights_path)

    def load_pretrained_weights(self, weights_path):
        print(f"[DINO-v3] Loading pretrained weights from {weights_path}...")
        try:
            ckpt = torch.load(weights_path, map_location='cpu')
            if 'state_dict' in ckpt:
                state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                state_dict = ckpt['model']
            else:
                state_dict = ckpt
            self.load_state_dict(state_dict, strict=False)
            print("[DINO-v3] Pretrained weights loaded successfully.")
        except Exception as e:
            print(f"[DINO-v3] Warning: could not load weights from {weights_path}: {e}")

    def forward(self, x, h=None, w=None, has_cls_token=False):
        """
        x: (Batch, Num_Tokens, Dim)
        h, w: Height and Width of patch grid (e.g. 12x12 for 24x24 image with patch_size 2)
        """
        if h is None or w is None:
            num_patches = x.shape[1] - (1 if has_cls_token else 0)
            h = int(num_patches ** 0.5)
            w = h

        for layer in self.layers:
            x = layer(x, rope=self.rope, h=h, w=w, has_cls_token=has_cls_token)

        return self.norm(x)


class DINOv3Backbone(nn.Module):
    """
    Full DINO-v3 2D Vision Backbone for Satellite Imagery.
    Accepts arbitrary number of input channels (e.g. 10 or 14 channels) and
    processes 2D frames with DINO-v3 ViT blocks.
    """
    def __init__(self, in_channels=10, img_size=24, patch_size=2, embed_dim=128,
                 depth=4, heads=4, dim_head=32, mlp_dim=512, dropout=0.):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches_1d = img_size // patch_size
        self.num_patches = self.num_patches_1d ** 2
        self.embed_dim = embed_dim

        # Patch embedding projection
        patch_dim = in_channels * (patch_size ** 2)
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(2)
        )

        self.encoder = DINOv3SpatialEncoder(
            dim=embed_dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout
        )

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        x_proj = self.patch_embed(x).transpose(1, 2) # (B, num_patches, embed_dim)
        h = H // self.patch_size
        w = W // self.patch_size
        feats = self.encoder(x_proj, h=h, w=w)
        return feats
