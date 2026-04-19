"""
transformer.py
--------------
StrategyTransformer: Egitim orneklerinden gorev stratejisi cikar.

Mimari: 4 katman, 4 head, dim=128, ~800K param

Giris:
  Egitim ciftleri [(in_1, out_1), ..., (in_k, out_k)]
  Her grid: one-hot [11,H,W] + PosEnc [16,H,W] → patch/pixel tokenization

Ciktilar:
  input_tokens : [B, N_tok, D]  SeedMLP cross-attention icin key/value
  task_emb     : [B, D]         NCA FiLM global embedding
  H_out, W_out : [B] int        Hedef boyut (1-30 arasi classification)
  boundary_mode: [B] int        {0=border, 1=zeros, 2=reflection}
  obj_mask_0   : [B, K, H_canvas, W_canvas]  Nesne slot maskeleri

Clifford (V1):
  task_emb icinde Clifford rotorlar kodlanir (donme/yansima yonu)
  Ayri bir CliffordHead: task_emb → 16D multivector coefficients
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, List

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from encoding import make_posenc
from color_codec import N_COLOR, NULL_IDX, CANVAS_H, CANVAS_W

# ────────────────────────────────────────────────────────────────────────
# Sabitler
# ────────────────────────────────────────────────────────────────────────

D_TRANS   = 128    # Transformer embedding boyutu
N_HEADS   = 4      # Attention head sayisi
N_LAYERS  = 4      # Transformer katman sayisi
PATCH_SZ  = 2      # Patch boyutu (2x2 → 1 token)
MAX_SIZE  = 30     # Maksimum grid boyutu (siniflandirma icin)
K_MAX     = 12     # Nesne slot sayisi
P_DIM     = 16     # PosEnc boyutu


# ────────────────────────────────────────────────────────────────────────
# Tokenizasyon
# ────────────────────────────────────────────────────────────────────────

class PatchTokenizer(nn.Module):
    """
    Grid'i PATCH_SZ x PATCH_SZ patch'lere ayirir, her patch bir token olur.

    Giris:
      grid_onehot : [B, 11, H, W]   renk one-hot
      posenc      : [B, P, H, W]    pozisyon (onceden hesaplanmis)

    Cikis:
      tokens : [B, N_tok, D_trans]
      N_tok  = ceil(H/PATCH_SZ) * ceil(W/PATCH_SZ)
    """

    def __init__(self, D_trans: int = D_TRANS, patch_sz: int = PATCH_SZ):
        super().__init__()
        self.patch_sz = patch_sz
        # Her patch icin: (N_COLOR + P) * patch_sz^2 → D_trans
        C_in = (N_COLOR + P_DIM) * patch_sz * patch_sz
        self.proj = nn.Linear(C_in, D_trans, bias=True)
        self.norm = nn.LayerNorm(D_trans)

    def forward(self, grid_onehot: torch.Tensor,
                posenc: torch.Tensor) -> torch.Tensor:
        """
        Parametreler
        ------------
        grid_onehot : [B, 11, H, W]
        posenc      : [B, P, H, W]

        Donus
        -----
        tokens : [B, N_tok, D_trans]
        """
        B, _, H, W = grid_onehot.shape
        p = self.patch_sz

        # Pad H, W to be divisible by patch_sz
        H_pad = math.ceil(H / p) * p
        W_pad = math.ceil(W / p) * p
        if H_pad > H or W_pad > W:
            grid_onehot = F.pad(grid_onehot, (0, W_pad - W, 0, H_pad - H))
            posenc      = F.pad(posenc,      (0, W_pad - W, 0, H_pad - H))

        # Concat color + posenc
        x = torch.cat([grid_onehot, posenc], dim=1)   # [B, 11+P, H_pad, W_pad]

        # Patch unfold: [B, C*(p^2), H//p, W//p]
        C = x.shape[1]
        x = x.unfold(2, p, p).unfold(3, p, p)
        # [B, C, H//p, W//p, p, p]
        x = x.contiguous().view(B, C * p * p, -1).permute(0, 2, 1)
        # [B, N_tok, C*p*p]

        tokens = self.norm(self.proj(x))   # [B, N_tok, D_trans]
        return tokens


# ────────────────────────────────────────────────────────────────────────
# Transformer Blogu
# ────────────────────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """Standart Transformer encoder blogu: MultiHeadAttn + FFN."""

    def __init__(self, D: int = D_TRANS, n_heads: int = N_HEADS,
                 ffn_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn  = nn.MultiheadAttention(D, n_heads, dropout=dropout, batch_first=True)
        self.ff    = nn.Sequential(
            nn.Linear(D, D * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(D * ffn_mult, D),
        )
        self.norm1 = nn.LayerNorm(D)
        self.norm2 = nn.LayerNorm(D)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention + residual
        x2, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + self.drop(x2))
        # FFN + residual
        x = self.norm2(x + self.drop(self.ff(x)))
        return x


# ────────────────────────────────────────────────────────────────────────
# Transformer Segmentasyon Kafa (ObjectMask_0 icin)
# ────────────────────────────────────────────────────────────────────────

class TransformerSegHead(nn.Module):
    """
    Transformer ciktisindan nesne slot maskeleri uretir.
    task_emb → [K, H_canvas, W_canvas] ObjectMask_0

    V1: Basit MLP ile uretim — her slot icin bir 'attention map'
    """

    def __init__(self, D_trans: int, K: int, canvas_H: int, canvas_W: int):
        super().__init__()
        self.K = K
        self.canvas_H = canvas_H
        self.canvas_W = canvas_W
        # task_emb → K * (H//4) * (W//4) → upsample → K x H x W
        H4, W4 = canvas_H // 4, canvas_W // 4
        self.mlp = nn.Sequential(
            nn.Linear(D_trans, D_trans * 2),
            nn.GELU(),
            nn.Linear(D_trans * 2, K * H4 * W4),
        )
        self.H4 = H4
        self.W4 = W4

    def forward(self, task_emb: torch.Tensor) -> torch.Tensor:
        """
        Parametreler
        ------------
        task_emb : [B, D_trans]

        Donus
        -----
        obj_mask : [B, K, canvas_H, canvas_W]  (softmax normalize)
        """
        B = task_emb.shape[0]
        x = self.mlp(task_emb)                   # [B, K*H4*W4]
        x = x.view(B, self.K, self.H4, self.W4)  # [B, K, H4, W4]
        # Bilinear upsample to full canvas
        x = F.interpolate(x, size=(self.canvas_H, self.canvas_W),
                          mode='bilinear', align_corners=False)  # [B, K, H, W]
        # Softmax slot boyutunda normalize et
        x = F.softmax(x, dim=1)                  # her piksel: slot agirliklari toplamı=1
        return x


# ────────────────────────────────────────────────────────────────────────
# Boyut Tahmini Kafasi
# ────────────────────────────────────────────────────────────────────────

class SizeHead(nn.Module):
    """
    task_emb → H_out ve W_out classification (1-30 arasi).

    30-sinifli classification icin logit vektor uretir.
    """

    def __init__(self, D_trans: int, max_size: int = MAX_SIZE):
        super().__init__()
        self.max_size = max_size
        self.h_head = nn.Linear(D_trans, max_size)
        self.w_head = nn.Linear(D_trans, max_size)

    def forward(self, task_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Donus: (H_logits [B, 30], W_logits [B, 30])
        Boyut tahmini: argmax + 1  (1-indexed)
        """
        return self.h_head(task_emb), self.w_head(task_emb)

    def predict(self, task_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Boyutlari tahmin et (greedy).
        Donus: (H_out [B], W_out [B]) int tensoru (1-30)
        """
        h_logits, w_logits = self.forward(task_emb)
        H_out = h_logits.argmax(dim=-1) + 1   # 0-indexed → 1-indexed
        W_out = w_logits.argmax(dim=-1) + 1
        return H_out, W_out


# ────────────────────────────────────────────────────────────────────────
# Sinir Modu Kafasi
# ────────────────────────────────────────────────────────────────────────

class BoundaryModeHead(nn.Module):
    """
    task_emb → boundary_mode {0=border, 1=zeros, 2=reflection}
    """

    def __init__(self, D_trans: int):
        super().__init__()
        self.head = nn.Linear(D_trans, 3)

    def forward(self, task_emb: torch.Tensor) -> torch.Tensor:
        """Donus: [B, 3] logit"""
        return self.head(task_emb)

    def predict(self, task_emb: torch.Tensor) -> List[str]:
        """Donus: list[str] — her ornek icin 'border'/'zeros'/'reflection'"""
        modes = ['border', 'zeros', 'reflection']
        idxs  = self.forward(task_emb).argmax(dim=-1).tolist()
        return [modes[i] for i in idxs]


# ────────────────────────────────────────────────────────────────────────
# Ana StrategyTransformer
# ────────────────────────────────────────────────────────────────────────

class StrategyTransformer(nn.Module):
    """
    Egitim orneklerinden gorev stratejisi cikar.

    Pipeline:
    1. Her (input, output) ciftini tokenlastir
    2. [CLS_in], input_tokens, [CLS_out], output_tokens birlestirilir
    3. N_LAYERS katman self-attention
    4. [CLS] tokenindan task_emb cikar
    5. task_emb'den tum kafalar uretilir

    V1: K_update=0 → sadece baslangicta calisir
    """

    def __init__(self,
                 D_trans: int    = D_TRANS,
                 n_heads: int    = N_HEADS,
                 n_layers: int   = N_LAYERS,
                 K_slots: int    = K_MAX,
                 canvas_H: int   = CANVAS_H,
                 canvas_W: int   = CANVAS_W,
                 patch_sz: int   = PATCH_SZ,
                 max_examples: int = 10,
                 dropout: float  = 0.1):
        super().__init__()
        self.D_trans     = D_trans
        self.K_slots     = K_slots
        self.canvas_H    = canvas_H
        self.canvas_W    = canvas_W

        # Tokenizasyon
        self.patch_tokenizer = PatchTokenizer(D_trans, patch_sz)

        # [CLS] tokenlar
        self.cls_in_token  = nn.Parameter(torch.randn(1, 1, D_trans) * 0.02)
        self.cls_out_token = nn.Parameter(torch.randn(1, 1, D_trans) * 0.02)
        self.pair_sep_token= nn.Parameter(torch.randn(1, 1, D_trans) * 0.02)
        # Test query tokeni (test_input'un farkli bir rol oynadigini belirtir)
        self.cls_test_token= nn.Parameter(torch.randn(1, 1, D_trans) * 0.02)

        # Ornek sira embedding (kac. egitim ornegi olduğunu kodlar)
        self.example_embed = nn.Embedding(max_examples, D_trans)

        # Transformer katmanlari
        self.layers = nn.ModuleList([
            TransformerBlock(D_trans, n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(D_trans)

        # Cikti kafalar
        self.size_head    = SizeHead(D_trans)
        self.bound_head   = BoundaryModeHead(D_trans)
        self.seg_head     = TransformerSegHead(D_trans, K_slots, canvas_H, canvas_W)

        # Boyut bilgisini task_emb'e enjekte eden lineer (SeedMLP icin)
        # H_out, W_out → sinusoidal encoding → D_trans/4 → task_emb'e eklenir
        self.size_embed = nn.Linear(P_DIM, D_trans)  # boyut sinusoidal → D_trans

    def _size_to_sinusoidal(self, size: torch.Tensor) -> torch.Tensor:
        """
        Boyut tamsayısını sinusoidal embedding'e cevir.
        size : [B] int
        Donus: [B, P_DIM]
        """
        B = size.shape[0]
        device = size.device
        n_freqs = P_DIM // 2
        freqs = torch.tensor(
            [2 * math.pi * (2 ** i) / MAX_SIZE for i in range(n_freqs)],
            dtype=torch.float32, device=device
        )
        s = size.float().unsqueeze(1)   # [B, 1]
        enc = torch.cat([torch.sin(s * freqs), torch.cos(s * freqs)], dim=-1)
        return enc  # [B, P_DIM]

    def _build_sequence(self, train_inputs: torch.Tensor,
                         train_outputs: torch.Tensor,
                         train_masks: torch.Tensor,
                         test_input: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Egitim orneklerinden token dizisi olusturur.

        Parametreler
        ------------
        train_inputs  : [B, max_ex, 11, H, W]
        train_outputs : [B, max_ex, 11, H, W]
        train_masks   : [B, max_ex] bool — gecerli ornekler

        Donus
        -----
        seq          : [B, N_total, D_trans]  token dizisi
        padding_mask : [B, N_total] bool      True = padding (yoksay)
        """
        B, max_ex, C, H, W = train_inputs.shape
        device = train_inputs.device

        # PosEnc canvas icin onceden hesapla
        posenc = make_posenc(H, W, P=P_DIM, device=device)  # [P, H, W]
        posenc_b = posenc.unsqueeze(0).expand(B * max_ex, -1, -1, -1)  # [B*max_ex, P, H, W]

        # Tum girisleri tokenize et
        all_in  = train_inputs.view(B * max_ex, C, H, W)     # [B*max_ex, 11, H, W]
        all_out = train_outputs.view(B * max_ex, C, H, W)    # [B*max_ex, 11, H, W]
        in_tok  = self.patch_tokenizer(all_in,  posenc_b)    # [B*max_ex, N_p, D]
        out_tok = self.patch_tokenizer(all_out, posenc_b)    # [B*max_ex, N_p, D]

        N_p = in_tok.shape[1]
        in_tok  = in_tok.view(B, max_ex, N_p, -1)            # [B, max_ex, N_p, D]
        out_tok = out_tok.view(B, max_ex, N_p, -1)           # [B, max_ex, N_p, D]

        # Her ornek icin [CLS_in | in_tokens | CLS_out | out_tokens | SEP]
        seq_parts = []
        mask_parts = []
        for ei in range(max_ex):
            valid = train_masks[:, ei].unsqueeze(1)  # [B, 1]
            ex_emb = self.example_embed(
                torch.tensor(ei, device=device)
            ).view(1, 1, -1).expand(B, -1, -1)       # [B, 1, D]

            cls_in  = self.cls_in_token.expand(B, -1, -1)    # [B, 1, D]
            cls_out = self.cls_out_token.expand(B, -1, -1)   # [B, 1, D]
            sep     = self.pair_sep_token.expand(B, -1, -1)  # [B, 1, D]

            # Giris bolumu
            pair_seq = torch.cat([
                cls_in + ex_emb,          # [B, 1, D]
                in_tok[:, ei],            # [B, N_p, D]
                cls_out,                  # [B, 1, D]
                out_tok[:, ei],           # [B, N_p, D]
                sep,                      # [B, 1, D]
            ], dim=1)                     # [B, 2+2*N_p, D]

            seq_len = pair_seq.shape[1]

            # Gecersiz ornekler icin maske
            pair_mask = (~valid.bool()).expand(B, seq_len)  # [B, seq_len] True=padding
            seq_parts.append(pair_seq)
            mask_parts.append(pair_mask)

        # Test input'u ek bir bolum olarak ekle (varsa)
        if test_input is not None:
            # test_input: [B, 11, H, W]  canvas uzerinde
            H_t, W_t = test_input.shape[-2], test_input.shape[-1]
            posenc_test = make_posenc(H_t, W_t, P=P_DIM, device=device)  # [P, H, W]
            posenc_test = posenc_test.unsqueeze(0).expand(B, -1, -1, -1)  # [B, P, H, W]
            cls_test = self.cls_test_token.expand(B, -1, -1)
            sep      = self.pair_sep_token.expand(B, -1, -1)
            test_tok = self.patch_tokenizer(test_input, posenc_test)   # [B, N_p, D]
            test_seq = torch.cat([cls_test, test_tok, sep], dim=1)  # [B, 2+N_p, D]
            test_mask= torch.zeros(B, test_seq.shape[1], dtype=torch.bool, device=device)
            seq_parts.append(test_seq)
            mask_parts.append(test_mask)

        seq          = torch.cat(seq_parts,  dim=1)  # [B, ..., D]
        padding_mask = torch.cat(mask_parts, dim=1)  # [B, N_total] bool

        # [CLS] global token en basa ekle
        global_cls = (self.cls_in_token + self.cls_out_token).expand(B, -1, -1)
        seq          = torch.cat([global_cls, seq], dim=1)
        padding_mask = torch.cat([
            torch.zeros(B, 1, dtype=torch.bool, device=device),
            padding_mask
        ], dim=1)

        return seq, padding_mask

    def forward(self,
                train_inputs:  torch.Tensor,
                train_outputs: torch.Tensor,
                train_masks:   torch.Tensor,
                test_input:    Optional[torch.Tensor] = None,
                ground_truth_H: Optional[torch.Tensor] = None,
                ground_truth_W: Optional[torch.Tensor] = None) -> dict:
        """
        Parametreler
        ------------
        train_inputs  : [B, max_ex, 11, H, W]
        train_outputs : [B, max_ex, 11, H, W]
        train_masks   : [B, max_ex] bool
        ground_truth_H: [B] int — egitimde boyut kaybı icin (None=test)
        ground_truth_W: [B] int

        Donus
        -----
        {
          'task_emb'     : [B, D_trans]
          'input_tokens' : [B, N_tok, D_trans]  SeedMLP icin
          'h_logits'     : [B, 30]  H boyut logitleri
          'w_logits'     : [B, 30]  W boyut logitleri
          'H_out'        : [B] int  boyut tahmini
          'W_out'        : [B] int
          'boundary_logits': [B, 3]
          'boundary_modes' : list[str]
          'obj_mask_0'   : [B, K, H_canvas, W_canvas]
        }
        """
        B = train_inputs.shape[0]
        device = train_inputs.device

        # Token dizisi olustur (test_input varsa o da eklenir)
        seq, pad_mask = self._build_sequence(train_inputs, train_outputs,
                                              train_masks, test_input=test_input)
        # seq: [B, N_total, D], pad_mask: [B, N_total] bool

        # Transformer katmanlari
        x = seq
        for layer in self.layers:
            x = layer(x, key_padding_mask=pad_mask)
        x = self.norm(x)   # [B, N_total, D]

        # [0] = global CLS token → task_emb
        task_emb     = x[:, 0, :]    # [B, D_trans]
        input_tokens = x             # [B, N_total, D] — SeedMLP tum sekansi kullanir

        # Boyut kafasi
        h_logits, w_logits = self.size_head(task_emb)
        H_out, W_out       = self.size_head.predict(task_emb)

        # Boyut bilgisini task_emb'e dahil et (SeedMLP icin daha zengin sinyal)
        h_sinenc = self._size_to_sinusoidal(H_out)   # [B, P_DIM]
        w_sinenc = self._size_to_sinusoidal(W_out)   # [B, P_DIM]
        size_enc = self.size_embed(h_sinenc + w_sinenc)  # [B, D_trans]
        task_emb = task_emb + 0.1 * size_enc             # yumusak ekleme

        # Sinir modu kafasi
        bound_logits = self.bound_head(task_emb)
        bound_modes  = self.bound_head.predict(task_emb)

        # Nesne maske kafasi
        obj_mask_0 = self.seg_head(task_emb)  # [B, K, canvas_H, canvas_W]

        return {
            'task_emb':        task_emb,
            'input_tokens':    input_tokens,
            'h_logits':        h_logits,
            'w_logits':        w_logits,
            'H_out':           H_out,
            'W_out':           W_out,
            'boundary_logits': bound_logits,
            'boundary_modes':  bound_modes,
            'obj_mask_0':      obj_mask_0,
        }


# ────────────────────────────────────────────────────────────────────────
# Hizli Test
# ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== transformer.py birim testi ===\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Cihaz: {device}")

    B        = 2
    max_ex   = 3
    H, W     = 8, 8

    # Sahte egitim ornekleri
    train_in  = torch.softmax(torch.randn(B, max_ex, 11, H, W, device=device), dim=2)
    train_out = torch.softmax(torch.randn(B, max_ex, 11, H, W, device=device), dim=2)
    train_mask= torch.ones(B, max_ex, dtype=torch.bool, device=device)
    train_mask[0, -1] = False   # Son ornek gecersiz

    model = StrategyTransformer(D_trans=D_TRANS).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"StrategyTransformer parametre sayisi: {param_count:,}")

    out = model(train_in, train_out, train_mask)

    print(f"\nCiktilar:")
    print(f"  task_emb     : {out['task_emb'].shape}")         # [2, 128]
    print(f"  input_tokens : {out['input_tokens'].shape}")     # [2, N_tok, 128]
    print(f"  h_logits     : {out['h_logits'].shape}")         # [2, 30]
    print(f"  H_out        : {out['H_out']}")                  # [2] int
    print(f"  W_out        : {out['W_out']}")                  # [2] int
    print(f"  bound_modes  : {out['boundary_modes']}")
    print(f"  obj_mask_0   : {out['obj_mask_0'].shape}")       # [2, 12, 30, 30]

    # Sekil kontrolleri
    assert out['task_emb'].shape  == (B, D_TRANS)
    assert out['h_logits'].shape  == (B, MAX_SIZE)
    assert out['obj_mask_0'].shape == (B, K_MAX, CANVAS_H, CANVAS_W)
    print("\n[OK] Sekil kontrolleri gecti")

    # Boyut tahmini: 1-30 araliginda olmali
    assert (out['H_out'] >= 1).all() and (out['H_out'] <= 30).all()
    assert (out['W_out'] >= 1).all() and (out['W_out'] <= 30).all()
    print("[OK] Boyut tahmini 1-30 araliginda")

    # Maske normalize: her piksel slot agirliklari toplamı = 1
    mask_sum = out['obj_mask_0'].sum(dim=1)  # [B, H, W]
    assert (mask_sum - 1.0).abs().max() < 1e-4
    print("[OK] ObjectMask normalizasyon dogru")

    # Gradient
    loss = out['task_emb'].mean() + out['h_logits'].mean()
    loss.backward()
    print("[OK] Backward gecti")

    print("\n[OK] Tum testler gecti.")
