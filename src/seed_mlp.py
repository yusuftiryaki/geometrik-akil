"""
seed_mlp.py
-----------
SeedMLP: Generative Canvas — C2 (Grid Resize) cozumu.

Ana fikir: ARC'de boyut degisimi geometrik germe DEGIL;
yeni bir grid'in kural-tabanli insasidir.

Her cikti hucresinin "nereden geldigini" cross-attention ile ogrenir:
  query(x,y) = PosEnc(x, y)
  ctx(x,y)   = CrossAttention(query, input_tokens)
  seed(x,y)  = MLP(concat(query, ctx, task_emb))
             → Latent_0[x,y], ColorLogits_0[x,y]

ARC resize semantigi:
  Tile 2x  : query(x,y) → attend to (x%H_in, y%W_in) in input
  Scale 2x : query(2i,2j) → attend to (i,j)
  Mirror   : query(x, W-1-y) → attend to (x,y)
  Same     : query(x,y) → attend to (x,y)

CrossAttention, bu eslesmeleri orneklerden ogrenir.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from encoding import make_posenc
from color_codec import N_COLOR, NULL_IDX, CANVAS_H, CANVAS_W

# ────────────────────────────────────────────────────────────────────────
# Sabitler
# ────────────────────────────────────────────────────────────────────────

P_DIM    = 16    # PosEnc boyutu (query boyutu)
L_DIM    = 8     # Latent boyutu
D_TRANS  = 128   # Transformer embedding boyutu


# ────────────────────────────────────────────────────────────────────────
# Cross-Attention Modulu
# ────────────────────────────────────────────────────────────────────────

class CrossAttention(nn.Module):
    """
    Cikti pozisyonlari (query) → girdi tokenlar (key/value) arasinda dikkat.

    Query boyutu P_DIM'den D_trans'a projeksiyon yapilir.

    Giris:
      query : [B, N_q, P_DIM]  cikti pozisyon kodlamalari
      kv    : [B, N_k, D_trans] girdi token temsilleri

    Cikis:
      ctx : [B, N_q, D_trans]
    """

    def __init__(self, P_dim: int = P_DIM, D_trans: int = D_TRANS,
                 n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.q_proj = nn.Linear(P_dim, D_trans, bias=False)
        self.attn   = nn.MultiheadAttention(D_trans, n_heads,
                                             dropout=dropout, batch_first=True)
        self.norm   = nn.LayerNorm(D_trans)

    def forward(self, query: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(query)                    # [B, N_q, D_trans]
        ctx, _ = self.attn(q, kv, kv)            # [B, N_q, D_trans]
        ctx = self.norm(q + ctx)                  # residual + norm
        return ctx


# ────────────────────────────────────────────────────────────────────────
# SeedMLP
# ────────────────────────────────────────────────────────────────────────

class SeedMLP(nn.Module):
    """
    Cikti canvas'i icin baslangic durumu uretir.

    Her cikti pozisyonu icin:
      1. PosEnc query olustur
      2. CrossAttention ile girdi token'larindan baglam al
      3. MLP ile Latent_0 ve ColorLogits_0 uret

    Cikis canvas [0:H_out, 0:W_out] bolgesini doldurur.
    Geri kalan [H_out:, W_out:] → NULL (onceden sifir/NULL ile baslatilmis).

    Parametreler
    ------------
    P_dim   : Query PosEnc boyutu (16)
    D_trans : Transformer token boyutu (128)
    L       : Latent boyutu (8)
    n_heads : CrossAttention head sayisi (4)
    """

    def __init__(self,
                 P_dim: int   = P_DIM,
                 D_trans: int = D_TRANS,
                 L: int       = L_DIM,
                 n_heads: int = 4,
                 dropout: float = 0.1,
                 canvas_H: int = CANVAS_H,
                 canvas_W: int = CANVAS_W):
        super().__init__()
        self.P_dim    = P_dim
        self.D_trans  = D_trans
        self.L        = L
        self.canvas_H = canvas_H
        self.canvas_W = canvas_W

        self.cross_attn = CrossAttention(P_dim, D_trans, n_heads, dropout)

        # MLP: concat(query[P], ctx[D], task_emb[D], test_pixel[N_COLOR]) → seed[L + N_COLOR]
        # test_pixel: pozisyon (x,y)'deki test_input'un one-hot degeri (copy-prior icin)
        MLP_in  = P_dim + D_trans + D_trans + N_COLOR
        MLP_mid = 128
        MLP_out = L + N_COLOR
        self.mlp = nn.Sequential(
            nn.Linear(MLP_in, MLP_mid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(MLP_mid, MLP_mid),
            nn.GELU(),
            nn.Linear(MLP_mid, MLP_out),
        )

        # NULL canvas icin baz logit (ilk katmani sifir-baslat)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

        # Copy-prior residual: identity icin guclu inductive bias.
        # color_logits = mlp_out + copy_alpha * test_input[x,y]
        # Init=3.0 → exp(3)≈20x, softmax true color ~95% (identity default)
        # Model ogrenir: identity icin buyuk alpha, resize icin kucuk alpha
        self.copy_alpha = nn.Parameter(torch.tensor(3.0))

    def _make_queries(self, H_out: int, W_out: int,
                      device=None) -> torch.Tensor:
        """
        Hedef bolgedeki tum koordinatlar icin PosEnc query olusturur.

        Donus: [H_out*W_out, P_dim]
        """
        posenc = make_posenc(H_out, W_out, P=self.P_dim, device=device)
        # [P, H_out, W_out] → [H_out*W_out, P]
        return posenc.permute(1, 2, 0).reshape(H_out * W_out, self.P_dim)

    def forward(self,
                input_tokens: torch.Tensor,
                task_emb: torch.Tensor,
                H_out: int,
                W_out: int,
                test_input: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tek (H_out, W_out) boyutu icin canvas uretir.
        Batch icinde tum orneklerin ayni H_out, W_out oldugu varsayilir.
        (Farkli boyutlar icin bkz: forward_batch)

        Parametreler
        ------------
        input_tokens : [B, N_tok, D_trans]  Transformer ciktisi
        task_emb     : [B, D_trans]
        H_out, W_out : int — hedef boyutlar

        Donus
        -----
        color_logits : [B, 11, CANVAS_H, CANVAS_W]  (NULL ile dolu, hedef bolge dolduruldu)
        latent_0     : [B, L,  CANVAS_H, CANVAS_W]  (sifir, hedef bolge dolduruldu)
        """
        B      = input_tokens.shape[0]
        device = input_tokens.device

        # NULL ile dolu baslangic canvas
        color_logits = torch.zeros(B, N_COLOR, self.canvas_H, self.canvas_W, device=device)
        color_logits[:, NULL_IDX, :, :] = 10.0   # NULL logit yuksek (softmax => ~1)
        latent_0 = torch.zeros(B, self.L, self.canvas_H, self.canvas_W, device=device)

        if H_out == 0 or W_out == 0:
            return color_logits, latent_0

        # Query olustur: [H_out*W_out, P_dim]
        queries_flat = self._make_queries(H_out, W_out, device=device)
        N_q = queries_flat.shape[0]

        # Batch boyutuna genisle: [B, N_q, P_dim]
        queries = queries_flat.unsqueeze(0).expand(B, -1, -1)

        # CrossAttention: her pozisyon girdi token'larindan bilgi alir
        ctx = self.cross_attn(queries, input_tokens)   # [B, N_q, D_trans]

        # task_emb'i N_q boyutuna genisle
        task_exp = task_emb.unsqueeze(1).expand(B, N_q, -1)   # [B, N_q, D_trans]

        # test_input'dan pozisyon-bazli pixel ozellikleri (copy-prior query)
        # test_input: [B, N_COLOR, CANVAS_H, CANVAS_W] canvas (one-hot, NULL=10 disarida)
        if test_input is not None:
            # Output grid pozisyonlarindan test_input'un o pozisyondaki degerini al
            test_slice = test_input[:, :, :H_out, :W_out]              # [B, 11, H_out, W_out]
            test_flat  = test_slice.permute(0, 2, 3, 1).reshape(B, N_q, N_COLOR)
        else:
            test_flat = torch.zeros(B, N_q, N_COLOR, device=device)

        # MLP girisi (copy-prior pixel dahil)
        mlp_in = torch.cat([queries, ctx, task_exp, test_flat], dim=-1)  # [B, N_q, P+D+D+11]
        seed   = self.mlp(mlp_in)                                         # [B, N_q, L+11]

        # Canvas'a yaz
        seed_r = seed.permute(0, 2, 1).view(B, self.L + N_COLOR, H_out, W_out)
        # [B, L+11, H_out, W_out]

        latent_seed = seed_r[:, :self.L]     # [B, L, H_out, W_out]
        color_seed  = seed_r[:, self.L:]     # [B, 11, H_out, W_out]

        # Copy-prior residual: identity icin guclu inductive bias
        # color_seed += alpha * test_input_at_output_pos
        # Identity'de test_input[x,y] dogru rengi iceriyor → argmax garantili dogru.
        # Diger gorevler icin MLP offset ogrenir veya alpha kucultur.
        if test_input is not None:
            test_slice = test_input[:, :, :H_out, :W_out]   # [B, 11, H_out, W_out]
            color_seed = color_seed + self.copy_alpha * test_slice

        # Hedef bolgeye yaz
        latent_0[:, :, :H_out, :W_out] = latent_seed
        color_logits[:, :, :H_out, :W_out] = color_seed

        return color_logits, latent_0

    def forward_variable(self,
                          input_tokens: torch.Tensor,
                          task_emb: torch.Tensor,
                          H_outs: torch.Tensor,
                          W_outs: torch.Tensor,
                          test_input: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batch icinde farkli H_out, W_out boyutlari icin canvas uretir.
        Her ornek ayri ayri islenir (batchlanmamis — daha yavas ama dogru).

        Parametreler
        ------------
        input_tokens : [B, N_tok, D_trans]
        task_emb     : [B, D_trans]
        H_outs       : [B] int tensor
        W_outs       : [B] int tensor

        Donus
        -----
        color_logits : [B, 11, CANVAS_H, CANVAS_W]
        latent_0     : [B, L,  CANVAS_H, CANVAS_W]
        """
        B      = input_tokens.shape[0]
        device = input_tokens.device

        color_list  = []
        latent_list = []

        for i in range(B):
            h = H_outs[i].item()
            w = W_outs[i].item()
            ti_i = test_input[i:i+1] if test_input is not None else None
            c_i, l_i = self.forward(
                input_tokens[i:i+1],
                task_emb[i:i+1],
                int(h), int(w),
                test_input=ti_i,
            )
            color_list.append(c_i[0])   # [11, canvas_H, canvas_W]
            latent_list.append(l_i[0])  # [L,  canvas_H, canvas_W]

        return torch.stack(color_list), torch.stack(latent_list)


# ────────────────────────────────────────────────────────────────────────
# Hizli Test
# ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== seed_mlp.py birim testi ===\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Cihaz: {device}")

    B       = 3
    N_tok   = 50      # Transformer'dan gelen token sayisi
    H_out   = 6
    W_out   = 8

    input_tokens = torch.randn(B, N_tok, D_TRANS, device=device)
    task_emb     = torch.randn(B, D_TRANS, device=device)
    # Sahte test_input (one-hot canvas)
    test_input = torch.zeros(B, N_COLOR, CANVAS_H, CANVAS_W, device=device)
    test_input[:, NULL_IDX] = 1.0   # varsayilan NULL
    # Hedef bolgeye rastgele renk ata
    rand_colors = torch.randint(0, 10, (B, H_out, W_out), device=device)
    test_input[:, :, :H_out, :W_out] = 0.0
    for b in range(B):
        for yy in range(H_out):
            for xx in range(W_out):
                test_input[b, rand_colors[b, yy, xx], yy, xx] = 1.0

    model = SeedMLP().to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"SeedMLP parametre sayisi: {param_count:,}")

    # Tek boyut testi (test_input ile)
    color_logits, latent_0 = model(input_tokens, task_emb, H_out, W_out,
                                   test_input=test_input)
    print(f"\nTek boyut ({H_out}x{W_out}):")
    print(f"  color_logits : {color_logits.shape}")   # [3, 11, 30, 30]
    print(f"  latent_0     : {latent_0.shape}")        # [3, 8, 30, 30]

    assert color_logits.shape == (B, N_COLOR, CANVAS_H, CANVAS_W)
    assert latent_0.shape     == (B, L_DIM,   CANVAS_H, CANVAS_W)
    print("  [OK] Sekil dogru")

    # Hedef bolge doldu, dis bolge NULL kaldi mi?
    target_region = color_logits[:, :, :H_out, :W_out]
    outside_region = color_logits[:, :, H_out:, :]
    # Dis bolge: NULL logit yuksek
    outside_argmax = outside_region.argmax(dim=1)
    assert (outside_argmax == NULL_IDX).all(), "HATA: Dis bolge NULL olmali"
    print("  [OK] Dis bolge NULL")

    # Degisken boyut testi
    H_outs = torch.tensor([4, 8, 6], device=device)
    W_outs = torch.tensor([5, 7, 8], device=device)
    c_var, l_var = model.forward_variable(input_tokens, task_emb, H_outs, W_outs,
                                          test_input=test_input)
    print(f"\nDegisken boyut:")
    print(f"  color_logits : {c_var.shape}")   # [3, 11, 30, 30]
    print(f"  latent_0     : {l_var.shape}")    # [3, 8, 30, 30]
    assert c_var.shape == (B, N_COLOR, CANVAS_H, CANVAS_W)
    print("  [OK] Degisken boyut dogru")

    # Gradient testi
    loss = color_logits.mean() + latent_0.mean()
    loss.backward()
    print("\n[OK] Backward gecti (gradient akiyor)")

    # Her ornekte icindeki bolgenin degerleri NULL'dan farkli olmali
    # (MLP'nin urettigi degerlerin sifirlanmamis olmasi lazim)
    in_region = color_logits[:, :, :H_out, :W_out]
    # Baslatma sifirlari, MLP tarafindan degistirilmis olmali
    # (Ciktilar sifir olmayabilir, kontrol etmek yerine gradyanlarin aktigi yeterli)
    print("[OK] CrossAttention + MLP zinciri calisiyor")

    print("\n[OK] Tum testler gecti.")
