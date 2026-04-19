"""
color_codec.py
--------------
Grid <-> one-hot donusumleri ve NULL sinifi yonetimi.

ARC renk sistemi:
  0-9  : Gercek ARC renkleri (siyah=0 dahil, gercek renklerdir)
  10   : NULL/padding (canvas dolgu, egitim verisinde gercek renk DEGIL)

N_COLOR = 11 (10 gercek + 1 NULL)

Fonksiyonlar:
  grid_to_onehot   : [H,W] int   → [11,H,W] float32  (one-hot)
  onehot_to_grid   : [11,H,W]    → [H,W]    int       (argmax)
  null_canvas      : 30x30 NULL dolu canvas uretir
  place_grid       : Grid'i canvas'a [0:H, 0:W] konumuna yerlestir
  extract_grid     : Canvas'tan [0:H_out, 0:W_out] bolgesini al
  gumbel_decode    : [11,H,W] logit → [H,W] int (sicaklikli ornekleme)
  color_to_null_mask: NULL pikselleri isaretleyen maske
"""

import torch
import torch.nn.functional as F

# ────────────────────────────────────────────────────────────────────────
# Sabitler
# ────────────────────────────────────────────────────────────────────────

N_COLOR   = 11    # Toplam sinif sayisi (0-9 gercek + 10=NULL)
NULL_IDX  = 10    # NULL/padding sinifi indeksi
CANVAS_H  = 30    # Sabit canvas yuksekligi
CANVAS_W  = 30    # Sabit canvas genisligi


# ────────────────────────────────────────────────────────────────────────
# Grid <-> One-hot Donusumu
# ────────────────────────────────────────────────────────────────────────

def grid_to_onehot(grid: torch.Tensor) -> torch.Tensor:
    """
    Integer grid'i one-hot tensore donusturur.

    Parametreler
    ------------
    grid : [H, W] int64 ya da [B, H, W] int64, degerler 0-10

    Donus
    -----
    onehot : [11, H, W] float32 ya da [B, 11, H, W] float32

    Notlar
    ------
    - NULL (idx=10) gercek bir ARC rengi DEGIL, canvas dolgusu.
    - Grid degerlerinin 0-10 araliginda oldugu varsayilir.
    """
    if grid.dim() == 2:
        # [H, W] → [11, H, W]
        return F.one_hot(grid.long(), num_classes=N_COLOR).float().permute(2, 0, 1)
    elif grid.dim() == 3:
        # [B, H, W] → [B, 11, H, W]
        return F.one_hot(grid.long(), num_classes=N_COLOR).float().permute(0, 3, 1, 2)
    else:
        raise ValueError(f"grid boyutu 2 veya 3 olmali, alindi: {grid.dim()}")


def onehot_to_grid(onehot: torch.Tensor) -> torch.Tensor:
    """
    One-hot (veya logit) tensoru integer grid'e donusturur (argmax).

    Parametreler
    ------------
    onehot : [11, H, W] ya da [B, 11, H, W]

    Donus
    -----
    grid : [H, W] int64 ya da [B, H, W] int64
    """
    if onehot.dim() == 3:
        return onehot.argmax(dim=0)     # [H, W]
    elif onehot.dim() == 4:
        return onehot.argmax(dim=1)     # [B, H, W]
    else:
        raise ValueError(f"onehot boyutu 3 veya 4 olmali, alindi: {onehot.dim()}")


# ────────────────────────────────────────────────────────────────────────
# Canvas Islemleri
# ────────────────────────────────────────────────────────────────────────

def null_canvas(B: int = 1, device=None) -> torch.Tensor:
    """
    30x30 NULL dolu canvas one-hot tensoru uretir.

    Donus
    -----
    canvas : [B, 11, 30, 30] float32
             canvas[:, NULL_IDX, :, :] == 1.0  (hepsi NULL)
    """
    canvas = torch.zeros(B, N_COLOR, CANVAS_H, CANVAS_W, device=device)
    canvas[:, NULL_IDX, :, :] = 1.0
    return canvas


def null_latent(B: int, L: int = 8, device=None) -> torch.Tensor:
    """
    30x30 sifir latent tensoru uretir.

    Donus
    -----
    latent : [B, L, 30, 30] float32, tum sifir
    """
    return torch.zeros(B, L, CANVAS_H, CANVAS_W, device=device)


def place_grid(canvas: torch.Tensor, grid: torch.Tensor,
               H_in: int, W_in: int) -> torch.Tensor:
    """
    Integer grid'i canvas'in [0:H_in, 0:W_in] bolgesine yerlestirir.
    Geri kalan bolge NULL olarak kalir.

    Parametreler
    ------------
    canvas : [B, 11, 30, 30] float32 (NULL dolu)
    grid   : [H_in, W_in] int64 ya da [B, H_in, W_in] int64
    H_in   : Giriş grid yuksekligi
    W_in   : Giriş grid genisligi

    Donus
    -----
    canvas : [B, 11, 30, 30] float32 (guncellendi, in-place degil)
    """
    assert H_in <= CANVAS_H and W_in <= CANVAS_W, \
        f"Grid boyutu ({H_in},{W_in}) canvas'tan ({CANVAS_H},{CANVAS_W}) buyuk olamaz"

    canvas = canvas.clone()
    B = canvas.shape[0]

    if grid.dim() == 2:
        grid = grid.unsqueeze(0).expand(B, -1, -1)  # [B, H_in, W_in]

    onehot_in = grid_to_onehot(grid)  # [B, 11, H_in, W_in]
    canvas[:, :, :H_in, :W_in] = onehot_in
    return canvas


def place_latent(latent: torch.Tensor, seed_latent: torch.Tensor,
                 H_out: int, W_out: int) -> torch.Tensor:
    """
    SeedMLP ciktisini latent canvas'a yerlestirir.

    Parametreler
    ------------
    latent      : [B, L, 30, 30] float32 (sifir dolu)
    seed_latent : [B, L, H_out, W_out] float32
    H_out, W_out: Hedef bolge boyutu

    Donus
    -----
    latent : [B, L, 30, 30] (guncellendi, in-place degil)
    """
    latent = latent.clone()
    latent[:, :, :H_out, :W_out] = seed_latent
    return latent


def extract_output(color_tensor: torch.Tensor,
                   H_out: int, W_out: int,
                   as_grid: bool = True) -> torch.Tensor:
    """
    NCA ciktisinin [0:H_out, 0:W_out] hedef bolgesini alir.

    Parametreler
    ------------
    color_tensor : [B, 11, 30, 30] ya da [11, 30, 30]
    H_out, W_out : Hedef boyut
    as_grid      : True → argmax uygula, int grid dondur
                   False → logit/prob olarak dondur

    Donus
    -----
    as_grid=True : [B, H_out, W_out] int64 ya da [H_out, W_out] int64
    as_grid=False: [B, 11, H_out, W_out] ya da [11, H_out, W_out]
    """
    if color_tensor.dim() == 3:
        region = color_tensor[:, :H_out, :W_out]    # [11, H_out, W_out]
        return region.argmax(dim=0) if as_grid else region
    elif color_tensor.dim() == 4:
        region = color_tensor[:, :, :H_out, :W_out] # [B, 11, H_out, W_out]
        return region.argmax(dim=1) if as_grid else region
    else:
        raise ValueError(f"color_tensor boyutu 3 veya 4 olmali, alindi: {color_tensor.dim()}")


# ────────────────────────────────────────────────────────────────────────
# Maske Yardimcilari
# ────────────────────────────────────────────────────────────────────────

def get_null_mask(color: torch.Tensor) -> torch.Tensor:
    """
    NULL pikselleri isaretleyen boolean maske dondurur.

    Parametreler
    ------------
    color : [B, 11, H, W] one-hot veya logit

    Donus
    -----
    mask : [B, H, W] bool  (True = NULL piksel)
    """
    return color.argmax(dim=1) == NULL_IDX


def get_active_mask(H_out: int, W_out: int,
                    B: int = 1, device=None) -> torch.Tensor:
    """
    [0:H_out, 0:W_out] bolgesini True, disini False yapan canvas maskesi.

    Donus
    -----
    mask : [B, 1, 30, 30] float32  (1.0 = aktif, 0.0 = NULL bolge)
    """
    mask = torch.zeros(B, 1, CANVAS_H, CANVAS_W, device=device)
    mask[:, :, :H_out, :W_out] = 1.0
    return mask


# ────────────────────────────────────────────────────────────────────────
# Gumbel-Softmax Decode (sadece cikti asamasinda)
# ────────────────────────────────────────────────────────────────────────

def gumbel_decode(logits: torch.Tensor,
                  temperature: float = 0.1,
                  hard: bool = True) -> torch.Tensor:
    """
    Gumbel-Softmax ile renk logitlerini ornekler.

    SADECE CIKTI ASAMASINDA kullanilir (warp sirasinda degil).
    Warp sirasinda one-hot olarak kalir, Straight-Through ile gradyan akar.

    Parametreler
    ------------
    logits      : [11, H, W] veya [B, 11, H, W]
    temperature : Dusuk = keskin, Yuksek = duzgun (default: 0.1 cikarim icin)
    hard        : True = one-hot (forward), soft (backward)
                  False = yumusak dagilim

    Donus
    -----
    sample : logits ile ayni sekil, float32
    """
    return F.gumbel_softmax(logits, tau=temperature, hard=hard, dim=-3 if logits.dim() == 3 else 1)


# ────────────────────────────────────────────────────────────────────────
# ARC JSON Format Yardimcilari
# ────────────────────────────────────────────────────────────────────────

def arc_list_to_tensor(grid_list: list, device=None) -> torch.Tensor:
    """
    ARC JSON formatindaki liste [[r,g,b,...], ...] → [H, W] int64 tensor.

    Parametreler
    ------------
    grid_list : ARC JSON'dan okunan liste (liste-in-liste)

    Donus
    -----
    tensor : [H, W] int64, degerler 0-9
    """
    return torch.tensor(grid_list, dtype=torch.int64, device=device)


def tensor_to_arc_list(grid: torch.Tensor) -> list:
    """
    [H, W] int64 tensor → ARC JSON formatindaki liste.

    Parametreler
    ------------
    grid : [H, W] int64

    Donus
    -----
    liste : [[int, ...], ...]
    """
    return grid.tolist()


# ────────────────────────────────────────────────────────────────────────
# Hizli test
# ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import torch

    print("=== color_codec.py birim testi ===\n")

    # --- grid_to_onehot testi ---
    g = torch.tensor([[0, 1, 2], [3, 0, 9]], dtype=torch.int64)
    oh = grid_to_onehot(g)
    print(f"grid_to_onehot : {g.shape} -> {oh.shape}")   # [11, 2, 3]
    assert oh.shape == (11, 2, 3)
    assert oh[:, 0, 0].argmax().item() == 0   # renk 0
    assert oh[:, 0, 1].argmax().item() == 1   # renk 1
    assert oh[:, 1, 2].argmax().item() == 9   # renk 9
    print("[OK] grid_to_onehot dogru")

    # --- onehot_to_grid testi ---
    g2 = onehot_to_grid(oh)
    print(f"\nonehot_to_grid : {oh.shape} -> {g2.shape}")  # [2, 3]
    assert torch.equal(g, g2), "HATA: geri donusum basarisiz"
    print("[OK] onehot_to_grid dogru")

    # --- Batch versiyonu ---
    gb = torch.stack([g, g], dim=0)   # [2, 2, 3]
    ohb = grid_to_onehot(gb)
    print(f"\nBatch one-hot  : {gb.shape} -> {ohb.shape}")  # [2, 11, 2, 3]
    assert ohb.shape == (2, 11, 2, 3)
    gb2 = onehot_to_grid(ohb)
    assert torch.equal(gb, gb2)
    print("[OK] Batch donusum dogru")

    # --- null_canvas testi ---
    canvas = null_canvas(B=2)
    print(f"\nnull_canvas    : {canvas.shape}")  # [2, 11, 30, 30]
    assert canvas[:, NULL_IDX, :, :].all(), "HATA: NULL kanali 1 olmali"
    assert (canvas[:, :NULL_IDX, :, :] == 0).all(), "HATA: Diger kanallar 0 olmali"
    print("[OK] null_canvas dogru")

    # --- place_grid testi ---
    small_grid = torch.tensor([[1, 2], [3, 4]], dtype=torch.int64)
    canvas = null_canvas(B=1)
    canvas = place_grid(canvas, small_grid, H_in=2, W_in=2)
    print(f"\nplace_grid     : grid=(2,2) canvas icine yerlestirildi")
    placed = canvas[0, :, :2, :2].argmax(dim=0)
    assert placed[0, 0] == 1 and placed[0, 1] == 2
    assert placed[1, 0] == 3 and placed[1, 1] == 4
    # Kalan bolge NULL olmali
    assert canvas[0, NULL_IDX, 2:, :].all(), "HATA: Dis bolge NULL olmali"
    print("[OK] place_grid dogru")

    # --- extract_output testi ---
    result = extract_output(canvas, H_out=2, W_out=2, as_grid=True)
    print(f"\nextract_output : {result.shape}")  # [1, 2, 2]
    assert result[0, 0, 0] == 1 and result[0, 1, 1] == 4
    print("[OK] extract_output dogru")

    # --- get_active_mask testi ---
    mask = get_active_mask(H_out=5, W_out=7, B=2)
    print(f"\nget_active_mask: {mask.shape}")  # [2, 1, 30, 30]
    assert mask[0, 0, :5, :7].all(), "HATA: Aktif bolge 1 olmali"
    assert not mask[0, 0, 5, 0].bool(), "HATA: Dis bolge 0 olmali"
    print("[OK] get_active_mask dogru")

    # --- gumbel_decode testi ---
    logits = torch.randn(11, 5, 5)
    decoded = gumbel_decode(logits, temperature=0.1, hard=True)
    print(f"\ngumbel_decode  : {logits.shape} -> {decoded.shape}")  # [11, 5, 5]
    # Hard one-hot: her piksel icin tam olarak bir kanalda 1.0
    assert (decoded.sum(dim=0) - 1.0).abs().max() < 1e-5, "HATA: one-hot degil"
    print("[OK] gumbel_decode dogru")

    # --- ARC format testi ---
    arc_list = [[0, 1, 2], [3, 4, 5]]
    t = arc_list_to_tensor(arc_list)
    back = tensor_to_arc_list(t)
    assert back == arc_list
    print(f"\narc_list <-> tensor: [OK]")

    print("\n[OK] Tum testler gecti.")
