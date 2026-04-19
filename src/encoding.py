"""
encoding.py
-----------
Statik pozisyon ve geometrik ozellik kodlamalari.

Mimarinin STATIK katmanini uretir:
  PosEnc(x,y)  [P=16, H, W]  Sinusoidal konum kodlamasi
  GeoFeat(x,y) [G=4,  H, W]  border_dist, center_dist, x_norm, y_norm

Bu tensörler NCA boyunca HICBIR ZAMAN guncellenmez.
Warp yapilmaz; sadece concat ile h_base'e eklenir.
"""

import torch
import math


# ────────────────────────────────────────────────────────────────────────
# PosEnc — Sinusoidal 2D Konum Kodlamasi
# ────────────────────────────────────────────────────────────────────────

def make_posenc(H: int, W: int, P: int = 16, device=None) -> torch.Tensor:
    """
    Sinusoidal 2D konum kodlamasi uretir.

    Her piksele (x,y) icin P boyutlu sabit bir vektor atar.
    Formul: sin/cos ciftleri, farkli frekanslarda.

    Parametreler
    ------------
    H, W  : Grid yukseklik ve genisligi (<=30 ARC icin)
    P     : Kanal sayisi. P//4 frekans ciftine bolunur (sin+cos, x+y).
            P=16 → 4 frekans, her biri (sin_x, cos_x, sin_y, cos_y) = 4 kanal.
    device: torch.device

    Donus
    -----
    posenc : [P, H, W] float32, sabit (no grad)
    """
    assert P % 4 == 0, f"P={P} 4'un kati olmali (sin_x,cos_x,sin_y,cos_y)"
    n_freqs = P // 4

    # Frekanslar: 2pi * [1, 2, 4, ...] / max_dim
    max_dim = max(H, W)
    freqs = torch.tensor(
        [2 * math.pi * (2 ** i) / max_dim for i in range(n_freqs)],
        dtype=torch.float32, device=device
    )  # [n_freqs]

    # Koordinat gridleri normalize: [0, 1) araliginda
    ys = torch.arange(H, dtype=torch.float32, device=device) / max(H - 1, 1)
    xs = torch.arange(W, dtype=torch.float32, device=device) / max(W - 1, 1)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')  # [H, W]

    # Her frekans icin sin/cos hesapla
    channels = []
    for f in freqs:
        channels.append(torch.sin(f * grid_x))   # sin_x
        channels.append(torch.cos(f * grid_x))   # cos_x
        channels.append(torch.sin(f * grid_y))   # sin_y
        channels.append(torch.cos(f * grid_y))   # cos_y

    posenc = torch.stack(channels, dim=0)  # [P, H, W]
    return posenc.detach()


def make_posenc_batch(H: int, W: int, B: int = 1, P: int = 16, device=None) -> torch.Tensor:
    """
    Batch boyutlu PosEnc: [B, P, H, W]
    Tum batch ornekleri ayni PosEnc'i paylasar.
    """
    enc = make_posenc(H, W, P=P, device=device)   # [P, H, W]
    return enc.unsqueeze(0).expand(B, -1, -1, -1)  # [B, P, H, W]


# ────────────────────────────────────────────────────────────────────────
# GeoFeat — Geometrik Ozellikler
# ────────────────────────────────────────────────────────────────────────

def make_geofeat(H: int, W: int, device=None) -> torch.Tensor:
    """
    4 kanalli geometrik ozellik tensoru uretir.

    Kanallar:
      0: border_dist  — en yakin kenara normalize mesafe [0, 1]
                        (1 = merkez, 0 = kenar)
      1: center_dist  — merkeze normalize mesafe [0, 1]
                        (0 = merkez, 1 = kose)
      2: x_norm       — normalize x koordinati [0, 1] (soldan saga)
      3: y_norm       — normalize y koordinati [0, 1] (yukten asa)

    Parametreler
    ------------
    H, W  : Grid boyutlari
    device: torch.device

    Donus
    -----
    geofeat : [4, H, W] float32, sabit (no grad)
    """
    ys = torch.arange(H, dtype=torch.float32, device=device)
    xs = torch.arange(W, dtype=torch.float32, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')  # [H, W]

    # Normalize koordinatlar [0, 1]
    x_norm = grid_x / max(W - 1, 1)   # [H, W]
    y_norm = grid_y / max(H - 1, 1)   # [H, W]

    # border_dist: en yakin kenara mesafe, [0, 1]
    # dist_left   = x / (W-1)
    # dist_right  = (W-1-x) / (W-1)
    # dist_top    = y / (H-1)
    # dist_bottom = (H-1-y) / (H-1)
    if W > 1 and H > 1:
        dist_x = torch.minimum(grid_x, (W - 1) - grid_x) / ((W - 1) / 2)
        dist_y = torch.minimum(grid_y, (H - 1) - grid_y) / ((H - 1) / 2)
        border_dist = torch.minimum(dist_x, dist_y).clamp(0.0, 1.0)
    else:
        border_dist = torch.zeros(H, W, device=device)

    # center_dist: merkeze Chebyshev mesafesi, [0, 1]
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    if W > 1 and H > 1:
        dx = (grid_x - cx).abs() / (cx)
        dy = (grid_y - cy).abs() / (cy)
        center_dist = torch.maximum(dx, dy).clamp(0.0, 1.0)
    else:
        center_dist = torch.zeros(H, W, device=device)

    geofeat = torch.stack([border_dist, center_dist, x_norm, y_norm], dim=0)  # [4, H, W]
    return geofeat.detach()


def make_geofeat_batch(H: int, W: int, B: int = 1, device=None) -> torch.Tensor:
    """
    Batch boyutlu GeoFeat: [B, 4, H, W]
    """
    feat = make_geofeat(H, W, device=device)         # [4, H, W]
    return feat.unsqueeze(0).expand(B, -1, -1, -1)   # [B, 4, H, W]


# ────────────────────────────────────────────────────────────────────────
# Yardimci: Sabit Canvas (30x30) icin onceden hesapla
# ────────────────────────────────────────────────────────────────────────

class StaticEncodings:
    """
    Sabit 30x30 canvas icin PosEnc ve GeoFeat'i bir kez hesaplayip saklar.
    NCA dongusunde her adimda yeniden hesaplamak yerine bu sinif kullanilir.

    Kullanim:
        enc = StaticEncodings(device=device)
        posenc = enc.posenc    # [1, 16, 30, 30]
        geofeat = enc.geofeat  # [1,  4, 30, 30]
    """

    def __init__(self, H: int = 30, W: int = 30, P: int = 16, device=None):
        self.H = H
        self.W = W
        self.P = P
        self.device = device
        self._posenc = make_posenc(H, W, P=P, device=device).unsqueeze(0)   # [1, P, H, W]
        self._geofeat = make_geofeat(H, W, device=device).unsqueeze(0)       # [1, 4, H, W]

    def get(self, B: int):
        """
        Verilen batch boyutu icin expand edilmis tensörleri dondur.
        Bellek paylasimi yapar (expand = view, kopya degil).
        """
        posenc  = self._posenc.expand(B, -1, -1, -1)   # [B, P, H, W]
        geofeat = self._geofeat.expand(B, -1, -1, -1)  # [B, 4, H, W]
        return posenc, geofeat

    @property
    def posenc(self):
        return self._posenc

    @property
    def geofeat(self):
        return self._geofeat


# ────────────────────────────────────────────────────────────────────────
# Hizli test
# ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import torch

    print("=== encoding.py birim testi ===\n")

    # PosEnc testi
    pe = make_posenc(10, 12, P=16)
    print(f"PosEnc shape  : {pe.shape}")         # bekle: [16, 10, 12]
    print(f"PosEnc range  : [{pe.min():.3f}, {pe.max():.3f}]")  # [-1, 1]
    assert pe.shape == (16, 10, 12), "HATA: PosEnc sekli yanlis"
    assert not pe.requires_grad, "HATA: PosEnc gradient istememeli"

    # GeoFeat testi
    gf = make_geofeat(10, 12)
    print(f"\nGeoFeat shape : {gf.shape}")        # bekle: [4, 10, 12]
    print(f"border_dist[0]: {gf[0]}")
    assert gf.shape == (4, 10, 12), "HATA: GeoFeat sekli yanlis"
    assert gf.min() >= 0.0 and gf.max() <= 1.0, "HATA: GeoFeat [0,1] disinda"

    # StaticEncodings testi
    enc = StaticEncodings(H=30, W=30, P=16)
    pe_b, gf_b = enc.get(B=4)
    print(f"\nBatch PosEnc  : {pe_b.shape}")      # bekle: [4, 16, 30, 30]
    print(f"Batch GeoFeat : {gf_b.shape}")         # bekle: [4, 4, 30, 30]
    assert pe_b.shape  == (4, 16, 30, 30)
    assert gf_b.shape  == (4, 4,  30, 30)

    # Merkez pikseli test: border_dist maksimum olmali
    gf30 = make_geofeat(30, 30)
    center_border = gf30[0, 15, 15]
    edge_border   = gf30[0, 0,  0]
    print(f"\nborder_dist merkez: {center_border:.3f}  (yuksek olmali)")
    print(f"border_dist kose  : {edge_border:.3f}    (0 olmali)")
    assert center_border > edge_border, "HATA: Merkez sinirdan uzak olmali"

    print("\n[OK] Tum testler gecti.")
