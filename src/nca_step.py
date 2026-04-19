"""
nca_step.py
-----------
NCA (Neural Cellular Automata) tek adim implementasyonu.

C1-onaylı tam mimari (12 adim):
  1.  Giris concat: Color + Latent + PosEnc + GeoFeat
  2.  LocalEncoder: DWConv3x3 + PWConv
  3.  GradFeatures: Sobel(Latent) — yardimci sinyal
  4.  h_base = concat(x, local_ctx, grad_feat)
  5.  Per-Object FiLM: ObjectMask ile agirlikli FiLM kosullama
  6.  FlowNet: h_obj -> v [2, H, W]
  7.  Akis sinirla: v = v_max * tanh(v)
  8.  BoundaryController: damping + mode
  9.  Color warp: ST-nearest
  10. Latent warp: bilinear
  11. ObjectMask warp: ST-nearest + normalize
  12. LogicGate: post-warp yerel MLP (cakisma, renk degisimi, NULL dolgu)

Boyutlar (V1):
  N_COLOR = 11  (0-9 gercek + 10=NULL)
  L       = 8   (Latent)
  P       = 16  (PosEnc)
  G       = 4   (GeoFeat)
  K_max   = 12  (Nesne slot)
  C_ctx   = 32  (LocalEncoder cikti)
  C_h     = 11+8+16+4 + 32 + 16 = 87
  C_flow  = 32  (FlowNet ara katman)
  v_max   = 0.5 (SeedMLP'den sonra dusuruldu — yerel duzeltici)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# ────────────────────────────────────────────────────────────────────────
# Sabitler
# ────────────────────────────────────────────────────────────────────────

N_COLOR  = 11
NULL_IDX = 10
L_DIM    = 8
P_DIM    = 16
G_DIM    = 4
K_MAX    = 12
C_CTX    = 32
C_FLOW   = 32
V_MAX    = 0.5     # SeedMLP sonrasi daha kucuk (yerel duzeltici rol)

# h_base boyutu: Color + Latent + PosEnc + GeoFeat + LocalCtx + GradFeat
C_H = N_COLOR + L_DIM + P_DIM + G_DIM + C_CTX + 2 * L_DIM  # = 11+8+16+4+32+16 = 87


# ────────────────────────────────────────────────────────────────────────
# Alt Modüller
# ────────────────────────────────────────────────────────────────────────

class LocalEncoder(nn.Module):
    """
    DWConv3x3 + PWConv: yerel komsuluk algilama.

    Giris : [B, C_in, H, W]  (C_in = N_COLOR+L+P+G = 39)
    Cikis : [B, C_ctx, H, W] (C_ctx = 32)
    """

    def __init__(self, C_in: int, C_ctx: int = C_CTX):
        super().__init__()
        self.dw_conv = nn.Conv2d(
            C_in, C_in,
            kernel_size=3, padding=1, groups=C_in,
            bias=False
        )  # Depthwise: her kanal bagimsiz
        self.pw_conv = nn.Conv2d(
            C_in, C_ctx,
            kernel_size=1, bias=True
        )  # Pointwise: kanal karisimi
        self.norm = nn.GroupNorm(min(8, C_ctx), C_ctx)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ctx = self.dw_conv(x)
        ctx = self.pw_conv(ctx)
        ctx = self.norm(ctx)
        ctx = F.relu(ctx)
        return ctx  # [B, C_ctx, H, W]


class GradFeatures(nn.Module):
    """
    Latent alaninin Sobel turevileri.

    Akisin KAYNAGI degil — yardimci geometrik ozellik.
    Kenar/yon bilgisi verir.

    Giris : [B, L, H, W]
    Cikis : [B, 2L, H, W]  (x ve y turevleri)
    """

    def __init__(self, L: int = L_DIM):
        super().__init__()
        self.L = L
        # Sobel kernelleri sabit (ogrenilebilir degil)
        self.register_buffer('sobel_x', self._make_sobel_x(L))
        self.register_buffer('sobel_y', self._make_sobel_y(L))

    @staticmethod
    def _make_sobel_x(L: int) -> torch.Tensor:
        kernel = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1],
        ], dtype=torch.float32)
        return kernel.view(1, 1, 3, 3).expand(L, 1, 3, 3).contiguous()

    @staticmethod
    def _make_sobel_y(L: int) -> torch.Tensor:
        kernel = torch.tensor([
            [-1, -2, -1],
            [0,  0,  0],
            [1,  2,  1],
        ], dtype=torch.float32)
        return kernel.view(1, 1, 3, 3).expand(L, 1, 3, 3).contiguous()

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        gx = F.conv2d(latent, self.sobel_x, padding=1, groups=self.L)
        gy = F.conv2d(latent, self.sobel_y, padding=1, groups=self.L)
        return torch.cat([gx, gy], dim=1)  # [B, 2L, H, W]


class PerObjectFiLM(nn.Module):
    """
    Per-object FiLM kosullama (Secenek A: tek adimda).

    Her nesne slotu icin gamma_k, beta_k uretir.
    Piksele ozel h_obj = sum_k mask_k(x,y) * (gamma_k * h_base(x,y) + beta_k)

    Giris:
      task_emb   : [B, D_trans]
      obj_tokens : [K, D_obj]  (statik tablo, embed olarak ogrenilebilir)
      h_base     : [B, C_h, H, W]
      obj_mask   : [B, K, H, W]  (ObjectMask)

    Cikis:
      h_obj : [B, C_h, H, W]
    """

    def __init__(self, D_trans: int, D_obj: int, K: int, C_h: int):
        super().__init__()
        self.K    = K
        self.C_h  = C_h
        # Her slot icin ayri FiLMHead: concat(task_emb, obj_token) → gamma, beta
        self.film_head = nn.Linear(D_trans + D_obj, 2 * C_h)
        # Nesne kimlik tablolari (V1'de statik — egitimle guncellenir ama adim adim degil)
        self.obj_tokens = nn.Embedding(K, D_obj)

    def forward(self, task_emb: torch.Tensor,
                h_base: torch.Tensor,
                obj_mask: torch.Tensor) -> torch.Tensor:
        """
        Parametreler
        ------------
        task_emb : [B, D_trans]
        h_base   : [B, C_h, H, W]
        obj_mask : [B, K, H, W]  soft maskeler, normalize edilmis

        Donus
        -----
        h_obj : [B, C_h, H, W]
        """
        B, C_h, H, W = h_base.shape
        K = self.K

        # Tum slot indisleri
        slot_ids = torch.arange(K, device=task_emb.device)  # [K]
        tokens   = self.obj_tokens(slot_ids)                  # [K, D_obj]

        # Her slot icin FiLM parametreleri
        task_exp = task_emb.unsqueeze(1).expand(B, K, -1)    # [B, K, D_trans]
        tok_exp  = tokens.unsqueeze(0).expand(B, K, -1)      # [B, K, D_obj]
        inp      = torch.cat([task_exp, tok_exp], dim=-1)     # [B, K, D_trans+D_obj]
        film_out = self.film_head(inp)                         # [B, K, 2*C_h]
        gamma    = film_out[:, :, :C_h]                        # [B, K, C_h]
        beta     = film_out[:, :, C_h:]                        # [B, K, C_h]

        # Piksele ozel FiLM: agirlikli toplam
        # gamma: [B, K, C_h] → [B, K, C_h, 1, 1] * mask [B, K, 1, H, W]
        gamma_4d = gamma.unsqueeze(-1).unsqueeze(-1)           # [B, K, C_h, 1, 1]
        beta_4d  = beta.unsqueeze(-1).unsqueeze(-1)            # [B, K, C_h, 1, 1]
        mask_4d  = obj_mask.unsqueeze(2)                       # [B, K, 1, H, W]
        h_base_5d= h_base.unsqueeze(1)                        # [B, 1, C_h, H, W]

        # sum_k mask_k * (gamma_k * h_base + beta_k)
        h_obj = (mask_4d * (gamma_4d * h_base_5d + beta_4d)).sum(dim=1)  # [B, C_h, H, W]
        return h_obj


class FlowNet(nn.Module):
    """
    h_obj'den flow vektoru uretir.

    Mimari: Conv1x1(C_h → C_flow) → ReLU → Conv1x1(C_flow → 2)

    Giris : [B, C_h, H, W]
    Cikis : [B, 2, H, W]
    """

    def __init__(self, C_h: int = C_H, C_flow: int = C_FLOW):
        super().__init__()
        self.conv1 = nn.Conv2d(C_h, C_flow, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(C_flow, 2,    kernel_size=1, bias=True)
        # Kucuk bir baslangic akisi icin baslatma
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        v = F.relu(self.conv1(h))
        v = self.conv2(v)
        return v  # [B, 2, H, W]


class BoundaryController(nn.Module):
    """
    3 katmanli sinir koruma:
    1. Akis tanh ile sinirlanir (FlowNet cikisinda zaten v_max ile carpilir)
    2. Sinir mesafesine bagli damping
    3. Boundary mode (border/zeros/reflection) — Transformer'dan gelir

    'border' modunda sinira yakin piksel akisi 0'a suruklenir.
    """

    def __init__(self, sharpness: float = 5.0):
        super().__init__()
        self.sharpness = sharpness

    def forward(self, v: torch.Tensor,
                active_mask: torch.Tensor,
                boundary_mode: str = 'border') -> Tuple[torch.Tensor, str]:
        """
        Parametreler
        ------------
        v           : [B, 2, H, W] akis vektoru
        active_mask : [B, 1, H, W] float, 1=aktif bolge
        boundary_mode: 'border' | 'zeros' | 'reflection'

        Donus
        -----
        v_damped : [B, 2, H, W]
        padding_mode : str (F.grid_sample padding_mode icin)
        """
        if boundary_mode == 'border':
            # Sinirdaki piksel akisi 0'a suruklenir (dauvara yapis)
            dist = self._boundary_distance(active_mask)   # [B, 1, H, W]
            damp = torch.sigmoid(dist * self.sharpness)
            v    = v * damp
            padding_mode = 'border'
        elif boundary_mode == 'zeros':
            # Sinir disina cikan pikseller sifir renk alir
            dist = self._boundary_distance(active_mask)
            damp = torch.sigmoid(dist * self.sharpness)
            v    = v * damp
            padding_mode = 'zeros'
        else:  # 'reflection'
            # Sinirda seker (reflecting)
            padding_mode = 'reflection'

        return v, padding_mode

    @staticmethod
    def _boundary_distance(active_mask: torch.Tensor) -> torch.Tensor:
        """
        Her aktif pikselin sinira olan normalize mesafesini hesaplar.
        active_mask: [B, 1, H, W], 1=aktif
        Donus: [B, 1, H, W] float, buyuk = sinirdan uzak
        """
        # Erosion ile sinir mesafesi yaklasimlama
        # Her erosion adimi siniri 1 piksel iceride yapar
        kernel = torch.ones(1, 1, 3, 3, device=active_mask.device)
        dist = active_mask.float().clone()
        # 3 adim erosion = 3 piksel derinlikli gradient
        for _ in range(3):
            eroded = F.conv2d(dist, kernel, padding=1) / 9.0
            dist   = eroded * active_mask  # aktif bolge disini sifir yap
        return dist


class LogicGate(nn.Module):
    """
    Post-warp yerel MLP: cakisma, renk degisimi, NULL dolgu, latent guncelleme.

    Mimari: DWConv3x3 + Conv1x1(C_in → C_mid → C_out)

    C_in = Color + Latent + ObjectMask + flow(2) + PosEnc + GeoFeat
         = 11 + L + K + 2 + P + G = 11+8+12+2+16+4 = 53
    C_out = Color + Latent = 11 + L = 19
    """

    def __init__(self, C_in: int, C_mid: int, C_out: int):
        super().__init__()
        self.dw_conv  = nn.Conv2d(C_in, C_in, kernel_size=3, padding=1, groups=C_in, bias=False)
        self.conv1    = nn.Conv2d(C_in * 2, C_mid, kernel_size=1, bias=True)
        self.conv2    = nn.Conv2d(C_mid, C_out, kernel_size=1, bias=True)
        self.norm     = nn.GroupNorm(min(8, C_mid), C_mid)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """
        Parametreler
        ------------
        inp : [B, C_in, H, W]  concat(Color', Latent', ObjectMask, v, PosEnc, GeoFeat)

        Donus
        -----
        out : [B, C_out, H, W]  concat(dColor, dLatent) — delta guncelleme
        """
        ctx = self.dw_conv(inp)
        cat = torch.cat([inp, ctx], dim=1)         # [B, 2*C_in, H, W]
        out = F.relu(self.norm(self.conv1(cat)))
        out = self.conv2(out)                       # [B, C_out, H, W]
        return out


# ────────────────────────────────────────────────────────────────────────
# Warp Yardimcilari
# ────────────────────────────────────────────────────────────────────────

def flow_to_grid(flow: torch.Tensor) -> torch.Tensor:
    """
    Flow vektorunu F.grid_sample icin normalize koordinat griline donusturur.

    Parametreler
    ------------
    flow : [B, 2, H, W]  (vx, vy) piksel cinsinden

    Donus
    -----
    grid : [B, H, W, 2]  normalize edilmis [-1, 1]
    """
    B, _, H, W = flow.shape
    # Baz grid: her pikselin normalize koordinati
    ys = torch.linspace(-1, 1, H, device=flow.device)
    xs = torch.linspace(-1, 1, W, device=flow.device)
    base_y, base_x = torch.meshgrid(ys, xs, indexing='ij')  # [H, W]
    base_grid = torch.stack([base_x, base_y], dim=-1)        # [H, W, 2]
    base_grid = base_grid.unsqueeze(0).expand(B, -1, -1, -1) # [B, H, W, 2]

    # Flow piksel cinsinden → normalize
    # vx: W yonunde, vy: H yonunde
    flow_norm = torch.stack([
        flow[:, 0] / (W / 2),   # vx normalize
        flow[:, 1] / (H / 2),   # vy normalize
    ], dim=-1)  # [B, H, W, 2]

    return base_grid + flow_norm  # [B, H, W, 2]


def warp_st_nearest(tensor: torch.Tensor, flow: torch.Tensor,
                    padding_mode: str = 'border') -> torch.Tensor:
    """
    Straight-Through Nearest Warp.

    Forward:  nearest neighbor (kategorik kimlik korunur)
    Backward: bilinear (Straight-Through Estimator ile gradyan)

    Parametreler
    ------------
    tensor       : [B, C, H, W]  (Color veya ObjectMask)
    flow         : [B, 2, H, W]
    padding_mode : 'border' | 'zeros' | 'reflection'

    Donus
    -----
    warped : [B, C, H, W]
    """
    grid = flow_to_grid(flow)  # [B, H, W, 2]
    # Forward: nearest (kategorik, karistirma yok)
    out_hard = F.grid_sample(tensor, grid, mode='nearest',
                              padding_mode=padding_mode, align_corners=True)
    # Backward: bilinear (gradyan akisi icin)
    out_soft = F.grid_sample(tensor, grid, mode='bilinear',
                              padding_mode=padding_mode, align_corners=True)
    # Straight-Through: forward=hard, backward=soft gradient
    return out_hard.detach() + (out_soft - out_soft.detach())


def warp_bilinear(tensor: torch.Tensor, flow: torch.Tensor,
                  padding_mode: str = 'border') -> torch.Tensor:
    """
    Standart bilinear warp (Latent icin).

    Parametreler
    ------------
    tensor       : [B, C, H, W]  (Latent)
    flow         : [B, 2, H, W]
    padding_mode : 'border' | 'zeros' | 'reflection'

    Donus
    -----
    warped : [B, C, H, W]
    """
    grid = flow_to_grid(flow)
    return F.grid_sample(tensor, grid, mode='bilinear',
                          padding_mode=padding_mode, align_corners=True)


# ────────────────────────────────────────────────────────────────────────
# Ana NCA Adim Modulu
# ────────────────────────────────────────────────────────────────────────

class NCAStep(nn.Module):
    """
    NCA tek adim: tam 12-adim C1-onaylı mimari.

    Giris:
      Color_t      [B, 11, H, W]  renk one-hot (11 sinif)
      Latent_t     [B, L,  H, W]  latent
      ObjectMask_t [B, K,  H, W]  nesne maskeleri (soft, normalize)
      posenc       [B, P,  H, W]  STATIK konum
      geofeat      [B, G,  H, W]  STATIK geometri
      task_emb     [B, D_trans]    Transformer embedding
      active_mask  [B, 1,  H, W]  aktif bolge maskesi
      boundary_mode str            'border' | 'zeros' | 'reflection'

    Cikis:
      Color_{t+1}      [B, 11, H, W]
      Latent_{t+1}     [B, L,  H, W]
      ObjectMask_{t+1} [B, K,  H, W]
    """

    def __init__(self,
                 D_trans: int = 128,
                 D_obj: int   = 32,
                 K: int       = K_MAX,
                 L: int       = L_DIM,
                 P: int       = P_DIM,
                 G: int       = G_DIM,
                 C_ctx: int   = C_CTX,
                 C_flow: int  = C_FLOW,
                 v_max: float = V_MAX):
        super().__init__()
        self.L     = L
        self.K     = K
        self.v_max = v_max

        # Giris concat boyutu (LocalEncoder girisi)
        C_in_enc = N_COLOR + L + P + G  # = 11+8+16+4 = 39

        # h_base boyutu
        C_h = C_in_enc + C_ctx + 2 * L  # = 39 + 32 + 16 = 87

        # LogicGate giris boyutu
        C_in_lg = N_COLOR + L + K + 2 + P + G  # = 11+8+12+2+16+4 = 53
        C_out_lg = N_COLOR + L                  # = 11 + 8 = 19
        C_mid_lg = 64

        self.local_encoder    = LocalEncoder(C_in_enc, C_ctx)
        self.grad_features    = GradFeatures(L)
        self.per_object_film  = PerObjectFiLM(D_trans, D_obj, K, C_h)
        self.flow_net         = FlowNet(C_h, C_flow)
        self.boundary_ctrl    = BoundaryController(sharpness=5.0)
        self.logic_gate       = LogicGate(C_in_lg, C_mid_lg, C_out_lg)

    def forward(self,
                color_t: torch.Tensor,
                latent_t: torch.Tensor,
                obj_mask_t: torch.Tensor,
                posenc: torch.Tensor,
                geofeat: torch.Tensor,
                task_emb: torch.Tensor,
                active_mask: torch.Tensor,
                boundary_mode: str = 'border') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # ── 1. GIRIS CONCAT ──────────────────────────────────────────
        x = torch.cat([color_t, latent_t, posenc, geofeat], dim=1)
        # [B, N_COLOR+L+P+G, H, W] = [B, 39, H, W]

        # ── 2. LOCAL ENCODER ─────────────────────────────────────────
        local_ctx = self.local_encoder(x)   # [B, C_ctx, H, W]

        # ── 3. GRAD FEATURES ─────────────────────────────────────────
        grad_feat = self.grad_features(latent_t)  # [B, 2L, H, W]

        # ── 4. h_base CONCAT ─────────────────────────────────────────
        h_base = torch.cat([x, local_ctx, grad_feat], dim=1)
        # [B, 39+32+16, H, W] = [B, 87, H, W]

        # ── 5. PER-OBJECT FiLM ───────────────────────────────────────
        h_obj = self.per_object_film(task_emb, h_base, obj_mask_t)
        # [B, 87, H, W]

        # ── 6. FLOW NET ──────────────────────────────────────────────
        v = self.flow_net(h_obj)   # [B, 2, H, W]

        # ── 7. AKIS SINIRLA ──────────────────────────────────────────
        v = self.v_max * torch.tanh(v)   # [-v_max, +v_max]

        # ── 8. BOUNDARY CONTROLLER ───────────────────────────────────
        v, padding_mode = self.boundary_ctrl(v, active_mask, boundary_mode)

        # ── 9. COLOR WARP (ST-nearest) ───────────────────────────────
        color_prime = warp_st_nearest(color_t, v, padding_mode)   # [B, 11, H, W]

        # ── 10. LATENT WARP (bilinear) ────────────────────────────────
        latent_prime = warp_bilinear(latent_t, v, padding_mode)    # [B, L, H, W]

        # ── 11. OBJECTMASK WARP + NORMALIZE ──────────────────────────
        mask_prime = warp_st_nearest(obj_mask_t, v, padding_mode)  # [B, K, H, W]
        mask_prime = mask_prime / (mask_prime.sum(dim=1, keepdim=True) + 1e-6)

        # ── 12. LOGIC GATE ────────────────────────────────────────────
        lg_in = torch.cat([
            color_prime,   # [B, 11, H, W]
            latent_prime,  # [B, L,  H, W]
            mask_prime,    # [B, K,  H, W]
            v,             # [B, 2,  H, W]
            posenc,        # [B, P,  H, W]
            geofeat,       # [B, G,  H, W]
        ], dim=1)
        # [B, 11+8+12+2+16+4, H, W] = [B, 53, H, W]

        delta = self.logic_gate(lg_in)          # [B, 19, H, W]
        d_color  = delta[:, :N_COLOR]           # [B, 11, H, W]
        d_latent = delta[:, N_COLOR:]           # [B, L,  H, W]

        # Renk guncelleme: warp sonrasi delta ekle (logit uzayinda)
        color_next  = color_prime  + 0.1 * d_color    # yumusak guncelleme
        latent_next = latent_prime + 0.1 * d_latent

        return color_next, latent_next, mask_prime


# ────────────────────────────────────────────────────────────────────────
# Hizli Test
# ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import torch

    print("=== nca_step.py birim testi ===\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Cihaz: {device}")

    B, H, W = 2, 10, 12
    D_trans  = 128

    # Sahte girdiler
    color_t   = torch.randn(B, 11, H, W, device=device)
    latent_t  = torch.randn(B,  8, H, W, device=device)
    obj_mask_t= torch.softmax(torch.randn(B, 12, H, W, device=device), dim=1)
    task_emb  = torch.randn(B, D_trans, device=device)
    active_mask = torch.ones(B, 1, H, W, device=device)

    # STATIK encodinglar
    import sys
    sys.path.insert(0, '.')
    from encoding import make_posenc_batch, make_geofeat_batch
    posenc  = make_posenc_batch(H, W, B=B, device=device)
    geofeat = make_geofeat_batch(H, W, B=B, device=device)

    print(f"color_t    : {color_t.shape}")
    print(f"latent_t   : {latent_t.shape}")
    print(f"obj_mask_t : {obj_mask_t.shape}")
    print(f"posenc     : {posenc.shape}")
    print(f"geofeat    : {geofeat.shape}")
    print(f"task_emb   : {task_emb.shape}")

    # NCA Adimi
    nca = NCAStep(D_trans=D_trans).to(device)
    param_count = sum(p.numel() for p in nca.parameters() if p.requires_grad)
    print(f"\nNCA parametre sayisi: {param_count:,}")

    color_next, latent_next, mask_next = nca(
        color_t, latent_t, obj_mask_t,
        posenc, geofeat, task_emb, active_mask,
        boundary_mode='border'
    )

    print(f"\nCikis sekillleri:")
    print(f"  color_next  : {color_next.shape}")   # [2, 11, 10, 12]
    print(f"  latent_next : {latent_next.shape}")   # [2, 8, 10, 12]
    print(f"  mask_next   : {mask_next.shape}")     # [2, 12, 10, 12]

    assert color_next.shape  == (B, 11, H, W), "HATA: color_next sekli yanlis"
    assert latent_next.shape == (B,  8, H, W), "HATA: latent_next sekli yanlis"
    assert mask_next.shape   == (B, 12, H, W), "HATA: mask_next sekli yanlis"
    print("\n[OK] Sekil kontrolleri gecti")

    # Gradient testi
    loss = color_next.mean() + latent_next.mean()
    loss.backward()
    print("[OK] Backward gecti (gradient akiyor)")

    # Maske normalize kontrol
    mask_sum = mask_next.sum(dim=1)   # [B, H, W]
    assert (mask_sum - 1.0).abs().max() < 0.1, "HATA: Maske normalizasyon hatalı"
    print("[OK] ObjectMask normalizasyon dogru")

    print("\n[OK] Tum testler gecti.")
