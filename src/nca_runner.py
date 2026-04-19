"""
nca_runner.py
-------------
NCA N-adim dongusu ve equilibrium tespiti.

Ana fonksiyon:
  run_nca(nca_step, state, N_steps, ...) -> final_state

Ozellikler:
  - Sabit adim sayisi veya equilibrium ile dur
  - K_update: Transformer her K adimda bir yenilenir (V1: K_update=0)
  - Gradient checkpointing: bellek tasarruflu backward icin
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple, Callable

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from nca_step import NCAStep
from encoding import StaticEncodings
from color_codec import get_active_mask, CANVAS_H, CANVAS_W


# ────────────────────────────────────────────────────────────────────────
# NCA Durum Veri Yapisi
# ────────────────────────────────────────────────────────────────────────

class NCAState:
    """NCA durumunu bir arada tutar."""
    __slots__ = ('color', 'latent', 'obj_mask')

    def __init__(self, color: torch.Tensor,
                 latent: torch.Tensor,
                 obj_mask: torch.Tensor):
        """
        Parametreler
        ------------
        color    : [B, 11, H, W]
        latent   : [B, L,  H, W]
        obj_mask : [B, K,  H, W]
        """
        self.color    = color
        self.latent   = latent
        self.obj_mask = obj_mask

    @property
    def B(self): return self.color.shape[0]

    @property
    def H(self): return self.color.shape[2]

    @property
    def W(self): return self.color.shape[3]


# ────────────────────────────────────────────────────────────────────────
# Equilibrium Tespiti
# ────────────────────────────────────────────────────────────────────────

def equilibrium_reached(state: NCAState,
                         prev_state: NCAState,
                         color_thr: float = 0.01,
                         latent_thr: float = 0.005) -> bool:
    """
    NCA durumunun degisip degismedigini kontrol eder.

    Renk dagilimi ve latent uzayi icin ayri esik degerleri.

    Parametreler
    ------------
    state, prev_state : NCAState
    color_thr         : Renk degisim esigi (argmax bazli)
    latent_thr        : Latent degisim esigi (L2 bazli)

    Donus
    -----
    bool : True = equilibrium ulasildi
    """
    with torch.no_grad():
        # Renk degisimi: argmax farklilasan piksel orani
        color_now  = state.color.argmax(dim=1)       # [B, H, W]
        color_prev = prev_state.color.argmax(dim=1)  # [B, H, W]
        color_change = (color_now != color_prev).float().mean().item()

        # Latent degisimi: ortalama L2
        latent_diff = (state.latent - prev_state.latent).norm(dim=1).mean().item()

    return color_change < color_thr and latent_diff < latent_thr


# ────────────────────────────────────────────────────────────────────────
# Ana Runner
# ────────────────────────────────────────────────────────────────────────

def run_nca(nca: NCAStep,
            init_state: NCAState,
            posenc: torch.Tensor,
            geofeat: torch.Tensor,
            task_emb: torch.Tensor,
            H_out: torch.Tensor,
            W_out: torch.Tensor,
            N_steps: int = 10,
            K_update: int = 0,
            boundary_mode: str = 'border',
            use_checkpoint: bool = False,
            check_equilibrium: bool = True,
            equil_start_step: int = 5,
            transformer_update_fn: Optional[Callable] = None) -> NCAState:
    """
    NCA N adim calistirir.

    Parametreler
    ------------
    nca           : NCAStep modulu
    init_state    : NCAState (Color_0, Latent_0, ObjectMask_0)
    posenc        : [B, P, H, W]  statik
    geofeat       : [B, G, H, W]  statik
    task_emb      : [B, D_trans]  Transformer embedding
    H_out, W_out  : [B] int tensoru — hedef canvas boyutu
    N_steps       : Maksimum adim sayisi
    K_update      : Transformer guncelleme periyodu (0=sadece baslangic)
    boundary_mode : 'border' | 'zeros' | 'reflection'
    use_checkpoint: True = gradient checkpointing (bellek tasarrufu)
    check_equilibrium: True = equilibrium ile erken dur
    equil_start_step: Bu adimdan sonra equilibrium kontrolu baslar
    transformer_update_fn: K_update>0 ise Transformer'dan yeni task_emb alinir

    Donus
    -----
    NCAState : Final durum (Color_N, Latent_N, ObjectMask_N)
    """
    B = init_state.B
    H, W = init_state.H, init_state.W

    # Aktif bolge maskesi: [B, 1, H, W]
    # Her ornek icin farkli H_out, W_out olabilir — en buyugunu al
    # (Batch icinde farkli boyutlar icin en buyuk olani kullan)
    H_max = H_out.max().item()
    W_max = W_out.max().item()
    active_mask = get_active_mask(int(H_max), int(W_max),
                                  B=B, device=posenc.device)

    # Durum kopyalari
    color    = init_state.color.clone()
    latent   = init_state.latent.clone()
    obj_mask = init_state.obj_mask.clone()

    prev_color  = color.clone()
    prev_latent = latent.clone()

    for step in range(N_steps):
        # Transformer guncelleme (K_update > 0 ise)
        if K_update > 0 and step > 0 and step % K_update == 0:
            if transformer_update_fn is not None:
                state_for_transformer = NCAState(color, latent, obj_mask)
                task_emb = transformer_update_fn(state_for_transformer)

        # NCA tek adim
        if use_checkpoint and color.requires_grad:
            # Gradient checkpointing: forward yeniden hesaplanir, aktivasyon saklanmaz
            color, latent, obj_mask = checkpoint(
                nca, color, latent, obj_mask,
                posenc, geofeat, task_emb, active_mask,
                use_reentrant=False
            )
        else:
            color, latent, obj_mask = nca(
                color, latent, obj_mask,
                posenc, geofeat, task_emb, active_mask,
                boundary_mode=boundary_mode
            )

        # Equilibrium kontrolu
        if check_equilibrium and step >= equil_start_step:
            cur_state  = NCAState(color, latent, obj_mask)
            prev_state = NCAState(prev_color, prev_latent, obj_mask)
            if equilibrium_reached(cur_state, prev_state):
                break

        prev_color  = color.detach()
        prev_latent = latent.detach()

    return NCAState(color, latent, obj_mask)


# ────────────────────────────────────────────────────────────────────────
# NCARunner Sinif (model icin kullanisli sarici)
# ────────────────────────────────────────────────────────────────────────

class NCARunner(nn.Module):
    """
    NCAStep + calisma dongusu + statik encodinglar.

    model.py'de kullanilacak: Transformer → SeedMLP → NCARunner
    """

    def __init__(self,
                 D_trans: int     = 128,
                 D_obj: int       = 32,
                 K: int           = 12,
                 L: int           = 8,
                 N_steps: int     = 10,
                 K_update: int    = 0,
                 use_checkpoint: bool = False,
                 canvas_H: int    = CANVAS_H,
                 canvas_W: int    = CANVAS_W):
        super().__init__()
        self.N_steps        = N_steps
        self.K_update       = K_update
        self.use_checkpoint = use_checkpoint

        self.nca_step = NCAStep(D_trans=D_trans, D_obj=D_obj, K=K, L=L)
        self.static_enc = StaticEncodings(H=canvas_H, W=canvas_W)

    def forward(self,
                init_state: NCAState,
                task_emb: torch.Tensor,
                H_out: torch.Tensor,
                W_out: torch.Tensor,
                boundary_mode: str = 'border',
                transformer_update_fn: Optional[Callable] = None) -> NCAState:
        """
        Parametreler
        ------------
        init_state : NCAState (SeedMLP'den gelen baslangic durumu)
        task_emb   : [B, D_trans]
        H_out, W_out: [B] int — hedef boyutlar
        boundary_mode: sinir davranisi
        transformer_update_fn: K_update>0 ise kullanilir

        Donus
        -----
        NCAState : final NCA durumu
        """
        B = init_state.B
        posenc, geofeat = self.static_enc.get(B)
        # Cihazi state ile esitle
        posenc  = posenc.to(init_state.color.device)
        geofeat = geofeat.to(init_state.color.device)

        return run_nca(
            nca                  = self.nca_step,
            init_state           = init_state,
            posenc               = posenc,
            geofeat              = geofeat,
            task_emb             = task_emb,
            H_out                = H_out,
            W_out                = W_out,
            N_steps              = self.N_steps,
            K_update             = self.K_update,
            boundary_mode        = boundary_mode,
            use_checkpoint       = self.use_checkpoint,
            transformer_update_fn= transformer_update_fn,
        )


# ────────────────────────────────────────────────────────────────────────
# Hizli Test
# ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import torch

    print("=== nca_runner.py birim testi ===\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Cihaz: {device}")

    B, H, W = 2, 30, 30
    D_trans  = 128
    K_slots  = 12
    L_dim    = 8
    N_steps  = 5

    # Sahte baslangic durumu
    color_0   = torch.softmax(torch.randn(B, 11, H, W, device=device), dim=1)
    latent_0  = torch.randn(B, L_dim, H, W, device=device) * 0.1
    obj_mask_0= torch.softmax(torch.randn(B, K_slots, H, W, device=device), dim=1)
    task_emb  = torch.randn(B, D_trans, device=device)
    H_out = torch.tensor([8, 6], device=device)
    W_out = torch.tensor([8, 6], device=device)

    init_state = NCAState(color_0, latent_0, obj_mask_0)

    # NCARunner
    runner = NCARunner(
        D_trans=D_trans,
        K=K_slots,
        L=L_dim,
        N_steps=N_steps,
        use_checkpoint=False
    ).to(device)

    param_count = sum(p.numel() for p in runner.parameters() if p.requires_grad)
    print(f"NCARunner parametre sayisi: {param_count:,}")

    # Forward
    final_state = runner(init_state, task_emb, H_out, W_out)

    print(f"\nFinal Color  : {final_state.color.shape}")    # [2, 11, 30, 30]
    print(f"Final Latent : {final_state.latent.shape}")    # [2, 8, 30, 30]
    print(f"Final Mask   : {final_state.obj_mask.shape}")  # [2, 12, 30, 30]

    assert final_state.color.shape  == (B, 11, H, W)
    assert final_state.latent.shape == (B, L_dim, H, W)
    assert final_state.obj_mask.shape == (B, K_slots, H, W)
    print("\n[OK] Sekil kontrolleri gecti")

    # Gradient
    loss = final_state.color.mean() + final_state.latent.mean()
    loss.backward()
    print("[OK] Backward gecti")

    # Gradient checkpointing testi
    color_0   = torch.softmax(torch.randn(B, 11, H, W, device=device), dim=1)
    latent_0  = torch.randn(B, L_dim, H, W, device=device) * 0.1
    obj_mask_0= torch.softmax(torch.randn(B, K_slots, H, W, device=device), dim=1)
    color_0.requires_grad_(True)
    latent_0.requires_grad_(True)

    runner_ckpt = NCARunner(
        D_trans=D_trans, K=K_slots, L=L_dim,
        N_steps=3, use_checkpoint=True
    ).to(device)

    init_ckpt = NCAState(color_0, latent_0, obj_mask_0)
    final_ckpt = runner_ckpt(init_ckpt, task_emb, H_out, W_out)
    loss_ckpt = final_ckpt.color.mean()
    loss_ckpt.backward()
    print("[OK] Gradient checkpointing testi gecti")

    print("\n[OK] Tum testler gecti.")
