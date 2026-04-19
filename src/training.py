"""
training.py
-----------
Egitim dongusu ve D1-onaylı loss fonksiyonlari.

Loss stack:
  L_total = 1.0 * L_recon
          + 0.5 * L_size
          + 0.2 * L_null_outside
          + 0.3 * L_mask
          + 0.3 * L_object_consistency

  L_recon: Piksel CE (NULL ignore). Eger NULL/kolay pikseller baskınsa focal.
  L_size : H_out, W_out classification (30-sinif)
  L_null_outside: hedef disi bolgeyi NULL tut
  L_mask : cover + overlap + compact + boundary + temporal + slot_sparse
  L_object_consistency: slot icindeki renk/dagilim tutarliligi

  L_conf (-entropy): VARSAYILAN DEGIL — sadece gec egitim fazinda, dusuk agirlik.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, Optional, List
import math

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from model import GeometrikAkil, ModelConfig
from data_loader import make_dataloader
from color_codec import NULL_IDX, CANVAS_H, CANVAS_W


# ────────────────────────────────────────────────────────────────────────
# Focal Loss
# ────────────────────────────────────────────────────────────────────────

def focal_cross_entropy(logits: torch.Tensor, targets: torch.Tensor,
                        gamma: float = 2.0,
                        ignore_index: int = -100,
                        reduction: str = 'mean') -> torch.Tensor:
    """
    Focal Cross-Entropy kaybi.

    Amac: Kolay ornekleri (yuksek p_t) asagi agirla.
    Uygun gerekce: NULL/background baskınsa veya kolay pikseller
                   loss'u domine ediyorsa kullan.
    YANLIS gerekce: "CE gradient vanishing" — p_t kucuk oldugunda CE zaten guclu.

    Parametreler
    ------------
    logits       : [B, C, H, W] veya [B, C]
    targets      : [B, H, W] veya [B] int64
    gamma        : Focal parametresi (0=standart CE, 2=tipik focal)
    ignore_index : Bu sinif kayıptan cıkar (NULL=10)
    reduction    : 'mean' | 'sum' | 'none'
    """
    # Standart CE kaybi
    ce = F.cross_entropy(logits, targets, ignore_index=ignore_index, reduction='none')
    # p_t = exp(-ce)  (dogru sinif olasiligi)
    p_t = torch.exp(-ce)
    # Focal agirlik: (1 - p_t)^gamma
    focal_w = (1 - p_t) ** gamma

    # ignore_index maskesi
    mask = (targets != ignore_index)
    loss = focal_w * ce * mask.float()

    if reduction == 'mean':
        n = mask.sum().clamp(min=1)
        return loss.sum() / n
    elif reduction == 'sum':
        return loss.sum()
    return loss


# ────────────────────────────────────────────────────────────────────────
# L_recon: Ana Yeniden Olusturma Kaybi
# ────────────────────────────────────────────────────────────────────────

def recon_loss(color_final: torch.Tensor,
               target_grids: list,
               H_outs: torch.Tensor,
               W_outs: torch.Tensor,
               use_focal: bool = False,
               gamma: float = 2.0) -> torch.Tensor:
    """
    Hedef bolgedeki [0:H_out, 0:W_out] pikseller icin CE/Focal loss.

    Parametreler
    ------------
    color_final : [B, 11, 30, 30]  NCA son renk ciktisi
    target_grids: list[torch.Tensor]  her ornek icin [H_out, W_out] int64
    H_outs, W_outs: [B] int
    use_focal   : True = Focal CE, False = standart CE
    gamma       : Focal gamma

    Donus
    -----
    loss : scalar
    """
    B = color_final.shape[0]
    device = color_final.device

    # Her ornek icin hedef bolgeyi cekip loss hesapla
    losses = []
    for i in range(B):
        H = H_outs[i].item()
        W = W_outs[i].item()
        if H == 0 or W == 0:
            continue

        logits_i = color_final[i:i+1, :, :H, :W]  # [1, 11, H, W]
        target_i = target_grids[i].to(device)        # [H, W] int64

        if target_i.shape != torch.Size([H, W]):
            # Boyut uyumsuzlugu: kes veya pad
            th, tw = target_i.shape
            H_use = min(H, th)
            W_use = min(W, tw)
            logits_i = logits_i[:, :, :H_use, :W_use]
            target_i = target_i[:H_use, :W_use]

        target_i = target_i.unsqueeze(0)  # [1, H, W]

        if use_focal:
            l = focal_cross_entropy(logits_i, target_i, gamma=gamma,
                                    ignore_index=NULL_IDX)
        else:
            l = F.cross_entropy(logits_i, target_i, ignore_index=NULL_IDX)
        losses.append(l)

    if not losses:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return torch.stack(losses).mean()


# ────────────────────────────────────────────────────────────────────────
# L_size: Boyut Siniflandirma Kaybi
# ────────────────────────────────────────────────────────────────────────

def size_loss(h_logits: torch.Tensor, w_logits: torch.Tensor,
              H_true: torch.Tensor, W_true: torch.Tensor) -> torch.Tensor:
    """
    H_out ve W_out tahminleri icin CE kaybi.

    Parametreler
    ------------
    h_logits, w_logits : [B, 30]  Transformer boyut logitleri
    H_true, W_true     : [B] int (1-30 arasi)

    Donus
    -----
    loss : scalar
    """
    # 0-indexed: boyut-1 hedef olur (1→0, 30→29)
    h_target = (H_true - 1).clamp(0, 29).long()
    w_target = (W_true - 1).clamp(0, 29).long()
    l_h = F.cross_entropy(h_logits, h_target)
    l_w = F.cross_entropy(w_logits, w_target)
    return (l_h + l_w) / 2


# ────────────────────────────────────────────────────────────────────────
# L_null_outside: Hedef Disi Bolgeyi NULL Tut
# ────────────────────────────────────────────────────────────────────────

def null_outside_loss(color_final: torch.Tensor,
                       H_outs: torch.Tensor,
                       W_outs: torch.Tensor) -> torch.Tensor:
    """
    [0:H_out, 0:W_out] disi pikseller NULL (idx=10) olmali.

    Parametreler
    ------------
    color_final : [B, 11, 30, 30]
    H_outs, W_outs : [B] int

    Donus
    -----
    loss : scalar
    """
    B, C, H, W = color_final.shape
    device = color_final.device
    losses = []

    for i in range(B):
        Ho = H_outs[i].item()
        Wo = W_outs[i].item()

        # Disi bolgeyi topla
        outside_pixels = []
        if Ho < H:
            # Alt satırlar tamamen dis
            outside_pixels.append(color_final[i, :, Ho:, :])   # [11, H-Ho, W]
        if Wo < W:
            # Sag sutunlar tamamen dis (Ho satirina kadar)
            outside_pixels.append(color_final[i, :, :Ho, Wo:]) # [11, Ho, W-Wo]

        if not outside_pixels:
            continue

        # Her dis piksel icin NULL class'ini yuksek yapmayi bekliyoruz
        # CE: NULL class = hedef
        for op in outside_pixels:
            # op: [11, ...]
            op_flat = op.permute(1, 0) if op.dim() == 2 else op.permute(1, 2, 0)
            # [N_pixels, 11]
            if op.dim() == 3:
                _, hh, ww = op.shape
                logits = op.permute(1, 2, 0).reshape(hh * ww, 11)  # [N, 11]
                targets = torch.full((hh * ww,), NULL_IDX, dtype=torch.long, device=device)
            else:  # dim=2
                _, n = op.shape
                logits = op.permute(1, 0)  # [n, 11]
                targets = torch.full((n,), NULL_IDX, dtype=torch.long, device=device)
            l = F.cross_entropy(logits, targets)
            losses.append(l)

    if not losses:
        return torch.tensor(0.0, device=device)
    return torch.stack(losses).mean()


# ────────────────────────────────────────────────────────────────────────
# L_mask: Nesne Segmentasyon Yardimci Kayiplari (C1)
# ────────────────────────────────────────────────────────────────────────

def mask_loss(obj_mask: torch.Tensor,
               color_final: torch.Tensor,
               obj_mask_prev: Optional[torch.Tensor] = None,
               lambda_cover:    float = 0.5,
               lambda_overlap:  float = 0.5,
               lambda_compact:  float = 0.2,
               lambda_boundary: float = 0.3,
               lambda_temporal: float = 0.3,
               lambda_sparse:   float = 0.1) -> Dict[str, torch.Tensor]:
    """
    Nesne maske regularizasyon kayip paketi.

    Parametreler
    ------------
    obj_mask    : [B, K, H, W]  NCA son nesne maskeleri
    color_final : [B, 11, H, W] NCA son renk
    obj_mask_prev: [B, K, H, W] onceki zaman adimi maskesi (None ise L_temporal=0)

    Donus
    -----
    {
      'L_cover':    scalar  — her piksel bir slota ait olmali
      'L_overlap':  scalar  — slotlar ust uste gelmemeli
      'L_compact':  scalar  — maskeler uzamsal tutarlilik gostermeli
      'L_boundary': scalar  — maske kenarlari renk kenarlariyla ortusur
      'L_temporal': scalar  — kimlik warp sonrasi korunur
      'L_sparse':   scalar  — kullanilmayan slotlar sifira yaklasir
      'total':      scalar  — agirlikli toplam
    }
    """
    B, K, H, W = obj_mask.shape
    device = obj_mask.device

    # L_cover: Her pikselde slot toplamı = 1 (zaten normalize, ancak bozulabilir)
    # Ek softmax yoksa sadece sapma kontrolu
    mask_sum = obj_mask.sum(dim=1)  # [B, H, W]
    L_cover = (mask_sum - 1.0).abs().mean()

    # L_overlap: Slotlar ust uste gelmemeli → min(slot_i, slot_j) toplami kucuk olmali
    # Yaklasim: argmax ile winner-take-all, sonra fark
    hard_mask = F.one_hot(obj_mask.argmax(dim=1), num_classes=K).float()
    hard_mask = hard_mask.permute(0, 3, 1, 2)   # [B, K, H, W]
    overlap   = (obj_mask * (1 - hard_mask)).mean()
    L_overlap = overlap

    # L_compact: Her slot maskesi uzamsal olarak kompakt olmali
    # Yaklasim: maske icindeki pikseller birbirine yakin olmali
    # Basit: maskenin x ve y varyansini hesapla, kucuk olsun
    ys = torch.linspace(0, 1, H, device=device)
    xs = torch.linspace(0, 1, W, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')   # [H, W]
    grid_y = grid_y.unsqueeze(0).unsqueeze(0)   # [1, 1, H, W]
    grid_x = grid_x.unsqueeze(0).unsqueeze(0)   # [1, 1, H, W]

    # Slot kitle merkezi
    mask_norm = obj_mask / (obj_mask.sum(dim=(-2,-1), keepdim=True) + 1e-6)
    cy = (mask_norm * grid_y).sum(dim=(-2,-1))  # [B, K]
    cx = (mask_norm * grid_x).sum(dim=(-2,-1))  # [B, K]
    # Varyans: E[(x-cx)^2]
    var_y = (mask_norm * (grid_y - cy.unsqueeze(-1).unsqueeze(-1))**2).sum(dim=(-2,-1))
    var_x = (mask_norm * (grid_x - cx.unsqueeze(-1).unsqueeze(-1))**2).sum(dim=(-2,-1))
    L_compact = (var_y + var_x).mean()

    # L_boundary: Maske kenarlari renk kenarlariyla ortusur
    # Sobel ile renk gradyani hesapla
    color_gray = color_final[:, :10].argmax(dim=1).float().unsqueeze(1) / 9.0
    # [B, 1, H, W]
    sobel_k = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]],
                            dtype=torch.float32, device=device).view(1,1,3,3)
    color_edge_x = F.conv2d(color_gray, sobel_k, padding=1)
    color_edge_y = F.conv2d(color_gray, sobel_k.permute(0,1,3,2), padding=1)
    color_edge   = (color_edge_x**2 + color_edge_y**2).sqrt()  # [B, 1, H, W]

    # Maske kenarlari: obj_mask'in her slotunun gradyani
    mask_edge_x = F.conv2d(obj_mask.view(B*K,1,H,W), sobel_k, padding=1).view(B,K,H,W)
    mask_edge_y = F.conv2d(obj_mask.view(B*K,1,H,W), sobel_k.permute(0,1,3,2), padding=1).view(B,K,H,W)
    mask_edge   = (mask_edge_x**2 + mask_edge_y**2).sqrt().max(dim=1)[0].unsqueeze(1)  # [B,1,H,W]

    # Maske kenarlarinin renk kenarlarıyla ortusmesi: ters yonde ceza ver
    # Yuksek maske kenari + dusuk renk kenari = ceza
    boundary_mismatch = (mask_edge * (1 - color_edge.sigmoid())).mean()
    L_boundary = boundary_mismatch

    # L_temporal: Kimlik warp sonrasi korunur
    if obj_mask_prev is not None:
        L_temporal = F.mse_loss(obj_mask, obj_mask_prev.detach())
    else:
        L_temporal = torch.tensor(0.0, device=device)

    # L_sparse: Kullanilmayan slotlar sifira yaklasir
    # Slot basvuru olasiligi: maskenin ortalamasi
    slot_usage = obj_mask.mean(dim=(-2,-1))  # [B, K]
    # Dusuk kullanimli slotlar (< ortalama/2) icin ceza yok, yuksek olanlar kalsin
    # Basit: entropy tesvik et (slotlarin birkacinda yogunlas)
    # slot entropisini minimize etmek = bir-kac slot kullan = sparse
    slot_entropy = -(slot_usage * (slot_usage + 1e-6).log()).mean()
    L_sparse = slot_entropy   # Kucuk entropy = sparse

    # Agirlikli toplam
    total = (lambda_cover    * L_cover +
             lambda_overlap  * L_overlap +
             lambda_compact  * L_compact +
             lambda_boundary * L_boundary +
             lambda_temporal * L_temporal +
             lambda_sparse   * L_sparse)

    return {
        'L_cover':    L_cover,
        'L_overlap':  L_overlap,
        'L_compact':  L_compact,
        'L_boundary': L_boundary,
        'L_temporal': L_temporal,
        'L_sparse':   L_sparse,
        'total':      total,
    }


# ────────────────────────────────────────────────────────────────────────
# L_object_consistency: Nesne Seviyesi Tutarlilik (D1)
# ────────────────────────────────────────────────────────────────────────

def object_consistency_loss(obj_mask: torch.Tensor,
                             color_final: torch.Tensor) -> torch.Tensor:
    """
    Ayni nesne slotu icindeki pikseller tutarli renk gostermeli.

    ARC'nin kritik hata turu: "bir nesnenin tamamini yanlis konuma koymak"
    CE bunu zayif yakalar; bu loss nesne seviyesinde tutarliligi tesvik eder.

    Yaklasim:
    - Her slot icin renk dagilimini hesapla (slot-level color histogram)
    - Her slottaki pikseller o slotin ortalama renk dagilimına yakin olmali
    - Varyasyon kucukse tutarli → ceza az

    Parametreler
    ------------
    obj_mask    : [B, K, H, W]  soft maske
    color_final : [B, 11, H, W] renk logitleri/softmax

    Donus
    -----
    loss : scalar
    """
    B, K, H, W = obj_mask.shape
    _, C, _, _  = color_final.shape

    color_prob = F.softmax(color_final, dim=1)  # [B, 11, H, W]

    # Her slot icin agirlikli ortalama renk dagilimi
    # mask: [B, K, H, W], color_prob: [B, 11, H, W]
    # → [B, K, 11]
    mask_norm = obj_mask / (obj_mask.sum(dim=(-2,-1), keepdim=True) + 1e-6)
    slot_color = torch.einsum('bkhw, bchw -> bkc', mask_norm, color_prob)
    # slot_color: [B, K, 11]

    # Her pikselin rengi, bulundugu slotun ortalama rengiyle ne kadar tutarli?
    # slot_color_map: [B, 11, H, W] — her piksel icin slot-agirlikli hedef renk
    slot_color_map = torch.einsum('bkc, bkhw -> bchw', slot_color, obj_mask)
    # slot_color_map: [B, 11, H, W]

    # Tutarsizlik: pikselin gercek rengi ile slot hedefi arasindaki KL divergence
    # KL(color_prob || slot_color_map)
    # = sum_c color_prob_c * log(color_prob_c / slot_color_map_c)
    log_pred = (color_prob + 1e-6).log()
    log_tgt  = (slot_color_map.detach() + 1e-6).log()
    kl       = (color_prob * (log_pred - log_tgt)).sum(dim=1)  # [B, H, W]

    return kl.mean()


# ────────────────────────────────────────────────────────────────────────
# Tam Loss Hesaplamasi
# ────────────────────────────────────────────────────────────────────────

def compute_loss(model_output: dict,
                 batch: dict,
                 use_focal: bool = False,
                 lambda_size:   float = 0.5,
                 lambda_out:    float = 0.2,
                 lambda_mask:   float = 0.3,
                 lambda_obj:    float = 0.3,
                 lambda_conf:   float = 0.0,    # VARSAYILAN: KAPALI
                 step: int = 0,
                 conf_warmup_steps: int = 5000) -> Dict[str, torch.Tensor]:
    """
    Tam D1-onaylı loss stack.

    lambda_conf=0: L_conf (confidence entropy) varsayilan olarak kapali.
    Gec fazda acmak icin: conf_warmup_steps'ten sonra kucuk agirlikla.

    Donus
    -----
    {
      'total':     scalar
      'L_recon':   scalar
      'L_size':    scalar
      'L_out':     scalar
      'L_mask':    scalar (agirlikli toplam)
      'L_obj':     scalar
      'L_conf':    scalar (eger aciksa)
      'accuracy':  float  (piksel dogruluk, bilgi icin)
    }
    """
    device = model_output['color_final'].device

    # --- L_recon ---
    L_recon = recon_loss(
        model_output['color_final'],
        batch['target_output'],
        model_output['H_out_true'],
        model_output['W_out_true'],
        use_focal = use_focal,
    )

    # --- L_size ---
    L_size = size_loss(
        model_output['h_logits'],
        model_output['w_logits'],
        model_output['H_out_true'],
        model_output['W_out_true'],
    )

    # --- L_null_outside ---
    L_out = null_outside_loss(
        model_output['color_final'],
        model_output['H_out_true'],
        model_output['W_out_true'],
    )

    # --- L_mask ---
    mask_losses = mask_loss(
        model_output['obj_mask_final'],
        model_output['color_final'],
    )
    L_mask_total = mask_losses['total']

    # --- L_object_consistency ---
    L_obj = object_consistency_loss(
        model_output['obj_mask_final'],
        model_output['color_final'],
    )

    # --- L_conf (opsiyonel, gec faz) ---
    L_conf = torch.tensor(0.0, device=device)
    eff_lambda_conf = 0.0
    if lambda_conf > 0 and step >= conf_warmup_steps:
        H_out = model_output['H_out_true']
        W_out = model_output['W_out_true']
        B = model_output['color_final'].shape[0]
        conf_losses = []
        for i in range(B):
            H = H_out[i].item()
            W = W_out[i].item()
            if H > 0 and W > 0:
                logits_i = model_output['color_final'][i, :, :H, :W]
                prob_i   = F.softmax(logits_i, dim=0)
                entropy  = -(prob_i * (prob_i + 1e-6).log()).sum(dim=0).mean()
                conf_losses.append(-entropy)   # negatif entropi = guvven tesvik
        if conf_losses:
            L_conf = torch.stack(conf_losses).mean()
            eff_lambda_conf = lambda_conf

    # --- Toplam ---
    L_total = (1.0 * L_recon +
               lambda_size * L_size +
               lambda_out  * L_out +
               lambda_mask * L_mask_total +
               lambda_obj  * L_obj +
               eff_lambda_conf * L_conf)

    # --- Piksel dogruluk (bilgi icin, loss degil) ---
    with torch.no_grad():
        pred_grid  = model_output['color_final'].argmax(dim=1)  # [B, 30, 30]
        correct = 0
        total   = 0
        for i in range(pred_grid.shape[0]):
            H = model_output['H_out_true'][i].item()
            W = model_output['W_out_true'][i].item()
            if H > 0 and W > 0 and i < len(batch['target_output']):
                tgt = batch['target_output'][i].to(device)
                th, tw = tgt.shape
                H_u = min(H, th)
                W_u = min(W, tw)
                pred_i = pred_grid[i, :H_u, :W_u]
                tgt_i  = tgt[:H_u, :W_u]
                correct += (pred_i == tgt_i).sum().item()
                total   += H_u * W_u
        accuracy = correct / max(total, 1)

    return {
        'total':   L_total,
        'L_recon': L_recon.detach(),
        'L_size':  L_size.detach(),
        'L_out':   L_out.detach(),
        'L_mask':  L_mask_total.detach(),
        'L_obj':   L_obj.detach(),
        'L_conf':  L_conf.detach(),
        'accuracy': accuracy,
    }


# ────────────────────────────────────────────────────────────────────────
# Egitim Dongusu
# ────────────────────────────────────────────────────────────────────────

def train_epoch(model: GeometrikAkil,
                loader,
                optimizer: torch.optim.Optimizer,
                device: torch.device,
                step_counter: list,
                use_focal: bool = False,
                grad_clip: float = 1.0,
                log_interval: int = 50) -> Dict[str, float]:
    """
    Tek epoch egitimi.

    Donus: ortalama kayip degerleri
    """
    model.train()
    totals = {}
    n_batches = 0

    for batch_idx, batch in enumerate(loader):
        # Batch'i device'a tasi (gerekirse)
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        optimizer.zero_grad()

        # Forward
        model_out = model.forward_train(batch)

        # Loss
        losses = compute_loss(model_out, batch,
                               use_focal=use_focal,
                               step=step_counter[0])
        loss = losses['total']

        # Backward
        loss.backward()

        # Gradient clipping
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        step_counter[0] += 1

        # Kayiplari topla
        for k, v in losses.items():
            if k == 'accuracy':
                totals[k] = totals.get(k, 0.0) + v
            elif isinstance(v, torch.Tensor):
                totals[k] = totals.get(k, 0.0) + v.item()
            else:
                totals[k] = totals.get(k, 0.0) + v
        n_batches += 1

        if log_interval > 0 and batch_idx % log_interval == 0:
            print(f"  step={step_counter[0]:5d} | "
                  f"loss={losses['total'].item():.4f} | "
                  f"recon={losses['L_recon'].item():.4f} | "
                  f"acc={losses['accuracy']:.3f}")

    # Ortalamalar
    return {k: v / max(n_batches, 1) for k, v in totals.items()}


def evaluate(model: GeometrikAkil,
             loader,
             device: torch.device) -> Dict[str, float]:
    """
    Degerlendirme dongusu (grad hesaplama yok).
    """
    model.eval()
    totals = {}
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            model_out = model.forward_train(batch)
            losses    = compute_loss(model_out, batch)
            for k, v in losses.items():
                if isinstance(v, torch.Tensor):
                    totals[k] = totals.get(k, 0.0) + v.item()
                else:
                    totals[k] = totals.get(k, 0.0) + v
            n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


def exact_match_accuracy(model: GeometrikAkil,
                          tasks: dict,
                          device: torch.device,
                          max_tasks: int = 50) -> float:
    """
    ARC tam eslesme dogrulugu.
    Egitim orneklerinden biri test olarak kullanilir.

    Donus: exact-match orani [0, 1]
    """
    from color_codec import null_canvas, place_grid
    model.eval()
    correct = 0
    total   = 0

    task_list = list(tasks.values())[:max_tasks]
    with torch.no_grad():
        for task in task_list:
            if task.n_train < 2:
                continue

            # Son train ornegini test olarak kullan
            test_ex = task.train_examples[-1]
            train_exs = task.train_examples[:-1]
            max_ex = len(train_exs)

            # Transformer girisi hazirla
            H, W = task.train_examples[0].H_in, task.train_examples[0].W_in
            train_in_list, train_out_list = [], []
            for ex in train_exs:
                inp_c = place_grid(null_canvas(B=1, device=device),
                                   ex.input_grid, ex.H_in, ex.W_in)
                out_c = place_grid(null_canvas(B=1, device=device),
                                   ex.output_grid, ex.H_out, ex.W_out)
                train_in_list.append(inp_c[0])
                train_out_list.append(out_c[0])

            train_in  = torch.stack(train_in_list).unsqueeze(0)   # [1, max_ex, 11, 30, 30]
            train_out = torch.stack(train_out_list).unsqueeze(0)
            train_mask= torch.ones(1, max_ex, dtype=torch.bool, device=device)

            # Test girisi
            test_in_c = place_grid(null_canvas(B=1, device=device),
                                   test_ex.input_grid, test_ex.H_in, test_ex.W_in)

            preds = model.predict(train_in, train_out, train_mask, test_in_c)

            # Herhangi bir tahmin dogru mu?
            true_grid = test_ex.output_grid.to(device)
            matched = False
            for pred in preds:
                if pred.shape == true_grid.shape and torch.equal(pred, true_grid):
                    matched = True
                    break
            if matched:
                correct += 1
            total += 1

    return correct / max(total, 1)


# ────────────────────────────────────────────────────────────────────────
# Hizli Test
# ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import random

    print("=== training.py birim testi ===\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Cihaz: {device}")

    # Minimal model
    config = ModelConfig(D_trans=64, n_heads=2, n_layers=2, N_steps=2, dropout=0.0)
    model = GeometrikAkil(config).to(device)
    counts = model.param_count()
    print(f"Toplam parametre: {counts['total']:,}")

    # Sahte veri
    from data_loader import ArcTask, ArcExample, make_dataloader
    def make_fake():
        train_exs = [
            ArcExample(
                torch.randint(0, 10, (4, 4)),
                torch.randint(0, 10, (3, 3))
            )
            for _ in range(3)
        ]
        test_ex = ArcExample(torch.randint(0, 10, (4, 4)))
        return ArcTask("fake", train_exs, [test_ex])

    tasks = {f"t{i}": make_fake() for i in range(16)}
    loader = make_dataloader(tasks, batch_size=4, shuffle=False)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    step_counter = [0]

    print("\nEgitim dongusu testi...")
    avg_losses = train_epoch(model, loader, optimizer, device,
                             step_counter, log_interval=10)
    print(f"\nOrtalama kayiplar:")
    for k, v in avg_losses.items():
        print(f"  {k:15s}: {v:.4f}")

    print("\n[OK] Egitim dongusu calisiyor")

    # Kayip sifirdan buyuk olmali
    assert avg_losses['total'] > 0, "HATA: Total loss sifir"
    assert avg_losses['L_recon'] > 0, "HATA: Recon loss sifir"
    print("[OK] Kayip degerleri makul")

    # Exact match testi
    print("\nExact-match dogrulugu testi...")
    acc = exact_match_accuracy(model, tasks, device, max_tasks=5)
    print(f"Exact-match dogrulugu: {acc:.3f} (egitilmemis model, dusuk beklenir)")
    print("[OK] Exact-match fonksiyonu calisiyor")

    print("\n[OK] Tum testler gecti.")
