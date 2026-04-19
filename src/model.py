"""
model.py
--------
GeometrikAkil: Tam sistem entegrasyonu.

4 asamali pipeline:
  Stage 1: StrategyTransformer  — egitim orneklerinden gorev stratejisi cikar
  Stage 2: SeedMLP              — cikti canvas'ini kural-tabanli olarak tohum ek
  Stage 3: NCARunner            — yerel fizik duzeltme (N adim)
  Stage 4: Decode               — argmax → grid

Konfigürasyon:
  Tum hiperparametreler ModelConfig dataclass'ında tutulur.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from transformer import StrategyTransformer
from seed_mlp    import SeedMLP
from nca_runner  import NCARunner, NCAState
from color_codec import (
    null_canvas, null_latent, extract_output,
    N_COLOR, NULL_IDX, CANVAS_H, CANVAS_W
)


# ────────────────────────────────────────────────────────────────────────
# Model Konfigürasyonu
# ────────────────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    """GeometrikAkil V1 konfigürasyonu."""

    # Transformer
    D_trans:    int   = 128
    n_heads:    int   = 4
    n_layers:   int   = 4
    patch_sz:   int   = 2
    max_train:  int   = 10      # maksimum egitim ornegi sayisi

    # SeedMLP
    P_dim:      int   = 16
    seed_heads: int   = 4

    # NCA
    L_dim:      int   = 8
    K_slots:    int   = 12
    D_obj:      int   = 32
    N_steps:    int   = 10
    K_update:   int   = 0       # V1: sadece baslangicta
    v_max:      float = 0.5

    # Canvas
    canvas_H:   int   = 30
    canvas_W:   int   = 30

    # Egitim
    use_checkpoint: bool = False
    dropout:    float = 0.1

    def __post_init__(self):
        assert self.D_trans % self.n_heads == 0, \
            f"D_trans={self.D_trans} n_heads={self.n_heads} ile bolunebilmeli"


# ────────────────────────────────────────────────────────────────────────
# Ana Model
# ────────────────────────────────────────────────────────────────────────

class GeometrikAkil(nn.Module):
    """
    ARC-AGI V1 cozucu sistem.

    Kullanim (egitim):
        model = GeometrikAkil(config)
        output = model.forward_train(batch)
        loss   = compute_loss(output, batch)

    Kullanim (test/inference):
        with torch.no_grad():
            grid = model.predict(task)   # ArcTask → [[int]] (ARC JSON format)
    """

    def __init__(self, config: ModelConfig = None):
        super().__init__()
        if config is None:
            config = ModelConfig()
        self.config = config

        # Stage 1: StrategyTransformer
        self.transformer = StrategyTransformer(
            D_trans    = config.D_trans,
            n_heads    = config.n_heads,
            n_layers   = config.n_layers,
            K_slots    = config.K_slots,
            canvas_H   = config.canvas_H,
            canvas_W   = config.canvas_W,
            patch_sz   = config.patch_sz,
            max_examples = config.max_train,
            dropout    = config.dropout,
        )

        # Stage 2: SeedMLP
        self.seed_mlp = SeedMLP(
            P_dim    = config.P_dim,
            D_trans  = config.D_trans,
            L        = config.L_dim,
            n_heads  = config.seed_heads,
            dropout  = config.dropout,
            canvas_H = config.canvas_H,
            canvas_W = config.canvas_W,
        )

        # Stage 3: NCARunner
        self.nca_runner = NCARunner(
            D_trans        = config.D_trans,
            D_obj          = config.D_obj,
            K              = config.K_slots,
            L              = config.L_dim,
            N_steps        = config.N_steps,
            K_update       = config.K_update,
            use_checkpoint = config.use_checkpoint,
            canvas_H       = config.canvas_H,
            canvas_W       = config.canvas_W,
        )
        # v_max'i set et
        self.nca_runner.nca_step.v_max = config.v_max

    def _encode_task(self, batch: dict) -> dict:
        """
        Stage 1: Egitim orneklerini Transformer'dan gecir.
        test_input de sequence'a eklenir ki SeedMLP cross-attend edebilsin.

        Donus: transformer ciktilari
        """
        return self.transformer(
            train_inputs  = batch['train_inputs'],    # [B, max_ex, 11, H, W]
            train_outputs = batch['train_outputs'],   # [B, max_ex, 11, H, W]
            train_masks   = batch['train_masks'],     # [B, max_ex] bool
            test_input    = batch.get('target_input', None),  # [B, 11, 30, 30]
        )

    def _seed_canvas(self, trans_out: dict,
                      H_outs: torch.Tensor,
                      W_outs: torch.Tensor,
                      test_input: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Stage 2: SeedMLP ile cikti canvas'i olustur.

        test_input: [B, 11, 30, 30] copy-prior icin (identity/color_swap/mirror)

        Donus: (color_logits [B,11,30,30], latent_0 [B,L,30,30])
        """
        return self.seed_mlp.forward_variable(
            input_tokens = trans_out['input_tokens'],  # [B, N_tok, D]
            task_emb     = trans_out['task_emb'],      # [B, D]
            H_outs       = H_outs,
            W_outs       = W_outs,
            test_input   = test_input,                  # [B, 11, 30, 30]
        )

    def _run_nca(self, color_0: torch.Tensor,
                 latent_0: torch.Tensor,
                 obj_mask_0: torch.Tensor,
                 task_emb: torch.Tensor,
                 H_out: torch.Tensor,
                 W_out: torch.Tensor,
                 boundary_modes: List[str]) -> NCAState:
        """
        Stage 3: NCA N adim calistir.

        boundary_modes: list[str] — her batch ornegi icin
        """
        # Batch icinde tum ornekler ayni boundary_mode kullanir (basitlik icin)
        # V2'de per-example mod desteklenecek
        mode = boundary_modes[0] if boundary_modes else 'border'

        # Color_0 softmax -> one-hot baslangic (NCA float one-hot ister)
        color_soft = torch.softmax(color_0, dim=1)   # [B, 11, 30, 30]

        init_state = NCAState(
            color    = color_soft,
            latent   = latent_0,
            obj_mask = obj_mask_0,
        )

        return self.nca_runner(
            init_state    = init_state,
            task_emb      = task_emb,
            H_out         = H_out,
            W_out         = W_out,
            boundary_mode = mode,
        )

    def forward_train(self, batch: dict) -> dict:
        """
        Egitim forward pass'i. Loss hesaplamak icin tum ara ciktilar dondurulur.

        Parametreler
        ------------
        batch : collate_arc_batch'ten gelen sozluk

        Donus
        -----
        {
          'color_final'    : [B, 11, 30, 30]  NCA son renk ciktisi
          'latent_final'   : [B, L,  30, 30]
          'obj_mask_final' : [B, K,  30, 30]
          'h_logits'       : [B, 30]   boyut logitleri
          'w_logits'       : [B, 30]
          'boundary_logits': [B, 3]
          'H_out_pred'     : [B] int   tahmin edilen boyutlar
          'W_out_pred'     : [B] int
          'H_out_true'     : [B] int   gercek boyutlar (loss icin)
          'W_out_true'     : [B] int
        }
        """
        device = batch['target_input'].device

        # Stage 1: Transformer
        trans_out = self._encode_task(batch)

        # Egitimde gercek boyutlari kullan (teacher forcing)
        # Bu sayede NCA dogru boyutta calisir
        H_out_true = batch['H_out'].to(device)   # [B]
        W_out_true = batch['W_out'].to(device)   # [B]

        # Stage 2: SeedMLP (test_input copy-prior ile)
        color_0, latent_0 = self._seed_canvas(
            trans_out, H_out_true, W_out_true,
            test_input=batch['target_input'],
        )

        # Stage 3: NCA
        boundary_modes = trans_out['boundary_modes']
        final_state = self._run_nca(
            color_0    = color_0,
            latent_0   = latent_0,
            obj_mask_0 = trans_out['obj_mask_0'],
            task_emb   = trans_out['task_emb'],
            H_out      = H_out_true,
            W_out      = W_out_true,
            boundary_modes = boundary_modes,
        )

        return {
            'color_final':     final_state.color,       # [B, 11, 30, 30]
            'latent_final':    final_state.latent,      # [B, L,  30, 30]
            'obj_mask_final':  final_state.obj_mask,    # [B, K,  30, 30]
            'h_logits':        trans_out['h_logits'],   # [B, 30]
            'w_logits':        trans_out['w_logits'],   # [B, 30]
            'boundary_logits': trans_out['boundary_logits'],
            'H_out_pred':      trans_out['H_out'],      # [B]
            'W_out_pred':      trans_out['W_out'],      # [B]
            'H_out_true':      H_out_true,
            'W_out_true':      W_out_true,
        }

    @torch.no_grad()
    def predict(self, train_inputs: torch.Tensor,
                train_outputs: torch.Tensor,
                train_masks: torch.Tensor,
                test_input: torch.Tensor,
                n_attempts: int = 2) -> List[torch.Tensor]:
        """
        Test/inference: [H_out, W_out] int grid dondurur.

        ARC yarismasi icin n_attempts=2 (iki tahmin hakki).

        Parametreler
        ------------
        train_inputs  : [1, max_ex, 11, H, W]
        train_outputs : [1, max_ex, 11, H, W]
        train_masks   : [1, max_ex] bool
        test_input    : [1, 11, 30, 30]  (canvas'ta)
        n_attempts    : Kac farkli tahmin yapilacak

        Donus
        -----
        list[torch.Tensor] : Her tahmin icin [H_out, W_out] int grid
        """
        self.eval()
        device = train_inputs.device
        B = 1  # Inference'ta hep tek ornek

        # Transformer encode (test_input da dizi sonuna eklenir)
        trans_out = self.transformer(train_inputs, train_outputs, train_masks,
                                     test_input=test_input)

        H_out = trans_out['H_out']   # [1]
        W_out = trans_out['W_out']   # [1]
        mode  = trans_out['boundary_modes'][0]

        predictions = []
        for attempt in range(n_attempts):
            # SeedMLP: test_input'a cross-attend + copy-prior residual
            # (Training ile AYNI dagitim; train-test mismatch olmamali)
            color_0, latent_0 = self._seed_canvas(
                trans_out, H_out, W_out,
                test_input=test_input,
            )

            # Attempt cesitliligi icin seed'e kucuk noise ekle (opsiyonel)
            if attempt > 0:
                color_0 = color_0 + 0.1 * torch.randn_like(color_0)

            final_state = self._run_nca(
                color_0    = color_0,       # SeedMLP ciktisi (training ile ayni)
                latent_0   = latent_0,
                obj_mask_0 = trans_out['obj_mask_0'],
                task_emb   = trans_out['task_emb'],
                H_out      = H_out,
                W_out      = W_out,
                boundary_modes = [mode],
            )

            # Decode
            grid = extract_output(
                final_state.color[0],   # [11, 30, 30]
                H_out=H_out[0].item(),
                W_out=W_out[0].item(),
                as_grid=True
            )  # [H_out, W_out] int64

            predictions.append(grid)

        return predictions

    def param_count(self) -> Dict[str, int]:
        """Her modülün parametre sayisini dondurur."""
        counts = {
            'transformer': sum(p.numel() for p in self.transformer.parameters()),
            'seed_mlp':    sum(p.numel() for p in self.seed_mlp.parameters()),
            'nca_runner':  sum(p.numel() for p in self.nca_runner.parameters()),
        }
        counts['total'] = sum(counts.values())
        return counts


# ────────────────────────────────────────────────────────────────────────
# Hizli Test
# ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== model.py birim testi ===\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Cihaz: {device}")

    config = ModelConfig(
        D_trans  = 128,
        N_steps  = 3,      # Test icin kisa tut
        K_slots  = 12,
        L_dim    = 8,
        dropout  = 0.0,    # Test icin kapat
    )

    model = GeometrikAkil(config).to(device)
    counts = model.param_count()
    print(f"Parametre sayilari:")
    for k, v in counts.items():
        print(f"  {k:15s}: {v:>10,}")

    # Sahte batch
    B = 2
    max_ex = 3
    H, W = 8, 8

    from color_codec import null_canvas
    train_in  = torch.softmax(torch.randn(B, max_ex, 11, H, W, device=device), dim=2)
    train_out = torch.softmax(torch.randn(B, max_ex, 11, H, W, device=device), dim=2)
    train_mask= torch.ones(B, max_ex, dtype=torch.bool, device=device)
    target_in = null_canvas(B=B, device=device)   # [B, 11, 30, 30]

    batch = {
        'train_inputs':  train_in,
        'train_outputs': train_out,
        'train_masks':   train_mask,
        'target_input':  target_in,
        'H_out': torch.tensor([6, 5], device=device),
        'W_out': torch.tensor([7, 4], device=device),
        'H_in':  torch.tensor([H, H], device=device),
        'W_in':  torch.tensor([W, W], device=device),
    }

    print("\nForward pass (egitim)...")
    out = model.forward_train(batch)

    print(f"\nCiktilar:")
    print(f"  color_final    : {out['color_final'].shape}")     # [2, 11, 30, 30]
    print(f"  latent_final   : {out['latent_final'].shape}")    # [2, 8, 30, 30]
    print(f"  obj_mask_final : {out['obj_mask_final'].shape}")  # [2, 12, 30, 30]
    print(f"  h_logits       : {out['h_logits'].shape}")        # [2, 30]
    print(f"  H_out_pred     : {out['H_out_pred']}")
    print(f"  H_out_true     : {out['H_out_true']}")

    # Sekil kontrolleri
    assert out['color_final'].shape   == (B, 11, CANVAS_H, CANVAS_W)
    assert out['latent_final'].shape  == (B, 8,  CANVAS_H, CANVAS_W)
    assert out['obj_mask_final'].shape== (B, 12, CANVAS_H, CANVAS_W)
    assert out['h_logits'].shape      == (B, 30)
    print("\n[OK] Sekil kontrolleri gecti")

    # Gradient
    loss = out['color_final'].mean() + out['h_logits'].mean()
    loss.backward()
    print("[OK] Backward gecti")

    # Predict (inference)
    print("\nPredict (inference)...")
    with torch.no_grad():
        preds = model.predict(
            train_inputs  = train_in[0:1],
            train_outputs = train_out[0:1],
            train_masks   = train_mask[0:1],
            test_input    = target_in[0:1],
            n_attempts    = 2,
        )
    print(f"Tahmin sayisi: {len(preds)}")
    print(f"Tahmin 1 sekli: {preds[0].shape}")   # [H_out, W_out]
    assert len(preds) == 2
    assert preds[0].dim() == 2
    print("[OK] Predict dogru")

    print("\n[OK] Tum testler gecti.")
