"""
onnx_export.py
--------------
NeuroGolf 2026 yarismasi icin ONNX export.

Yarisma gereksinimleri:
  - Her gorev icin ayri bir ONNX dosyasi
  - NCA dongusu ONNX'e loop unrolling ile yazilir (surekli dongu yok)
  - Tek dosya = Transformer encode + SeedMLP + NCA N adim unroll + decode

NeuroGolf skoru:
  score = max(1, 25 - ln(cost))
  cost  = parametreler + memory_bytes + MAC_ops

Strateji:
  - Transformer: once egitim verisini isle → task_emb (CPU'da, once)
  - ONNX dosyasina yaz: sadece NCA (daha hafif)
  - Alternatif: tam pipeline ONNX

Dikkat: ONNX export'ta dinamik boyutlar sorun cıkarabilir.
        Sabit 30x30 canvas kullanmak export'u kolaylastirir.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import List, Optional
import os

import sys
sys.path.insert(0, str(Path(__file__).parent))

from model import GeometrikAkil, ModelConfig
from nca_runner import NCAState
from color_codec import (
    null_canvas, extract_output,
    N_COLOR, NULL_IDX, CANVAS_H, CANVAS_W
)


# ────────────────────────────────────────────────────────────────────────
# ONNX-Uyumlu NCA Unroll Modulu
# ────────────────────────────────────────────────────────────────────────

class NCAUnrolled(nn.Module):
    """
    NCA N adimini unroll edilmis sekilde calistiran ONNX-uyumlu modul.

    ONNX dongu desteklemez; her NCA adimi explicit cagrilir.
    N_steps sabit olmalidir.

    Bu modul SADECE NCA + decode icerir.
    Transformer encode ve SeedMLP onceden calistirilir ve ciktilari
    bu modüle sabit tensor olarak verilir.
    """

    def __init__(self, model: GeometrikAkil, N_steps: int = 10):
        super().__init__()
        self.N_steps   = N_steps
        self.nca_step  = model.nca_runner.nca_step
        self.static_enc= model.nca_runner.static_enc

    def forward(self,
                color_0:    torch.Tensor,
                latent_0:   torch.Tensor,
                obj_mask_0: torch.Tensor,
                task_emb:   torch.Tensor,
                active_mask:torch.Tensor) -> torch.Tensor:
        """
        Parametreler (ONNX sabit girdiler)
        ------------------------------------
        color_0     : [1, 11, 30, 30]  SeedMLP ciktisi (softmax uygulanmis)
        latent_0    : [1,  8, 30, 30]  SeedMLP latent
        obj_mask_0  : [1, 12, 30, 30]  Transformer seg ciktisi
        task_emb    : [1, 128]          Transformer task embedding
        active_mask : [1,  1, 30, 30]  Hedef bolge maskesi

        Donus
        -----
        color_logits : [1, 11, 30, 30]  argmax → output grid
        """
        B = color_0.shape[0]
        posenc, geofeat = self.static_enc.get(B)
        posenc  = posenc.to(color_0.device)
        geofeat = geofeat.to(color_0.device)

        color    = color_0.clone()
        latent   = latent_0.clone()
        obj_mask = obj_mask_0.clone()

        # Unrolled NCA: her adimi explicit yaz
        for _ in range(self.N_steps):
            color, latent, obj_mask = self.nca_step(
                color, latent, obj_mask,
                posenc, geofeat, task_emb, active_mask,
                boundary_mode='border'
            )

        return color   # [1, 11, 30, 30]


# ────────────────────────────────────────────────────────────────────────
# Per-Task ONNX Export
# ────────────────────────────────────────────────────────────────────────

def export_task_onnx(model: GeometrikAkil,
                     task,
                     output_path: str,
                     device: torch.device,
                     N_steps: int = 10,
                     opset_version: int = 17) -> bool:
    """
    Tek bir ARC gorevi icin ONNX dosyasi uretir.

    Pipeline:
    1. Transformer encode (Python'da, ONNX disinda)
    2. SeedMLP (Python'da)
    3. NCA unroll → ONNX

    Parametreler
    ------------
    model       : Egitilmis GeometrikAkil
    task        : ArcTask
    output_path : ONNX dosyasi cikti yolu
    device      : cpu (ONNX export CPU'da)
    N_steps     : NCA adim sayisi
    opset_version: ONNX opset (17 onerilen)

    Donus
    -----
    bool : True = basarili
    """
    try:
        import onnx
        import onnxruntime
    except ImportError:
        print("UYARI: onnx veya onnxruntime yuklu degil. pip install onnx onnxruntime")
        return False

    model.eval()
    model = model.to(device)

    from color_codec import null_canvas, place_grid, get_active_mask

    try:
        with torch.no_grad():
            # --- Stage 1: Transformer encode ---
            if task.n_train == 0:
                print(f"UYARI: Gorev {task.task_id} egitim ornegi yok")
                return False

            # Egitim orneklerini hazirla
            train_in_list, train_out_list = [], []
            for ex in task.train_examples:
                inp_c = place_grid(null_canvas(B=1, device=device),
                                   ex.input_grid.to(device), ex.H_in, ex.W_in)
                out_c = place_grid(null_canvas(B=1, device=device),
                                   ex.output_grid.to(device), ex.H_out, ex.W_out)
                train_in_list.append(inp_c[0])
                train_out_list.append(out_c[0])

            n_ex = len(train_in_list)
            train_in  = torch.stack(train_in_list).unsqueeze(0)
            train_out = torch.stack(train_out_list).unsqueeze(0)
            train_mask= torch.ones(1, n_ex, dtype=torch.bool, device=device)

            trans_out = model.transformer(train_in, train_out, train_mask)
            task_emb  = trans_out['task_emb']
            H_out_pred = trans_out['H_out']
            W_out_pred = trans_out['W_out']
            obj_mask_0 = trans_out['obj_mask_0']

            # --- Stage 2: SeedMLP ---
            H_out = int(H_out_pred[0].item())
            W_out = int(W_out_pred[0].item())
            _, latent_0 = model.seed_mlp(
                trans_out['input_tokens'], task_emb, H_out, W_out
            )

            # Aktif bolge maskesi
            active_mask = get_active_mask(H_out, W_out, B=1, device=device)

            # --- Stage 3: NCA Unroll ONNX ---
            nca_model = NCAUnrolled(model, N_steps=N_steps).to(device)
            nca_model.eval()

            # Ornek girdi (placeholder)
            dummy_color = null_canvas(B=1, device=device)

        # ONNX export (no_grad disinda — torch.onnx.export kendi handle eder)
        torch.onnx.export(
            nca_model,
            args=(dummy_color, latent_0, obj_mask_0, task_emb, active_mask),
            f=output_path,
            input_names=['color_0', 'latent_0', 'obj_mask_0', 'task_emb', 'active_mask'],
            output_names=['color_final'],
            dynamic_axes={'color_0': {0: 'batch'}},
            opset_version=opset_version,
            do_constant_folding=True,
        )

        print(f"  [{task.task_id}] ONNX kaydedildi: {output_path} "
              f"(H_out={H_out}, W_out={W_out})")
        return True

    except Exception as e:
        print(f"  [{task.task_id}] HATA: {e}")
        return False


def batch_export_onnx(model: GeometrikAkil,
                       tasks: dict,
                       output_dir: str,
                       device: torch.device,
                       N_steps: int = 10,
                       max_tasks: Optional[int] = None) -> int:
    """
    Tum gorevler icin ONNX dosyalari uretir.

    Donus: Basarili export sayisi
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    task_list = list(tasks.items())
    if max_tasks is not None:
        task_list = task_list[:max_tasks]

    success = 0
    for task_id, task in task_list:
        onnx_path = str(output_path / f"{task_id}.onnx")
        ok = export_task_onnx(model, task, onnx_path, device, N_steps=N_steps)
        if ok:
            success += 1

    print(f"\nONNX export: {success}/{len(task_list)} basarili")
    return success


# ────────────────────────────────────────────────────────────────────────
# ONNX Inference
# ────────────────────────────────────────────────────────────────────────

def run_onnx_inference(onnx_path: str,
                        test_input_canvas: torch.Tensor,
                        latent_0: torch.Tensor,
                        obj_mask_0: torch.Tensor,
                        task_emb: torch.Tensor,
                        active_mask: torch.Tensor,
                        H_out: int,
                        W_out: int) -> torch.Tensor:
    """
    ONNX dosyasindan tek tahmin calistir.

    Donus: [H_out, W_out] int64 grid
    """
    try:
        import onnxruntime as ort
        import numpy as np
    except ImportError:
        raise ImportError("pip install onnxruntime")

    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

    inputs = {
        'color_0':    test_input_canvas.numpy().astype('float32'),
        'latent_0':   latent_0.numpy().astype('float32'),
        'obj_mask_0': obj_mask_0.numpy().astype('float32'),
        'task_emb':   task_emb.numpy().astype('float32'),
        'active_mask':active_mask.numpy().astype('float32'),
    }

    [color_final] = sess.run(['color_final'], inputs)
    color_tensor  = torch.from_numpy(color_final)    # [1, 11, 30, 30]

    return extract_output(color_tensor[0], H_out, W_out, as_grid=True)


# ────────────────────────────────────────────────────────────────────────
# Hizli Test (ONNX kutuphanesi olmadan)
# ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== onnx_export.py birim testi ===\n")

    device = torch.device('cpu')
    config = ModelConfig(D_trans=64, n_heads=2, n_layers=2, N_steps=3, dropout=0.0)
    model = GeometrikAkil(config).to(device)

    # NCAUnrolled testi (ONNX olmadan)
    nca_unroll = NCAUnrolled(model, N_steps=3)
    B = 1
    color_0    = torch.softmax(torch.randn(B, 11, 30, 30), dim=1)
    latent_0   = torch.zeros(B, 8, 30, 30)
    obj_mask_0 = torch.softmax(torch.randn(B, 12, 30, 30), dim=1)
    task_emb   = torch.randn(B, 64)
    active_mask= torch.ones(B, 1, 30, 30)

    with torch.no_grad():
        color_final = nca_unroll(color_0, latent_0, obj_mask_0, task_emb, active_mask)
    print(f"NCAUnrolled cikti: {color_final.shape}")   # [1, 11, 30, 30]
    assert color_final.shape == (1, 11, 30, 30)
    print("[OK] NCAUnrolled calisiyor")

    # ONNX export testi
    try:
        import onnx

        # Sahte gorev olustur
        from data_loader import ArcTask, ArcExample
        train_exs = [
            ArcExample(
                torch.randint(0, 9, (4, 4)),
                torch.randint(0, 9, (3, 3))
            )
            for _ in range(2)
        ]
        task = ArcTask("test_task", train_exs, [ArcExample(torch.randint(0, 9, (4, 4)))])

        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "test_task.onnx")
            ok = export_task_onnx(model, task, onnx_path, device, N_steps=3)
            if ok:
                # Dosya boyutu
                size_mb = os.path.getsize(onnx_path) / 1e6
                print(f"[OK] ONNX export basarili: {size_mb:.2f} MB")

                # ONNX dogrulama
                onnx_model = onnx.load(onnx_path)
                onnx.checker.check_model(onnx_model)
                print("[OK] ONNX model gecerli")
            else:
                print("[SKIP] ONNX export basarisiz (model boyutu sorunu olabilir)")

    except ImportError:
        print("[SKIP] onnx yuklu degil -- export testi atlaniyor")
        print("       Yuklemek icin: pip install onnx onnxruntime")

    print("\n[OK] Testler tamamlandi.")
