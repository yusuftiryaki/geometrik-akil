"""
kaggle_notebook.py
------------------
GeometrikAkil V1 — NeuroGolf 2026 Kaggle Notebook

Bu dosya Kaggle'da cell cell calistirmak icin tasarlanmistir.
Tum kaynak dosyalar /kaggle/working/src/ altina yazilir,
ardindan egitim ve submission olusturulur.

Kaggle ayarları:
  - GPU: T4 x1 (veya P100)
  - RAM: 16GB
  - Disk: 20GB
  - Hedef sure: ~20-25 saat
"""

# ═══════════════════════════════════════════════════════
# CELL 1: Kurulum ve Import
# ═══════════════════════════════════════════════════════

import os, sys, json, time, random
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Cihaz: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Bellek: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ═══════════════════════════════════════════════════════
# CELL 2: Kaynak Dosyalari Yukle
# ═══════════════════════════════════════════════════════

# /kaggle/working/src altindaki modulleri import et
# (Notebook'u calistirmadan once src/ klasorunu upload etmek gerekir)
# Alternatif: her modulu ayri notebook cell olarak yaz

SRC_DIR = Path('/kaggle/working/src')
if not SRC_DIR.exists():
    # Yerel gelistirme ortamı
    SRC_DIR = Path('./src')

sys.path.insert(0, str(SRC_DIR))

from encoding      import StaticEncodings
from color_codec   import N_COLOR, NULL_IDX, CANVAS_H, CANVAS_W, null_canvas, place_grid
from data_loader   import load_arc_json, load_arc_gen, make_dataloader
from model         import GeometrikAkil, ModelConfig
from training      import train_epoch, evaluate, exact_match_accuracy, compute_loss
from onnx_export   import batch_export_onnx

print("Moduller yuklendi.")


# ═══════════════════════════════════════════════════════
# CELL 3: Veri Yukleme
# ═══════════════════════════════════════════════════════

ARC_BASE = '/kaggle/input/arc-prize-2024'
GEN_BASE = '/kaggle/input/arc-agi-gen-100k'  # Varsa

print("ARC verileri yukleniyor...")

# Egitim verisi
train_tasks = {}
eval_tasks  = {}
test_tasks  = {}

if os.path.exists(f"{ARC_BASE}/arc-agi_training_challenges.json"):
    train_tasks = load_arc_json(
        f"{ARC_BASE}/arc-agi_training_challenges.json",
        f"{ARC_BASE}/arc-agi_training_solutions.json",
        device=device
    )
    print(f"  Train: {len(train_tasks)} gorev")

if os.path.exists(f"{ARC_BASE}/arc-agi_evaluation_challenges.json"):
    eval_tasks = load_arc_json(
        f"{ARC_BASE}/arc-agi_evaluation_challenges.json",
        f"{ARC_BASE}/arc-agi_evaluation_solutions.json",
        device=device
    )
    print(f"  Eval:  {len(eval_tasks)} gorev")

if os.path.exists(f"{ARC_BASE}/arc-agi_test_challenges.json"):
    test_tasks = load_arc_json(
        f"{ARC_BASE}/arc-agi_test_challenges.json",
        device=device
    )
    print(f"  Test:  {len(test_tasks)} gorev")

# ARC-GEN (eger indirilmisse)
gen_tasks = {}
if os.path.exists(GEN_BASE):
    gen_tasks = load_arc_gen(GEN_BASE, max_tasks=100_000, device=device)
    print(f"  ARC-GEN: {len(gen_tasks)} gorev")

# Tum egitim verisini birlestir
all_train = {**train_tasks, **gen_tasks}
print(f"\nToplam egitim gorevi: {len(all_train)}")


# ═══════════════════════════════════════════════════════
# CELL 4: Model Olustur
# ═══════════════════════════════════════════════════════

config = ModelConfig(
    D_trans   = 128,
    n_heads   = 4,
    n_layers  = 4,
    K_slots   = 12,
    L_dim     = 8,
    N_steps   = 10,
    K_update  = 0,      # V1: sadece baslangic
    v_max     = 0.5,
    dropout   = 0.1,
    use_checkpoint = True,   # Bellek tasarrufu
)

model = GeometrikAkil(config).to(device)
counts = model.param_count()
print(f"Model olusturuldu:")
for k, v in counts.items():
    print(f"  {k:15s}: {v:>10,}")


# ═══════════════════════════════════════════════════════
# CELL 5: Egitim
# ═══════════════════════════════════════════════════════

# Veri yukleyici
train_loader = make_dataloader(
    all_train,
    batch_size = 8,
    examples_per_task = 2,
    shuffle = True,
    num_workers = 0,
    device = device,
)

eval_loader = make_dataloader(
    eval_tasks,
    batch_size = 4,
    examples_per_task = 1,
    shuffle = False,
    num_workers = 0,
    device = device,
) if eval_tasks else None

# Optimizer
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4, betas=(0.9, 0.999))

N_EPOCHS   = 5   # Baslangic icin; arttirilabilir
total_steps= N_EPOCHS * len(train_loader)
scheduler  = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)

CHECKPOINT_DIR = Path('/kaggle/working/checkpoints')
CHECKPOINT_DIR.mkdir(exist_ok=True)

print(f"Egitim basliyor:")
print(f"  Epoch sayisi  : {N_EPOCHS}")
print(f"  Adim/epoch    : {len(train_loader)}")
print(f"  Toplam adim   : {total_steps}")

step_counter = [0]
best_eval_loss = float('inf')

for epoch in range(N_EPOCHS):
    t0 = time.time()

    avg = train_epoch(
        model, train_loader, optimizer, device,
        step_counter, use_focal=False, grad_clip=1.0,
        log_interval=100
    )
    scheduler.step()

    t1 = time.time()
    print(f"\nEpoch {epoch+1}/{N_EPOCHS}: "
          f"loss={avg['total']:.4f} | "
          f"recon={avg['L_recon']:.4f} | "
          f"acc={avg['accuracy']:.3f} | "
          f"sure={t1-t0:.0f}s")

    # Eval
    if eval_loader is not None and (epoch + 1) % 2 == 0:
        eval_avg = evaluate(model, eval_loader, device)
        print(f"  Eval: loss={eval_avg['total']:.4f} | acc={eval_avg['accuracy']:.3f}")

        if eval_avg['total'] < best_eval_loss:
            best_eval_loss = eval_avg['total']
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'config': config,
            }, CHECKPOINT_DIR / 'best_model.pt')
            print(f"  [Kaydedildi] En iyi model: loss={best_eval_loss:.4f}")

    # Checkpoint
    if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), CHECKPOINT_DIR / f'epoch_{epoch+1}.pt')

print("\nEgitim tamamlandi!")


# ═══════════════════════════════════════════════════════
# CELL 6: Exact-Match Dogrulugu
# ═══════════════════════════════════════════════════════

# En iyi modeli yukle
best_ckpt = CHECKPOINT_DIR / 'best_model.pt'
if best_ckpt.exists():
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    print(f"En iyi model yuklendi (epoch {ckpt['epoch']+1})")

print("\nExact-match dogrulugu hesaplaniyor...")
if train_tasks:
    em_train = exact_match_accuracy(model, train_tasks, device, max_tasks=100)
    print(f"Train exact-match: {em_train:.3f} ({em_train*100:.1f}%)")

if eval_tasks:
    em_eval = exact_match_accuracy(model, eval_tasks, device, max_tasks=100)
    print(f"Eval  exact-match: {em_eval:.3f} ({em_eval*100:.1f}%)")


# ═══════════════════════════════════════════════════════
# CELL 7: Test-Time Fine-Tuning (Per-Task)
# ═══════════════════════════════════════════════════════

def finetune_on_task(model: GeometrikAkil,
                     task,
                     n_steps: int = 50,
                     lr: float = 1e-4) -> GeometrikAkil:
    """
    Tek bir gorev uzerinde hizli fine-tuning.
    Test orneklerini cikarmadan sadece train orneklerini kullanir.
    """
    from data_loader import ArcTask, make_dataloader

    if task.n_train < 1:
        return model

    # Mini loader
    mini_tasks = {task.task_id: task}
    loader = make_dataloader(mini_tasks, batch_size=1, examples_per_task=task.n_train)

    model.train()
    opt = AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    for step, batch in enumerate(loader):
        if step >= n_steps:
            break
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        opt.zero_grad()
        out = model.forward_train(batch)
        losses = compute_loss(out, batch)
        losses['total'].backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    model.eval()
    return model


# ═══════════════════════════════════════════════════════
# CELL 8: Submission Olustur
# ═══════════════════════════════════════════════════════

def create_submission(model: GeometrikAkil,
                       tasks: dict,
                       output_file: str = '/kaggle/working/submission.json',
                       n_attempts: int = 2,
                       do_finetune: bool = True) -> dict:
    """
    ARC submission JSON olustur.
    """
    submission = {}

    for task_id, task in tasks.items():
        print(f"Islemde: {task_id}", end=' ')

        # Per-task fine-tuning
        if do_finetune and task.n_train >= 1:
            # Model agirliklarini kopyala (fine-tune orijinali bozmasin)
            import copy
            task_model = copy.deepcopy(model)
            task_model = finetune_on_task(task_model, task, n_steps=50)
        else:
            task_model = model

        task_model.eval()

        # Her test ornegi icin tahmin
        task_preds = []
        for test_ex in task.test_examples:
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
            train_in  = torch.stack(train_in_list).unsqueeze(0).to(device)
            train_out = torch.stack(train_out_list).unsqueeze(0).to(device)
            train_mask= torch.ones(1, n_ex, dtype=torch.bool, device=device)

            # Test girdisi
            test_c = place_grid(null_canvas(B=1, device=device),
                                test_ex.input_grid.to(device),
                                test_ex.H_in, test_ex.W_in)

            with torch.no_grad():
                preds = task_model.predict(
                    train_in, train_out, train_mask, test_c,
                    n_attempts=n_attempts
                )

            # Her tahmini listeye cevir
            ex_preds = []
            for pred in preds:
                ex_preds.append({'attempt_1': pred.tolist()})

            # 2 tahmin formatı
            if len(ex_preds) >= 2:
                task_preds.append({
                    'attempt_1': preds[0].tolist(),
                    'attempt_2': preds[1].tolist(),
                })
            elif len(ex_preds) == 1:
                task_preds.append({
                    'attempt_1': preds[0].tolist(),
                    'attempt_2': preds[0].tolist(),
                })
            else:
                # Fallback: bos grid
                task_preds.append({
                    'attempt_1': [[0]],
                    'attempt_2': [[0]],
                })

        submission[task_id] = task_preds
        print(f"-> {len(task_preds)} tahmin")

    # JSON kaydet
    with open(output_file, 'w') as f:
        json.dump(submission, f)
    print(f"\nSubmission kaydedildi: {output_file}")
    print(f"Toplam gorev: {len(submission)}")

    return submission


# Test submission (kucuk ornekle)
if test_tasks:
    print("\nTest submission olusturuluyor...")
    submission = create_submission(
        model,
        test_tasks,
        output_file='/kaggle/working/submission.json',
        n_attempts=2,
        do_finetune=True,
    )
else:
    print("Test verisi yok, submission atlaniyor.")


# ═══════════════════════════════════════════════════════
# CELL 9: ONNX Export (NeuroGolf icin)
# ═══════════════════════════════════════════════════════

ONNX_DIR = Path('/kaggle/working/onnx_models')
ONNX_DIR.mkdir(exist_ok=True)

if test_tasks:
    print("ONNX modelleri olusturuluyor...")
    n_success = batch_export_onnx(
        model,
        test_tasks,
        str(ONNX_DIR),
        device=torch.device('cpu'),  # ONNX export CPU'da
        N_steps=config.N_steps,
    )
    print(f"ONNX export: {n_success}/{len(test_tasks)} basarili")
else:
    print("Test verisi yok, ONNX export atlaniyor.")

print("\nNotebook tamamlandi!")
