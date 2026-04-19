"""
kaggle_notebook.py
------------------
GeometrikAkil V1 -- Kaggle Notebook (3 Fazli Egitim)

Egitim sirasi (plan'a gore):
  Faz 1: Sentetik veri on-egitim  (synthetic_data.py)
  Faz 2: ARC-AGI-GEN 100K        (arc-agi-gen-100k, varsa)
  Faz 3: Yarisma verisi fine-tune (arc-prize-2024)

Her faz sonunda:
  - Loss curve
  - Exact-match accuracy (faz verisinde + arc eval'de)
  - Sentetik faz icin: tip bazinda accuracy breakdown

Kaggle ayarlari:
  - GPU: T4 x1 (veya P100)
  - RAM: 16GB
  - Hedef sure: ~20-25 saat
"""

# ═══════════════════════════════════════════════════════
# CELL 1: Kurulum ve Seed
# ═══════════════════════════════════════════════════════

import os, sys, json, time, random, copy
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

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

SRC_DIR = Path('/kaggle/working/src')
if not SRC_DIR.exists():
    SRC_DIR = Path('./src')  # Yerel gelistirme

sys.path.insert(0, str(SRC_DIR))

from encoding        import StaticEncodings
from color_codec     import N_COLOR, NULL_IDX, CANVAS_H, CANVAS_W, null_canvas, place_grid
from data_loader     import load_arc_json, load_arc_gen, make_dataloader
from model           import GeometrikAkil, ModelConfig
from training        import train_epoch, evaluate, exact_match_accuracy, compute_loss
from synthetic_data  import generate_synthetic_tasks, TASK_GENERATORS
from onnx_export     import batch_export_onnx

print("Moduller yuklendi.")


# ═══════════════════════════════════════════════════════
# CELL 3: Veri Yukleme
# ═══════════════════════════════════════════════════════

ARC_BASE = '/kaggle/input/arc-prize-2024'
GEN_BASE = '/kaggle/input/arc-agi-gen-100k'

print("ARC verileri yukleniyor...")

train_tasks = {}
eval_tasks  = {}
test_tasks  = {}
gen_tasks   = {}

if os.path.exists(f"{ARC_BASE}/arc-agi_training_challenges.json"):
    train_tasks = load_arc_json(
        f"{ARC_BASE}/arc-agi_training_challenges.json",
        f"{ARC_BASE}/arc-agi_training_solutions.json",
        device=device
    )
    print(f"  ARC Train : {len(train_tasks)} gorev")

if os.path.exists(f"{ARC_BASE}/arc-agi_evaluation_challenges.json"):
    eval_tasks = load_arc_json(
        f"{ARC_BASE}/arc-agi_evaluation_challenges.json",
        f"{ARC_BASE}/arc-agi_evaluation_solutions.json",
        device=device
    )
    print(f"  ARC Eval  : {len(eval_tasks)} gorev")

if os.path.exists(f"{ARC_BASE}/arc-agi_test_challenges.json"):
    test_tasks = load_arc_json(
        f"{ARC_BASE}/arc-agi_test_challenges.json",
        device=device
    )
    print(f"  ARC Test  : {len(test_tasks)} gorev (label yok)")

if os.path.exists(GEN_BASE):
    gen_tasks = load_arc_gen(GEN_BASE, max_tasks=100_000, device=device)
    print(f"  ARC-GEN   : {len(gen_tasks)} gorev")
else:
    print("  ARC-GEN   : bulunamadi, atlanacak")

print(f"\nVeri yuklemesi tamamlandi.")


# ═══════════════════════════════════════════════════════
# CELL 4: Model Olustur
# ═══════════════════════════════════════════════════════

config = ModelConfig(
    D_trans        = 128,
    n_heads        = 4,
    n_layers       = 4,
    K_slots        = 12,
    L_dim          = 8,
    N_steps        = 10,
    K_update       = 0,
    v_max          = 0.5,
    dropout        = 0.1,
    use_checkpoint = True,
)

model = GeometrikAkil(config).to(device)
counts = model.param_count()
print("Model olusturuldu:")
for k, v in counts.items():
    print(f"  {k:20s}: {v:>10,}")

CHECKPOINT_DIR = Path('/kaggle/working/checkpoints')
CHECKPOINT_DIR.mkdir(exist_ok=True)

# Gecmis checkpoint varsa yukle
resume_path = CHECKPOINT_DIR / 'phase3_best.pt'
if resume_path.exists():
    ckpt = torch.load(resume_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    print(f"\nCheckpoint yuklendi: {resume_path}")


# ═══════════════════════════════════════════════════════
# CELL 5: Ogrenme Izleme Yardimcilari
# ═══════════════════════════════════════════════════════

def make_optimizer(model, lr=3e-4):
    from torch.optim import AdamW
    return AdamW(model.parameters(), lr=lr, weight_decay=1e-4, betas=(0.9, 0.999))


def run_training_phase(phase_name, model, train_data, n_epochs, lr,
                       eval_tasks_dict=None, log_interval=50,
                       save_prefix=None, batch_size=8):
    """
    Tek bir egitim fazi calistir ve sonuclari rapor et.

    Donus: (model, history) -- history: list of epoch dicts
    """
    from torch.optim.lr_scheduler import CosineAnnealingLR

    print(f"\n{'='*60}")
    print(f"FAZ: {phase_name}")
    print(f"  Gorev sayisi : {len(train_data)}")
    print(f"  Epoch sayisi : {n_epochs}")
    print(f"  LR           : {lr}")
    print(f"{'='*60}")

    loader = make_dataloader(
        train_data,
        batch_size=batch_size,
        examples_per_task=2,
        shuffle=True,
        num_workers=0,
        device=device,
    )

    if len(loader) == 0:
        print("  UYARI: Veri yukleyici bos, faz atlaniyor.")
        return model, []

    eval_loader = make_dataloader(
        eval_tasks_dict,
        batch_size=4,
        examples_per_task=1,
        shuffle=False,
        num_workers=0,
        device=device,
    ) if eval_tasks_dict else None

    optimizer  = make_optimizer(model, lr=lr)
    total_steps= n_epochs * len(loader)
    scheduler  = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=lr * 0.01)

    history = []
    best_eval_loss = float('inf')
    step_counter = [0]

    for epoch in range(n_epochs):
        t0 = time.time()
        avg = train_epoch(
            model, loader, optimizer, device,
            step_counter, use_focal=False, grad_clip=1.0,
            log_interval=log_interval,
        )
        scheduler.step()
        elapsed = time.time() - t0

        row = {
            'phase'  : phase_name,
            'epoch'  : epoch + 1,
            'loss'   : avg['total'],
            'L_recon': avg['L_recon'],
            'acc'    : avg.get('accuracy', 0.0),
            'time_s' : elapsed,
        }

        # Eval
        if eval_loader is not None:
            eval_avg = evaluate(model, eval_loader, device)
            row['eval_loss'] = eval_avg['total']
            row['eval_acc']  = eval_avg.get('accuracy', 0.0)
            eval_str = f" | eval_loss={row['eval_loss']:.4f} eval_acc={row['eval_acc']:.3f}"

            # Checkpoint
            if eval_avg['total'] < best_eval_loss:
                best_eval_loss = eval_avg['total']
                if save_prefix:
                    ckpt_path = CHECKPOINT_DIR / f'{save_prefix}_best.pt'
                    torch.save({
                        'model_state': model.state_dict(),
                        'epoch': epoch,
                        'loss': best_eval_loss,
                        'config': config,
                    }, ckpt_path)
                    print(f"  -> Kaydedildi: {ckpt_path.name}")
        else:
            eval_str = ""

        history.append(row)
        print(f"Epoch {epoch+1:3d}/{n_epochs}: "
              f"loss={row['loss']:.4f} | recon={row['L_recon']:.4f} | "
              f"acc={row['acc']:.3f}{eval_str} | {elapsed:.0f}s")

    return model, history


def print_history_summary(history):
    """Faz sonunda kisa ozet tablosu."""
    if not history:
        return
    print(f"\n{'Epoch':>6} {'Loss':>8} {'Recon':>8} {'Acc':>6}", end="")
    if 'eval_loss' in history[0]:
        print(f" {'EvalLoss':>10} {'EvalAcc':>8}", end="")
    print()
    for row in history:
        print(f"{row['epoch']:>6} {row['loss']:>8.4f} {row['L_recon']:>8.4f} "
              f"{row['acc']:>6.3f}", end="")
        if 'eval_loss' in row:
            print(f" {row['eval_loss']:>10.4f} {row['eval_acc']:>8.3f}", end="")
        print()


# ═══════════════════════════════════════════════════════
# CELL 6: FAZ 1 — Sentetik On-Egitim
# ═══════════════════════════════════════════════════════

# Konfigurasyonlar (Kaggle sure kotasina gore ayarla)
SYNTH_N_TASKS  = 5000   # Uretilecek sentetik gorev sayisi
SYNTH_N_EPOCHS = 20     # Sentetik faz epoch sayisi
SYNTH_LR       = 3e-4

print(f"Sentetik veri uretiliyor: {SYNTH_N_TASKS} gorev...")
synth_tasks = generate_synthetic_tasks(
    n_tasks=SYNTH_N_TASKS,
    n_train_per_task=3,
    seed=SEED,
)
print(f"Uretildi: {len(synth_tasks)} gorev")

# Tip dagilimi goster
from collections import Counter
type_counts = Counter(tid.split('_')[1] for tid in synth_tasks.keys())
print("Tip dagilimi:")
for t, c in sorted(type_counts.items()):
    print(f"  {t:15s}: {c}")

# Egitim
model, hist_synth = run_training_phase(
    phase_name  = "Sentetik On-Egitim",
    model       = model,
    train_data  = synth_tasks,
    n_epochs    = SYNTH_N_EPOCHS,
    lr          = SYNTH_LR,
    eval_tasks_dict = eval_tasks if eval_tasks else None,
    log_interval    = 100,
    save_prefix     = "phase1",
    batch_size      = 8,
)
print_history_summary(hist_synth)

# Sentetik faz sonunda: tip bazinda accuracy
print("\n--- Sentetik Tip Bazinda Accuracy (Faz 1 Sonu) ---")
synth_type_acc = {}
for gen_name, gen_fn in TASK_GENERATORS:
    # Her tip icin 20 test gorevi uret
    type_tasks = generate_synthetic_tasks(
        n_tasks=20,
        n_train_per_task=3,
        weights={gen_name: 1.0},   # sadece bu tip
        seed=999,
    )
    if not type_tasks:
        continue
    acc = exact_match_accuracy(model, type_tasks, device, max_tasks=20)
    synth_type_acc[gen_name] = acc
    print(f"  {gen_name:15s}: {acc:.3f} ({acc*100:.1f}%)")

learned = [n for n, a in synth_type_acc.items() if a > 0.1]
print(f"\nOgrenen tipler (>10%): {learned if learned else 'henuz yok -- beklenen'}")
print("NOT: Sentetik dogru ogrenmeden gercek ARC'de basari beklenmez.")


# ═══════════════════════════════════════════════════════
# CELL 7: FAZ 2 — ARC-AGI-GEN 100K Fine-Tune
# ═══════════════════════════════════════════════════════

GEN_N_EPOCHS = 15
GEN_LR       = 1e-4   # Faz 1'den daha dusuk LR (katastrofik unutma azalt)

if gen_tasks:
    model, hist_gen = run_training_phase(
        phase_name      = "ARC-GEN 100K Fine-Tune",
        model           = model,
        train_data      = gen_tasks,
        n_epochs        = GEN_N_EPOCHS,
        lr              = GEN_LR,
        eval_tasks_dict = eval_tasks if eval_tasks else None,
        log_interval    = 200,
        save_prefix     = "phase2",
        batch_size      = 8,
    )
    print_history_summary(hist_gen)

    # Faz 2 sonu: sentetik accuracy hala oluyor mu? (unutma kontrolu)
    print("\n--- Sentetik Accuracy Kontrolu (Faz 2 Sonu) ---")
    for gen_name, acc_before in synth_type_acc.items():
        type_tasks = generate_synthetic_tasks(
            n_tasks=20, n_train_per_task=3,
            weights={gen_name: 1.0}, seed=888,
        )
        acc_after = exact_match_accuracy(model, type_tasks, device, max_tasks=20)
        delta = acc_after - acc_before
        flag = " [UNUTTU!]" if acc_after < acc_before - 0.05 else ""
        print(f"  {gen_name:15s}: {acc_before:.3f} -> {acc_after:.3f} "
              f"({delta:+.3f}){flag}")
else:
    print("ARC-GEN bulunamadi, Faz 2 atlaniyor.")
    hist_gen = []


# ═══════════════════════════════════════════════════════
# CELL 8: FAZ 3 — Yarisma Verisi Fine-Tune
# ═══════════════════════════════════════════════════════

ARC_N_EPOCHS = 30
ARC_LR       = 5e-5   # En dusuk LR, ogrenilenler korunsun

if train_tasks:
    model, hist_arc = run_training_phase(
        phase_name      = "ARC Yarisma Fine-Tune",
        model           = model,
        train_data      = train_tasks,
        n_epochs        = ARC_N_EPOCHS,
        lr              = ARC_LR,
        eval_tasks_dict = eval_tasks if eval_tasks else None,
        log_interval    = 20,
        save_prefix     = "phase3",
        batch_size      = 4,   # Kucuk dataset, kucuk batch
    )
    print_history_summary(hist_arc)
else:
    print("ARC train verisi yok, Faz 3 atlaniyor.")
    hist_arc = []


# ═══════════════════════════════════════════════════════
# CELL 9: Nihai Degerlendirme
# ═══════════════════════════════════════════════════════

# En iyi Phase 3 checkpoint'i yukle (yoksa mevcut agirliklari kullan)
best_ckpt = CHECKPOINT_DIR / 'phase3_best.pt'
if not best_ckpt.exists():
    best_ckpt = CHECKPOINT_DIR / 'phase2_best.pt'
if not best_ckpt.exists():
    best_ckpt = CHECKPOINT_DIR / 'phase1_best.pt'

if best_ckpt.exists():
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    print(f"En iyi checkpoint yuklendi: {best_ckpt.name}")

model.eval()
print("\n=== NIHAI DEGERLENDIRME ===")

# 1. Sentetik tip bazinda
print("\n[1] Sentetik Donusum Tip Bazinda Accuracy:")
print(f"  {'Tip':15s} {'Acc':>6} {'Yorum':}")
for gen_name, gen_fn in TASK_GENERATORS:
    type_tasks = generate_synthetic_tasks(
        n_tasks=30, n_train_per_task=3,
        weights={gen_name: 1.0}, seed=777,
    )
    acc = exact_match_accuracy(model, type_tasks, device, max_tasks=30)
    if acc >= 0.5:
        yorum = "iyi"
    elif acc >= 0.2:
        yorum = "kismi"
    elif acc >= 0.05:
        yorum = "zayif sinyal"
    else:
        yorum = "ogrenilemedi"
    print(f"  {gen_name:15s} {acc:>6.3f} {yorum}")

# 2. ARC Train exact-match
if train_tasks:
    em_train = exact_match_accuracy(model, train_tasks, device, max_tasks=200)
    print(f"\n[2] ARC Train exact-match : {em_train:.3f} ({em_train*100:.1f}%)")

# 3. ARC Eval exact-match
if eval_tasks:
    em_eval = exact_match_accuracy(model, eval_tasks, device, max_tasks=200)
    print(f"[3] ARC Eval  exact-match : {em_eval:.3f} ({em_eval*100:.1f}%)")

# 4. Ozet: ne ogrendik?
print("\n[4] Egitim Ozeti:")
all_hist = (
    [(r['epoch'], "synth", r['loss'], r['acc']) for r in hist_synth] +
    [(r['epoch'], "gen",   r['loss'], r['acc']) for r in hist_gen] +
    [(r['epoch'], "arc",   r['loss'], r['acc']) for r in hist_arc]
)
if all_hist:
    print(f"  {'Faz':6} {'Epoch':>6} {'Loss':>8} {'TrainAcc':>9}")
    for ep, phase, loss, acc in all_hist[::max(1, len(all_hist)//10)]:
        print(f"  {phase:6} {ep:>6} {loss:>8.4f} {acc:>9.3f}")

# V2'ye gecis karari
print("\n[5] V2 Ablation Karari:")
if eval_tasks and em_eval > 0.20:
    print(f"  ARC eval %{em_eval*100:.1f} > %20 -> V2 BASLATILABILIR")
    print("  V2: CliffordFlowHead + dinamik token + GlobalReadout")
elif eval_tasks and em_eval > 0.05:
    print(f"  ARC eval %{em_eval*100:.1f} -> V1'de kalarak hiper-param tuning onerilir")
else:
    print("  ARC eval dusuk -- debug: once sentetik tipleri kontrol et")
    print("  Beklenen: tile, scale, identity >= %50 olmali")


# ═══════════════════════════════════════════════════════
# CELL 10: Test-Time Fine-Tuning + Submission
# ═══════════════════════════════════════════════════════

def finetune_on_task(base_model, task, n_steps=50, lr=1e-4):
    """
    Tek gorev icin test-time fine-tuning.
    base_model kopyalanir, orijinal bozulmaz.
    """
    from data_loader import make_dataloader
    if task.n_train < 1:
        return base_model

    task_model = copy.deepcopy(base_model)
    mini_tasks = {task.task_id: task}
    loader = make_dataloader(mini_tasks, batch_size=1,
                             examples_per_task=task.n_train)
    task_model.train()
    opt = torch.optim.AdamW(task_model.parameters(), lr=lr, weight_decay=1e-5)

    for step, batch in enumerate(loader):
        if step >= n_steps:
            break
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        opt.zero_grad()
        out = task_model.forward_train(batch)
        losses = compute_loss(out, batch)
        losses['total'].backward()
        nn.utils.clip_grad_norm_(task_model.parameters(), 1.0)
        opt.step()

    task_model.eval()
    return task_model


def create_submission(model, tasks, output_file='/kaggle/working/submission.json',
                      do_finetune=True, finetune_steps=50):
    """ARC submission JSON olustur (2 attempt per test example)."""
    submission = {}

    for task_id, task in tasks.items():
        print(f"  {task_id}", end='  ')

        task_model = finetune_on_task(model, task, n_steps=finetune_steps) \
                     if do_finetune else model

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
            train_in   = torch.stack(train_in_list).unsqueeze(0).to(device)
            train_out  = torch.stack(train_out_list).unsqueeze(0).to(device)
            train_mask = torch.ones(1, n_ex, dtype=torch.bool, device=device)
            test_c     = place_grid(null_canvas(B=1, device=device),
                                    test_ex.input_grid.to(device),
                                    test_ex.H_in, test_ex.W_in)

            with torch.no_grad():
                preds = task_model.predict(
                    train_in, train_out, train_mask, test_c, n_attempts=2
                )

            if len(preds) >= 2:
                task_preds.append({
                    'attempt_1': preds[0].tolist(),
                    'attempt_2': preds[1].tolist(),
                })
            elif len(preds) == 1:
                task_preds.append({
                    'attempt_1': preds[0].tolist(),
                    'attempt_2': preds[0].tolist(),
                })
            else:
                task_preds.append({'attempt_1': [[0]], 'attempt_2': [[0]]})

        submission[task_id] = task_preds
        print(f"{len(task_preds)} tahmin")

    with open(output_file, 'w') as f:
        json.dump(submission, f)
    print(f"\nSubmission kaydedildi: {output_file}  ({len(submission)} gorev)")
    return submission


if test_tasks:
    print("\nSubmission olusturuluyor...")
    submission = create_submission(
        model, test_tasks,
        output_file='/kaggle/working/submission.json',
        do_finetune=True,
        finetune_steps=50,
    )
else:
    print("Test verisi yok, submission atlaniyor.")


# ═══════════════════════════════════════════════════════
# CELL 11: ONNX Export (NeuroGolf icin)
# ═══════════════════════════════════════════════════════

ONNX_DIR = Path('/kaggle/working/onnx_models')
ONNX_DIR.mkdir(exist_ok=True)

if test_tasks:
    print("ONNX modelleri olusturuluyor...")
    n_success = batch_export_onnx(
        model, test_tasks, str(ONNX_DIR),
        device=torch.device('cpu'),
        N_steps=config.N_steps,
    )
    print(f"ONNX export: {n_success}/{len(test_tasks)} basarili")
else:
    print("Test verisi yok, ONNX export atlaniyor.")

print("\n=== NOTEBOOK TAMAMLANDI ===")
