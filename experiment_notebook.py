"""
experiment_notebook.py
----------------------
GeometrikAkil V1 -- Curriculum Deney Notebook

Amac:
  Modelin gercekten ogrenip ogrenmedigini ispatlamak + hatalari tespit etmek.

Yaklasim:
  Curriculum learning -- kolaydan zora 6 seviye:
    L0: identity                (en kolay: out=in)
    L1: color_swap              (renk haritalama)
    L2: reflection              (yatay/dikey ayna, boyut sabit)
    L3: translation             (kaydirma, boyut sabit)
    L4: rotation + scale        (geometrik donusum, boyut degisir)
    L5: tile + gravity + outline (cok-adim mantik)

Her seviye icin:
  - Sadece o tipin gorevleriyle egit (kucuk, hizli)
  - Exact-match accuracy izle
  - BASARI KRITERI: >=50% exact-match
  - Basarisizsa: detayli tanilama (gorev printing, pred vs gt, loss breakdown)
  - Basariliysa: bir sonraki seviye ile birlestirerek devam

Cikti:
  - Curve: seviye x epoch x accuracy
  - Her seviye icin 3-5 ornek tahmin (gorsel + grid)
  - Ortalama loss breakdown her adimi
  - Basarisiz tiplerin gorev printleri (debug icin)
"""

# ═══════════════════════════════════════════════════════
# CELL 1: Kurulum
# ═══════════════════════════════════════════════════════

import os, sys, json, time, random, copy
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

SEED = 123
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Cihaz: {device}")

SRC_DIR = Path('/kaggle/working/src')
if not SRC_DIR.exists():
    SRC_DIR = Path('./src')
sys.path.insert(0, str(SRC_DIR))

from color_codec     import null_canvas, place_grid, NULL_IDX, N_COLOR
from data_loader     import make_dataloader, ArcTask, ArcExample
from model           import GeometrikAkil, ModelConfig
from training        import train_epoch, evaluate, exact_match_accuracy, compute_loss
from synthetic_data  import generate_synthetic_tasks, TASK_GENERATORS

print("Moduller yuklendi.\n")


# ═══════════════════════════════════════════════════════
# CELL 2: Kucuk Model (Hizli Deney)
# ═══════════════════════════════════════════════════════

# Curriculum deneyi icin kucultulmus model (hizli iterasyon)
config = ModelConfig(
    D_trans        = 64,      # 128 yerine 64 (deney hizi)
    n_heads        = 4,
    n_layers       = 2,       # 4 yerine 2
    K_slots        = 8,       # 12 yerine 8
    L_dim          = 8,
    N_steps        = 6,       # 10 yerine 6
    K_update       = 0,
    v_max          = 0.5,
    dropout        = 0.0,     # Deney: overfit bile etmesine izin ver
    use_checkpoint = False,
)
model = GeometrikAkil(config).to(device)
print("Deney modeli:")
for k, v in model.param_count().items():
    print(f"  {k:20s}: {v:>10,}")


# ═══════════════════════════════════════════════════════
# CELL 3: Curriculum Seviyeleri
# ═══════════════════════════════════════════════════════

CURRICULUM = [
    ("L0_identity",     ["identity"]),
    ("L1_color",        ["identity", "color_swap"]),
    ("L2_reflection",   ["identity", "color_swap", "reflection"]),
    ("L3_translation",  ["identity", "color_swap", "reflection", "translation"]),
    ("L4_geometric",    ["reflection", "translation", "rotation", "scale"]),
    ("L5_mantik",       ["tile", "gravity", "outline", "recolor_size", "crop"]),
]

SUCCESS_THRESHOLD = 0.50   # %50 exact-match = seviye basarili


# ═══════════════════════════════════════════════════════
# CELL 4: Tanilama Yardimcilari
# ═══════════════════════════════════════════════════════

def grid_to_str(grid: torch.Tensor) -> str:
    """Kucuk grid'i terminal icin basilabilir hale getir."""
    if isinstance(grid, torch.Tensor):
        grid = grid.cpu().numpy()
    lines = []
    for row in grid:
        lines.append(' '.join(str(int(c)) if c >= 0 and c <= 9 else '.'
                              for c in row))
    return '\n'.join(lines)


def predict_single(model, task, device):
    """Tek test ornegi icin tahmin uret; (input, gt, pred) dondur."""
    model.eval()
    test_ex = task.test_examples[0]

    # Train orneklerini hazirla
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

    test_c = place_grid(null_canvas(B=1, device=device),
                        test_ex.input_grid.to(device),
                        test_ex.H_in, test_ex.W_in)

    with torch.no_grad():
        preds = model.predict(train_in, train_out, train_mask, test_c, n_attempts=1)

    return test_ex.input_grid, test_ex.output_grid, preds[0]


def diagnose_failures(model, tasks_dict, device, max_samples=3):
    """Basarisiz gorevlerden orneklediginde hata tipini teshis et."""
    print("\n  --- Hata Tanilama ---")
    shown = 0
    for task_id, task in tasks_dict.items():
        if shown >= max_samples:
            break
        try:
            inp, gt, pred = predict_single(model, task, device)
        except Exception as e:
            print(f"    [{task_id}] predict() hata: {e}")
            continue

        if gt is None:
            continue
        correct = (pred.shape == gt.shape) and torch.equal(pred.cpu(), gt.cpu())
        if correct:
            continue

        shown += 1
        print(f"\n  [{task_id}]")
        print(f"    Input ({tuple(inp.shape)}):")
        for line in grid_to_str(inp).split('\n'):
            print(f"      {line}")
        print(f"    Ground truth ({tuple(gt.shape)}):")
        for line in grid_to_str(gt).split('\n'):
            print(f"      {line}")
        print(f"    Prediction ({tuple(pred.shape)}):")
        for line in grid_to_str(pred).split('\n'):
            print(f"      {line}")

        # Hata turu sezgisi
        if pred.shape != gt.shape:
            print(f"    HATA TURU: Boyut yanlis ({tuple(pred.shape)} != {tuple(gt.shape)})")
        else:
            diff = (pred.cpu() != gt.cpu()).float().mean().item()
            print(f"    HATA TURU: Renk/piksel yanlis, fark orani={diff:.2%}")


def show_success_examples(model, tasks_dict, device, max_samples=2):
    """Basarili tahminlerden ornek goster (model ogrendi ispatı)."""
    print("\n  --- Basarili Tahmin Ornekleri ---")
    shown = 0
    for task_id, task in tasks_dict.items():
        if shown >= max_samples:
            break
        try:
            inp, gt, pred = predict_single(model, task, device)
        except Exception:
            continue
        if gt is None:
            continue
        correct = (pred.shape == gt.shape) and torch.equal(pred.cpu(), gt.cpu())
        if not correct:
            continue

        shown += 1
        print(f"\n  [OK] {task_id}")
        print(f"    in={tuple(inp.shape)}  out={tuple(gt.shape)}  [exact match]")


def measure_accuracy_per_type(model, type_names, device, n_tasks_per_type=20, seed=999):
    """Her tip icin bagimsiz 20 gorev uret ve accuracy hesapla."""
    results = {}
    for t in type_names:
        tasks = generate_synthetic_tasks(
            n_tasks=n_tasks_per_type, n_train_per_task=3,
            weights={t: 1.0}, seed=seed,
        )
        if not tasks:
            results[t] = 0.0
            continue
        acc = exact_match_accuracy(model, tasks, device, max_tasks=n_tasks_per_type)
        results[t] = acc
    return results


# ═══════════════════════════════════════════════════════
# CELL 5: Curriculum Egitim Dongusu
# ═══════════════════════════════════════════════════════

def train_on_types(model, type_names, n_epochs=10, lr=3e-4,
                   n_tasks=500, batch_size=8, log_interval=50):
    """Belirli tiplerdeki sentetik gorevlerle egit."""
    tasks = generate_synthetic_tasks(
        n_tasks=n_tasks, n_train_per_task=3,
        weights={t: 1.0 for t in type_names},
        seed=SEED,
    )
    if not tasks:
        print("  UYARI: Gorev uretilemedi.")
        return []

    loader = make_dataloader(
        tasks, batch_size=batch_size, examples_per_task=2,
        shuffle=True, num_workers=0, device=device,
    )
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    opt = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=n_epochs * len(loader), eta_min=lr * 0.01)

    history = []
    step_counter = [0]
    for ep in range(n_epochs):
        t0 = time.time()
        avg = train_epoch(
            model, loader, opt, device,
            step_counter, use_focal=False, grad_clip=1.0,
            log_interval=log_interval,
        )
        sched.step()
        row = {
            'epoch': ep + 1,
            'loss':    avg['total'],
            'L_recon': avg.get('L_recon', 0.0),
            'L_size':  avg.get('L_size',  0.0),
            'L_out':   avg.get('L_out',   0.0),
            'L_mask':  avg.get('L_mask',  0.0),
            'L_obj':   avg.get('L_obj',   0.0),
            'acc':     avg.get('accuracy', 0.0),
            'time':    time.time() - t0,
        }
        history.append(row)
        print(f"    Epoch {ep+1:2d}/{n_epochs}: "
              f"tot={row['loss']:.3f} "
              f"rec={row['L_recon']:.3f} siz={row['L_size']:.3f} "
              f"out={row['L_out']:.3f} msk={row['L_mask']:.3f} "
              f"obj={row['L_obj']:.3f} | pix={row['acc']:.3f} "
              f"({row['time']:.0f}s)")
    return history


def measure_sizehead_accuracy(model, type_names, device, n_tasks=30, seed=555):
    """SizeHead'in H,W tahmininin dogruluk orani."""
    tasks = generate_synthetic_tasks(
        n_tasks=n_tasks, n_train_per_task=3,
        weights={t: 1.0 for t in type_names}, seed=seed,
    )
    if not tasks:
        return 0.0, 0.0
    model.eval()
    h_correct = w_correct = total = 0
    with torch.no_grad():
        for task in tasks.values():
            test_ex = task.test_examples[0]
            if test_ex.output_grid is None:
                continue
            # Transformer encode (test_input dahil)
            train_in_list, train_out_list = [], []
            for ex in task.train_examples:
                inp_c = place_grid(null_canvas(B=1, device=device),
                                   ex.input_grid.to(device), ex.H_in, ex.W_in)
                out_c = place_grid(null_canvas(B=1, device=device),
                                   ex.output_grid.to(device), ex.H_out, ex.W_out)
                train_in_list.append(inp_c[0])
                train_out_list.append(out_c[0])
            train_in   = torch.stack(train_in_list).unsqueeze(0)
            train_out  = torch.stack(train_out_list).unsqueeze(0)
            train_mask = torch.ones(1, len(train_in_list), dtype=torch.bool, device=device)
            test_c     = place_grid(null_canvas(B=1, device=device),
                                    test_ex.input_grid.to(device),
                                    test_ex.H_in, test_ex.W_in)
            trans_out = model.transformer(train_in, train_out, train_mask,
                                          test_input=test_c)
            h_pred = trans_out['H_out'][0].item()
            w_pred = trans_out['W_out'][0].item()
            if h_pred == test_ex.H_out:
                h_correct += 1
            if w_pred == test_ex.W_out:
                w_correct += 1
            total += 1
    if total == 0:
        return 0.0, 0.0
    return h_correct / total, w_correct / total


# ═══════════════════════════════════════════════════════
# CELL 6: ANA DENEY — Curriculum Yurutme
# ═══════════════════════════════════════════════════════

EPOCHS_PER_LEVEL = 10
TASKS_PER_LEVEL  = 500

overall_results = {}
level_accuracy_history = []

for level_name, types in CURRICULUM:
    print(f"\n{'='*60}")
    print(f"SEVIYE: {level_name}")
    print(f"  Tipler: {types}")
    print(f"{'='*60}")

    # Egit
    hist = train_on_types(
        model, types,
        n_epochs=EPOCHS_PER_LEVEL,
        n_tasks=TASKS_PER_LEVEL,
        lr=3e-4,
    )

    # SizeHead accuracy (test-time)
    h_acc, w_acc = measure_sizehead_accuracy(model, types, device, n_tasks=50)
    print(f"\n  --- SizeHead Test Accuracy ---")
    print(f"    H tahmini dogru: {h_acc:.2%}   W tahmini dogru: {w_acc:.2%}")

    # Bu seviyedeki tum tipler icin ayri accuracy
    print(f"\n  --- {level_name} Test Accuracy (tip bazinda, exact match) ---")
    acc_per_type = measure_accuracy_per_type(model, types, device,
                                              n_tasks_per_type=30, seed=777)
    for t, a in acc_per_type.items():
        status = "[OK]" if a >= SUCCESS_THRESHOLD else "[X]"
        print(f"    {status} {t:15s}: {a:.3f} ({a*100:.1f}%)")

    overall_acc = np.mean(list(acc_per_type.values()))
    overall_results[level_name] = {
        'types': types,
        'per_type_acc': acc_per_type,
        'overall_acc': overall_acc,
        'history': hist,
    }
    level_accuracy_history.append((level_name, overall_acc))

    # Basariliysa: basari ornekleri. Basarisizsa: hata tanilama.
    eval_tasks = generate_synthetic_tasks(
        n_tasks=30, n_train_per_task=3,
        weights={t: 1.0 for t in types}, seed=555,
    )

    if overall_acc >= SUCCESS_THRESHOLD:
        print(f"\n  OVERALL: {overall_acc:.3f} >= {SUCCESS_THRESHOLD} -> BASARILI")
        show_success_examples(model, eval_tasks, device, max_samples=2)
    else:
        print(f"\n  OVERALL: {overall_acc:.3f} < {SUCCESS_THRESHOLD} -> BASARISIZ")
        diagnose_failures(model, eval_tasks, device, max_samples=3)

        # Sorulmasi gereken sorular
        print("\n  --- Olasi Hata Nedenleri ---")
        all_zero = all(a < 0.05 for a in acc_per_type.values())
        mixed = any(a > 0.3 for a in acc_per_type.values())
        if all_zero:
            print("    * Hic bir sey ogrenilmemis -> loss collapsed mi? "
                  "Gradient flow, init, LR kontrol et.")
        elif mixed:
            print("    * Bazi tipler ogrenildi digerleri yok -> kapasite "
                  "yetersiz veya egitim epoch'u az.")
        if acc_per_type.get('identity', 0) < 0.5:
            print("    * Identity bile ogrenilmemis -> SeedMLP canvas "
                  "init veya NCA stability problemi.")

        # Bu seviye basarisizsa: devam edilsin mi?
        print("\n  Seviye atlanabilir, ama ileriki seviyeler muhtemelen "
              "daha kotu olacak.")


# ═══════════════════════════════════════════════════════
# CELL 7: Ozet Rapor
# ═══════════════════════════════════════════════════════

print(f"\n{'='*60}")
print("CURRICULUM DENEY OZETI")
print(f"{'='*60}\n")

print(f"{'Seviye':20s} {'Overall Acc':>12s} {'Durum':>10s}")
print("-" * 45)
for level_name, acc in level_accuracy_history:
    status = "OK" if acc >= SUCCESS_THRESHOLD else "BASARISIZ"
    print(f"{level_name:20s} {acc:>12.3f} {status:>10s}")

# Tip bazinda nihai durum (tum seviyelerden)
print("\n--- Tip Bazinda Nihai Accuracy ---")
final_type_acc = {}
for level_name, result in overall_results.items():
    for t, a in result['per_type_acc'].items():
        # En son egitilmis seviyedeki degeri al
        final_type_acc[t] = a
for t in sorted(final_type_acc.keys()):
    a = final_type_acc[t]
    status = "OK" if a >= SUCCESS_THRESHOLD else "X"
    print(f"  [{status}] {t:15s}: {a:.3f}")

# Karar
n_ok = sum(1 for a in final_type_acc.values() if a >= SUCCESS_THRESHOLD)
n_total = len(final_type_acc)
print(f"\nOgrenilen tip sayisi: {n_ok}/{n_total}")

if n_ok == 0:
    print("\n[KARAR] Model temel sinyal gostermiyor.")
    print("  Hata teshisi: arch / training / data hangisinde sorun?")
    print("  1) Once identity ogrenilmeli. Ogrenilmediyse:")
    print("     - SeedMLP canvas init kontrolu")
    print("     - NCA stability (v_max, N_steps dusur)")
    print("     - Loss weights (L_out cok baskin mi?)")
elif n_ok < n_total / 2:
    print("\n[KARAR] Kismi ogrenme var.")
    print("  Hangi tipler ogreniliyor/ogrenilmiyor analiz et.")
    print("  - Basit olanlar (identity/color_swap) OK ise mimari calisiyor")
    print("  - Karmasik olanlar (tile/rotation) OK degilse kapasite yetersiz")
elif n_ok < n_total:
    print("\n[KARAR] Cogunlukla ogrenildi, V1 sagligi iyi.")
    print("  Eksik tipler hedef alinarak daha uzun egitim denenebilir.")
else:
    print("\n[KARAR] TUM tipler ogrenildi. Mimari saglam.")
    print("  Gercek ARC-AGI egitimi guvenli bir sekilde baslatilabilir.")

# Checkpoint kaydet
save_path = Path('/kaggle/working/experiment_model.pt')
if not save_path.parent.exists():
    save_path = Path('./experiment_model.pt')
torch.save({
    'model_state': model.state_dict(),
    'config': config,
    'results': overall_results,
}, save_path)
print(f"\nDeney checkpoint: {save_path}")
print("\n=== DENEY TAMAMLANDI ===")
