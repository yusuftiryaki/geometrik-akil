"""
diagnose.py
-----------
Tek bir 3x3 identity gorevi uzerinde mikro-tani.

Amac: Sorun hangi asamadà? Transformer? SeedMLP? NCA? Gradient?

Test: Ayni gorevi 200 step boyunca OVERFIT etmeye zorla.
  - Basarili olmali (tek gorev, sonsuz kapasite)
  - Basarili olmazsa: mimari/gradient/loss problemi
"""

import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import torch
import torch.nn.functional as F
import numpy as np

from color_codec    import null_canvas, place_grid, N_COLOR, NULL_IDX
from data_loader    import ArcTask, ArcExample, make_dataloader
from model          import GeometrikAkil, ModelConfig
from training       import compute_loss

torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")

# ────────────────────────────────────────────────────────────────────────
# 1. En basit gorev: 3x3 identity, tek ornek
# ────────────────────────────────────────────────────────────────────────

def make_simple_identity_task():
    """Tek bir 3x3 identity gorevi (train=test)."""
    grids = [
        torch.tensor([[1,0,0],[0,2,0],[0,0,3]], dtype=torch.long),
        torch.tensor([[0,4,0],[5,0,5],[0,4,0]], dtype=torch.long),
        torch.tensor([[6,6,0],[0,7,0],[0,0,8]], dtype=torch.long),
    ]
    train_exs = [ArcExample(g, g.clone()) for g in grids]
    test_ex   = ArcExample(grids[0], grids[0].clone())
    return ArcTask("diag_id", train_exs, [test_ex])

task = make_simple_identity_task()
print(f"Task: {task.task_id}, train={len(task.train_examples)}, test={len(task.test_examples)}")
print(f"Input grid:\n{task.train_examples[0].input_grid.numpy()}\n")


# ────────────────────────────────────────────────────────────────────────
# 2. Kucuk model
# ────────────────────────────────────────────────────────────────────────

config = ModelConfig(
    D_trans=64, n_heads=4, n_layers=2,
    K_slots=8, L_dim=8, N_steps=4,
    dropout=0.0, use_checkpoint=False, v_max=0.5,
)
model = GeometrikAkil(config).to(device)
print(f"Model params: {sum(p.numel() for p in model.parameters()):,}\n")


# ────────────────────────────────────────────────────────────────────────
# 3. Batch hazirla (tek gorev, tek yon)
# ────────────────────────────────────────────────────────────────────────

def make_batch(task, device):
    train_in_list, train_out_list = [], []
    for ex in task.train_examples:
        inp_c = place_grid(null_canvas(B=1, device=device),
                           ex.input_grid.to(device), ex.H_in, ex.W_in)
        out_c = place_grid(null_canvas(B=1, device=device),
                           ex.output_grid.to(device), ex.H_out, ex.W_out)
        train_in_list.append(inp_c[0])
        train_out_list.append(out_c[0])

    train_in   = torch.stack(train_in_list).unsqueeze(0)  # [1, 3, 11, 30, 30]
    train_out  = torch.stack(train_out_list).unsqueeze(0)
    train_mask = torch.ones(1, 3, dtype=torch.bool, device=device)

    test_ex = task.test_examples[0]
    target_in = place_grid(null_canvas(B=1, device=device),
                            test_ex.input_grid.to(device),
                            test_ex.H_in, test_ex.W_in)

    batch = {
        'train_inputs':  train_in,
        'train_outputs': train_out,
        'train_masks':   train_mask,
        'target_input':  target_in,
        'target_output': [test_ex.output_grid.to(device)],  # list icin batch[0]
        'H_out': torch.tensor([test_ex.H_out], device=device),
        'W_out': torch.tensor([test_ex.W_out], device=device),
        'H_in':  torch.tensor([test_ex.H_in], device=device),
        'W_in':  torch.tensor([test_ex.W_in], device=device),
    }
    return batch

batch = make_batch(task, device)
print("Batch shapes:")
for k, v in batch.items():
    if isinstance(v, torch.Tensor):
        print(f"  {k:15s}: {tuple(v.shape)}")
print()


# ────────────────────────────────────────────────────────────────────────
# 4. ILK FORWARD: Her asamanin ciktisini incele
# ────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("FORWARD TANI (egitim oncesi)")
print("=" * 60)

model.eval()
with torch.no_grad():
    # Stage 1: Transformer
    trans_out = model.transformer(
        batch['train_inputs'], batch['train_outputs'], batch['train_masks'],
        test_input=batch['target_input'],
    )
    print(f"\n[Transformer]")
    print(f"  task_emb:     {tuple(trans_out['task_emb'].shape)}   "
          f"norm={trans_out['task_emb'].norm().item():.3f}")
    print(f"  input_tokens: {tuple(trans_out['input_tokens'].shape)}  "
          f"(her ornek ~30-40 token)")
    print(f"  H_out pred: {trans_out['H_out'].item()} (gt=3)")
    print(f"  W_out pred: {trans_out['W_out'].item()} (gt=3)")

    # Stage 2: SeedMLP
    color_0, latent_0 = model._seed_canvas(
        trans_out, batch['H_out'], batch['W_out']
    )
    print(f"\n[SeedMLP]")
    print(f"  color_0: {tuple(color_0.shape)}")
    # 3x3 bolgedeki argmax dagilimi
    region = color_0[0, :, :3, :3].argmax(dim=0)
    print(f"  argmax in 3x3 region:\n{region.cpu().numpy()}")
    print(f"  target:\n{task.test_examples[0].output_grid.numpy()}")
    # Dagilim
    probs_region = torch.softmax(color_0[0, :, :3, :3], dim=0)
    print(f"  max prob/pixel (3x3 avg): {probs_region.max(dim=0)[0].mean().item():.3f}")
    print(f"  prob class 0 (bg) avg:   {probs_region[0].mean().item():.3f}")
    print(f"  prob class 10 (NULL) avg:{probs_region[10].mean().item():.3f}")

    # Stage 3: NCA
    final_state = model._run_nca(
        color_0=torch.softmax(color_0, dim=1),
        latent_0=latent_0,
        obj_mask_0=trans_out['obj_mask_0'],
        task_emb=trans_out['task_emb'],
        H_out=batch['H_out'],
        W_out=batch['W_out'],
        boundary_modes=trans_out['boundary_modes'],
    )
    print(f"\n[NCA after {config.N_steps} steps]")
    region_nca = final_state.color[0, :, :3, :3].argmax(dim=0)
    print(f"  argmax in 3x3 region:\n{region_nca.cpu().numpy()}")
    probs_nca = final_state.color[0, :, :3, :3]
    print(f"  max prob/pixel (3x3 avg): {probs_nca.max(dim=0)[0].mean().item():.3f}")
    print(f"  prob class 0 (bg) avg:   {probs_nca[0].mean().item():.3f}")
    print(f"  prob class 10 (NULL) avg:{probs_nca[10].mean().item():.3f}")


# ────────────────────────────────────────────────────────────────────────
# 5. OVERFIT DENEMESI: 200 step tek gorev
# ────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("OVERFIT DENEMESI (200 step, ayni gorev)")
print("=" * 60)

opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
model.train()

for step in range(1, 201):
    opt.zero_grad()
    out = model.forward_train(batch)
    losses = compute_loss(out, batch)
    losses['total'].backward()

    # Gradient norms per module
    if step in (1, 10, 50, 100, 200):
        grads = {}
        for name, module in [
            ('transformer', model.transformer),
            ('seed_mlp',    model.seed_mlp),
            ('nca_runner',  model.nca_runner),
        ]:
            total = 0.0
            count = 0
            for p in module.parameters():
                if p.grad is not None:
                    total += p.grad.norm().item() ** 2
                    count += 1
            grads[name] = (total ** 0.5, count)

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

    if step in (1, 5, 10, 25, 50, 100, 150, 200):
        # Teacher-forcing pix accuracy
        pred = out['color_final'][0, :, :3, :3].argmax(dim=0)
        gt   = batch['target_output'][0]
        correct = (pred == gt).float().mean().item()

        # Argmax distribution
        pred_np = pred.cpu().numpy()

        print(f"\nStep {step:3d}: total={losses['total'].item():.3f} "
              f"recon={losses['L_recon'].item():.3f} "
              f"size={losses['L_size'].item():.3f} "
              f"out={losses['L_out'].item():.3f} "
              f"pix_acc(3x3)={correct:.2f}")
        print(f"  Pred:\n{pred_np}")
        if step in (1, 10, 50, 100, 200):
            print(f"  Grad norms: trans={grads['transformer'][0]:.4f} "
                  f"seed={grads['seed_mlp'][0]:.4f} "
                  f"nca={grads['nca_runner'][0]:.4f}")


# ────────────────────────────────────────────────────────────────────────
# 6. SONUC YORUMU
# ────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("YORUM")
print("=" * 60)

# Son tahmin
model.eval()
with torch.no_grad():
    out = model.forward_train(batch)
    pred = out['color_final'][0, :, :3, :3].argmax(dim=0)
    gt   = batch['target_output'][0]
    final_acc = (pred == gt).float().mean().item()

print(f"\nSon 3x3 pix_acc: {final_acc:.2%}")
print(f"Beklenen: %100 (tek gorev, 200 step overfit)\n")

if final_acc > 0.95:
    print("[OK] Mimari overfit edebiliyor. Sorun: generalization veya veri cesitliligi.")
elif final_acc > 0.5:
    print("[KISMI] Ogreniyor ama tam ezberleyemedi.")
    print("  - Gradient NCA'ya zayif akiyor olabilir")
    print("  - NCA steps cok yuksek (gradient vanish)")
else:
    print("[HATA] Overfit BILE edemiyor. Mimari/gradient problemi.")
    print("  - Loss backward hic gradient uretiyor mu?")
    print("  - SeedMLP/NCA arasinda kesinti var mi?")
    print("  - Softmax'in hangi eksende uygulandigi dogru mu?")
