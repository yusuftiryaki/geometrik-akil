"""
data_loader.py
--------------
ARC-AGI ve ARC-GEN veri yukleme ve on-isleme.

Kaggle dizin yapisi:
  /kaggle/input/arc-prize-2024/
    arc-agi_training_challenges.json
    arc-agi_training_solutions.json
    arc-agi_evaluation_challenges.json
    arc-agi_evaluation_solutions.json
    arc-agi_test_challenges.json

  /kaggle/input/arc-agi-gen-100k/  (eger indirilmisse)
    *.json

Her gorev sozu:
  {
    "train": [{"input": [[...]], "output": [[...]]}, ...],
    "test":  [{"input": [[...]]}, ...]
  }

Bu dosya:
  - ARC JSON'larini yukler
  - Her taski Task nesnesine cevrilir
  - PyTorch Dataset + DataLoader uretir
  - Dinamik padding: her batch icin minimum canvas boyutu
"""

import json
import os
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

# Diger modüllerimiz
import sys
sys.path.insert(0, str(Path(__file__).parent))
from color_codec import (
    arc_list_to_tensor, grid_to_onehot, null_canvas, place_grid,
    N_COLOR, NULL_IDX, CANVAS_H, CANVAS_W
)
from encoding import make_posenc, make_geofeat


# ────────────────────────────────────────────────────────────────────────
# Veri Yapilari
# ────────────────────────────────────────────────────────────────────────

class ArcExample:
    """Tek bir (input, output) cifti."""
    __slots__ = ('input_grid', 'output_grid', 'H_in', 'W_in', 'H_out', 'W_out')

    def __init__(self, input_grid: torch.Tensor,
                 output_grid: Optional[torch.Tensor] = None):
        """
        Parametreler
        ------------
        input_grid  : [H_in, W_in] int64
        output_grid : [H_out, W_out] int64 veya None (test ornekleri icin)
        """
        self.input_grid  = input_grid
        self.output_grid = output_grid
        self.H_in, self.W_in   = input_grid.shape
        if output_grid is not None:
            self.H_out, self.W_out = output_grid.shape
        else:
            self.H_out = self.W_out = None


class ArcTask:
    """Tek bir ARC gorevi: birden fazla egitim cifti + test girisleri."""
    __slots__ = ('task_id', 'train_examples', 'test_examples')

    def __init__(self, task_id: str,
                 train_examples: List[ArcExample],
                 test_examples:  List[ArcExample]):
        self.task_id        = task_id
        self.train_examples = train_examples
        self.test_examples  = test_examples

    @property
    def n_train(self):
        return len(self.train_examples)

    @property
    def n_test(self):
        return len(self.test_examples)


# ────────────────────────────────────────────────────────────────────────
# JSON Yukleme
# ────────────────────────────────────────────────────────────────────────

def load_arc_json(challenges_path: str,
                  solutions_path: Optional[str] = None,
                  device=None) -> Dict[str, ArcTask]:
    """
    ARC JSON dosyalarindan gorev sozlugu yukler.

    Parametreler
    ------------
    challenges_path : Zorluklar JSON yolu
    solutions_path  : Cozumler JSON yolu (None = test seti)
    device          : torch.device

    Donus
    -----
    tasks : {task_id: ArcTask}
    """
    with open(challenges_path, 'r') as f:
        challenges = json.load(f)

    solutions = {}
    if solutions_path and os.path.exists(solutions_path):
        with open(solutions_path, 'r') as f:
            solutions = json.load(f)

    tasks = {}
    for task_id, task_data in challenges.items():
        # Egitim ornekleri
        train_examples = []
        for ex in task_data.get('train', []):
            inp  = arc_list_to_tensor(ex['input'], device=device)
            outp = arc_list_to_tensor(ex['output'], device=device)
            train_examples.append(ArcExample(inp, outp))

        # Test ornekleri
        test_examples = []
        task_solutions = solutions.get(task_id, [])
        for i, ex in enumerate(task_data.get('test', [])):
            inp  = arc_list_to_tensor(ex['input'], device=device)
            outp = None
            if i < len(task_solutions):
                outp = arc_list_to_tensor(task_solutions[i], device=device)
            test_examples.append(ArcExample(inp, outp))

        tasks[task_id] = ArcTask(task_id, train_examples, test_examples)

    return tasks


def load_arc_gen(gen_dir: str, max_tasks: Optional[int] = None,
                 device=None) -> Dict[str, ArcTask]:
    """
    ARC-GEN JSON dosyalarindan gorev sozlugu yukler.
    Her JSON dosyasi bir gorevdir.

    Parametreler
    ------------
    gen_dir   : JSON dosyalarinin bulundugu dizin
    max_tasks : Maksimum yuklenecek gorev sayisi (None = hepsi)
    device    : torch.device

    Donus
    -----
    tasks : {task_id: ArcTask}
    """
    tasks = {}
    gen_path = Path(gen_dir)
    json_files = sorted(gen_path.glob('*.json'))

    if max_tasks is not None:
        json_files = json_files[:max_tasks]

    for jf in json_files:
        task_id = jf.stem
        with open(jf, 'r') as f:
            task_data = json.load(f)

        train_examples = []
        for ex in task_data.get('train', []):
            inp  = arc_list_to_tensor(ex['input'], device=device)
            outp = arc_list_to_tensor(ex['output'], device=device)
            train_examples.append(ArcExample(inp, outp))

        test_examples = []
        for ex in task_data.get('test', []):
            inp  = arc_list_to_tensor(ex['input'], device=device)
            outp = None
            if 'output' in ex:
                outp = arc_list_to_tensor(ex['output'], device=device)
            test_examples.append(ArcExample(inp, outp))

        tasks[task_id] = ArcTask(task_id, train_examples, test_examples)

    return tasks


# ────────────────────────────────────────────────────────────────────────
# Dataset
# ────────────────────────────────────────────────────────────────────────

class ArcDataset(Dataset):
    """
    Egitim icin ArcTask listesinden ornekleme yapan Dataset.

    Her __getitem__ cagrisi bir (task, train_example_idx) cifti dondurur:
      - task'in tum train ornekleri Transformer'a giris olarak kullanilir
      - train_example_idx'li ornek NCA cikti dogrulama hedefi olarak kullanilir
      (Hem girdi hem cikti bilinmekte — self-consistency egitimi)

    Bu yaklasim:
      - Her epoch'ta her gorev birden fazla kez orneklenir
      - Her seferinde farkli bir train ornegi hedef secilir
    """

    def __init__(self, tasks: Dict[str, ArcTask],
                 examples_per_task: int = 1,
                 augment: bool = False):
        """
        Parametreler
        ------------
        tasks             : {task_id: ArcTask}
        examples_per_task : Epoch basina her gorevden kac ornek alinir
        augment           : Veri buyutme (donme, yansima)
        """
        self.task_list        = list(tasks.values())
        self.examples_per_task = examples_per_task
        self.augment          = augment

        # Her (task_idx, example_idx) cifti icin indeks olustur
        self.index = []
        for ti, task in enumerate(self.task_list):
            if task.n_train == 0:
                continue
            for _ in range(examples_per_task):
                ei = random.randint(0, task.n_train - 1)
                self.index.append((ti, ei))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        """
        Donus: (task, target_example_idx) demeti
        Collate_fn sonrasi tensor'lara donusturulur.
        """
        task_idx, ex_idx = self.index[idx]
        return self.task_list[task_idx], ex_idx


# ────────────────────────────────────────────────────────────────────────
# Collate Fonksiyonu
# ────────────────────────────────────────────────────────────────────────

def collate_arc_batch(samples: list, device=None) -> dict:
    """
    ArcDataset orneklerini batch tensörlerine donusturur.

    Her ornek icin:
      - Tum train ornekleri Transformer'a giris olarak hazirlanir
        (input + output, one-hot, 30x30 canvas'a pad)
      - Secilen train orneginin output'u NCA hedefi olarak kullanilir

    Donus sozlugu:
      'task_ids'     : [B] str listesi
      'train_inputs' : [B, max_train, 11, 30, 30] float - Egitim giris canvas'lari
      'train_outputs': [B, max_train, 11, 30, 30] float - Egitim cikis canvas'lari
      'train_masks'  : [B, max_train] bool - Gecerli egitim ornegi maskesi
      'target_input' : [B, 11, 30, 30] float - NCA baslangic girisi
      'target_output': [B, H_out, W_out] int64 - NCA beklenen cikti
      'H_out'        : [B] int - Hedef yukseklik
      'W_out'        : [B] int - Hedef genislik
      'H_in'         : [B] int - Giris yuksekligi
      'W_in'         : [B] int - Giris genisligi
    """
    B = len(samples)
    max_train = max(s[0].n_train for s in samples)

    task_ids     = []
    train_inputs  = []
    train_outputs = []
    train_masks   = []
    target_inputs  = []
    target_outputs = []
    h_outs, w_outs = [], []
    h_ins,  w_ins  = [], []

    for task, ex_idx in samples:
        task_ids.append(task.task_id)

        # Hedef ornek (NCA dogrulama)
        target_ex = task.train_examples[ex_idx]
        H_in, W_in   = target_ex.H_in, target_ex.W_in
        H_out, W_out = target_ex.H_out, target_ex.W_out
        h_ins.append(H_in);   w_ins.append(W_in)
        h_outs.append(H_out); w_outs.append(W_out)

        # Hedef giris canvas'i (NCA baslangic durumu)
        t_inp_canvas = null_canvas(B=1, device=device)
        t_inp_canvas = place_grid(t_inp_canvas, target_ex.input_grid, H_in, W_in)
        target_inputs.append(t_inp_canvas[0])  # [11, 30, 30]
        target_outputs.append(target_ex.output_grid)  # [H_out, W_out]

        # Tum egitim ornekleri (Transformer girisi)
        t_in_list, t_out_list, mask_list = [], [], []
        for i in range(max_train):
            if i < task.n_train:
                ex = task.train_examples[i]
                inp_canvas = null_canvas(B=1, device=device)
                inp_canvas = place_grid(inp_canvas, ex.input_grid, ex.H_in, ex.W_in)
                out_canvas = null_canvas(B=1, device=device)
                out_canvas = place_grid(out_canvas, ex.output_grid, ex.H_out, ex.W_out)
                t_in_list.append(inp_canvas[0])   # [11, 30, 30]
                t_out_list.append(out_canvas[0])  # [11, 30, 30]
                mask_list.append(True)
            else:
                # Padding (gecersiz ornek)
                t_in_list.append(null_canvas(B=1, device=device)[0])
                t_out_list.append(null_canvas(B=1, device=device)[0])
                mask_list.append(False)

        train_inputs.append(torch.stack(t_in_list))   # [max_train, 11, 30, 30]
        train_outputs.append(torch.stack(t_out_list)) # [max_train, 11, 30, 30]
        train_masks.append(torch.tensor(mask_list))   # [max_train] bool

    return {
        'task_ids':     task_ids,
        'train_inputs': torch.stack(train_inputs),    # [B, max_train, 11, 30, 30]
        'train_outputs':torch.stack(train_outputs),   # [B, max_train, 11, 30, 30]
        'train_masks':  torch.stack(train_masks),     # [B, max_train] bool
        'target_input': torch.stack(target_inputs),   # [B, 11, 30, 30]
        'target_output':target_outputs,               # liste[B] of [H_out, W_out]
        'H_out':        torch.tensor(h_outs),         # [B]
        'W_out':        torch.tensor(w_outs),         # [B]
        'H_in':         torch.tensor(h_ins),          # [B]
        'W_in':         torch.tensor(w_ins),          # [B]
    }


def make_dataloader(tasks: Dict[str, ArcTask],
                    batch_size: int = 8,
                    examples_per_task: int = 1,
                    shuffle: bool = True,
                    num_workers: int = 0,
                    device=None) -> DataLoader:
    """
    ArcDataset'ten DataLoader uretir.

    Parametreler
    ------------
    tasks             : {task_id: ArcTask}
    batch_size        : Gorev basina batch boyutu
    examples_per_task : Epoch basina her gorevden kac ornek
    shuffle           : Karistirma
    num_workers       : Paralel veri yukleme (Kaggle: 0 onerilen)
    device            : Tensörlerin yerlestirilecegi cihaz

    Donus
    -----
    DataLoader
    """
    dataset = ArcDataset(tasks, examples_per_task=examples_per_task)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda samples: collate_arc_batch(samples, device=device)
    )


# ────────────────────────────────────────────────────────────────────────
# Kaggle Yollari
# ────────────────────────────────────────────────────────────────────────

KAGGLE_ARC_BASE = "/kaggle/input/arc-prize-2024"
KAGGLE_GEN_BASE = "/kaggle/input/arc-agi-gen-100k"

ARC_PATHS = {
    'train_challenges': f"{KAGGLE_ARC_BASE}/arc-agi_training_challenges.json",
    'train_solutions':  f"{KAGGLE_ARC_BASE}/arc-agi_training_solutions.json",
    'eval_challenges':  f"{KAGGLE_ARC_BASE}/arc-agi_evaluation_challenges.json",
    'eval_solutions':   f"{KAGGLE_ARC_BASE}/arc-agi_evaluation_solutions.json",
    'test_challenges':  f"{KAGGLE_ARC_BASE}/arc-agi_test_challenges.json",
}


def load_all_arc(device=None) -> Dict[str, Dict[str, ArcTask]]:
    """
    Tum ARC bolumlerini yukler.

    Donus
    -----
    {
      'train': {...},
      'eval':  {...},
      'test':  {...}
    }
    """
    result = {}
    splits = [
        ('train', ARC_PATHS['train_challenges'], ARC_PATHS['train_solutions']),
        ('eval',  ARC_PATHS['eval_challenges'],  ARC_PATHS['eval_solutions']),
        ('test',  ARC_PATHS['test_challenges'],  None),
    ]
    for name, chal, sol in splits:
        if os.path.exists(chal):
            result[name] = load_arc_json(chal, sol, device=device)
            print(f"[data_loader] {name}: {len(result[name])} gorev yuklendi")
        else:
            print(f"[data_loader] {name}: dosya bulunamadi ({chal})")
            result[name] = {}

    return result


# ────────────────────────────────────────────────────────────────────────
# Hizli Test (JSON dosyasi olmadan)
# ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== data_loader.py birim testi ===\n")

    # Sahte gorev olustur
    def make_fake_task(task_id, n_train=3):
        train_exs = []
        for i in range(n_train):
            h = random.randint(2, 8)
            w = random.randint(2, 8)
            inp  = torch.randint(0, 10, (h, w))
            outp = torch.randint(0, 10, (h, w))
            train_exs.append(ArcExample(inp, outp))
        test_inp = torch.randint(0, 10, (5, 5))
        test_exs = [ArcExample(test_inp)]
        return ArcTask(task_id, train_exs, test_exs)

    tasks = {f"task_{i:03d}": make_fake_task(f"task_{i:03d}") for i in range(20)}
    print(f"Sahte gorev sayisi: {len(tasks)}")

    # DataLoader testi
    loader = make_dataloader(tasks, batch_size=4, examples_per_task=2, shuffle=False)
    print(f"DataLoader uzunlugu: {len(loader)}")

    batch = next(iter(loader))
    print(f"\nBatch anahtarlari: {list(batch.keys())}")
    print(f"train_inputs  : {batch['train_inputs'].shape}")   # [4, max_train, 11, 30, 30]
    print(f"train_outputs : {batch['train_outputs'].shape}")  # [4, max_train, 11, 30, 30]
    print(f"train_masks   : {batch['train_masks'].shape}")    # [4, max_train]
    print(f"target_input  : {batch['target_input'].shape}")   # [4, 11, 30, 30]
    print(f"H_out         : {batch['H_out']}")
    print(f"W_out         : {batch['W_out']}")

    # one-hot dogrulama
    ti = batch['train_inputs']
    assert ti.shape[2] == 11, "HATA: 11 renk kanali olmali"
    assert ti.shape[3] == 30 and ti.shape[4] == 30, "HATA: 30x30 canvas olmali"
    print("\n[OK] Sekil kontrolleri gecti")

    # NULL canvas dogrulama
    # Eklenen grid disindaki pikseller NULL olmali
    t0 = batch['target_input'][0]  # [11, 30, 30]
    H0 = batch['H_in'][0].item()
    W0 = batch['W_in'][0].item()
    outside_pixel = t0[:, H0:, W0:]
    if outside_pixel.numel() > 0:
        assert outside_pixel.argmax(dim=0).max().item() == NULL_IDX, \
            "HATA: Dis bolge NULL olmali"
        print("[OK] Dis bolge NULL kontrol gecti")

    print("\n[OK] Tum testler gecti.")
