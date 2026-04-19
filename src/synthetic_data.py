"""
synthetic_data.py
-----------------
Sentetik ARC benzeri gorev ureticisi (on-egitim icin).

Amac:
  - ARC-AGI veri seti kucuk (~400 egitim gorevi)
  - Transformer + NCA'yi sifirdan egitmek zor
  - Bilinen donusumlerle sentetik gorev uret: model temel fizik kurallarini
    once basit orneklerle ogrenir, sonra gercek ARC'ye fine-tune yapilir

Uretilen donusum turleri:
  1. Identity           : out = in
  2. Translation        : nesneleri (dx, dy) kadar kaydir
  3. Rotation           : 90, 180, 270 derece dondur
  4. Reflection         : yatay / dikey ayna
  5. Color swap         : renk A -> renk B
  6. Tile               : girdi grid'i N×M kere tekrarla
  7. Scale              : her piksel -> N×N blok
  8. Crop               : en buyuk non-bg nesneyi kirp
  9. Fill               : ic bolgeleri doldur
 10. Gravity            : tum nesneleri bir yone dusur
 11. Outline            : nesnelerin etrafina cerceve ekle
 12. Recolor by size    : buyuk nesneler A, kucuk nesneler B

Her gorev bir ArcTask olarak uretilir (train_examples + test_examples).
"""

import random
from typing import List, Tuple, Optional, Callable
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np

from data_loader import ArcExample, ArcTask


# ────────────────────────────────────────────────────────────────────────
# Grid Yardimcilari (numpy tabanli, hizli)
# ────────────────────────────────────────────────────────────────────────

N_REAL_COLORS = 10  # 0-9


def random_grid(H: int, W: int,
                n_colors: int = 3,
                bg_color: int = 0,
                obj_density: float = 0.3,
                rng: Optional[random.Random] = None) -> np.ndarray:
    """Rastgele basit grid: arka plan + bir kac renkli piksel."""
    rng = rng or random
    grid = np.full((H, W), bg_color, dtype=np.int64)
    colors = [c for c in range(N_REAL_COLORS) if c != bg_color]
    rng.shuffle(colors)
    active_colors = colors[:n_colors]

    n_obj_pixels = int(H * W * obj_density)
    for _ in range(n_obj_pixels):
        y, x = rng.randint(0, H - 1), rng.randint(0, W - 1)
        grid[y, x] = rng.choice(active_colors)
    return grid


def random_rect_objects(H: int, W: int,
                        n_objects: int = 3,
                        bg_color: int = 0,
                        min_size: int = 1,
                        max_size: int = 4,
                        rng: Optional[random.Random] = None) -> np.ndarray:
    """Rastgele dikdortgen nesnelerden olusan grid."""
    rng = rng or random
    grid = np.full((H, W), bg_color, dtype=np.int64)
    colors = [c for c in range(N_REAL_COLORS) if c != bg_color]
    rng.shuffle(colors)

    for i in range(n_objects):
        h = rng.randint(min_size, max_size)
        w = rng.randint(min_size, max_size)
        y = rng.randint(0, max(0, H - h))
        x = rng.randint(0, max(0, W - w))
        c = colors[i % len(colors)]
        grid[y:y+h, x:x+w] = c
    return grid


def find_bbox(grid: np.ndarray, bg: int = 0) -> Optional[Tuple[int, int, int, int]]:
    """Non-background bolgenin sinirlayici kutusu (y1, x1, y2, x2) veya None."""
    mask = (grid != bg)
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    return ys.min(), xs.min(), ys.max() + 1, xs.max() + 1


# ────────────────────────────────────────────────────────────────────────
# Donusum Fonksiyonlari (her biri bir ArcTask uretir)
# ────────────────────────────────────────────────────────────────────────

def task_identity(rng: random.Random, n_train: int = 3,
                  fixed_size: Optional[Tuple[int, int]] = None) -> ArcTask:
    """out = in (temel baseline).
    fixed_size=(H,W) verilirse tum gorevler o boyutta olur (generalization testi icin)."""
    examples = []
    for i in range(n_train + 1):
        if fixed_size is not None:
            H, W = fixed_size
        else:
            H, W = rng.randint(3, 10), rng.randint(3, 10)
        inp = random_grid(H, W, n_colors=rng.randint(2, 4), rng=rng)
        out = inp.copy()
        examples.append(_to_example(inp, out))
    return _build_task("synth_identity", examples, n_train)


def task_translation(rng: random.Random, n_train: int = 3) -> ArcTask:
    """Tum nesneleri sabit (dy, dx) kadar kaydir."""
    dy = rng.randint(-2, 2)
    dx = rng.randint(-2, 2)
    if dy == 0 and dx == 0:
        dy = 1
    bg = rng.randint(0, 9)
    examples = []
    for i in range(n_train + 1):
        H, W = rng.randint(5, 10), rng.randint(5, 10)
        inp = random_rect_objects(H, W, n_objects=rng.randint(1, 3),
                                  bg_color=bg, rng=rng)
        out = np.full_like(inp, bg)
        for y in range(H):
            for x in range(W):
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W:
                    out[ny, nx] = inp[y, x]
        examples.append(_to_example(inp, out))
    return _build_task(f"synth_translate_{dy}_{dx}", examples, n_train)


def task_rotation(rng: random.Random, n_train: int = 3) -> ArcTask:
    """90, 180, 270 derece dondurme."""
    k = rng.choice([1, 2, 3])
    bg = rng.randint(0, 9)
    examples = []
    for i in range(n_train + 1):
        S = rng.randint(3, 8)
        inp = random_rect_objects(S, S, n_objects=rng.randint(1, 3),
                                  bg_color=bg, rng=rng)
        out = np.rot90(inp, k=k).copy()
        examples.append(_to_example(inp, out))
    return _build_task(f"synth_rot{90*k}", examples, n_train)


def task_reflection(rng: random.Random, n_train: int = 3) -> ArcTask:
    """Yatay veya dikey ayna."""
    axis = rng.choice(['horizontal', 'vertical'])
    bg = rng.randint(0, 9)
    examples = []
    for i in range(n_train + 1):
        H, W = rng.randint(3, 8), rng.randint(3, 8)
        inp = random_rect_objects(H, W, n_objects=rng.randint(1, 3),
                                  bg_color=bg, rng=rng)
        if axis == 'horizontal':
            out = np.fliplr(inp).copy()
        else:
            out = np.flipud(inp).copy()
        examples.append(_to_example(inp, out))
    return _build_task(f"synth_flip_{axis}", examples, n_train)


def task_color_swap(rng: random.Random, n_train: int = 3) -> ArcTask:
    """Renk A -> Renk B (diger renkler aynen kalir)."""
    c_from = rng.randint(1, 9)
    c_to   = rng.randint(1, 9)
    while c_to == c_from:
        c_to = rng.randint(1, 9)
    bg = 0

    examples = []
    for i in range(n_train + 1):
        H, W = rng.randint(4, 10), rng.randint(4, 10)
        # c_from mutlaka bulunsun
        inp = random_rect_objects(H, W, n_objects=rng.randint(2, 4),
                                  bg_color=bg, rng=rng)
        # Bir nesneyi c_from yap
        y = rng.randint(0, H - 1)
        x = rng.randint(0, W - 1)
        inp[y, x] = c_from

        out = inp.copy()
        out[out == c_from] = c_to
        examples.append(_to_example(inp, out))
    return _build_task(f"synth_swap_{c_from}_{c_to}", examples, n_train)


def task_tile(rng: random.Random, n_train: int = 3) -> ArcTask:
    """Girdi grid'i N×M kere tekrarla."""
    n_rep_y = rng.randint(2, 3)
    n_rep_x = rng.randint(2, 3)
    examples = []
    for i in range(n_train + 1):
        H, W = rng.randint(2, 4), rng.randint(2, 4)
        if H * n_rep_y > 30 or W * n_rep_x > 30:
            continue
        inp = random_grid(H, W, n_colors=rng.randint(2, 3),
                          obj_density=0.5, rng=rng)
        out = np.tile(inp, (n_rep_y, n_rep_x))
        examples.append(_to_example(inp, out))
        if len(examples) == n_train + 1:
            break
    while len(examples) < n_train + 1:
        H, W = 2, 2
        inp = random_grid(H, W, n_colors=2, obj_density=0.5, rng=rng)
        out = np.tile(inp, (n_rep_y, n_rep_x))
        examples.append(_to_example(inp, out))
    return _build_task(f"synth_tile_{n_rep_y}x{n_rep_x}", examples, n_train)


def task_scale(rng: random.Random, n_train: int = 3) -> ArcTask:
    """Her piksel -> k×k blok (nearest upsampling)."""
    k = rng.randint(2, 3)
    examples = []
    for i in range(n_train + 1):
        H, W = rng.randint(2, 5), rng.randint(2, 5)
        if H * k > 30 or W * k > 30:
            H = W = 30 // k
        inp = random_grid(H, W, n_colors=rng.randint(2, 4),
                          obj_density=0.5, rng=rng)
        out = np.repeat(np.repeat(inp, k, axis=0), k, axis=1)
        examples.append(_to_example(inp, out))
    return _build_task(f"synth_scale_{k}x", examples, n_train)


def task_crop(rng: random.Random, n_train: int = 3) -> ArcTask:
    """Girdide non-bg bolgeyi kirp."""
    bg = 0
    examples = []
    for i in range(n_train + 1):
        H, W = rng.randint(6, 12), rng.randint(6, 12)
        inp = np.full((H, W), bg, dtype=np.int64)
        # Tek bir dikdortgen nesne yerlestir
        h = rng.randint(2, min(5, H - 2))
        w = rng.randint(2, min(5, W - 2))
        y = rng.randint(1, H - h - 1)
        x = rng.randint(1, W - w - 1)
        c = rng.randint(1, 9)
        inp[y:y+h, x:x+w] = c

        out = inp[y:y+h, x:x+w].copy()
        examples.append(_to_example(inp, out))
    return _build_task("synth_crop", examples, n_train)


def task_gravity(rng: random.Random, n_train: int = 3) -> ArcTask:
    """Nesneleri bir yone dusur (tum renkli pikseller alta/uste/saga/sola)."""
    direction = rng.choice(['down', 'up', 'left', 'right'])
    bg = 0
    examples = []
    for i in range(n_train + 1):
        H, W = rng.randint(4, 8), rng.randint(4, 8)
        inp = random_grid(H, W, n_colors=rng.randint(2, 3),
                          bg_color=bg, obj_density=0.25, rng=rng)
        out = _apply_gravity(inp, direction, bg)
        examples.append(_to_example(inp, out))
    return _build_task(f"synth_gravity_{direction}", examples, n_train)


def _apply_gravity(grid: np.ndarray, direction: str, bg: int) -> np.ndarray:
    """Stabil kaydirma: her sutunda/satirda bg olmayanlari yogunlastir."""
    H, W = grid.shape
    out = grid.copy()
    if direction in ('down', 'up'):
        for x in range(W):
            col = [out[y, x] for y in range(H) if out[y, x] != bg]
            if direction == 'down':
                new_col = [bg] * (H - len(col)) + col
            else:
                new_col = col + [bg] * (H - len(col))
            for y in range(H):
                out[y, x] = new_col[y]
    else:
        for y in range(H):
            row = [out[y, x] for x in range(W) if out[y, x] != bg]
            if direction == 'right':
                new_row = [bg] * (W - len(row)) + row
            else:
                new_row = row + [bg] * (W - len(row))
            for x in range(W):
                out[y, x] = new_row[x]
    return out


def task_outline(rng: random.Random, n_train: int = 3) -> ArcTask:
    """Dolu dikdortgenlerin etrafina farkli renk cerceve."""
    outline_color = rng.randint(1, 9)
    bg = 0
    examples = []
    for i in range(n_train + 1):
        H, W = rng.randint(6, 10), rng.randint(6, 10)
        inp = np.full((H, W), bg, dtype=np.int64)
        out = inp.copy()
        n_obj = rng.randint(1, 2)
        for _ in range(n_obj):
            h = rng.randint(2, 4)
            w = rng.randint(2, 4)
            y = rng.randint(1, H - h - 1)
            x = rng.randint(1, W - w - 1)
            c = rng.randint(1, 9)
            while c == outline_color:
                c = rng.randint(1, 9)
            inp[y:y+h, x:x+w] = c
            # Cerceve
            out[y-1:y+h+1, x-1:x+w+1] = outline_color
            out[y:y+h, x:x+w] = c
        examples.append(_to_example(inp, out))
    return _build_task(f"synth_outline_{outline_color}", examples, n_train)


def task_recolor_by_size(rng: random.Random, n_train: int = 3) -> ArcTask:
    """Buyuk nesneler renk A, kucuk nesneler renk B."""
    c_big = rng.randint(1, 9)
    c_small = rng.randint(1, 9)
    while c_small == c_big:
        c_small = rng.randint(1, 9)
    bg = 0

    examples = []
    for i in range(n_train + 1):
        H, W = rng.randint(8, 12), rng.randint(8, 12)
        inp = np.full((H, W), bg, dtype=np.int64)
        out = inp.copy()
        n_obj = rng.randint(2, 4)
        placed = []
        for _ in range(n_obj):
            h = rng.randint(1, 4)
            w = rng.randint(1, 4)
            if H - h - 1 < 1 or W - w - 1 < 1:
                continue
            y = rng.randint(0, H - h)
            x = rng.randint(0, W - w)
            c = rng.randint(1, 9)
            while c in (c_big, c_small, bg):
                c = rng.randint(1, 9)
            inp[y:y+h, x:x+w] = c
            placed.append((y, x, h, w, h*w))
        if not placed:
            continue
        median_size = sorted([p[4] for p in placed])[len(placed)//2]
        for (y, x, h, w, size) in placed:
            target_c = c_big if size >= median_size else c_small
            out[y:y+h, x:x+w] = target_c
        examples.append(_to_example(inp, out))
    while len(examples) < n_train + 1:
        examples.append(examples[-1])
    return _build_task(f"synth_resize_{c_big}_{c_small}", examples, n_train)


# ────────────────────────────────────────────────────────────────────────
# Ic Yardimcilar
# ────────────────────────────────────────────────────────────────────────

def _to_example(inp_np: np.ndarray, out_np: np.ndarray) -> ArcExample:
    """numpy grid'leri torch tensora cevir ve ArcExample yap."""
    inp_t = torch.from_numpy(np.ascontiguousarray(inp_np)).long()
    out_t = torch.from_numpy(np.ascontiguousarray(out_np)).long()
    return ArcExample(inp_t, out_t)


def _build_task(task_id: str, examples: List[ArcExample], n_train: int) -> ArcTask:
    """Ilk n_train -> train, kalan -> test."""
    train_exs = examples[:n_train]
    test_exs  = examples[n_train:]
    # Test orneklerinde output'u tutuyoruz (sentetik de cikti ogreniliyor);
    # istersek None'a cevirebiliriz ama degerlendirme icin lazim.
    return ArcTask(task_id, train_exs, test_exs)


# ────────────────────────────────────────────────────────────────────────
# Toplu Uretim API
# ────────────────────────────────────────────────────────────────────────

TASK_GENERATORS: List[Tuple[str, Callable]] = [
    ("identity",       task_identity),
    ("translation",    task_translation),
    ("rotation",       task_rotation),
    ("reflection",     task_reflection),
    ("color_swap",     task_color_swap),
    ("tile",           task_tile),
    ("scale",          task_scale),
    ("crop",           task_crop),
    ("gravity",        task_gravity),
    ("outline",        task_outline),
    ("recolor_size",   task_recolor_by_size),
]


def generate_synthetic_tasks(n_tasks: int = 1000,
                             n_train_per_task: int = 3,
                             weights: Optional[dict] = None,
                             seed: int = 42,
                             fixed_size: Optional[Tuple[int, int]] = None) -> dict:
    """
    Sentetik ARC benzeri gorevler uret.

    Parametreler
    ------------
    n_tasks           : Uretilecek toplam gorev sayisi
    n_train_per_task  : Her gorevde kac egitim ornegi
    weights           : {generator_name: weight} dict, None -> esit
    seed              : RNG seed
    fixed_size        : (H, W) verilirse tum gorevler o boyutta (sadece identity
                        ve boyut koruyan generator'larda etkili)

    Donus
    -----
    dict[task_id -> ArcTask]  (data_loader.load_arc_json ile ayni format)
    """
    rng = random.Random(seed)

    if weights is None:
        names = [n for n, _ in TASK_GENERATORS]
        gen_weights = [1.0] * len(names)
    else:
        names = list(weights.keys())
        gen_weights = [weights[n] for n in names]
    name_to_fn = dict(TASK_GENERATORS)

    # fixed_size destekleyen generator'lar
    fixed_size_supported = {'identity'}

    tasks = {}
    for i in range(n_tasks):
        name = rng.choices(names, weights=gen_weights, k=1)[0]
        fn = name_to_fn[name]
        try:
            if fixed_size is not None and name in fixed_size_supported:
                task = fn(rng, n_train=n_train_per_task, fixed_size=fixed_size)
            else:
                task = fn(rng, n_train=n_train_per_task)
        except Exception as e:
            continue
        task_id = f"{task.task_id}_{i:05d}"
        tasks[task_id] = ArcTask(task_id, task.train_examples, task.test_examples)

    return tasks


# ────────────────────────────────────────────────────────────────────────
# Hizli Test
# ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== synthetic_data.py birim testi ===\n")
    rng = random.Random(0)

    # Her generator'i tek tek test et
    for name, fn in TASK_GENERATORS:
        task = fn(rng, n_train=2)
        ex = task.train_examples[0]
        print(f"  [{name:14s}] train={len(task.train_examples)} "
              f"test={len(task.test_examples)}  "
              f"in={tuple(ex.input_grid.shape)} out={tuple(ex.output_grid.shape)}")
    print()

    # Toplu uretim
    tasks = generate_synthetic_tasks(n_tasks=100, seed=7)
    print(f"Toplu uretim: {len(tasks)} gorev")

    # Ornek bir gorev goster
    first_id = next(iter(tasks))
    first = tasks[first_id]
    print(f"\nIlk gorev: {first_id}")
    print(f"  train: {len(first.train_examples)} ornek")
    print(f"  test : {len(first.test_examples)} ornek")
    ex0 = first.train_examples[0]
    print(f"  Ornek 1 input  ({tuple(ex0.input_grid.shape)}):")
    print(ex0.input_grid.numpy())
    print(f"  Ornek 1 output ({tuple(ex0.output_grid.shape)}):")
    print(ex0.output_grid.numpy())

    # Tip kontrolu
    assert isinstance(ex0.input_grid, torch.Tensor)
    assert ex0.input_grid.dtype == torch.int64
    assert ex0.input_grid.min() >= 0 and ex0.input_grid.max() <= 9
    print("\n[OK] Tum testler basarili.")
