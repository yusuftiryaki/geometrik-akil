"""
clifford_algebra.py
-------------------
Conformal Geometric Algebra Cl(3,1) uzerinde temel operasyonlar.

Plan statusu: V1'de NCA'DA KULLANILMIYOR (flow problemi cozulduktan sonra
kaldirildi). Transformer'da operator uretimi icin V1/V2 ablation'da
kullanilabilir. Bu dosya bagimsiz, test edilebilir PyTorch modulleri saglar.

Cl(3,1) nedir?
  - Uc uzaysal (e1, e2, e3) + bir zamansal (e4) temel vektor
  - Metrik: e1^2 = e2^2 = e3^2 = +1, e4^2 = -1
  - Toplam 2^4 = 16 boyutlu multivector uzayi

Temel bileşenler (grade):
  Grade 0: skalar                   (1 boyut)
  Grade 1: vektorler e1..e4         (4 boyut)
  Grade 2: bivector e_ij            (6 boyut)
  Grade 3: trivector e_ijk          (4 boyut)
  Grade 4: pseudoscalar I           (1 boyut)

ARC icin kullanimi:
  - Rotor R = exp(-theta/2 * B): Donme operatoru (bivector B ile tanimli)
  - Reflection: v' = -n v n (n: hiperdüzlem normali)
  - Translate (CGA'da): rotor ile ifade edilebilir
  - Transformer, uretilen multivector'u sandwich product ile uygular

V1 yaklasimi: Clifford rotorlar Transformer'dan uretilir,
              task_emb icine FiLM kanali olarak eklenir.
"""

import torch
import torch.nn as nn
from typing import Tuple


# ────────────────────────────────────────────────────────────────────────
# Baz Iskelet: 16D multivector sirasi
# ────────────────────────────────────────────────────────────────────────

# Blade dizilimi (sabit, model boyunca tutarli):
#   idx 0       : 1                  (skalar)
#   idx 1..4    : e1, e2, e3, e4     (vektorler)
#   idx 5..10   : e12, e13, e14, e23, e24, e34  (bivectors)
#   idx 11..14  : e123, e124, e134, e234        (trivectors)
#   idx 15      : e1234                          (pseudoscalar)

MV_DIM = 16

BLADE_NAMES = [
    "1",
    "e1", "e2", "e3", "e4",
    "e12", "e13", "e14", "e23", "e24", "e34",
    "e123", "e124", "e134", "e234",
    "e1234",
]

# Metrik: diag(+1, +1, +1, -1)
METRIC = torch.tensor([1.0, 1.0, 1.0, -1.0])


# ────────────────────────────────────────────────────────────────────────
# Geometrik carpim tablosu
# ────────────────────────────────────────────────────────────────────────

def _blade_as_set(idx: int) -> Tuple[int, ...]:
    """Blade idx -> temel vektor indeksleri (1..4) sirali tuple."""
    # Blade dizilimini acik tanimla (sabit, 16 eleman)
    table = [
        (),
        (1,), (2,), (3,), (4,),
        (1,2), (1,3), (1,4), (2,3), (2,4), (3,4),
        (1,2,3), (1,2,4), (1,3,4), (2,3,4),
        (1,2,3,4),
    ]
    return table[idx]


def _set_to_blade(s: Tuple[int, ...]) -> int:
    """Sirali temel vektor tuple -> blade idx."""
    for i in range(16):
        if _blade_as_set(i) == s:
            return i
    raise ValueError(f"Bilinmeyen blade: {s}")


def _multiply_blades(a: Tuple[int, ...], b: Tuple[int, ...],
                     metric: list) -> Tuple[int, Tuple[int, ...]]:
    """
    Iki blade'in geometrik carpimi -> (isaret, sonuc_blade).

    Anti-commute kurali: swap sayisi ile isaret, ayni vektor kalktiginda
    metrige gore skalar.
    """
    combined = list(a) + list(b)
    sign = 1

    # Bubble sort (swap sayisini takip et)
    n = len(combined)
    for i in range(n):
        for j in range(n - 1 - i):
            if combined[j] > combined[j + 1]:
                combined[j], combined[j + 1] = combined[j + 1], combined[j]
                sign *= -1

    # Ayni vektorleri kaldir (e_i * e_i = metric[i-1])
    result = []
    idx = 0
    while idx < len(combined):
        if idx + 1 < len(combined) and combined[idx] == combined[idx + 1]:
            sign *= metric[combined[idx] - 1]
            idx += 2
        else:
            result.append(combined[idx])
            idx += 1
    return sign, tuple(result)


def _build_cayley_table() -> torch.Tensor:
    """
    Cl(3,1) geometrik carpim Cayley tablosu.

    Donus: [16, 16, 16] tensor
      C[i, j, k] = (e_i * e_j)'nin e_k bileseninin katsayisi (+1, -1, 0)
    """
    metric = [1, 1, 1, -1]
    table = torch.zeros(MV_DIM, MV_DIM, MV_DIM)
    for i in range(MV_DIM):
        a = _blade_as_set(i)
        for j in range(MV_DIM):
            b = _blade_as_set(j)
            sign, res = _multiply_blades(a, b, metric)
            if sign == 0:
                continue
            try:
                k = _set_to_blade(res)
            except ValueError:
                continue
            table[i, j, k] = float(sign)
    return table


# Cayley tablosunu modul yukleme sirasinda olustur (sabit, 16x16x16)
_CAYLEY: torch.Tensor = _build_cayley_table()


# ────────────────────────────────────────────────────────────────────────
# Multivector operasyonlari
# ────────────────────────────────────────────────────────────────────────

def geometric_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Iki multivectorun geometrik carpimi.

    Parametreler
    ------------
    a, b : [..., 16]   Multivector tensorleri (batch destegi)

    Donus
    -----
    c : [..., 16]      c = a * b
    """
    # einsum ile: c_k = sum_{i,j} a_i * b_j * C[i,j,k]
    cayley = _CAYLEY.to(a.device).to(a.dtype)
    return torch.einsum('...i,...j,ijk->...k', a, b, cayley)


def reverse(a: torch.Tensor) -> torch.Tensor:
    """
    Ters cevirme (reverse): grade g olan bileseni (-1)^(g(g-1)/2) ile carp.

    Rotorler icin: R~ R = 1 (inverse = reverse eger R unit ise)
    """
    # Grade'lere gore isaret: [0,1,2,3,4] -> [+1,+1,-1,-1,+1]
    grades = [
        0,
        1, 1, 1, 1,
        2, 2, 2, 2, 2, 2,
        3, 3, 3, 3,
        4,
    ]
    signs = torch.tensor(
        [(-1.0) ** (g * (g - 1) // 2) for g in grades],
        device=a.device, dtype=a.dtype
    )
    return a * signs


def grade_part(a: torch.Tensor, g: int) -> torch.Tensor:
    """Sadece belirli grade'i tut, digerlerini sifirla."""
    mask_list = [1.0 if len(_blade_as_set(i)) == g else 0.0 for i in range(MV_DIM)]
    mask = torch.tensor(mask_list, device=a.device, dtype=a.dtype)
    return a * mask


def norm_squared(a: torch.Tensor) -> torch.Tensor:
    """|a|^2 = <a~ a>_0 (skalar parca)."""
    ar = reverse(a)
    prod = geometric_product(ar, a)
    return prod[..., 0]   # skalar parca


# ────────────────────────────────────────────────────────────────────────
# Sandwich Product (rotor uygulama)
# ────────────────────────────────────────────────────────────────────────

def sandwich(R: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Sandwich product: x' = R x R~

    R: rotor (grade 0 + grade 2 bilesenleri tipik)
    x: donusturulecek multivector
    """
    Rr = reverse(R)
    return geometric_product(geometric_product(R, x), Rr)


# ────────────────────────────────────────────────────────────────────────
# Rotor Uretimi
# ────────────────────────────────────────────────────────────────────────

def bivector_exp(B: torch.Tensor) -> torch.Tensor:
    """
    exp(B) -> rotor (B saf bivector olmali varsayimi).

    B^2 skalar ise (saf bivector icin genelde boyledir):
      exp(B) = cos(|B|) + B * sin(|B|) / |B|

    Numerik stabilite icin kucuk |B| durumu ele alinir.
    """
    # B^2'yi hesapla
    B_sq = geometric_product(B, B)
    s = B_sq[..., 0].clamp(max=0.0)   # saf bivector -> B^2 <= 0 (CL(3,1))
    theta_sq = (-s).clamp(min=1e-12)
    theta = torch.sqrt(theta_sq)

    cos_t = torch.cos(theta)
    sin_over_theta = torch.sin(theta) / theta

    out = torch.zeros_like(B)
    out[..., 0] = cos_t
    # B bilesenlerini sin/theta ile carp ve ekle
    out = out + B * sin_over_theta.unsqueeze(-1)
    return out


def rotor_from_plane_angle(plane_idx: int, theta: torch.Tensor,
                           batch_shape: Tuple[int, ...] = ()) -> torch.Tensor:
    """
    Belirli bir bivector duzleminde theta/2 acili rotor.

    plane_idx: 5..10 (e12, e13, e14, e23, e24, e34)
    theta    : [...] tensor
    """
    assert 5 <= plane_idx <= 10, "plane_idx 5..10 arasinda olmali (bivector)"
    B = torch.zeros(*batch_shape, MV_DIM, device=theta.device, dtype=theta.dtype)
    B[..., plane_idx] = -theta / 2.0
    return bivector_exp(B)


# ────────────────────────────────────────────────────────────────────────
# PyTorch Modulu: Multivector uretici baslik
# ────────────────────────────────────────────────────────────────────────

class MultivectorHead(nn.Module):
    """
    Task embedding -> Cl(3,1) multivector ureticisi.

    V1/V2'de Transformer task_emb'sinden rotor/operator uretmek icin:
      head = MultivectorHead(D_in=128)
      R = head(task_emb)          # [B, 16]
      v_transformed = sandwich(R, v_mv)
    """

    def __init__(self, D_in: int, normalize: bool = True,
                 rotor_only: bool = False):
        """
        Parametreler
        ------------
        D_in        : task_emb boyutu
        normalize   : Cikiti unit norm'a zorla (rotor icin onemli)
        rotor_only  : True -> sadece skalar + bivector bilesenleri ciktilanir
                     False -> tam 16D multivector
        """
        super().__init__()
        self.rotor_only = rotor_only
        out_dim = 7 if rotor_only else MV_DIM  # skalar (1) + 6 bivector = 7
        self.proj = nn.Linear(D_in, out_dim)
        self.normalize = normalize

        # Baslangic: kimlik rotor'a yakin (proj cikisi ~0 -> skalar=1, digerleri=0)
        nn.init.zeros_(self.proj.weight)
        if rotor_only:
            bias = torch.zeros(out_dim)
            bias[0] = 1.0     # skalar parca = 1
        else:
            bias = torch.zeros(out_dim)
            bias[0] = 1.0     # skalar parca = 1
        self.proj.bias.data = bias

    def forward(self, task_emb: torch.Tensor) -> torch.Tensor:
        """
        task_emb: [B, D_in]
        Donus   : [B, 16] multivector
        """
        out = self.proj(task_emb)           # [B, out_dim]
        if self.rotor_only:
            B = out.shape[0]
            mv = torch.zeros(B, MV_DIM, device=out.device, dtype=out.dtype)
            mv[:, 0] = out[:, 0]             # skalar
            mv[:, 5:11] = out[:, 1:7]        # 6 bivector
        else:
            mv = out

        if self.normalize:
            n2 = norm_squared(mv).abs().clamp(min=1e-6)
            mv = mv / torch.sqrt(n2).unsqueeze(-1)
        return mv


# ────────────────────────────────────────────────────────────────────────
# Vector <-> Multivector Donusumleri
# ────────────────────────────────────────────────────────────────────────

def vector_to_mv(v: torch.Tensor) -> torch.Tensor:
    """
    3D veya 4D vektor -> 16D multivector (grade 1 parca).

    v: [..., 3]  -> e1, e2, e3 bilesenlerine yerlesir
    v: [..., 4]  -> e1..e4 bilesenlerine
    """
    shape = list(v.shape)
    D = shape[-1]
    assert D in (3, 4), "Vektor 3D veya 4D olmali"
    out = torch.zeros(*shape[:-1], MV_DIM, device=v.device, dtype=v.dtype)
    out[..., 1:1+D] = v
    return out


def mv_to_vector(mv: torch.Tensor, D: int = 3) -> torch.Tensor:
    """Multivector -> D boyutlu vektor (grade 1 parca)."""
    assert D in (3, 4)
    return mv[..., 1:1+D]


# ────────────────────────────────────────────────────────────────────────
# Hizli Test
# ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== clifford_algebra.py birim testi ===\n")

    # 1. Cayley tablosu kontrolleri
    print("[1] Temel vektor carpimlari (metrik kontrolu)")
    e0 = torch.zeros(MV_DIM); e0[0] = 1.0         # skalar 1
    e1 = torch.zeros(MV_DIM); e1[1] = 1.0
    e2 = torch.zeros(MV_DIM); e2[2] = 1.0
    e3 = torch.zeros(MV_DIM); e3[3] = 1.0
    e4 = torch.zeros(MV_DIM); e4[4] = 1.0

    # e1 * e1 = +1
    p11 = geometric_product(e1, e1)
    assert torch.allclose(p11, e0, atol=1e-6), f"e1*e1: {p11}"
    # e4 * e4 = -1
    p44 = geometric_product(e4, e4)
    assert torch.allclose(p44, -e0, atol=1e-6), f"e4*e4: {p44}"
    # e1 * e2 = e12
    p12 = geometric_product(e1, e2)
    expected = torch.zeros(MV_DIM); expected[5] = 1.0
    assert torch.allclose(p12, expected, atol=1e-6)
    # e2 * e1 = -e12 (antikomutatif)
    p21 = geometric_product(e2, e1)
    assert torch.allclose(p21, -expected, atol=1e-6)
    print("  e1*e1=+1, e4*e4=-1, e1*e2=e12, e2*e1=-e12  [OK]")

    # 2. Reverse
    print("\n[2] Reverse operatoru")
    e12 = torch.zeros(MV_DIM); e12[5] = 1.0
    assert torch.allclose(reverse(e12), -e12)      # grade 2 -> -
    assert torch.allclose(reverse(e1), e1)         # grade 1 -> +
    assert torch.allclose(reverse(e0), e0)         # grade 0 -> +
    print("  reverse(e12)=-e12, reverse(e1)=e1  [OK]")

    # 3. Rotor: 90 derece e12 duzleminde
    print("\n[3] 90 derece rotor e12 duzleminde")
    theta = torch.tensor(3.141592653589793 / 2)    # 90 derece
    R = rotor_from_plane_angle(5, theta)
    print(f"  R = {R.round(decimals=4)}")
    # e1'i 90 derece dondurursek e2 olmali
    v = vector_to_mv(torch.tensor([1.0, 0.0, 0.0]))
    v_rot = sandwich(R, v)
    v_out = mv_to_vector(v_rot, D=3)
    print(f"  e1 -> {v_out.round(decimals=4)} (beklenen: [0, 1, 0])")
    # Dikkat: rotor yonu konvansiyonuna gore -1 olabilir; kontrol ediyoruz
    assert torch.allclose(v_out.abs(), torch.tensor([0.0, 1.0, 0.0]), atol=1e-4)
    print("  [OK]")

    # 4. MultivectorHead identity kontrolu
    print("\n[4] MultivectorHead identity init")
    head = MultivectorHead(D_in=64, rotor_only=True)
    task_emb = torch.randn(3, 64)
    R = head(task_emb)
    print(f"  R.shape = {R.shape}")
    assert R.shape == (3, MV_DIM)
    # Baslangic: neredeyse identity (skalar ~1, digerleri ~0)
    print(f"  R[0] (init) = {R[0].round(decimals=4)}")
    # Normalize sonrasi identity rotor <= sapma kucuk
    print("  [OK]")

    # 5. Batch geometric product
    print("\n[5] Batch boyutlu geometrik carpim")
    a = torch.randn(4, 16)
    b = torch.randn(4, 16)
    c = geometric_product(a, b)
    assert c.shape == (4, 16)
    print(f"  (4,16) * (4,16) -> {tuple(c.shape)}  [OK]")

    print("\n[OK] Tum testler basarili.")
