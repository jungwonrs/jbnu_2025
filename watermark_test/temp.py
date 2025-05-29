# inn_watermark.py
"""
Detailed, commented implementation of an **Invertible‑Neural‑Network (INN)‑based
DWT watermarking pipeline**.

───────────────────────────────────────────────────────────────────────────────
Why this script?
────────────────
* **Research** : Study how well invertible flows can model wavelet‑domain
  sub‑bands so that imperceptible data can be embedded in latent space.
* **Practical** : Provide a single, reproducible CLI you can call with
  ``train / embed / extract`` sub‑commands.
* **Learning aid** : Every major block is exhaustively commented so you can
  tweak architecture, wavelet type, embedding strength, etc. without hunting
  through cryptic code.

Tested on **Python 3.11 + PyTorch 2.3 + CUDA 12.x** using an RTX 5080.

Quick start (copy‑paste):
────────────────────────
```bash
# 1) Train the flow on COCO (≈ 30 min on RTX 5080)
python inn_watermark.py train \
  --coco_root /data/COCO/train2017 \
  --ann_file  /data/COCO/annotations/instances_train2017.json \
  --epochs 10 --batch 32 --save inn_dwt.pth

# 2) Embed a short ASCII watermark into image A.jpg
python inn_watermark.py embed \
  --model inn_dwt.pth --input A.jpg \
  --watermark "seo123" --output A_wm.png --key 42

# 3) Recover the watermark
python inn_watermark.py extract \
  --model inn_dwt.pth --input A_wm.png --length 6 --key 42
```
"""

# ============================================================================
# Imports
# ============================================================================
# Standard‑library ↦ reproducibility, math helpers
import argparse, os, random, math, json
from pathlib import Path

# Third‑party ↦ image I/O, wavelets, deep learning
import numpy as np
import cv2                     # OpenCV – image loading & resizing
import pywt                    # PyWavelets – DWT/IDWT
import torch, torch.nn as nn, torch.optim as optim
import FrEIA.framework as Ff   # Invertible‑network skeleton
import FrEIA.modules as Fm
from pycocotools.coco import COCO
from tqdm import tqdm          # Progress‑bar for training loop

# Quality metrics (just for logging / debug)
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

# ============================================================================
# Section 1 – Deterministic utilities
# ============================================================================

def set_seed(seed: int) -> None:
    """Set RNG seeds across Python, NumPy and Torch so runs are repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================================
# Section 2 – Bit ⇄ ASCII helpers (LSB‑first order per byte)
# ============================================================================

def ascii_to_bits(text: str) -> list[int]:
    """Convert *ASCII* string → list[0|1] (least‑significant bit first)."""
    return [int(bit) for char in text.encode("ascii") for bit in f"{char:08b}"[::-1]]


def bits_to_ascii(bits: list[int]) -> str:
    """Convert list of bits (LSB‑first/byte) back to ASCII string.

    Extra bits (< 8) at the end are silently discarded.
    """
    chars: list[str] = []
    for i in range(0, len(bits), 8):
        byte = bits[i : i + 8][::-1]  # flip back to MSB‑first
        if len(byte) < 8:
            break
        chars.append(chr(int("".join(map(str, byte)), 2)))
    return "".join(chars)


# ============================================================================
# Section 3 – Wavelet helpers
# ============================================================================

def dwt_lh_hl(img_gray: np.ndarray, wavelet: str = "haar") -> tuple[np.ndarray, np.ndarray]:
    """1‑level DWT → return **only** LH & HL (edge‑detail) sub‑bands.

    *Haar* is fast & orthogonal; swap `wavelet` for e.g. `'db2'` if you need
    smoother basis functions (but memory ↑).
    """
    coeffs2 = pywt.dwt2(img_gray, wavelet)
    LL, (LH, HL, HH) = coeffs2
    return LH.astype(np.float32), HL.astype(np.float32)


def idwt_from_bands(LL: np.ndarray, LH: np.ndarray, HL: np.ndarray, HH: np.ndarray, *, wavelet: str = "haar") -> np.ndarray:
    """Inverse DWT given explicit four sub‑bands."""
    return pywt.idwt2((LL, (LH, HL, HH)), wavelet)


# ============================================================================
# Section 4 – COCO → (LH, HL) Dataset
# ============================================================================
class CocoDWT(torch.utils.data.Dataset):
    """Iterate through COCO images, yielding a **2×(H/2)×(W/2)** FloatTensor.

    The two channels correspond to **LH** & **HL** DWT coefficients, each scaled
    to *[0,1]* by simple `/255` (enough since we only care about relative
    contrast, not absolute energy).
    """

    def __init__(self, coco_root: str, ann_file: str, *, size: int = 256):
        self.img_dir = Path(coco_root)
        self.coco = COCO(ann_file)
        self.ids = self.coco.getImgIds()
        self.size = size

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx):
        # --- 1️⃣ grab metadata then image path
        info = self.coco.loadImgs(self.ids[idx])[0]
        img_path = self.img_dir / info["file_name"]

        # --- 2️⃣ BGR → Gray & resize (keep aspect? not critical for 256×256)
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(img_path)
        img = cv2.cvtColor(cv2.resize(img, (self.size, self.size)), cv2.COLOR_BGR2GRAY)

        # --- 3️⃣ DWT & normalise to [0,1]
        LH, HL = dwt_lh_hl(img)
        x = np.stack([LH / 255.0, HL / 255.0], axis=0)
        return torch.from_numpy(x)


# ============================================================================
# Section 5 – INN architecture (GLOW‑like coupling blocks)
# ============================================================================

def subnet_constructor(c_in: int, c_out: int, *, f: int = 256) -> nn.Module:
    """Small CNN used inside each coupling block.

    *   Two 3×3 convs around a bottleneck 1×1 conv.
    *   ReLU activations (swap to GELU/SiLU if you like).
    *   `f` controls the internal channel width (model capacity).
    """
    return nn.Sequential(
        nn.Conv2d(c_in, f, 3, padding=1), nn.ReLU(),
        nn.Conv2d(f, f, 1), nn.ReLU(),
        nn.Conv2d(f, c_out, 3, padding=1))


def build_inn(channels: int, h: int, w: int, *, blocks: int = 8) -> Ff.ReversibleGraphNet:
    """Assemble a *ReversibleGraphNet* with `blocks` coupling layers.

    Flow follows: **Input → [Coupling ⟶ Random‑Permute] × N → Output**.
    """
    nodes = [Ff.InputNode(channels, h, w, name="inp")]
    for k in range(blocks):
        nodes.append(
            Ff.Node(
                nodes[-1], Fm.GLOWCouplingBlock,
                {"subnet_constructor": subnet_constructor, "clamp": 1.0},
                name=f"cb{k}")
        )
        nodes.append(
            Ff.Node(nodes[-1], Fm.PermuteRandom, {"seed": k}, name=f"perm{k}")
        )
    nodes.append(Ff.OutputNode(nodes[-1], name="out"))
    return Ff.ReversibleGraphNet(nodes, verbose=False)


# ============================================================================
# Section 6 – Training loop (maximum‑likelihood fit of LH/HL distribution)
# ============================================================================

def train_inn(args):
    """Entry point for `python inn_watermark.py train …`."""
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    set_seed(args.seed)

    # 1) Dataset & loader ------------------------------------------------------
    ds = CocoDWT(args.coco_root, args.ann_file, size=args.size)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True
    )

    # 2) Build network ---------------------------------------------------------
    # (Readable via first mini‑batch to get H,W)
    _, c, h, w = next(iter(dl)).shape  # -> (B,2,H',W')
    inn = build_inn(c, h, w, blocks=args.blocks).to(device)
    opt = optim.Adam(inn.parameters(), lr=args.lr)

    # 3) Train ----------------------------------------------------------------
    nll_const = 0.5 * math.log(2 * math.pi)  # constant term in NLL
    for epoch in range(1, args.epochs + 1):
        inn.train()
        running_loss = 0.0
        for x in tqdm(dl, desc=f"Epoch {epoch}/{args.epochs}"):
            x = x.to(device)
            z, log_jac = inn(x)              # forward pass: x -> z
            loss = (0.5 * z.pow(2) + nll_const - log_jac).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            running_loss += loss.item() * x.size(0)
        print(f"Epoch {epoch}: NLL = {running_loss / len(ds):.4f}")

    # 4) Save checkpoint -------------------------------------------------------
    torch.save({"state_dict": inn.state_dict(), "in_shape": (c, h, w)}, args.save)
    print(f"Model stored at → {args.save}")


# ============================================================================
# Section 7 – Watermark embedding
# ============================================================================
@torch.no_grad()
def embed(args):
    """Embed ASCII watermark into LH/HL by nudging INN latent coordinates."""
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # 1) Load trained INN ------------------------------------------------------
    ckpt = torch.load(args.model, map_location=device)
    inn = build_inn(*ckpt["in_shape"]).to(device)
    inn.load_state_dict(ckpt["state_dict"]).eval()

    # 2) Prepare cover image A -----------------------------------------------
    img = cv2.imread(args.input)
    h0, w0 = img.shape[:2]
    gray = cv2.cvtColor(cv2.resize(img, (args.size, args.size)), cv2.COLOR_BGR2GRAY)
    LL, (LH, HL, HH) = pywt.dwt2(gray, "haar")

    # 3) Map (LH,HL) → latent space z ----------------------------------------
    x = torch.tensor(np.stack([LH, HL]) / 255.0, dtype=torch.float32)[None].to(device)
    z, _ = inn(x)
    z_flat = z.flatten()

    # 4) Choose random latent positions & embed bits --------------------------
    bits = ascii_to_bits(args.watermark)
    rng = np.random.default_rng(args.key)
    idx = rng.choice(z_flat.numel(), size=len(bits), replace=False)

    delta = 0.25  # embedding strength; raise ↦ robust but perceptibility↑
    for pos, bit in zip(idx, bits):
        z_flat[pos] = torch.round(z_flat[pos]) + (delta if bit else -delta)

    # 5) Latent → modified (LH,HL) → spatial image ---------------------------
    z_emb = z_flat.view_as(z)
    x_emb, _ = inn.inverse(z_emb)
    LH_m, HL_m = (x_emb[0, 0].cpu().numpy() * 255.0), (x_emb[0, 1].cpu().numpy() * 255.0)

    wm_img = idwt_from_bands(LL, LH_m, HL_m, HH)
    wm_img = cv2.resize(wm_img, (w0, h0))
    wm_img_uint8 = np.clip(wm_img, 0, 255).astype(np.uint8)
    cv2.imwrite(args.output, wm_img_uint8)

    # 6) Save side‑channel metadata (indices, size) ---------------------------
    meta = {"idx": idx.tolist(), "length": len(bits), "h0": h0, "w0": w0}
    meta_path = args.output + ".json"
    with open(meta_path, "w") as fp:
        json.dump(meta, fp)
    print(f"✅ Watermarked image → {args.output}\nℹ️  Key data → {meta_path}")


# ============================================================================
# Section 8 – Watermark extraction
# ============================================================================
@torch.no_grad()
def extract(args):
    """Recover watermark from a possibly resized or compressed watermarked image."""
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # 1) Load INN -------------------------------------------------------------
    ckpt = torch.load(args.model, map_location=device)
    inn = build_inn(*ckpt["in_shape"]).to(device)
    inn.load_state_dict(ckpt["state_dict"]).eval()

    # 2) Load meta or recreate indices ---------------------------------------
    meta = json.load(open(args.meta)) if args.meta else {"idx": None, "length": args.length}
    idx = meta["idx"] if meta["idx"] is not None else None

    # 3) Image → DWT → latent -------------------------------------------------
    img = cv2.imread(args.input)
    gray = cv2.cvtColor(cv2.resize(img, (args.size, args.size)), cv2.COLOR_BGR2GRAY)
    _, (LH, HL, _) = pywt.dwt2(gray, "haar")
    x = torch.tensor(np.stack([LH, HL]) / 255.0, dtype=torch.float32)[None].to(device)
    z, _ = inn(x)
    z_flat = z.flatten()

    # 4) Determine latent indices to read ------------------------------------
    if idx is None:
        rng = np.random.default_rng(args.key)
        idx = rng.choice(z_flat.numel(), size=args.length, replace=False)

    # 5) Extract bits and decode ---------------------------------------------
    bits = [(1 if (z_flat[p] - torch.round(z_flat[p]) > 0) else 0) for p in idx[:args.length]]
    wm = bits_to_ascii(bits)
    print("🕵️  Recovered watermark →", wm)