import argparse, os, random, math, json
from pathlib import Path

import numpy as np
import cv2, pywt, torch, torch.nn as nn, torch.optim as optim
import FrEIA.framework as Ff
import FrEIA.modules as Fm
from pycocotools.coco import COCO
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

# Utility Helper

# Convert ASCII string to a list of bits
def ascii_to_bits(text: str):
    return [int(bit) for char in text.encode("ascii") for bit in f"{char:08b}"[::-1]]

# Convert list of bits back to ASCII string
def bits_to_ascii(bits):
    chars = []
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8][::-1]
        if len(byte) < 8:
            break
        chars.append(chr(int("".join(map(str, byte)), 2)))
    return "".join(chars)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Wavelet Helpers
def dwt_lh_hl(img_gray, wavelet="haar"):
    coeffs2 = pywt.dwt2(img_gray, wavelet)
    _, (LH, HL, _) = coeffs2
    return LH.astype(np.float32), HL.astype(np.float32)

def idwt_from_bands(LL, LH, HL, HH, wavelet="haar"):
    return pywt.idwt2((LL, (LH, HL, HH)), wavelet)


# COCO Dataset that yields (LH, HL) tensors

class CocoDWT(torch.utils.data.Dataset):
    def __init__(self, coco_root: str, ann_file:str, size: int = 256):
        self.img_dir = Path(coco_root)
        self.coco = COCO(ann_file)
        self.ids = self.coco.getImgIds()
        self.size = size

    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        info = self.coco.loadImgs(self.ids[idx])[0]
        img_path = self.img_dir / info["file_name"]
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(img_path)
        img = cv2.cvtColor(cv2.resize(img, (self.size, self.size)), cv2.COLOR_BGR2GRAY)
        LH, HL = dwt_lh_hl(img)
        x = np.stack([LH / 255.0, HL / 255.0], axis = 0)
        return torch.from_numpy(x)
    
# INN definition

def subnet_constructor(c_in, c_out, f=256):
    return nn.Sequential(
        nn.Conv2d(c_in, f, 3, padding=1), nn.ReLU(),
        nn.Conv2d(f, f, 1), nn.ReLu(), 
        nn.Conv2d(f, c_out, 3, padding=1))

def build_inn(channels: int, h: int, w: int, blocks: int = 8):
    nodes = [Ff.InputNode(channels, h, w, name="inp")]
    for k in range(blocks):
        nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,
                             {"subnet_constructor": subnet_constructor,
                              "clamp": 1.0},
                              name=f"cb{k}"))
        nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {"seed": k}, name=f"perm{k}"))
    nodes.append(Ff.OutputNode(nodes[-1], name="out"))
    return Ff.ReversibleGraphNet(nodes, verbose=False)

# Training Loop

def train_inn(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cup else "cpu")
    set_seed(args.seed)

    ds = CocoDWT(args.coco_root, args.ann_file, args.size)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)

    _, c, h, w = nex(iter(dl)).shape
    inn = build_inn(c, h, w, args.blocks).to(device)
    opt = optim.Adam(inn.parameters(), lr=args.lr)

    nll_coef = 0.5 * math.log(2 * math.pi)

    for epoch in range(1, args.epochs + 1):
        inn.train()
        epoch_loss = 0.0
        for x in tqdm(dl, desc=f"Epoch {epoch}/{args.epochs}"):
            x = x.to(device)
            z, log_jac = inn(x)
            loss = (0.5 * z.pow(2) + nll_coef - log_jac).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * x.size(0)
        print(f"Epoch {epoch}: NLL = {epoch_loss / len(ds):.4f}")
    torch.save({"state_dict": inn.state_dict(), "in_shape": (c, h, w)}, args.save)
    print(f"Model saved to {args.save}")

#Watermark Embedding & Extraction

@torch.no_grad()
def embed(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    ckpt = torch.load(args.model, map_location=device)
    inn = build_inn(*ckpt["in_shape"]).to(device)
    inn.load_state_dict(ckpt["state_dict"])
    inn.eval()

    img = cv2.imread(args.input)
    h0, w0 = img.shape[:2]
    gray = cv2.cvtColor(cv2.resize(img, (args.size, args.size)), cv2.COLOR_BGR2GRAY)
    LL, (LH, HL, HH) = pywt.dwt2(gray, "haar")

    x = torch.tensor(np.stack([LH, HL]) / 255.0, dtype=torch.float32).unsqueeze(0).to(device)
    z, _ = inn(x)
    z_flat = z.view(-1)

    bits = ascii_to_bits(args.watermark)
    rng = np.random.default_rng(args.key)
    idx = rng.choice(z_flat.numel(), size=len(bits), replace=False)

    delta = 0.25  # embedding strength
    for pos, bit in zip(idx, bits):
        z_flat[pos] = torch.round(z_flat[pos]) + (delta if bit else -delta)

    z_emb = z_flat.view_as(z)
    x_emb, _ = inn.inverse(z_emb)
    LH_m, HL_m = (x_emb[0, 0].cpu().numpy() * 255.0), (x_emb[0, 1].cpu().numpy() * 255.0)

    wm_img = idwt_from_bands(LL, LH_m, HL_m, HH)
    wm_img = cv2.resize(wm_img, (w0, h0))
    wm_img = np.clip(wm_img, 0, 255).astype(np.uint8)
    cv2.imwrite(args.output, wm_img)

    meta = {"idx": idx.tolist(), "length": len(bits), "h0": h0, "w0": w0}
    json_path = args.output + ".json"
    with open(json_path, "w") as fp:
        json.dump(meta, fp)
    print(f"Watermarked image saved to {args.output}; key data in {json_path}")


@torch.no_grad()
def extract(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    ckpt = torch.load(args.model, map_location=device)
    inn = build_inn(*ckpt["in_shape"]).to(device)
    inn.load_state_dict(ckpt["state_dict"])
    inn.eval()

    meta = json.load(open(args.meta)) if args.meta else {"idx": None, "length": args.length}
    idx = meta["idx"] if meta["idx"] is not None else None

    img = cv2.imread(args.input)
    gray = cv2.cvtColor(cv2.resize(img, (args.size, args.size)), cv2.COLOR_BGR2GRAY)
    _, (LH, HL, _) = pywt.dwt2(gray, "haar")

    x = torch.tensor(np.stack([LH, HL]) / 255.0, dtype=torch.float32).unsqueeze(0).to(device)
    z, _ = inn(x)
    z_flat = z.view(-1)

    if idx is None:
        # Regenerate indices deterministically
        rng = np.random.default_rng(args.key)
        idx = rng.choice(z_flat.numel(), size=args.length, replace=False)

    bits = [(1 if (z_flat[p] - torch.round(z_flat[p]) > 0) else 0) for p in idx[:args.length]]
    wm = bits_to_ascii(bits)
    print("Recovered watermark:", wm)

#CLI
def main():
    p = argparse.ArgumentParser(description="INN‑based DWT watermarking")
    sub = p.add_subparsers(dest="cmd", required=True)

    # ---- train ----
    sp = sub.add_parser("train")
    sp.add_argument("--coco_root", required=True)
    sp.add_argument("--ann_file", required=True)
    sp.add_argument("--save", default="inn_dwt.pth")
    sp.add_argument("--size", type=int, default=256)
    sp.add_argument("--epochs", type=int, default=5)
    sp.add_argument("--batch", type=int, default=16)
    sp.add_argument("--blocks", type=int, default=8)
    sp.add_argument("--lr", type=float, default=1e-4)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--cpu", action="store_true")

    # ---- embed ----
    se = sub.add_parser("embed")
    se.add_argument("--model", required=True)
    se.add_argument("--input", required=True)
    se.add_argument("--watermark", required=True)
    se.add_argument("--output", default="watermarked.png")
    se.add_argument("--size", type=int, default=256)
    se.add_argument("--key", type=int, default=123)
    se.add_argument("--cpu", action="store_true")

    # ---- extract ----
    sx = sub.add_parser("extract")
    sx.add_argument("--model", required=True)
    sx.add_argument("--input", required=True)
    sx.add_argument("--length", type=int, default=48, help="bit length (8×chars)")
    sx.add_argument("--key", type=int, default=123)
    sx.add_argument("--meta", help="optional JSON produced during embedding")
    sx.add_argument("--size", type=int, default=256)
    sx.add_argument("--cpu", action="store_true")

    args = p.parse_args()

    if args.cmd == "train":
        train_inn(args)
    elif args.cmd == "embed":
        embed(args)
    elif args.cmd == "extract":
        extract(args)


if __name__ == "__main__":
    main()
