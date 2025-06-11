from sympy import randprime
import secrets
from charm.toolbox.pairinggroup import PairingGroup, G1, pair
import hashlib
from copy import deepcopy

# CONFIGURATION
PAIRING_TYPE     = 'SS512'
TOTAL_FILES      = 100
BLOCKS_PER_FILE  = 10
NUM_POLYNOMIALS  = 10
assert TOTAL_FILES % NUM_POLYNOMIALS == 0, "TOTAL_FILES must be divisible by NUM_POLYNOMIALS"
FILES_PER_POLY = TOTAL_FILES // NUM_POLYNOMIALS

# ────────────────────────────────────────────────────────
# HELPERS: modular arithmetic & polynomials
# ────────────────────────────────────────────────────────

def modinv(a: int, p: int) -> int:
    return pow(a, -1, p)

def poly_add(A, B, p):
    m = max(len(A), len(B)); out = [0]*m
    for i,a in enumerate(A): out[i] = (out[i] + a) % p
    for i,b in enumerate(B): out[i] = (out[i] + b) % p
    while len(out)>1 and out[-1] == 0: out.pop()
    return out

def poly_mul(A, B, p):
    out = [0]*(len(A)+len(B)-1)
    for i,a in enumerate(A):
        for j,b in enumerate(B): out[i+j] = (out[i+j] + a*b) % p
    while len(out)>1 and out[-1] == 0: out.pop()
    return out

def poly_eval(coeffs, x, p):
    res = 0
    for i, c in enumerate(coeffs):
        res = (res + c * pow(x, i, p)) % p
    return res

def syn_div(coeffs, r, p):
    acc = 0
    out = []
    for c in reversed(coeffs):
        acc = (acc * r + c) % p
        out.append(acc)
    rem = out[-1]
    quot = list(reversed(out[:-1]))
    return quot, rem

def poly_from_points(points, p):
    poly = [0]
    for i,(xi, yi) in enumerate(points):
        numer = [1]; denom = 1
        for j,(xj,_) in enumerate(points):
            if i == j: continue
            numer = poly_mul(numer, [(-xj) % p, 1], p)
            denom = (denom * ((xi - xj) % p)) % p
        invd = modinv(denom, p)
        term = [(c * yi * invd) % p for c in numer]
        poly = poly_add(poly, term, p)
    return poly

def sha256_int(data: bytes) -> int:
    return int.from_bytes(hashlib.sha256(data).digest(), 'big')

def make_point(fid: int, bid: int, sk: int, p: int) -> tuple[int,int]:
    seed = f"{fid}:{bid}".encode()
    x = sha256_int(seed) % p or 1
    y = sha256_int(seed + str(sk).encode()) % p
    return x, y

def gen_dataset(nf: int, nb: int, sk: int, p: int):
    used = set(); data = []
    for fid in range(1, nf+1):
        pts = []
        for bid in range(nb):
            x,y = make_point(fid, bid, sk, p)
            while x in used:
                x = sha256_int(f"{fid}:{bid}:{len(pts)}".encode()) % p or 1
                y = sha256_int(str(x).encode() + str(sk).encode()) % p
            used.add(x); pts.append((x,y))
        data.append(pts)
    return data

# ────────────────────────────────────────────────────────
# PEDERSEN SETUP & COMMIT
# ────────────────────────────────────────────────────────
def pedersen_setup(max_degree: int, group):
    order = group.order()
    alpha = secrets.randbelow(order - 1) + 1
    g = group.random(G1)
    h = group.random(G1)
    g_pows = [g ** pow(alpha, i, order) for i in range(max_degree+1)]
    h_pows = [h ** pow(alpha, i, order) for i in range(max_degree+1)]
    return alpha, g, h, g_pows, h_pows

def pedersen_commit(F_coeffs, R_coeffs, g_pows, h_pows, order):
    C = g_pows[0] ** 0  # group.init(G1,1)
    for i, a in enumerate(F_coeffs):
        if a: C *= g_pows[i] ** (a % order)
    for i, b in enumerate(R_coeffs):
        if b: C *= h_pows[i] ** (b % order)
    return C

# ────────────────────────────────────────────────────────
# OPEN & VERIFY
# ────────────────────────────────────────────────────────
def witness_standard(F_coeffs, R_coeffs, x, yF, r_val, g_pows, h_pows, p, group, order):
    psiF, remF = syn_div(F_coeffs, x, p)
    psiR, remR = syn_div(R_coeffs, x, p)
    g_pows = g_pows[:len(psiF)]
    h_pows = h_pows[:len(psiR)]
    Wf = group.init(G1,1)
    for i,c in enumerate(psiF):
        if c: Wf *= g_pows[i] ** (c % order)
    Wr = group.init(G1,1)
    for i,d in enumerate(psiR):
        if d: Wr *= h_pows[i] ** (d % order)
    return Wf, Wr

def verify_standard(C, x, yF, r_val, Wf, Wr, g, h, p, group, alpha, order):
    # C = g^{F(α)} * h^{R(α)}, yF = F(x), r_val = R(x)

    # 1) 분모 제거: C / (g^{yF} h^{r_val})
    g_inv = g ** ((order - (yF % order)) % order)
    h_inv = h ** ((order - (r_val % order)) % order)
    numerator = C * g_inv * h_inv  # = g^{ψ_F(α)*(α-x)} * h^{ψ_R(α)*(α-x)}

    # 2) 페어링 검증 베이스
    g_exp = g ** ((alpha - x) % order)

    # 3) 올바른 RHS: Wf, Wr 모두 g_exp 와 페어링
    lhs = pair(numerator, g)
    rhs = pair(Wf, g_exp) * pair(Wr, g_exp)

    print(f"[DEBUG] verify_standard: lhs={lhs}, rhs={rhs}, ok={lhs==rhs}")
    return lhs == rhs

def witness_trapdoor(F_coeffs, R_coeffs, x, yF, r_val, alpha, g, h, p, order):
    Fα = poly_eval(F_coeffs, alpha, p)
    Rα = poly_eval(R_coeffs, alpha, p)
    inv = modinv((alpha - x) % p, p)
    aF = ((Fα - yF) * inv) % order
    aR = ((Rα - r_val) * inv) % order
    return g ** aF, h ** aR

def verify_trapdoor(C, x, yF, r_val, Wf, Wr, alpha, g, h, p, order):
    part = (g ** (yF % order)) * (h ** (r_val % order))
    return C == part * (Wf ** ((alpha - x) % order)) * (Wr ** ((alpha - x) % order))

if __name__ == '__main__':
    group = PairingGroup(PAIRING_TYPE)
    p = group.order()
    sk = secrets.randbelow(p-1) + 1
    data = gen_dataset(TOTAL_FILES, BLOCKS_PER_FILE, sk, p)
    polys = []
    for i in range(NUM_POLYNOMIALS):
        files_group = data[i*FILES_PER_POLY:(i+1)*FILES_PER_POLY]
        points = [pt for file_pts in files_group for pt in file_pts]
        polys.append(poly_from_points(points, p))
    F = deepcopy(polys[0])
    for poly in polys[1:]: F = poly_mul(F, poly, p)
    R = [secrets.randbelow(p) for _ in range(len(F))]
    alpha, g, h, g_pows, h_pows = pedersen_setup(len(F)-1, group)
    order = group.order()
    C = pedersen_commit(F, R, g_pows, h_pows, order)
    x, _ = data[0][0]
    yF = poly_eval(F, x, p)
    r_val = poly_eval(R, x, p)
    Wf, Wr = witness_standard(F, R, x, yF, r_val, g_pows, h_pows, p, group, order)
    ok1 = verify_standard(C, x, yF, r_val, Wf, Wr, g, h, p, group, alpha, order)
    print("Standard verify →", "✔" if ok1 else "✘")
    Wf2, Wr2 = witness_trapdoor(F, R, x, yF, r_val, alpha, g, h, p, order)
    ok2 = verify_trapdoor(C, x, yF, r_val, Wf2, Wr2, alpha, g, h, p, order)
    print("Trapdoor verify →", "✔" if ok2 else "✘")
    print(len(F), len(g_pows))
    print(len(R), len(h_pows))

    F = [5, 2, 7]
    x = 3
    p = 13
    val = poly_eval(F, x, p)      # 9
    psiF, remF = syn_div(F, x, p) # remF == 9 나와야!
    print(f"poly_eval(F, {x}, {p}) = {val}, syn_div rem = {remF}")
