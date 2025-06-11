from sympy import randprime
import secrets
from charm.toolbox.pairinggroup import PairingGroup, G1, pair
import hashlib

# Config
PRIME_BIT = 256
NUM_FILES = 10
NUM_BLOCKS = 10
PAIRING_TYPE = 'SS512'

# Generate Fp*
def generate_prime(bits: int) -> int:
    if bits < 2:
        raise ValueError('bit length must be > 2')
    return randprime(2 ** (bits - 1), 2 ** bits - 1)

def generate_fp(p: int) -> int:
    return secrets.randbelow(p - 1) + 1

# Pairing setup
def setup_pairing(bit_len: int):
    """
    Initialize pairing group and return group order as field modulus p,
    secret key sk in Z_p, pairing group instance, and generator g.
    """
    group = PairingGroup(PAIRING_TYPE)
    # Use the pairing group's order for polynomial arithmetic
    p = group.order()
    sk = secrets.randbelow(p - 1) + 1
    g = group.random(G1)
    return p, sk, group, g

# File Handling
def index_to_filename(i: int) -> str:
    out = []
    while True:
        i, r = divmod(i, 26)
        out.append(chr(97 + r))
        if i == 0:
            break
        i -= 1
    return 'file' + ''.join(reversed(out))

def sha256_int(data: bytes) -> int:
    return int.from_bytes(hashlib.sha256(data).digest(), 'big')

def block_point(file_idx: int, blk_idx: int, sk: int, p: int, retry_nonce: int = 0):
    while True:
        fb = str(file_idx).encode()
        bb = str(blk_idx).encode()
        sb = str(sk).encode()
        nb = str(retry_nonce).encode()
        x = sha256_int(fb + bb + nb) % p
        y = sha256_int(fb + bb + sb + nb) % p
        if x != 0:
            return x, y
        retry_nonce += 1

def generate_dataset(num_files: int, blocks_per_file: int, sk: int, p: int):
    dataset = []
    used_xs = set()
    for fi in range(num_files):
        entry = {
            'name': index_to_filename(fi),
            'idx': fi + 1,
            'blocks': []
        }
        for bi in range(blocks_per_file):
            retry = 0
            while True:
                x, y = block_point(entry['idx'], bi, sk, p, retry)
                if x not in used_xs:
                    used_xs.add(x)
                    break
                retry += 1
            entry['blocks'].append({'blk': bi, 'x': x, 'y': y})
        dataset.append(entry)
    return dataset

# polynomial operations
def modinv(a: int, p: int) -> int:
    return pow(a, -1, p)

def poly_add(A, B, p):
    m = max(len(A), len(B))
    out = [0] * m
    for i, c in enumerate(A): out[i] = (out[i] + c) % p
    for i, c in enumerate(B): out[i] = (out[i] + c) % p
    while out and out[-1] == 0: out.pop()
    return out or [0]

def poly_mul(A, B, p):
    out = [0] * (len(A) + len(B) - 1)
    for i, a in enumerate(A):
        for j, b in enumerate(B):
            out[i + j] = (out[i + j] + a * b) % p
    return out

def poly_eval(coeffs, x, p):
    acc = 0
    for c in reversed(coeffs): acc = (acc * x + c) % p
    return acc

def poly_from_points(points, p):
    poly = [0]
    for i, (xi, yi) in enumerate(points):
        numer = [1]
        denom = 1
        for j, (xj, _) in enumerate(points):
            if i == j: continue
            numer = poly_mul(numer, [(-xj) % p, 1], p)
            denom = (denom * ((xi - xj) % p)) % p
        factor = (yi * modinv(denom, p)) % p
        term = [(c * factor) % p for c in numer]
        poly = poly_add(poly, term, p)
    return poly or [0]

def synthetic_division(coeffs, r, p):
    """Divide φ(x) by (x - r) over F_p. Returns (quotient_coeffs, remainder)."""
    # φ(x) = a0 + a1 x + ... + a_n x^n
    n = len(coeffs) - 1  # degree
    # quotient will have degree n-1
    quotient = [0] * n
    # synthetic division: b[n-1] = a_n
    quotient[n-1] = coeffs[n] % p
    # b[k-1] = a_k + r * b[k]
    for k in range(n-1, 0, -1):
        quotient[k-1] = (coeffs[k] + r * quotient[k]) % p
    # remainder = a0 + r * b[0]
    remainder = (coeffs[0] + r * quotient[0]) % p
    return quotient, remainder

# polycommit functions
def public_parameters(g, sk, t, p, group):
    g_pows = [group.init(G1, 1)]
    g_sk = g ** sk
    g_pows.append(g_sk)
    for _ in range(2, t + 1): g_sk = g_sk ** sk; g_pows.append(g_sk)
    return g_pows

def commit_polynomial(coeffs, sk, g, p, group):
    return g ** poly_eval(coeffs, sk, p)

def create_witness(coeffs, x_i, y_i, sk, g, p, group):
    # φ(x) - y_i를 (x - x_i)로 나누어 목함수 생성
    shifted = coeffs.copy()
    shifted[0] = (shifted[0] - y_i) % p
    quotient, remainder = synthetic_division(shifted, x_i, p)
    assert remainder == 0, 'φ(x_i) - y_i must be divisible by (x - x_i)'
    return g ** poly_eval(quotient, sk, p)

def verify_eval(commitment, x_i, y_i, witness, g, sk, p, group):
    """
    Verify that the opening (witness) proves φ(x_i) = y_i under the commitment C.
    Checks e(C, g) == e(witness, g^{sk - x_i}) * e(g, g)^{y_i}.
    """
    # Compute pairings using standalone 'pair' function
    lhs = pair(commitment, g)
    # g^{sk - x_i}
    g_exp = g ** ((sk - x_i) % p)
    # pairing(witness, g_exp)
    rhs = pair(witness, g_exp)
    # multiply by e(g, g)^{y_i}
    rhs *= pair(g, g) ** y_i
    return lhs == rhs

# main execution
if __name__ == '__main__':
    p, sk, group, g = setup_pairing(PRIME_BIT)
    print(f"Field prime p  = {p}\nsecret sk      = {sk}")

    dataset = generate_dataset(NUM_FILES, NUM_BLOCKS, sk, p)
    points = [(blk['x'], blk['y']) for entry in dataset for blk in entry['blocks']]
    print(f"Total points   = {len(points)} (degree ≤ {len(points) - 1})")

    φ_coeffs = poly_from_points(points, p)
    C = commit_polynomial(φ_coeffs, sk, g, p, group)
    print('Commitment   C =', C)

    sample = dataset[0]['blocks'][0]
    x_i, y_i = sample['x'], sample['y']
    w_i = create_witness(φ_coeffs, x_i, y_i, sk, g, p, group)
    ok = verify_eval(C, x_i, y_i, w_i, g, sk, p, group)
    print(f"VerifyEval for block (x={x_i}) →", '✔︎ OK' if ok else '✘ failed')
