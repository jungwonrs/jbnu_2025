import random
import hashlib
FIELD_PRIME = 5831

#===========Polynomial Function==============================================
def generate_coeffs(secret_key):
    hash_hex = hashlib.sha256(secret_key.encode()).hexdigest()
    coeffs = [int(c, 16) for c in hash_hex]
    return coeffs

def calculate_polynomial(coeffs, x, mod = FIELD_PRIME):
    result = 0
    for i, coeff in enumerate(coeffs):
        term = coeff * pow(x, i, mod) if mod else coeff * (x ** i)
        result = (result + term) if not mod else (result + term) % mod
    return result
#===========================================================================
#==========Fuzzy Vault - LOCK===============================================
def lock(secret_key, A, r):
    coeffs = generate_coeffs(secret_key)
    R = []
    used_x = set()
    
    for a in A: # correct point
        R.append((a, calculate_polynomial(coeffs, a)))
        used_x.add(a)
        
    while len(R) < r: # chaff point
        x = random.randint(0, FIELD_PRIME - 1)
        if x in used_x: continue
        y = random.randint(0, FIELD_PRIME - 1)
        if y == calculate_polynomial(coeffs, x): continue
        R.append((x, y))
        used_x.add(x)
    return R
#===========================================================================
#==========Fuzzy Vault - UNLOCK=============================================
def unlock(vault, secret_key, B, threshold = 40):
    coeffs = generate_coeffs(secret_key)
    vault_dict = dict(vault)
    match_count = 0
    
    for b in B:
        if b in vault_dict and vault_dict[b] == calculate_polynomial(coeffs, b): match_count += 1
    
    return match_count >= threshold
#===========================================================================
#===============TEST========================================================
A = set(range(100))
B = set(random.sample(list(A), 50))
C = set(random.sample(list(A), 50))
C.update(random.sample(range(1000, 2000), 20)) 
D = set(random.sample(range(1000, 2000), 40))
vault = lock("secret_key", A, 500)

print("Vault Unlocking B with same secret key...")
if unlock(vault, "secret_key", B): print("Vault unlocked.\n")
else: print("You can't unlock vault.\n")

print("Vault Unlocking B with different secret key...")
if unlock(vault, "secret_key2", B): print("Vault unlocked.\n")
else: print("You can't unlock vault.\n")

print("Vault Unlocking C with same secret key...")
if unlock(vault, "secret_key", C): print("Vault unlocked.\n")
else: print("You can't unlock vault.\n")

print("Vault Unlocking C with different secret key...")
if unlock(vault, "secret_key2", C): print("Vault unlocked.\n")
else: print("You can't unlock vault.\n")

print("Vault Unlocking D with same secret key...")
if unlock(vault, "secret_key", D): print("Vault unlocked.\n")
else: print("You can't unlock vault.\n")
#===========================================================================