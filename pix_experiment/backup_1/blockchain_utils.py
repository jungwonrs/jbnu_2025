import ecdsa
import hashlib
import os

PRIVATE_KEY_FILE = "blockchain_private.key"

def generate_private_key(force_new=False):
    if os.path.exists(PRIVATE_KEY_FILE) and not force_new:
        print(f"🔑 기존 블록체인 개인키 '{PRIVATE_KEY_FILE}'를 로드합니다.")
        with open(PRIVATE_KEY_FILE, "rb") as f:
            private_key = ecdsa.SigningKey.from_string(f.read(), curve=ecdsa.SECP256k1)
    else:
        print(f"🔑 새로운 블록체인 개인키를 생성하여 '{PRIVATE_KEY_FILE}'에 저장합니다.")
        private_key = ecdsa.SigningKey.generate(curve=ecdsa.SECP256k1)
        with open(PRIVATE_KEY_FILE, "wb") as f:
            f.write(private_key.to_string())
    return private_key

def load_private_key():
    if not os.path.exists(PRIVATE_KEY_FILE):
        raise FileNotFoundError(f"'{PRIVATE_KEY_FILE}'이 없습니다. 먼저 키를 생성해주세요.")
    with open(PRIVATE_KEY_FILE, "rb") as f:
        return ecdsa.SigningKey.from_string(f.read(), curve=ecdsa.SECP256k1)

def private_key_to_seed(private_key):
    """개인키를 고유한 정수 시드 값으로 변환합니다."""
    key_bytes = private_key.to_string()
    seed = int.from_bytes(hashlib.sha256(key_bytes).digest()[:4], 'big')
    return seed