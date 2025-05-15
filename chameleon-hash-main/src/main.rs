use std::{fmt, time};

use num_bigint::{BigInt, ToBigUint};
use num_primes::{BigUint, Generator};
use num_traits::{One, Zero};

/// Chameleon hash function parameters
#[derive(Clone)]
pub struct ChameleonHash {
    p: BigUint,     // Large prime p
    q: BigUint,     // Order of the group (p = kq + 1)
    g: BigUint,     // Generator of order q
    x: BigUint,     // Private key (trapdoor)
    inv_x: BigUint, // Private key (trapdoor)
    y: BigUint,     // Public key y = g^x mod p
}

fn mod_inverse(x: &BigInt, n: &BigInt) -> Option<BigInt> {
    let (gcd, inv, _) = extended_gcd(x, n);

    // If gcd is not 1, inverse does not exist
    if gcd != BigInt::one() {
        return None;
    }

    // Ensure the inverse is positive
    let inv = (inv % n + n) % n;
    Some(inv)
}

/// Extended Euclidean Algorithm to find gcd and the coefficients for Bézout's identity.
/// It returns (gcd, x, y) such that gcd(a, b) = a*x + b*y.
fn extended_gcd(a: &BigInt, b: &BigInt) -> (BigInt, BigInt, BigInt) {
    if b.is_zero() {
        return (a.clone(), BigInt::one(), BigInt::zero());
    }

    let (gcd, x1, y1) = extended_gcd(b, &(a % b));
    let x = y1.clone();
    let y = x1 - (a / b) * y1;

    (gcd, x, y)
}

impl fmt::Display for ChameleonHash {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "ChameleonHash:\n  p: {}\n  q: {}\n  g: {}\n  x: {}\n  inv_x:{}\n  y: {}",
            self.p, self.q, self.g, self.x, self.inv_x, self.y
        )
    }
}

impl fmt::Debug for ChameleonHash {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "ChameleonHash:\n  p: {}\n  q: {}\n  g: {}\n  x: {}\n  inv_x:{}\n  y: {}",
            self.p, self.q, self.g, self.x, self.inv_x, self.y
        )
    }
}

impl ChameleonHash {
    /// Generate new parameters for the chameleon hash
    pub fn new(bits: usize) -> Self {
        let (p,g) = match bits {
            2048 => ("27920844145687281913857878805326597913957107490217608907572037738159537609281739892246247603665214499058619426958597190388458161715725627232301555503414007569124774391232828175307061795756908938159675529593026298777258045483005578932932125388253738555349149461769891676907380887082080307921773358399189533494435997834699554747102156518261852713920179058527455161435390849849488051009777451429513412843013082765645590391492284467424677299915189902591916711206589684391040146499947826793314813714762937720483138761849235911115094911454041366580074942541493715265895971730845150860064392085134138368635762014155394256023"
                .parse::<BigUint>()
                     .unwrap(), 2.to_biguint().unwrap()),
            1024 => ("179769313486231590770839156793787453197860296048756011706444423684197180216158519368947833795864925541502180565485980503646440548199239100050792877003355816639229553136239076508735759914822574862575007425302077447712589550957937778424442426617334727629299387668709205606050270810842907692932019128194467627007"
                .parse::<BigUint>()
                .unwrap(), 2.to_biguint().unwrap()),
            // 160 bits
            160 => ("1110993139090855285586132226382129815148739954519"
                .parse::<BigUint>()
                    .unwrap(), 5.to_biguint().unwrap()),
            bits => (Generator::safe_prime(bits), 5.to_biguint().unwrap()),
        } ;

        let one = BigUint::one();
        let two = &one + &one;

        let q = (&p - &one) / &two;
        // let g = 5.to_biguint().unwrap();

        let x = Generator::new_uint(q.bits() - 1);
        let y = g.modpow(&x, &p);
        let ix = BigInt::from_biguint(num_bigint::Sign::Plus, x.clone());
        let iq = BigInt::from_biguint(num_bigint::Sign::Plus, q.clone());
        let inv_x = mod_inverse(&ix, &iq)
            .expect("failed to compute inverse of x in q")
            .to_biguint()
            .expect("failed to convert inverse to BigUint");

        assert!(
            (&inv_x * &x) % &q == one,
            "failed to verify inverse of x in q: {}",
            (&inv_x * &x) % &q
        );

        ChameleonHash {
            p,
            q,
            g,
            x,
            y,
            inv_x,
        }
    }

    pub fn hash(&self, m: &BigUint, r: &BigUint) -> BigUint {
        let gm = self.g.modpow(m, &self.p);
        let yr = self.y.modpow(r, &self.p);
        (gm * yr) % &self.p
    }

    pub fn find_collision(&self, m1: &BigUint, r: &BigUint, m2: &BigUint) -> BigUint {
        let left = m1 + &self.q - m2;
        let left = left * &self.inv_x % &self.q;
        let left = left + r % &self.q;

        left % &self.q
    }

    pub fn public_keys(&self) -> (BigUint, BigUint, BigUint, BigUint) {
        (
            self.p.clone(),
            self.q.clone(),
            self.g.clone(),
            self.y.clone(),
        )
    }

    pub fn verify(
        (p, _q, g, y): (BigUint, BigUint, BigUint, BigUint),
        m: &BigUint,
        r: &BigUint,
        h: &BigUint,
    ) -> bool {
        let gm = g.modpow(m, &p);
        let yr = y.modpow(r, &p);
        let gh = (gm * yr) % &p;
        gh == *h
    }
}

fn test_chameleon_hash(bits: usize) {
    println!("Testing Chameleon Hash with {} bits", bits);
    let start = time::SystemTime::now();
    let t = time::SystemTime::now();
    let chameleon_hash = ChameleonHash::new(bits);
    println!(
        "Time elapsed(Setup): {:?}us",
        t.elapsed().unwrap().as_micros()
    );
    // println!("Chameleon hash: {:?}", chameleon_hash);

    let msg_bits = chameleon_hash.q.bits() / 2;

    let m1 = Generator::new_uint(msg_bits);
    let r1 = Generator::new_uint(msg_bits);

    let t = time::SystemTime::now();
    let hash1 = chameleon_hash.hash(&m1, &r1);
    println!(
        "Time elapsed(Hash): {:?}us",
        t.elapsed().unwrap().as_micros()
    );
    // println!("Hash of (m1({}), r1({})): {}", m1, r1, hash1);

    let t = time::SystemTime::now();
    assert!(
        ChameleonHash::verify(chameleon_hash.public_keys(), &m1, &r1, &hash1),
        "failed to verify hash"
    );
    println!(
        "Time elapsed(Verification): {:?}us",
        t.elapsed().unwrap().as_micros()
    );

    let m2 = Generator::new_uint(msg_bits);
    let t = time::SystemTime::now();
    let r2 = chameleon_hash.find_collision(&m1, &r1, &m2);
    println!(
        "Time elapsed(Finding Collision): {:?}us",
        t.elapsed().unwrap().as_micros()
    );

    let t = time::SystemTime::now();
    let hash2 = chameleon_hash.hash(&m2, &r2);
    println!(
        "Time elapsed(Hash): {:?}us",
        t.elapsed().unwrap().as_micros()
    );

    // println!("Hash of (m2({}), r2({})): {}", m2, r2, hash2);

    println!(
        "Total time elapsed: {:?}us",
        start.elapsed().unwrap().as_micros()
    );

    assert_eq!(hash1, hash2, "Collision failed!");
    println!("");
}

fn main() {
    for bits in vec![160, 1024, 2048] {
        test_chameleon_hash(bits);
    }
}
