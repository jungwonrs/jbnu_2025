import torch, torch.nn as nn, torch.optim as optim
import FrEIA.framework as Ff, FrEIA.modules as Fm


# ── 작은 MLP 서브넷 ───────────────────────────────────
def subnet_fc(d_in, d_out):
    return nn.Sequential(nn.Linear(d_in, 64), nn.ReLU(), nn.Linear(64, d_out))


# ── INN 그래프 ───────────────────────────────────────
nodes = [Ff.InputNode(2, name='in')]           # (batch, 2)

for k in range(3):
    # 1) CouplingBlock
    cb = Ff.Node(
        nodes[-1], Fm.GLOWCouplingBlock,
        {'subnet_constructor': subnet_fc, 'clamp': 1.0},
        name=f'cb_{k}'
    )
    nodes.append(cb)

    # 2) PermuteRandom  ─ 반드시 "방금 만든 cb"를 입력으로 사용
    perm = Ff.Node(
        nodes[-1], Fm.PermuteRandom,
        {'seed': k},
        name=f'perm_{k}'
    )
    nodes.append(perm)

# Split 없이 직접 2-차원 출력   (out[:,0]=y, out[:,1]=z)
nodes.append(Ff.OutputNode(nodes[-1], name='out'))
inn = Ff.ReversibleGraphNet(nodes)


# ── 학습 (self-reconstruction) ───────────────────────
opt, mse = optim.Adam(inn.parameters(), 1e-3), nn.MSELoss()

for ep in range(300):
    x  = torch.randn(128, 2)
    out = inn(x)[0]          # (B,2)

    y, z = out[:, :1], out[:, 1:]        # y, z 분리
    x_rec = inn(torch.cat([y, z], 1), rev=True)[0]

    loss = mse(x_rec, x)
    opt.zero_grad(); loss.backward(); opt.step()
    if ep % 100 == 0:
        print(f'epoch {ep:3d}  loss={loss.item():.3e}')


# ── 테스트: y, z 출력 확인 ────────────────────────────
x_t   = torch.randn(5, 2)
out_t = inn(x_t)[0]
y_t, z_t = out_t[:, :1], out_t[:, 1:]
x_back   = inn(torch.cat([y_t, z_t], 1), rev=True)[0]

print('\n[x → (y,z)]')
print('y:\n', y_t)
print('z:\n', z_t)
print('[ (y,z) → x ]  복원 오차:\n', (x_t - x_back).abs())
