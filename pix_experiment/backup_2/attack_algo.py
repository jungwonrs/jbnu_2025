import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------
# 공통 유틸
# ----------------------------------------------------------------------
def _forward_logits(model, x):
    """
    model(x) 결과에서 logits 텐서를 꺼내는 공통 함수
    (HF 모델과 torchvision 모델 둘 다 지원)
    """
    outputs = model(x)
    if hasattr(outputs, "logits"):
        return outputs.logits
    return outputs


def _cross_entropy_loss(model, x, labels):
    logits = _forward_logits(model, x)
    return F.cross_entropy(logits, labels)


# ----------------------------------------------------------------------
# 1) Single Attack 들 (FGSM / PGD / EOT / BPDA / C&W / SPSA)
# ----------------------------------------------------------------------
class sing_attack:
    # ---------------- FGSM ----------------
    @staticmethod
    def fgsm_attack(model, img, labels, epsilon):
        """
        기존 코드와 인터페이스 유지:
        - img: (1, C, H, W), [0,1]
        - labels: LongTensor (batch class index)
        - epsilon: L_inf budget
        """
        x = img.clone().detach()
        x.requires_grad = True

        loss = _cross_entropy_loss(model, x, labels)
        model.zero_grad()
        loss.backward()

        if x.grad is None:
            print("[FGSM] Error: Gradient is None. Check Model/Loss.")
            return img.detach()

        noise = epsilon * x.grad.sign()
        adv = torch.clamp(x + noise, 0.0, 1.0)
        return adv.detach()

    # ---------------- PGD (L_inf) ----------------
    @staticmethod
    def pgd_attack(
        model,
        img,
        labels,
        epsilon,
        alpha,
        num_steps,
        random_start=True,
    ):
        """
        PGD (Projected Gradient Descent, L_inf)
        - epsilon: 허용 노름 범위 (L_inf)
        - alpha: step size
        - num_steps: 반복 횟수
        - random_start: True면 eps 범위 내 무작위 초기화
        """
        x_orig = img.detach()
        if random_start:
            x = x_orig + torch.empty_like(x_orig).uniform_(-epsilon, epsilon)
            x = torch.clamp(x, 0.0, 1.0)
        else:
            x = x_orig.clone()

        for _ in range(num_steps):
            x.requires_grad = True

            loss = _cross_entropy_loss(model, x, labels)
            model.zero_grad()
            if x.grad is not None:
                x.grad.zero_()
            loss.backward()

            if x.grad is None:
                print("[PGD] Warning: grad is None, break.")
                break

            grad_sign = x.grad.sign()
            x = x + alpha * grad_sign

            # L_inf projection
            x = torch.max(torch.min(x, x_orig + epsilon), x_orig - epsilon)
            x = torch.clamp(x, 0.0, 1.0).detach()

        return x.detach()

    # ---------------- EOT + PGD ----------------
    @staticmethod
    def eot_pgd_attack(
        model,
        img,
        labels,
        epsilon,
        alpha,
        num_steps,
        eot_samples,
        eot_sigma=0.0,
    ):
        """
        EOT (Expectation over Transformation) + PGD
        - eot_samples: 각 step마다 몇 번 샘플링해서 gradient 평균낼지
        - eot_sigma: transformation으로 쓰는 가우시안 noise scale
        """
        x_orig = img.detach()
        x = x_orig.clone()

        for _ in range(num_steps):
            grad_accum = torch.zeros_like(x)

            for _ in range(eot_samples):
                x_noisy = x.clone()
                if eot_sigma > 0.0:
                    x_noisy = x_noisy + torch.randn_like(x_noisy) * eot_sigma
                    x_noisy = torch.clamp(x_noisy, 0.0, 1.0)

                x_noisy.requires_grad = True
                loss = _cross_entropy_loss(model, x_noisy, labels)
                model.zero_grad()
                if x_noisy.grad is not None:
                    x_noisy.grad.zero_()
                loss.backward()

                if x_noisy.grad is None:
                    print("[EOT-PGD] Warning: grad is None, skip sample.")
                    continue
                grad_accum += x_noisy.grad

            grad_mean = grad_accum / max(eot_samples, 1)
            x = x + alpha * grad_mean.sign()

            # L_inf projection
            x = torch.max(torch.min(x, x_orig + epsilon), x_orig - epsilon)
            x = torch.clamp(x, 0.0, 1.0).detach()

        return x.detach()

    # ---------------- BPDA + PGD ----------------
    @staticmethod
    def bpda_pgd_attack(
        model,
        img,
        labels,
        epsilon,
        alpha,
        num_steps,
        defense_fn=None,
    ):
        """
        BPDA + PGD
        - defense_fn: 비분리가능(Non-differentiable) 방어 (예: JPEG, median filter 등)
          forward 에서는 defense_fn(x)를 쓰고,
          backward 에서는 identity gradient 로 근사하는 형태.
        """
        x_orig = img.detach()
        x = x_orig.clone()

        for _ in range(num_steps):
            # forward 는 defense_fn(x) 통과 (gradient는 끊고 다시 붙임)
            if defense_fn is not None:
                x_def = defense_fn(x).detach()
            else:
                x_def = x.detach()

            x_def.requires_grad = True
            loss = _cross_entropy_loss(model, x_def, labels)
            model.zero_grad()
            if x_def.grad is not None:
                x_def.grad.zero_()
            loss.backward()

            if x_def.grad is None:
                print("[BPDA-PGD] Warning: grad is None, break.")
                break

            # gradient 를 x 쪽으로 copy (identity backward 근사)
            grad = x_def.grad
            x = x + alpha * grad.sign()

            # L_inf projection
            x = torch.max(torch.min(x, x_orig + epsilon), x_orig - epsilon)
            x = torch.clamp(x, 0.0, 1.0).detach()

        return x.detach()

    # ---------------- Carlini & Wagner (L2 variant – 간단 버전) ----------------
    @staticmethod
    def cw_attack(
        model,
        img,
        labels,
        c=1.0,
        kappa=0.0,
        num_steps=100,
        lr=0.01,
    ):
        """
        간단화된 C&W L2 (untargeted) 버전
        - c: regularization weight
        - kappa: confidence margin
        - num_steps: 최적화 step 수
        - lr: gradient descent learning rate
        """
        # tanh space 로 변환 (0~1 -> -1~1)
        x_orig = img.detach()
        x_var = x_orig.clone().detach()
        x_var.requires_grad = True

        for _ in range(num_steps):
            logits = _forward_logits(model, x_var)
            one_hot = F.one_hot(labels, num_classes=logits.shape[1]).float()

            # true class logit, max other logit
            real = (one_hot * logits).sum(dim=1)
            other = ((1 - one_hot) * logits - one_hot * 1e4).max(dim=1)[0]

            # f(x) = max(other - real + kappa, 0)
            f_x = torch.clamp(other - real + kappa, min=0.0)

            # L2 distance
            l2 = torch.sum((x_var - x_orig) ** 2, dim=[1, 2, 3])

            loss = l2 + c * f_x
            loss = loss.mean()

            model.zero_grad()
            if x_var.grad is not None:
                x_var.grad.zero_()
            loss.backward()

            if x_var.grad is None:
                print("[C&W] Warning: grad is None, break.")
                break

            # gradient descent
            x_var = x_var - lr * x_var.grad
            x_var = torch.clamp(x_var, 0.0, 1.0).detach()
            x_var.requires_grad = True

        return x_var.detach()

    # ---------------- SPSA ----------------
    @staticmethod
    def spsa_attack(
        model,
        img,
        labels,
        epsilon,
        num_steps,
        spsa_samples,
        spsa_delta,
        step_size,
    ):
        """
        SPSA (Simultaneous Perturbation Stochastic Approximation)
        - gradient-free
        - epsilon: L_inf budget
        - num_steps: 반복 횟수
        - spsa_samples: 각 step마다 gradient 추정을 위한 샘플 수
        - spsa_delta: finite-difference perturbation scale
        - step_size: update step size
        """
        x_orig = img.detach()
        x = x_orig.clone()

        for _ in range(num_steps):
            grad_est = torch.zeros_like(x)

            for _ in range(spsa_samples):
                rnd = torch.randint_like(x, low=0, high=2) * 2 - 1  # {-1, +1}
                delta = spsa_delta * rnd

                x_pos = torch.clamp(x + delta, 0.0, 1.0)
                x_neg = torch.clamp(x - delta, 0.0, 1.0)

                loss_pos = _cross_entropy_loss(model, x_pos, labels)
                loss_neg = _cross_entropy_loss(model, x_neg, labels)

                # gradient approximation
                grad_est += ((loss_pos - loss_neg) / (2.0 * spsa_delta)) * rnd

            grad_est /= max(spsa_samples, 1)
            x = x + step_size * grad_est.sign()

            # projection to L_inf ball
            x = torch.max(torch.min(x, x_orig + epsilon), x_orig - epsilon)
            x = torch.clamp(x, 0.0, 1.0).detach()

        return x.detach()


# ----------------------------------------------------------------------
# 2) Hybrid Attack 들: single_attack 조합
#    (모든 건 sing_attack.*를 재사용)
# ----------------------------------------------------------------------
class hybrid_attack:
    """
    single_attack 안의 공격들을 조합해서 쓰는 클래스.
    single_attack 쪽 파라미터를 바꾸면, 여기 조합들도 그대로 영향을 받음.
    """

    # 1) FGSM -> PGD
    @staticmethod
    def fgsm_then_pgd(
        model,
        img,
        labels,
        epsilon,
        pgd_alpha,
        pgd_steps,
        pgd_random_start=True,
    ):
        x = sing_attack.fgsm_attack(model, img, labels, epsilon)
        x = sing_attack.pgd_attack(
            model,
            x,
            labels,
            epsilon=epsilon,
            alpha=pgd_alpha,
            num_steps=pgd_steps,
            random_start=pgd_random_start,
        )
        return x

    # 2) FGSM -> SPSA
    @staticmethod
    def fgsm_then_spsa(
        model,
        img,
        labels,
        epsilon,
        spsa_steps,
        spsa_samples,
        spsa_delta,
        spsa_step_size,
    ):
        x = sing_attack.fgsm_attack(model, img, labels, epsilon)
        x = sing_attack.spsa_attack(
            model,
            x,
            labels,
            epsilon=epsilon,
            num_steps=spsa_steps,
            spsa_samples=spsa_samples,
            spsa_delta=spsa_delta,
            step_size=spsa_step_size,
        )
        return x

    # 3) PGD -> C&W
    @staticmethod
    def pgd_then_cw(
        model,
        img,
        labels,
        epsilon,
        pgd_alpha,
        pgd_steps,
        cw_c,
        cw_kappa,
        cw_steps,
        cw_lr,
    ):
        x = sing_attack.pgd_attack(
            model,
            img,
            labels,
            epsilon=epsilon,
            alpha=pgd_alpha,
            num_steps=pgd_steps,
            random_start=True,
        )
        x = sing_attack.cw_attack(
            model,
            x,
            labels,
            c=cw_c,
            kappa=cw_kappa,
            num_steps=cw_steps,
            lr=cw_lr,
        )
        return x

    # 4) EOT-PGD (그 자체도 하이브리드 느낌이라 그대로 제공)
    @staticmethod
    def eot_pgd(
        model,
        img,
        labels,
        epsilon,
        alpha,
        num_steps,
        eot_samples,
        eot_sigma,
    ):
        return sing_attack.eot_pgd_attack(
            model,
            img,
            labels,
            epsilon=epsilon,
            alpha=alpha,
            num_steps=num_steps,
            eot_samples=eot_samples,
            eot_sigma=eot_sigma,
        )

    # 5) BPDA-PGD (non-diff defense 연결용)
    @staticmethod
    def bpda_pgd(
        model,
        img,
        labels,
        epsilon,
        alpha,
        num_steps,
        defense_fn,
    ):
        return sing_attack.bpda_pgd_attack(
            model,
            img,
            labels,
            epsilon=epsilon,
            alpha=alpha,
            num_steps=num_steps,
            defense_fn=defense_fn,
        )

    # 6) PGD -> SPSA
    @staticmethod
    def pgd_then_spsa(
        model,
        img,
        labels,
        epsilon,
        pgd_alpha,
        pgd_steps,
        spsa_steps,
        spsa_samples,
        spsa_delta,
        spsa_step_size,
    ):
        x = sing_attack.pgd_attack(
            model,
            img,
            labels,
            epsilon=epsilon,
            alpha=pgd_alpha,
            num_steps=pgd_steps,
            random_start=True,
        )
        x = sing_attack.spsa_attack(
            model,
            x,
            labels,
            epsilon=epsilon,
            num_steps=spsa_steps,
            spsa_samples=spsa_samples,
            spsa_delta=spsa_delta,
            step_size=spsa_step_size,
        )
        return x

    # 7) FGSM -> PGD -> C&W
    @staticmethod
    def fgsm_pgd_cw(
        model,
        img,
        labels,
        epsilon,
        pgd_alpha,
        pgd_steps,
        cw_c,
        cw_kappa,
        cw_steps,
        cw_lr,
    ):
        x = sing_attack.fgsm_attack(model, img, labels, epsilon)
        x = sing_attack.pgd_attack(
            model,
            x,
            labels,
            epsilon=epsilon,
            alpha=pgd_alpha,
            num_steps=pgd_steps,
            random_start=True,
        )
        x = sing_attack.cw_attack(
            model,
            x,
            labels,
            c=cw_c,
            kappa=cw_kappa,
            num_steps=cw_steps,
            lr=cw_lr,
        )
        return x

    # 8) FGSM -> PGD -> SPSA
    @staticmethod
    def fgsm_pgd_spsa(
        model,
        img,
        labels,
        epsilon,
        pgd_alpha,
        pgd_steps,
        spsa_steps,
        spsa_samples,
        spsa_delta,
        spsa_step_size,
    ):
        x = sing_attack.fgsm_attack(model, img, labels, epsilon)
        x = sing_attack.pgd_attack(
            model,
            x,
            labels,
            epsilon=epsilon,
            alpha=pgd_alpha,
            num_steps=pgd_steps,
            random_start=True,
        )
        x = sing_attack.spsa_attack(
            model,
            x,
            labels,
            epsilon=epsilon,
            num_steps=spsa_steps,
            spsa_samples=spsa_samples,
            spsa_delta=spsa_delta,
            step_size=spsa_step_size,
        )
        return x

    # 9) EOT-PGD -> C&W
    @staticmethod
    def eot_pgd_cw(
        model,
        img,
        labels,
        epsilon,
        alpha,
        num_steps,
        eot_samples,
        eot_sigma,
        cw_c,
        cw_kappa,
        cw_steps,
        cw_lr,
    ):
        x = sing_attack.eot_pgd_attack(
            model,
            img,
            labels,
            epsilon=epsilon,
            alpha=alpha,
            num_steps=num_steps,
            eot_samples=eot_samples,
            eot_sigma=eot_sigma,
        )
        x = sing_attack.cw_attack(
            model,
            x,
            labels,
            c=cw_c,
            kappa=cw_kappa,
            num_steps=cw_steps,
            lr=cw_lr,
        )
        return x

    # 10) EOT-PGD -> SPSA
    @staticmethod
    def eot_pgd_spsa(
        model,
        img,
        labels,
        epsilon,
        alpha,
        num_steps,
        eot_samples,
        eot_sigma,
        spsa_steps,
        spsa_samples,
        spsa_delta,
        spsa_step_size,
    ):
        x = sing_attack.eot_pgd_attack(
            model,
            img,
            labels,
            epsilon=epsilon,
            alpha=alpha,
            num_steps=num_steps,
            eot_samples=eot_samples,
            eot_sigma=eot_sigma,
        )
        x = sing_attack.spsa_attack(
            model,
            x,
            labels,
            epsilon=epsilon,
            num_steps=spsa_steps,
            spsa_samples=spsa_samples,
            spsa_delta=spsa_delta,
            step_size=spsa_step_size,
        )
        return x
