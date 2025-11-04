import math
import random
from typing import Tuple

def wilson_interval(successes: int, trials: int, z: float = 1.96) -> Tuple[float, float]:
    if trials == 0:
        return (0.0, 1.0)
    phat = successes / trials
    denom = 1 + z*z/trials
    centre = phat + z*z/(2*trials)
    margin = z*math.sqrt((phat*(1-phat) + z*z/(4*trials))/trials)
    low = (centre - margin)/denom
    high = (centre + margin)/denom
    return max(0.0, low), min(1.0, high)

def simulate_device_survival(p1: float, p2: float, p3: float, n: int, trials: int, seed: int = 42) -> Tuple[float, Tuple[float, float], int]:
    rnd = random.Random(seed)
    success = 0
    for _ in range(trials):
        faults = [0, 0, 0]  # накопленные неисправности в узлах 1..3
        for _step in range(n):
            # независимые неисправности на этом включении
            if rnd.random() < p1: faults[0] += 1
            if rnd.random() < p2: faults[1] += 1
            if rnd.random() < p3: faults[2] += 1
        # прибор выжил, если ни в одном узле нет >=2 неисправностей
        if all(f < 2 for f in faults):
            success += 1
    phat = success / trials
    ci = wilson_interval(success, trials, z=1.96)
    return phat, ci, success

def analytic_survival(p1: float, p2: float, p3: float, n: int) -> float:
    """Точное значение по биномиальной модели накопления неисправностей."""
    def node_survive(p):
        # P(X=0)+P(X=1) для X~Bin(n,p) = (1-p)^n + n p (1-p)^(n-1)
        return (1 - p)**n + n*p*(1 - p)**(n-1)
    return node_survive(p1) * node_survive(p2) * node_survive(p3)

if __name__ == "__main__":
    # ---- ПАРАМЕТРЫ ЗАДАНИЯ ----
    # Пример: можно подставить свои значения p1, p2, p3 и n
    p1, p2, p3 = 0.10, 0.07, 0.05   # вероятности появления неисправности при одном включении
    n = 12                          # число включений
    trials = 200_000                # число имитаций (можно увеличить для большей точности)
    seed = 123                      # фиксируем зерно для воспроизводимости
    # ----------------------------

    mc_p, (ci_lo, ci_hi), succ = simulate_device_survival(p1, p2, p3, n, trials, seed)
    an_p = analytic_survival(p1, p2, p3, n)

    print("Параметры:")
    print(f"  p1={p1:.5f}, p2={p2:.5f}, p3={p3:.5f}, n={n}, trials={trials}")
    print("\nРезультаты:")
    print(f"  Монте-Карло:   P_hat = {mc_p:.6f}  (Wilson 95% CI: [{ci_lo:.6f}, {ci_hi:.6f}])")
    print(f"  Аналитически:  P     = {an_p:.6f}")
    print(f"  Попадание точного значения в ДИ?  {'ДА' if ci_lo <= an_p <= ci_hi else 'НЕТ'}")
