#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np

# ----------------------- НАСТРОЙКИ -----------------------
M = 10            # число шагов блуждания (по заданию)
RNG_SEED = 42     # сид генератора
N_PILOT = 5000    # объём пробного эксперимента (>=1000)

# Требования точности/достоверности
eps_mean = 0.10   # абсолютная точность для E[R] (полуширина)
beta_mean = 0.95  # достоверность для среднего

delta_var = 0.15  # относительная точность для Var(R): полуширина / оценка
beta_var  = 0.95  # достоверность для дисперсии

# Планирование bootstrap (если распределение не нормальное)
BOOT_B = 800      # число бутстрапов для ДИ дисперсии (баланс точность/время)
STEP_N  = 100     # шаг наращивания n при поиске объёма для дисперсии (ненорм.)
N_MAX   = 200000  # защитное ограничение

# ---------------------- ВСПОМОГАТЕЛЬНОЕ ----------------------
def rng_steps_3d(N, M, rng):
    """N траекторий по M шагов в 3D (±X, ±Y, ±Z). Возвращает R и компоненты."""
    dirs = np.array([[ 1, 0, 0], [-1, 0, 0],
                     [ 0, 1, 0], [ 0,-1, 0],
                     [ 0, 0, 1], [ 0, 0,-1]], dtype=int)
    idx   = rng.integers(0, 6, size=(N, M))
    steps = dirs[idx]              # (N, M, 3)
    pos   = steps.sum(axis=1)      # (N, 3)
    r     = np.sqrt((pos*pos).sum(axis=1))
    return r, pos[:,0], pos[:,1], pos[:,2]

def mean_var(a):
    return float(np.mean(a)), float(np.var(a, ddof=0))

# --- нормквантиль (Аппроксимация А́клема) ---
def norm_ppf(p):
    if not (0.0 < p < 1.0):
        raise ValueError("p must be in (0,1)")
    a = [-3.969683028665376e+01,  2.209460984245205e+02,
         -2.759285104469687e+02,  1.383577518672690e+02,
         -3.066479806614716e+01,  2.506628277459239e+00]
    b = [-5.447609879822406e+01,  1.615858368580409e+02,
         -1.556989798598866e+02,  6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
          4.374664141464968e+00,  2.938163982698783e+00]
    d = [ 7.784695709041462e-03,  3.224671290700398e-01,
          2.445134137142996e+00,  3.754408661907416e+00]
    plow  = 0.02425; phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2*math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2*math.log(1-p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                 ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5; r = q*q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)

# --- KS тест к Normal(mu, sigma): асимптотическое p-value ---
def ks_test_normal(x, mu, sigma):
    import math
    x = np.sort(x)
    n = x.size
    # теоретическая CDF нормального распределения
    cdf = 0.5 * (1 + np.array([math.erf((xi - mu) / (sigma * math.sqrt(2))) for xi in x]))
    ecdf = np.arange(1, n + 1) / n
    d_plus  = np.max(ecdf - cdf)
    d_minus = np.max(cdf - (np.arange(0, n) / n))
    d = max(float(d_plus), float(d_minus))
    # асимптотическая оценка p-value
    t = (math.sqrt(n) + 0.12 + 0.11 / math.sqrt(n)) * d
    s = 0.0
    for k in range(1, 101):
        s += (-1) ** (k - 1) * math.exp(-2 * (k * k) * (t * t))
    p = max(0.0, min(1.0, 2 * s))
    return d, p


# --- Jarque–Bera: p-value для χ²_2 равно exp(-JB/2) ---
def jarque_bera_pvalue(x):
    m = float(np.mean(x))
    s = float(np.std(x, ddof=0))
    if s == 0.0:
        return 1.0
    z = (x - m) / s
    skew = float(np.mean(z**3))
    kurt = float(np.mean(z**4))
    JB = (len(x)/6.0) * (skew**2 + ((kurt-3.0)**2)/4.0)
    p = math.exp(-JB/2.0)  # точно для df=2
    return JB, p

# --- Чебышёв: объём на среднее ---
def plan_n_mean_chebyshev(s, eps, beta):
    # P(|X̄ - μ| ≥ eps) ≤ s^2 / (n * eps^2) ≤ 1-β
    n = math.ceil((s*s) / (eps*eps*(1-beta)))
    return max(n, 2)

# --- Нормальный ДИ для среднего ---
def plan_n_mean_normal(s, eps, beta):
    z = norm_ppf(1 - (1-beta)/2)
    n = math.ceil((z*s/eps)**2)
    return max(n, 2), z

def ci_mean_normal(xbar, s, n, beta):
    z = norm_ppf(1 - (1-beta)/2)
    half = z * s / math.sqrt(n)
    return (xbar - half, xbar + half, half, z)

def ci_mean_chebyshev(xbar, s, n, beta):
    half = s * math.sqrt(1.0 / (n*(1-beta)))
    return (xbar - half, xbar + half, half)

# --- χ²-интервал для дисперсии (нормальный случай) ---
def chi2_ppf(p, df):
    z = norm_ppf(p)
    return df * (1 - 2/(9*df) + z*math.sqrt(2/(9*df)))**3  # Уилсон–Хилферти

def ci_var_chi2(s2, n, beta):
    df = n - 1
    q_low  = chi2_ppf((1-beta)/2, df)
    q_high = chi2_ppf(1 - (1-beta)/2, df)
    L = df * s2 / q_high
    U = df * s2 / q_low
    return L, U

def rel_halfwidth_var_chi2(n, beta):
    df = n - 1
    q_low  = chi2_ppf((1-beta)/2, df)
    q_high = chi2_ppf(1 - (1-beta)/2, df)
    # относительная полуширина не зависит от S^2
    return 0.5 * ((df/q_low) - (df/q_high))

# --- Bootstrap-поиск n для дисперсии (ненорм.) ---
def bootstrap_var_ci_relative_width(pilot, n, beta, B=800, rng=None):
    """Перцентильный ДИ дисперсии на уровне beta; возвращает относительную полуширину."""
    if rng is None:
        rng = np.random.default_rng()
    idx = rng.integers(0, pilot.size, size=(B, n))
    samples = pilot[idx]
    v = np.var(samples, axis=1, ddof=0)
    alpha = 1 - beta
    L, U = np.quantile(v, [alpha/2, 1-alpha/2])
    vhat = float(np.var(pilot[:n], ddof=0))  # грубая оценка для отнесения
    rel_half = 0.5 * (U - L) / vhat if vhat > 0 else float('inf')
    return rel_half, L, U

def plan_n_var_nonnormal(pilot, delta_target, beta, step=100, n_start=100, n_max=N_MAX, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    n = max(n_start, 30)
    while n <= n_max:
        rel, _, _ = bootstrap_var_ci_relative_width(pilot, n, beta, B=BOOT_B, rng=rng)
        if rel <= delta_target:
            return n
        n += step
    raise RuntimeError("Не удалось подобрать n_var (bootstrap) в заданном диапазоне.")

# -------------------- ПРОБНЫЙ ЭКСПЕРИМЕНТ --------------------
rng = np.random.default_rng(RNG_SEED)
R_pilot, *_ = rng_steps_3d(N_PILOT, M, rng)
xbar_p, s2_p = mean_var(R_pilot); s_p = math.sqrt(s2_p)

# Тесты нормальности
d_ks, p_ks = ks_test_normal(R_pilot, xbar_p, s_p if s_p>0 else 1.0)
JB, p_jb   = jarque_bera_pvalue(R_pilot)
is_normal  = (p_ks >= 0.05) and (p_jb >= 0.05)

print("ПРОБНЫЙ ЭКСПЕРИМЕНТ")
print("-------------------")
print(f"n_pilot = {N_PILOT},   x̄_p = {xbar_p:.5f},   s_p^2 = {s2_p:.5f}")
print(f"Нормальность параметра R:  KS p={p_ks:.3f},  JB p={p_jb:.3f}  -> {'Нормаль' if is_normal else 'НЕ нормаль'}")
print()

# ---------------------- ПЛАНИРОВАНИЕ ОБЪЁМА ----------------------
if is_normal:
    n_mean, z_used = plan_n_mean_normal(s_p, eps_mean, beta_mean)
    plan_mean_line = f"Метод для среднего: нормальное приближение (z={z_used:.3f}), требуемо n_mean = {n_mean}"
else:
    n_mean = plan_n_mean_chebyshev(s_p, eps_mean, beta_mean)
    plan_mean_line = f"Метод для среднего: неравенство Чебышёва, требуемо n_mean = {n_mean}"

if is_normal:
    # подберём n_var по относит. полуширине χ²-интервала
    n_var = 10
    while rel_halfwidth_var_chi2(n_var, beta_var) > delta_var:
        n_var += 1
    plan_var_line = f"Метод для дисперсии: χ²-интервал (норма), требуемо n_var = {n_var}"
else:
    # bootstrap-подбор из пилота
    n_var = plan_n_var_nonnormal(R_pilot, delta_var, beta_var, step=STEP_N, n_start=STEP_N, rng=rng)
    plan_var_line = f"Метод для дисперсии: бутстрап (распределение не нормальное), требуемо n_var = {n_var}"

n_final = max(n_mean, n_var)

print("ПЛАНИРОВАНИЕ ОБЪЁМА")
print("-------------------")
print(f"Точность среднего: ε = {eps_mean}, доверие β = {beta_mean}")
print(plan_mean_line)
print(f"Точность дисперсии: относительная ±{delta_var*100:.1f}%, доверие β = {beta_var}")
print(plan_var_line)
print(f"Итого берём n = max(n_mean, n_var) = {n_final}")
print()

# ----------------------- ОСНОВНОЙ ЭКСПЕРИМЕНТ -----------------------
R_main, *_ = rng_steps_3d(n_final, M, rng)
xbar = float(np.mean(R_main))
s2   = float(np.var(R_main, ddof=0))
s    = math.sqrt(s2)

print("ОСНОВНОЙ ЭКСПЕРИМЕНТ")
print("--------------------")
# Среднее
if is_normal:
    L,U,half,z = ci_mean_normal(xbar, s, n_final, beta_mean)
    print(f"Оценка среднего:  x̄ = {xbar:.5f},   {int(beta_mean*100)}%-ДИ = [{L:.5f}, {U:.5f}]")
else:
    L,U,half = ci_mean_chebyshev(xbar, s, n_final, beta_mean)
    print(f"Оценка среднего:  x̄ = {xbar:.5f},   {int(beta_mean*100)}%-ДИ (Чебышёв) = [{L:.5f}, {U:.5f}]")

# Дисперсия
if is_normal:
    Lv, Uv = ci_var_chi2(s2, n_final, beta_var)
    print(f"Оценка дисперсии: s^2 = {s2:.5f},   {int(beta_var*100)}%-ДИ = [{Lv:.5f}, {Uv:.5f}]")
else:
    # бутстрап-ДИ дисперсии на выборке основного эксперимента
    # (можно было бы реиспользовать пилот, но корректнее считать по основной выборке)
    rel, Lv, Uv = bootstrap_var_ci_relative_width(R_main, n_final, beta_var, B=BOOT_B, rng=rng)
    print(f"Оценка дисперсии: s^2 = {s2:.5f},   {int(beta_var*100)}%-ДИ (bootstrap) = [{Lv:.5f}, {Uv:.5f}]")
