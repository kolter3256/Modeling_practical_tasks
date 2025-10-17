# main.py
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2


N = 1000        # объём выборки (>= 1000)
K = 15          # число интервалов (15 или 25)
a, b = 0.0, 3.4 # носитель распределения


A = 165
Mu = 3463
Mm = 4096 * 4
Y  = 3887

def rnd():
    global Y
    Y = (A * Y + Mu) % Mm
    return Y / Mm  # в [0,1)


def F(x):
    if x < 0.0:      return 0.0
    if x <= 1.0:     return 0.15*x
    if x <= 2.0:     return 0.35*x - 0.2
    if x <= 2.4:     return 0.875*x - 1.25
    if x <= 3.4:     return 0.15*x + 0.49
    return 1.0

def f(x):
    if 0.0 <= x <= 1.0:   return 0.15
    if 1.0 <= x <= 2.0:   return 0.35
    if 2.0 <= x <= 2.4:   return 0.875
    if 2.4 <= x <= 3.4:   return 0.15
    return 0.0

def Finv(u):
    if u < 0.15:
        return u / 0.15
    elif u < 0.5:
        return (u + 0.2) / 0.35
    elif u < 0.85:
        return (u + 1.25) / 0.875
    else:
        return (u - 0.49) / 0.15


theor_mean = 1.805
theor_var  = 0.5813083333333342


x = np.array([Finv(rnd()) for _ in range(N)])
x_sorted = np.sort(x)


edges = np.linspace(a, b, K + 1)
counts, _ = np.histogram(x, bins=edges)
widths = np.diff(edges)
centers = 0.5*(edges[:-1] + edges[1:])


mean = float(np.mean(x))
var_biased = float(np.var(x, ddof=0))
var_unbiased = float(np.var(x, ddof=1))  # несмещённая


def expected_prob_on_bin(left, right):
    segs = [(0.0, 1.0, 0.15), (1.0, 2.0, 0.35), (2.0, 2.4, 0.875), (2.4, 3.4, 0.15)]
    p = 0.0
    for L, R, dens in segs:
        ll = max(left, L)
        rr = min(right, R)
        if rr > ll:
            p += dens * (rr - ll)
    return p

expected_probs  = np.array([expected_prob_on_bin(edges[i], edges[i+1]) for i in range(K)])
expected_counts = N * expected_probs

with np.errstate(divide='ignore', invalid='ignore'):
    chi2_stat = np.nansum((counts - expected_counts) ** 2 / np.where(expected_counts > 0, expected_counts, np.nan))

df = K - 1
alpha = 0.05
chi2_crit = chi2.ppf(1 - alpha, df)
pearson_ok = chi2_stat <= chi2_crit


Fn_vals = np.arange(1, N + 1) / N
F_vals  = np.array([F(val) for val in x_sorted])
D = np.max(np.maximum(np.abs(Fn_vals - F_vals), np.abs((np.arange(N)) / N - F_vals)))

K_alpha = 1.36
D_crit = K_alpha / math.sqrt(N)
kolmogorov_ok = D <= D_crit


print(f"Объём выборки: {N}\n")

print("Выборочные оценки:")
print(f"Оценка математического ожидания (выборочное среднее): {mean:.6f}")
print(f"Оценка дисперсии (несмещённая): {var_unbiased:.6f}\n")

print("Теоретические значения (анализ):")
print(f"Теоретическое мат. ожидание  E[X] = {theor_mean:.6f}")
print(f"Теоретическая дисперсия Var(X) = {theor_var:.6f}\n")

print("Распределение чисел по интервалам:")
cum = 0
for i, c in enumerate(counts, 1):
    cum += c
    norm_freq = c / N
    Fn_cum    = cum / N
    print(f"#{i:<2d}-й интервал: {c:<4d} | норм. частота: {norm_freq:0.6f} | меньше или равно: {Fn_cum:0.6f}")

print("\nКритерий Колмогорова:")
print(f"Статистика D = {D:.6f}")
print(f"Критическое значение (alpha=0.05) ~ {D_crit:.6f}")
if kolmogorov_ok:
    print("=> Нет оснований отвергнуть гипотезу о соответствии распределению.")
else:
    print("=> Гипотеза о соответствии распределению отвергается.")

print("\nКритерий Пирсона (хи-квадрат):")
print(f"Статистика χ² = {chi2_stat:.6f}")
print(f"Степени свободы = {df}")
if pearson_ok:
    print("=> Нет оснований отвергнуть гипотезу о соответствии распределению.")
else:
    print("=> Гипотеза о соответствии распределению отвергается.")


plt.figure(figsize=(9,4.6))
plt.subplot(1,2,1)
plt.title(f"Гистограмма частот (K={K})")
plt.bar(centers, counts, width=widths, align='center', edgecolor='k')
plt.step(centers, expected_counts, where='mid', linewidth=2)
plt.xlabel("x"); plt.ylabel("Частоты")
plt.legend(["Ожидаемо (теория)", "Наблюдаемо"], loc="best")
plt.subplot(1,2,2)
xx = np.linspace(a, b, 800)
plt.title("Статистическая функция распределения")
Fn_at_edges = np.searchsorted(x_sorted, edges[1:], side="right") / N
plt.step(edges[1:], Fn_at_edges, where='post', label="F_n (эмп.)")
plt.plot(xx, [F(t) for t in xx], linewidth=2, label="F (теор.)")
plt.xlabel("x"); plt.ylabel("F"); plt.legend(loc="best")
plt.tight_layout(); plt.show()
