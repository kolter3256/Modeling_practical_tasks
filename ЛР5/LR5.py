# ЛР5, задание 9: Распределение Вейбулла и Бета-распределение
# Исправленный вариант без предупреждения "divide by zero"
# Автор: Кондратьев П.А., гр.242

import math
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------- Параметры ----------------
N = 5000       # объём выборки (>=1000)
k_bins = 25    # число интервалов (15 или 25)

# Параметры распределений
weibull_k = 1.8
weibull_lambda = 2.0
beta_a = 2.5
beta_b = 4.0

rng = np.random.default_rng(42)

# --- Опционально SciPy (для p-value) ---
try:
    from scipy import stats as sps
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False


# ---------------- Служебные функции ----------------
def print_title():
    print("МИНИСТЕРСТВО НАУКИ И ВЫСШЕГО ОБРАЗОВАНИЯ РФ")
    print("ФГБОУ ВО «РГРТУ им. В.Ф. Уткина»")
    print("Отчёт по практической работе №5 — «Моделирование»")
    print("Тема: Генерирование случайных величин с часто используемыми законами распределения")
    print("Задание 9: Распределение Вейбулла, Бета-распределение")
    print("Выполнил: студент гр. 242 Кондратьев П.А.  |  Проверил: Овечкин Г.В., Анастасьев А.А.")
    print("-" * 80)


def describe_sample(name, x):
    mu = float(np.mean(x))
    var = float(np.var(x, ddof=1))
    print(f"[{name}] Объём выборки: {len(x)}")
    print(f"[{name}] Мат. ожидание: {mu:.6f}")
    print(f"[{name}] Дисперсия:     {var:.6f}")
    return mu, var


# ===== Вейбулл =====
def weibull_pdf(x, k, lam):
    x = np.asarray(x)
    f = np.zeros_like(x, dtype=float)
    mask = x >= 0
    z = (x[mask] / lam) ** k
    f[mask] = (k / lam) * (x[mask] / lam) ** (k - 1) * np.exp(-z)
    return f

def weibull_cdf(x, k, lam):
    x = np.asarray(x)
    F = np.zeros_like(x, dtype=float)
    mask = x >= 0
    F[mask] = 1 - np.exp(-(x[mask] / lam) ** k)
    return F

def sample_weibull(n, k, lam, rng):
    u = rng.random(n)
    return lam * (-np.log1p(-u)) ** (1 / k)


# ===== Бета =====
def beta_norm_const(a, b):
    return math.gamma(a + b) / (math.gamma(a) * math.gamma(b))

def beta_pdf(x, a, b):
    x = np.asarray(x)
    f = np.zeros_like(x, dtype=float)
    mask = (x >= 0) & (x <= 1)
    c = beta_norm_const(a, b)
    f[mask] = c * (x[mask] ** (a - 1)) * ((1 - x[mask]) ** (b - 1))
    return f

class BetaCDFNumeric:
    def __init__(self, a, b, grid_n=20000):
        xs = np.linspace(0, 1, grid_n)
        pdf = beta_pdf(xs, a, b)
        cdf = np.cumsum((pdf[1:] + pdf[:-1]) * (xs[1] - xs[0]) / 2)
        cdf = np.concatenate([[0.0], cdf])
        cdf /= cdf[-1]
        self.xs = xs
        self.cdf = cdf

    def F(self, x):
        x = np.asarray(x)
        below = x <= 0
        above = x >= 1
        mid = (~below) & (~above)
        out = np.zeros_like(x, dtype=float)
        out[below] = 0
        out[above] = 1
        if np.any(mid):
            out[mid] = np.interp(x[mid], self.xs, self.cdf)
        return out

def sample_beta(n, a, b, rng):
    return rng.beta(a, b, size=n)


# ---------------- Критерии ----------------
def ks_test_empirical(x, F_theor):
    x_sorted = np.sort(x)
    n = len(x_sorted)
    F_emp = np.arange(1, n + 1) / n
    F_th = F_theor(x_sorted)
    D = float(np.max(np.abs(F_emp - F_th)))
    D_crit = 1.36 / math.sqrt(n)
    return D, D_crit, D < D_crit

def chi_square_test(x, cdf_func, bins):
    N = len(x)
    hist, edges = np.histogram(x, bins=bins)
    p = np.clip(cdf_func(edges[1:]) - cdf_func(edges[:-1]), 1e-12, 1)
    expected = N * p
    chi2 = float(np.sum((hist - expected) ** 2 / expected))
    df = len(hist) - 1
    u = 1.6448536269514722
    chi2_crit = (u * math.sqrt(2 * df) + df - 2 / 3 + 2 / (9 * df)) ** 2 if df > 0 else float('inf')
    ok = chi2 < chi2_crit
    p_value = None
    if SCIPY_OK and df > 0:
        p_value = 1.0 - sps.chi2.cdf(chi2, df)
    return chi2, df, chi2_crit, ok, p_value


# ---------------- Визуализация ----------------
def plot_hist_with_pdf(x, pdf_func, x_min, x_max, title, xlabel, filename, bins):
    xs = np.linspace(x_min, x_max, 600)
    ys = pdf_func(xs)
    plt.figure()
    plt.hist(x, bins=bins, density=True, edgecolor='black')
    plt.plot(xs, ys)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Плотность")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()

def plot_ecdf_with_cdf(x, cdf_func, x_min, x_max, title, xlabel, filename):
    xs = np.sort(x)
    n = len(xs)
    F_emp = np.arange(1, n + 1) / n
    grid = np.linspace(x_min, x_max, 600)
    F_th = cdf_func(grid)
    plt.figure()
    plt.step(xs, F_emp, where='post', label="Эмпирическая CDF")
    plt.plot(grid, F_th, label="Теоретическая CDF")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("F(x)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()


# ---------------- Основные процедуры ----------------
def run_weibull():
    print("\n=== РАСПРЕДЕЛЕНИЕ ВЕЙБУЛЛА ===")
    print(f"Параметры: k={weibull_k}, λ={weibull_lambda}")
    x = sample_weibull(N, weibull_k, weibull_lambda, rng)
    describe_sample("Вейбулл", x)

    D, Dcrit, ok = ks_test_empirical(x, lambda t: weibull_cdf(t, weibull_k, weibull_lambda))
    print(f"[Колмогоров] D = {D:.5f}, Dкр = {Dcrit:.5f} → {'Согласуется' if ok else 'Отклоняется'}")

    probs = np.linspace(1e-6, 1 - 1e-6, k_bins + 1)
    edges = weibull_lambda * (-np.log1p(-probs)) ** (1 / weibull_k)
    chi2, df, chi2_crit, ok_chi, p_value = chi_square_test(
        x, lambda t: weibull_cdf(t, weibull_k, weibull_lambda), bins=edges
    )
    msg = f"[Хи-квадрат] χ² = {chi2:.3f}, df={df}, χ²кр≈{chi2_crit:.3f} → {'Согласуется' if ok_chi else 'Отклоняется'}"
    if p_value is not None:
        msg += f", p={p_value:.5g}"
    print(msg)

    xmax = np.quantile(x, 0.999)
    plot_hist_with_pdf(x, lambda t: weibull_pdf(t, weibull_k, weibull_lambda),
                       0, xmax, "Гистограмма и плотность Вейбулла", "x", "weibull_hist.png", k_bins)
    plot_ecdf_with_cdf(x, lambda t: weibull_cdf(t, weibull_k, weibull_lambda),
                       0, xmax, "Эмпирическая и теоретическая CDF Вейбулла", "x", "weibull_cdf.png")


def run_beta():
    print("\n=== БЕТА-РАСПРЕДЕЛЕНИЕ ===")
    print(f"Параметры: α={beta_a}, β={beta_b}")
    x = sample_beta(N, beta_a, beta_b, rng)
    describe_sample("Бета", x)

    beta_cdf_num = BetaCDFNumeric(beta_a, beta_b)
    D, Dcrit, ok = ks_test_empirical(x, beta_cdf_num.F)
    print(f"[Колмогоров] D = {D:.5f}, Dкр = {Dcrit:.5f} → {'Согласуется' if ok else 'Отклоняется'}")

    probs = np.linspace(1e-6, 1 - 1e-6, k_bins + 1)
    grid, Fg = beta_cdf_num.xs, beta_cdf_num.cdf
    edges = np.interp(probs, Fg, grid)

    chi2, df, chi2_crit, ok_chi, p_value = chi_square_test(x, beta_cdf_num.F, bins=edges)
    msg = f"[Хи-квадрат] χ² = {chi2:.3f}, df={df}, χ²кр≈{chi2_crit:.3f} → {'Согласуется' if ok_chi else 'Отклоняется'}"
    if p_value is not None:
        msg += f", p={p_value:.5g}"
    print(msg)

    plot_hist_with_pdf(x, lambda t: beta_pdf(t, beta_a, beta_b),
                       0, 1, "Гистограмма и плотность Бета-распределения", "x", "beta_hist.png", k_bins)
    plot_ecdf_with_cdf(x, beta_cdf_num.F,
                       0, 1, "Эмпирическая и теоретическая CDF Бета-распределения", "x", "beta_cdf.png")


if __name__ == "__main__":
    print_title()
    print("Цель: Составить подпрограммы генерации случайных величин с распределениями Вейбулла и Бета, "
          "построить гистограммы и функции распределения, оценить характеристики и проверить критерии.")
    print("-" * 80)
    run_weibull()
    print("-" * 80)
    run_beta()
    print("-" * 80)
    print("Готово! Рисунки сохранены как: weibull_hist.png, weibull_cdf.png, beta_hist.png, beta_cdf.png")
