# main_norm_var9_fixed.py
import math
import numpy as np
import matplotlib.pyplot as plt

# Пытаемся использовать точное критическое χ² из SciPy, иначе — приближение
try:
    from scipy.stats import chi2 as chi2_dist
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

# ---------------- Параметры задания ----------------
N = 5000       # объём выборки (>= 1000)
K = 25         # число интервалов (15 или 25)
mu = -1.5      # вариант 9
sigma = 1.7    # вариант 9 (это σ, не σ^2!)
SEED = 42

# Диапазон для гистограммы/теории: [mu-4σ, mu+4σ]
a = mu - 4*sigma
b = mu + 4*sigma

# ---------------- Теория (N(mu, sigma)) ----------------
def normal_pdf(x, mu, sigma):
    z = (x - mu)/sigma
    return (1.0/(sigma*np.sqrt(2*np.pi))) * np.exp(-0.5*z*z)

def normal_cdf(x, mu, sigma):
    z = (x - mu)/sigma
    return 0.5*(1.0 + math.erf(z/np.sqrt(2.0)))

theor_mean = mu
theor_var  = sigma**2

# ---------------- Генерация: ЦПТ и Марсальи–Брей ----------------
rng = np.random.default_rng(SEED)

def gen_clt(n, mu, sigma, rng):
    # Сумма 12 U(0,1) - 6 ≈ N(0,1)
    u = rng.random((n, 12))
    z = u.sum(axis=1) - 6.0
    return mu + sigma*z

def gen_marsaglia_bray(n, mu, sigma, rng):
    # Полярный метод Марсальи–Брея
    out = np.empty(n, dtype=float)
    i = 0
    while i < n:
        U = rng.uniform(-1.0, 1.0, size=2*(n - i))
        V = rng.uniform(-1.0, 1.0, size=2*(n - i))
        S = U*U + V*V
        mask = (S > 0) & (S < 1)
        if not np.any(mask):
            continue
        U, V, S = U[mask], V[mask], S[mask]
        Z = U * np.sqrt(-2.0*np.log(S)/S)  # N(0,1)
        take = min(len(Z), n - i)
        out[i:i+take] = Z[:take]
        i += take
    return mu + sigma*out

# ---------------- Статистика и критерии ----------------
def kolmogorov_test(x_sorted, mu, sigma, alpha=0.05):
    n = len(x_sorted)
    Fn_vals = np.arange(1, n + 1) / n
    F_vals  = np.array([normal_cdf(val, mu, sigma) for val in x_sorted])
    # Двусторонний критерий, как в примере
    D = np.max(np.maximum(np.abs(Fn_vals - F_vals),
                          np.abs((np.arange(n)) / n - F_vals)))
    K_alpha = 1.36  # ≈ для alpha=0.05
    D_crit = K_alpha / math.sqrt(n)
    ok = D <= D_crit
    return D, D_crit, ok

def chi2_critical(df, alpha=0.05):
    if SCIPY_OK:
        return float(chi2_dist.ppf(1 - alpha, df))
    # Приближение Уилсона–Хилферти
    from math import sqrt
    z = 1.6448536269514722  # квантиль 0.95 стандартной нормы
    return df * (1 - 2/(9*df) + z*sqrt(2/(9*df)))**3

def pearson_test(counts, edges, n, mu, sigma, alpha=0.05):
    # Ожидаемые частоты из теоретической CDF по интервалам
    exp = []
    for i in range(len(edges) - 1):
        p_i = normal_cdf(edges[i+1], mu, sigma) - normal_cdf(edges[i], mu, sigma)
        exp.append(n * p_i)
    exp = np.array(exp, dtype=float)

    # χ² и степени свободы
    with np.errstate(divide='ignore', invalid='ignore'):
        chi2_stat = np.nansum((counts - exp) ** 2 / np.where(exp > 0, exp, np.nan))

    df = len(exp) - 1          # параметры заданы, не оцениваем => df = m-1
    chi2_crit = chi2_critical(df, alpha=alpha)
    ok = chi2_stat <= chi2_crit
    return float(chi2_stat), int(df), float(chi2_crit), ok, exp  # <— возвращаем и exp!

# ---------------- Отрисовка ----------------
def plot_pair(x, counts, edges, expected_counts, title_left, title_right):
    widths  = np.diff(edges)
    centers = 0.5*(edges[:-1] + edges[1:])

    plt.figure(figsize=(9, 4.6))
    # Левая: гистограмма и ожидаемые частоты
    plt.subplot(1, 2, 1)
    plt.title(f"{title_left} (K={len(widths)})")
    plt.bar(centers, counts, width=widths, align='center', edgecolor='k')
    plt.step(centers, expected_counts, where='mid', linewidth=2)
    plt.xlabel("x"); plt.ylabel("Частоты")
    plt.legend(["Ожидаемо (теория)", "Наблюдаемо"], loc="best")

    # Правая: эмпирическая и теоретическая CDF
    plt.subplot(1, 2, 2)
    plt.title(title_right)
    x_sorted = np.sort(x)
    Fn_at_edges = np.searchsorted(x_sorted, edges[1:], side="right") / len(x_sorted)
    plt.step(edges[1:], Fn_at_edges, where='post', label="F_n (эмп.)")
    xx = np.linspace(edges[0], edges[-1], 800)
    plt.plot(xx, [normal_cdf(t, mu, sigma) for t in xx], linewidth=2, label="F (теор.)")
    plt.xlabel("x"); plt.ylabel("F"); plt.legend(loc="best")
    plt.tight_layout(); plt.show()

# ---------------- Единый раунд расчёта/печати ----------------
def run_block(method_name, sample):
    print(f"\n=== {method_name} ===")
    x = np.asarray(sample)
    x_sorted = np.sort(x)

    # Интервалы
    edges = np.linspace(a, b, K + 1)
    counts, _ = np.histogram(x, bins=edges)

    # Оценки по выборке
    mean = float(np.mean(x))
    var_unbiased = float(np.var(x, ddof=1))

    # χ² (с ожидаемыми по теории) — ВАЖНО: распаковываем 5 значений
    chi2_stat, df, chi2_crit, pearson_ok, expected_counts = pearson_test(
        counts, edges, len(x), mu, sigma, alpha=0.05
    )
    # Колмогоров
    D, D_crit, kolmogorov_ok = kolmogorov_test(x_sorted, mu, sigma, alpha=0.05)

    # Печать в стиле примера
    print(f"Объём выборки: {len(x)}\n")
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
        norm_freq = c / len(x)
        Fn_cum    = cum / len(x)
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
    print(f"Критическое значение χ²_0.95(df) ≈ {chi2_crit:.6f}")
    if pearson_ok:
        print("=> Нет оснований отвергнуть гипотезу о соответствии распределению.")
    else:
        print("=> Гипотеза о соответствии распределению отвергается.")

    # Графики
    plot_pair(x, counts, edges, expected_counts,
              title_left=f"Гистограмма частот ({method_name})",
              title_right="Статистическая функция распределения")

# ---------------- Точка входа ----------------
def main():
    # 1) ЦПТ
    x_clt = gen_clt(N, mu, sigma, rng)
    run_block("Метод ЦПТ", x_clt)

    # 2) Марсальи–Брей
    x_mb = gen_marsaglia_bray(N, mu, sigma, rng)
    run_block("Процедура Марсальи–Брея", x_mb)

if __name__ == "__main__":
    main()
