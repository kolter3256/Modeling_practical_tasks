
import matplotlib.pyplot as plt

import numpy as np

# ---------------- ПАРАМЕТРЫ ЭКСПЕРИМЕНТА ----------------
M = 10         # число шагов блуждания (по условию)
N = 20000      # число траекторий (экспериментов) >= 1000
K = 25         # число интервалов для гистограммы (15 или 25)
RNG_SEED = 42  # для воспроизводимости

# Печатать ли такие же таблицы гистограммы по каждой координате
PRINT_COORD_TABLES = False

rng = np.random.default_rng(RNG_SEED)

# ---------------- Кубическая решётка: 6 направлений ----------------
# Единичный шаг только вдоль осей: ±X, ±Y, ±Z
dirs = np.array([
    [ 1,  0,  0],
    [-1,  0,  0],
    [ 0,  1,  0],
    [ 0, -1,  0],
    [ 0,  0,  1],
    [ 0,  0, -1],
], dtype=int)  # shape=(6,3)

# ---------------- Генерация N траекторий по M шагов ----------------
idx = rng.integers(0, 6, size=(N, M))  # индексы выбранных направлений
steps = dirs[idx]                       # (N, M, 3)
pos = steps.sum(axis=1)                 # конечные координаты (x,y,z), shape=(N,3)

DX = pos[:, 0]
DY = pos[:, 1]
DZ = pos[:, 2]
R  = np.linalg.norm(pos, axis=1)        # расстояние от начала

# ---------------- Статистики выборки ----------------
def mean_var(a: np.ndarray):
    # Используем смещённую оценку дисперсии (ddof=0), как в «большой» сводке
    return float(np.mean(a)), float(np.var(a, ddof=0))

mean_r, var_r = mean_var(R)
mean_dx, var_dx = mean_var(DX)
mean_dy, var_dy = mean_var(DY)
mean_dz, var_dz = mean_var(DZ)

# ---------------- Форматирование и печать ----------------
def print_hist_table(sample: np.ndarray, bins: int, title: str):
    counts, edges = np.histogram(sample, bins=bins, range=(0.0, float(sample.max())))
    rel = counts / sample.size
    cum = np.cumsum(rel)

    print(title)
    print("Интервал                Кол-во   Норм.частота   Меньше или равно")
    print("-----------------------------------------------------------------")
    for i in range(bins):
        a, b = edges[i], edges[i + 1]
        print(f"[{a:6.3f} - {b:6.3f})    {counts[i]:6d}       {rel[i]:0.4f}           {cum[i]:0.4f}")

def print_signed_hist_table(sample: np.ndarray, bins: int, title: str):
    # Таблица для координатных компонент (могут быть отрицательные значения)
    lo, hi = float(sample.min()), float(sample.max())
    # защитимся от случая lo==hi
    if lo == hi:
        lo -= 0.5
        hi += 0.5
    counts, edges = np.histogram(sample, bins=bins, range=(lo, hi))
    rel = counts / sample.size
    cum = np.cumsum(rel)

    print(title)
    print("Интервал                Кол-во   Норм.частота   Меньше или равно")
    print("-----------------------------------------------------------------")
    for i in range(bins):
        a, b = edges[i], edges[i + 1]
        print(f"[{a:7.3f} - {b:7.3f})   {counts[i]:6d}       {rel[i]:0.4f}           {cum[i]:0.4f}")

# ---------------- Отчёт ----------------
print("РЕЗУЛЬТАТЫ СТАТИСТИЧЕСКОГО АНАЛИЗА")
print("==================================")
print(f"Число экспериментов: {N}")
print(f"Число шагов: {M}")
print(f"Математическое ожидание r: {mean_r:.6f}")
print(f"Дисперсия r:             {var_r:.6f}\n")

print_hist_table(R, K, "ГИСТОГРАММА РАСПРЕДЕЛЕНИЯ R")

print("\nСводка по координатам (ΔX, ΔY, ΔZ):")
print(f"  ΔX: mean = {mean_dx:.6f}, var = {var_dx:.6f}")
print(f"  ΔY: mean = {mean_dy:.6f}, var = {var_dy:.6f}")
print(f"  ΔZ: mean = {mean_dz:.6f}, var = {var_dz:.6f}")

if PRINT_COORD_TABLES:
    print()
    print_signed_hist_table(DX, K, "\nГИСТОГРАММА ΔX")
    print_signed_hist_table(DY, K, "\nГИСТОГРАММА ΔY")
    print_signed_hist_table(DZ, K, "\nГИСТОГРАММА ΔZ")

SAVE_FIGS = False           # True — сохранить PNG рядом со скриптом
PLOT_COORD_HISTS = False    # True — рисовать ещё ΔX/ΔY/ΔZ


plt.figure()
plt.hist(R, bins=K, density=True, edgecolor='black')
plt.title(f"Гистограмма R (M={M}, N={N})")
plt.xlabel("R"); plt.ylabel("Плотность")
if SAVE_FIGS: plt.savefig("R_hist.png", dpi=150, bbox_inches="tight")
plt.show()



if PLOT_COORD_HISTS:
    for name, data in (("ΔX", DX), ("ΔY", DY), ("ΔZ", DZ)):
        plt.figure()
        plt.hist(data, bins=K, density=True, edgecolor='black')
        plt.title(f"Гистограмма {name} (M={M}, N={N})")
        plt.xlabel(name); plt.ylabel("Плотность")
        if SAVE_FIGS: plt.savefig(f"{name}_hist.png", dpi=150, bbox_inches="tight")
        plt.show()
