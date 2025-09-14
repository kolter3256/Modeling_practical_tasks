

A = 165
Mu = 3463
Mm = 4096 * 4
Y = 3887
N = 4000   # объем выборки
K = 16     # число интервалов разбиения


def rnd():
    global Y
    Y = (A * Y + Mu) % Mm
    return Y / Mm


# генерируем последовательность
x = [rnd() for _ in range(N)]

# --- распределение по интервалам ---
interval_len = 1.0 / K
counts = [0] * K

for xi in x:
    idx = min(int(xi / interval_len), K - 1)
    counts[idx] += 1

print(f"Распределение чисел по интервалам [{interval_len:.6f}]:")
cum = 0
for i, c in enumerate(counts, 1):
    norm_freq = c / (N / K)
    cum += c
    border = (i * interval_len)
    print(f"{i:2d}-ый интервал: {c:3d}\tнорм. частота: {norm_freq:.3f}\tменьше или равно: {border:.6f}")

#
mean = sum(x) / N
variance = sum((xi - mean) ** 2 for xi in x) / N
moment2 = sum(xi ** 2 for xi in x) / N
moment3 = sum(xi ** 3 for xi in x) / N

# теоретические значения для равномерного распределения U[0,1)
theor_mean = 0.5
theor_var = 1 / 12
theor_m2 = 1 / 3
theor_m3 = 1 / 4

# хи-квадрат
expected = N / K
chi2 = sum(((c - expected) ** 2) / expected for c in counts)

print("\nВыборочная средняя:", round(mean, 6))
print("Математическое ожидание (теор.):", theor_mean)
print("Несмещенная оценка дисперсии:", round(variance, 6))
print("Требуемая дисперсия (теор.):", round(theor_var, 6))
print("Второй момент:", round(moment2, 6), f"(теор. 1/3 = {theor_m2:.6f})")
print("Третий момент:", round(moment3, 6), f"(теор. 1/4 = {theor_m3:.6f})")
print("Коэффициент ХИ-квадрат:", round(chi2, 6))
