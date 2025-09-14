from scipy.signal import square
from scipy.stats import chi2

#значения для гениратора случайных числел
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
xi2 = sum(((c - expected) ** 2) / expected for c in counts)

#Критерий Пирсона (хи квадрат):
step_freedom = K - 1
alpha = 0.05
crit = chi2.ppf(1 - alpha,step_freedom)

#Критерий Колмогорова
Ka = 1.36 # из-за альфа 0,05
d_crit = Ka/(N**0.5)
x_sorted = sorted(x)
d_plus = max((k + 1)/N - x_sorted[k] for k in range(N))   # k идёт с 0 до N-1
d_minus = max(x_sorted[k] - k/N for k in range(N))
d = max(d_plus, d_minus)
#Нулей
p = 0.25
bits = [1 if xi < p else 0 for xi in x]

# Подсчёт серий единиц
runs_ones = []
cur = 0
for b in bits:
    if b == 1:
        cur += 1
    else:
        if cur > 0:
            runs_ones.append(cur)
            cur = 0
if cur > 0:
    runs_ones.append(cur)

total_runs = len(runs_ones)
avg_run_length = sum(runs_ones) / total_runs if total_runs > 0 else 0
theor_avg_run_length = 1 / (1 - p)

# распределение длин серий
p = 0.5
bits = [0 if xi < p else 1 for xi in x]  # теперь 0 - "успех"

# Подсчёт серий нулей
runs_zeros = []
cur = 0
for b in bits:
    if b == 0:
        cur += 1
    else:
        if cur > 0:
            runs_zeros.append(cur)
            cur = 0
if cur > 0:
    runs_zeros.append(cur)

total_runs = len(runs_zeros)
avg_run_length = sum(runs_zeros) / total_runs if total_runs > 0 else 0
theor_avg_run_length = 1 / (1 - p)

# распределение длин серий
dist = {}
for L in runs_zeros:
    if L >= 9:
        dist['>=9'] = dist.get('>=9', 0) + 1
    else:
        dist[L] = dist.get(L, 0) + 1

print("\nВыборочная средняя:", round(mean, 6))
print("Математическое ожидание (теор.):", theor_mean)
print("Несмещенная оценка дисперсии:", round(variance, 6))
print("Требуемая дисперсия (теор.):", round(theor_var, 6))
print("Второй момент:", round(moment2, 6), f"(теор. 1/3 = {theor_m2:.6f})")
print("Третий момент:", round(moment3, 6), f"(теор. 1/4 = {theor_m3:.6f})")
print("Коэффициент ХИ-квадрат:", round(xi2, 6))
print()
print('================================================================================================================')
print('Статистические критерии:')
print('1. Критерий Пирсона:')
print('   Статистика хи-квадрат:',xi2)
print('   Степени свободы:',step_freedom)
print(f'   Критическое значение (alpha= 0,05:{crit:.6f}')
if xi2 <= crit:
    print('   Распределение равномерное(не отвергаем Н0)')
else:
    print('   Распределение не равномерное(отвергаем Н0)')
print()

print('2. Критерий Колмогорова:')
print(f'   Статистика Колмогорова: {d:.6f}')
print(f'   Критическое значение (alpha= 0,05):{d_crit:.6f}')
if d <= d_crit:
    print('   Распределение равномерное(не отвергаем Н0)')
else:
    print('   Распределение не равномерное(отвергаем Н0)')
print()

print(f"3.Тест длины серий нулей (p = {p:.4f}):")
print(f"   Общее количество серий: {total_runs}")
print(f"   Средняя длина серий: {avg_run_length:.3f}")
print(f"   Теоретическая средняя длина: {theor_avg_run_length:.3f}")
print("   Распределение длин серий:")
for L in range(1, 9):
    if L in dist:
        print(f"Длина {L}: {dist[L]} серий")
if '>=9' in dist:
    print(f"Длина >=9: {dist['>=9']} серий")
