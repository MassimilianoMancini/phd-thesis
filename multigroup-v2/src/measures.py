from math import sqrt
from statistics import mean, variance
from itertools import combinations, product
import matplotlib.pyplot as plt

def minmax(arr):
    mn = min(arr)
    mx = max(arr)
    return [(x - mn) / (mx - mn) if mx != mn else 0 for x in arr]


# ---------------------------------------------------------
# 1) Generazione dei valori di v secondo il tuo ciclo
# ---------------------------------------------------------

k = 8
step = 0.1

all_v = []

arr = [0.0] * k

for i in range(k):
    # prima fase: includi step 0
    if i == 0:
        steps = range(11)          # 0 → 10
    else:
        steps = range(1, 11)       # 1 → 10 (evita duplicati)

    for step in steps:
        arr[i] = step * 0.1
        all_v.append(arr.copy())


# ---------------------------------------------------------
# 2) Calcolo di s, variance, ep per ogni v
# ---------------------------------------------------------

SL2 = []
V = []
EP = []
DP = []

for v in all_v:

    d = sqrt(sum(i*i for i in v))

    # ---- caso speciale: vettore nullo ----
    if d == 0:
        SL2.append(0)
        V.append(0)
        EP.append(0)
        DP.append(0)
        continue

    # ---- calcolo di s ----
    s = sum(((i / d) - 1/sqrt(k))**2 for i in v)
    SL2.append(s)

    # ---- calcolo della variance ----
    V.append(variance(v))

    # ---- calcolo di ep ----
    ep = []
    j = 0
    for a, b in combinations(v, 2):
        ep.append(abs(a - b))
    EP.append(mean(ep))
    if k > 2:
        DP.append(variance(ep))
    else:
        DP.append(0)  # Per k=2, la varianza di ep non è definita (un solo valore)

# ---------------------------------------------------------
# 2c) Min-max normalization for all four arrays
# ---------------------------------------------------------

SL2_norm = minmax(SL2)
V_norm = minmax(V)
EP_norm = minmax(EP)
DP_norm = minmax(DP)

# ---------------------------------------------------------
# 3) Scatter plot without X-axis labels
# ---------------------------------------------------------

x = list(range(len(all_v)))

plt.figure(figsize=(18, 8))

plt.scatter(x, SL2_norm, label="s (normalized)", color="blue")
plt.scatter(x, V_norm, label="variance (normalized)", color="red")
plt.scatter(x, EP_norm, label="ep (normalized)", color="green")
plt.scatter(x, DP_norm, label="ep variance (normalized)", color="purple")
x_labels = [f"{tuple(round(x,1) for x in v)}" for v in all_v]

# Remove X-axis tick labels
plt.xticks(x, x_labels, rotation=90, fontsize=8)

plt.xlabel("Vector index")
plt.ylabel("Normalized values (0–1)")
plt.title("Normalized evolution of s, variance, ep and ep variance")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("normalized_plot_no_xlabels.png", dpi=300)
plt.show()

