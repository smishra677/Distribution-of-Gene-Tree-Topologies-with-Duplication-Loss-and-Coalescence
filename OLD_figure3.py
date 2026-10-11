import numpy as np
import matplotlib.pyplot as plt

ts = 5
td_values = np.linspace(0, 15, 1000)

blue_values = []
orange_values = []
green_values = []
red_values = []
purple_values = []

for td in td_values:
    tdts = td - ts
    tstd = ts - td

    if td < ts:
        expstd = np.exp(-tstd)
        blue_expr = 0
        orange_expr = 0
        green_expr = (1/6) * (3 - 2 * expstd)
        red_expr = (1/6) * expstd
        purple_expr = (1/6) * expstd
    else:
        expdts = np.exp(-tdts)
        expdt = np.exp(-2 * tdts)
        blue_expr = (1 - expdts) ** 2
        orange_expr = (1 - expdts) * (1/3) * expdts
        green_expr = (1/6) * expdt
        red_expr = (1/6) * (3 * expdts - 2 * expdt)
        purple_expr = (1/6) * expdt

    blue_values.append(blue_expr)
    orange_values.append(orange_expr)
    green_values.append(green_expr)
    red_values.append(red_expr)
    purple_values.append(purple_expr)

all_expressions = np.vstack([blue_values, orange_values, green_values, red_values, purple_values]).T
row_sums = all_expressions.sum(axis=1, keepdims=True)
normalized_expressions = all_expressions / row_sums

blue_norm, orange_norm, green_norm, red_norm, purple_norm = normalized_expressions.T

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(td_values, blue_norm, label="Tree 1:        ((A1,B1),(A2,B2))", linewidth=2, color='blue')
ax.plot(td_values, orange_norm, label="Tree 2,3:     (((A2,B_2),B1),A1) and ((A2,B2),A1),B1)", linewidth=2, color='orange')
ax.plot(td_values, green_norm, label="Tree 4,5:     ((B1,B2),A1) and ((A1,A2),B1)", linewidth=2, color='green')
ax.plot(td_values, red_norm, label="Tree 6,7:     ((A1,B1),B2) and ((A1,B1),A2)", linewidth=2, color='red')
ax.plot(td_values, purple_norm, label="Tree 8,9:     ((A2,B1),A1) and ((B2,A1),B1) ", linewidth=2, color='purple')


ax.axvline(x=ts, color='red', linestyle='--')
#ax.axhline(y=0.333, color='green', linestyle='--', label='Pr=0.33')


ax.set_xlabel("Time of Duplication") 
ax.set_ylabel("Probability")
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend().set_visible(False)
plt.legend()

plt.savefig('figure3.png',dpi=200)
plt.show()
