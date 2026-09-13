"""Generate FMSV loading bar chart from saved CSV; skip latent paths for time."""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TAB = Path("output/tables")
FIG = Path("output/figures")

load = pd.read_csv(TAB / "table10_fmsv_loadings_mean.csv", index_col=0)
print(load)

idx = np.arange(len(load))
W = 0.27
fig, ax = plt.subplots(figsize=(12, 4.5))
ax.bar(idx - W, load["F1"], width=W, color="navy", label="F1 (Европа)")
ax.bar(idx, load["F2"], width=W, color="darkred", label="F2 (Америки)")
ax.bar(idx + W, load["F3"], width=W, color="darkgreen", label="F3 (Тихоокеан/Global)")
ax.set_xticks(idx)
ax.set_xticklabels(load.index, rotation=45, ha="right")
ax.axhline(0, color="grey", lw=0.5)
ax.set_ylabel("Постериорная средняя нагрузка")
ax.set_title("Факторные нагрузки FMSV ($K=3$)")
ax.legend(loc="upper right")
plt.tight_layout()
plt.savefig(FIG / "fig8_fmsv_loadings.png", dpi=200)
plt.close()
print("Saved fig8_fmsv_loadings.png")
