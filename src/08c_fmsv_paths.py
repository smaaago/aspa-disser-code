"""Plot FMSV factor latent SDs and RTSI implied SD over time."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("output")
TAB = OUT / "tables"
FIG = OUT / "figures"

fac = pd.read_csv(TAB / "table13_fmsv_factor_logvol.csv", parse_dates=["date"]).set_index("date")
ass = pd.read_csv(TAB / "table12_fmsv_asset_logvol.csv", parse_dates=["date"]).set_index("date")
rsd = pd.read_csv(TAB / "table13b_fmsv_rtsi_sd.csv", parse_dates=["date"]).set_index("date")

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Convert log-vars to SD (exp(h/2))
fac_sd = np.exp(fac.values / 2)

fig, axes = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True)
colors = ["#003049", "#a40000", "#3a7d44"]
names = ["F1 (Европа)", "F2 (Америки)", "F3 (Тихоокеан/глобальный)"]

for k, (c, n) in enumerate(zip(colors, names)):
    axes[0].plot(fac.index, fac_sd[:, k], color=c, lw=0.6, label=n)
for vs, ve, c in [("2008-09-15", "2009-06-30", "#d62828"),
                  ("2020-02-20", "2020-06-30", "#7209b7"),
                  ("2022-02-21", "2022-06-30", "#f77f00")]:
    axes[0].axvspan(pd.Timestamp(vs), pd.Timestamp(ve), alpha=0.10, color=c)
axes[0].set_title("Латентные факторные SD (FMSV K=3, постериорная средняя)",
                  loc="left", fontweight="bold")
axes[0].set_ylabel("SD фактора (×100)")
axes[0].legend(loc="upper right", frameon=False, ncol=3)

# RTSI implied SD
axes[1].plot(rsd.index, rsd["sd_rtsi"], color="#a40000", lw=0.6,
             label="RTSI implied SD (FMSV)")
for vs, ve, c in [("2008-09-15", "2009-06-30", "#d62828"),
                  ("2020-02-20", "2020-06-30", "#7209b7"),
                  ("2022-02-21", "2022-06-30", "#f77f00")]:
    axes[1].axvspan(pd.Timestamp(vs), pd.Timestamp(ve), alpha=0.10, color=c)
axes[1].set_title("RTSI: подразумеваемая FMSV условная SD",
                  loc="left", fontweight="bold")
axes[1].set_ylabel("SD")
axes[1].set_xlabel("")

plt.tight_layout()
plt.savefig(FIG / "fig7_fmsv_factor_paths.png", dpi=200)
plt.close()
print("FMSV path fig saved.")
