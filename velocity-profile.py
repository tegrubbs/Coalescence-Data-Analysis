import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Plot inlet velocity profile for bubbles injected in experiment

Lx = 5.0*.001  # channel width, m

df = pd.read_csv("experimental-inlet-data.csv")
x = df["inlet_x_mm"].to_numpy()*.001


def profile(x, a, b):
    return a * (1 - (np.abs(x - Lx / 2) / (Lx / 2)) ** b)


def fit_and_plot(ax, x, v, ylabel, title):
    popt, pcov = curve_fit(profile, x, v, p0=[v.max(), 2.0])
    a_fit, b_fit = popt
    perr = np.sqrt(np.diag(pcov))
    print(f"{title}: a = {a_fit:.4f} +/- {perr[0]:.4f}, b = {b_fit:.4f} +/- {perr[1]:.4f}")

    x_fit = np.linspace(0, Lx, 200)
    v_fit = profile(x_fit, a_fit, b_fit)

    ax.scatter(x, v, s=10, alpha=0.5, label="Data")
    ax.plot(x_fit, v_fit, color="C1", lw=2,
            label=f"Fit: a={a_fit:.1f}, b={b_fit:.2f}")
    ax.set_xlabel("Inlet x coordinate (mm)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fit_and_plot(ax1, x, df["initial_vertical_velocity_mm_s"].to_numpy()*.001,
             "Initial vertical velocity (m/s)",
             "Vertical Velocity vs. Inlet X Coordinate")
fit_and_plot(ax2, x, df["initial_horizontal_velocity_mm_s"].to_numpy()*.001,
             "Initial horizontal velocity (m/s)",
             "Horizontal Velocity vs. Inlet X Coordinate")

fig.tight_layout()
plt.show()
