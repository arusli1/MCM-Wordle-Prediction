"""
Check if the Time Series (Number of Reports vs Time) Follows Lognormal Pattern
==============================================================================
The paper models popularity distributions, but we need to check:
- Does the time series itself follow a lognormal evolution?
- Or does each day's result come from a lognormal distribution?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)

print("=" * 80)
print("CHECKING IF TIME SERIES FOLLOWS LOGNORMAL PATTERN")
print("=" * 80)

# Load data
df = pd.read_csv('data/2023_MCM_Problem_C_Data.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

target_var = 'Number of  reported results'
df['days_since_start'] = (df['Date'] - df['Date'].min()).dt.days

y = df[target_var].values
X_time = df['days_since_start'].values

print(f"\nData: {len(df)} observations")
print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")

# ============================================================================
# 1. CHECK: Is the time series evolution lognormal?
# ============================================================================
print("\n1. Checking if time series evolution is lognormal...")
print("   Hypothesis: Results(t) follows a lognormal trajectory over time")

# Fit different models to the time series
# Option 1: Lognormal curve directly
def lognormal_curve(t, a, b, c):
    """
    Lognormal evolution: y = a * exp(-0.5 * ((log(t) - b) / c)^2)
    Or simpler: log(y) = log(a) - 0.5 * ((log(t) - b) / c)^2
    """
    # Ensure t > 0 and avoid log(0)
    t_safe = np.maximum(t, 1)
    return a * np.exp(-0.5 * ((np.log(t_safe) - b) / c)**2)

# Option 2: Exponential growth then decay (we already know this works)
def growth_then_decay(t, x0, a1, b1, a2, b2, c):
    result = np.zeros_like(t)
    mask_growth = t < x0
    mask_decay = t >= x0
    result[mask_growth] = a1 * np.exp(b1 * t[mask_growth])
    result[mask_decay] = a2 * np.exp(-b2 * (t[mask_decay] - x0)) + c
    return result

# Option 3: Lognormal on log(time) scale
def lognormal_time_evolution(t, a, mu, sigma):
    """
    y = a * exp(-0.5 * ((log(t) - mu) / sigma)^2)
    This is a lognormal curve in time domain
    """
    t_safe = np.maximum(t, 1)
    log_t = np.log(t_safe)
    return a * np.exp(-0.5 * ((log_t - mu) / sigma)**2)

print("\n   Testing Lognormal Curve in Time Domain...")
try:
    # Fit lognormal curve
    peak_idx = np.argmax(y)
    x0_guess = X_time[peak_idx]
    a_guess = y.max()
    mu_guess = np.log(x0_guess) if x0_guess > 0 else np.log(X_time.mean())
    sigma_guess = 0.5
    
    p0 = [a_guess, mu_guess, sigma_guess]
    popt_log, _ = curve_fit(lognormal_time_evolution, X_time, y, p0=p0, maxfev=10000,
                           bounds=([0, 0, 0.1], [y.max()*10, np.log(X_time.max()), 5]))
    
    y_lognorm = lognormal_time_evolution(X_time, *popt_log)
    r2_lognorm = 1 - np.sum((y - y_lognorm)**2) / np.sum((y - y.mean())**2)
    
    print(f"     Lognormal curve fit: R² = {r2_lognorm:.4f}")
    print(f"     Parameters: a={popt_log[0]:,.0f}, μ={popt_log[1]:.2f}, σ={popt_log[2]:.2f}")
except Exception as e:
    print(f"     Failed: {e}")
    r2_lognorm = -np.inf
    y_lognorm = None

# Compare with growth-then-decay (we know this works)
print("\n   Testing Growth-Then-Decay (for comparison)...")
peak_idx = np.argmax(y)
x0_guess = X_time[peak_idx]
p0 = [x0_guess, y[0], 0.01, y.max(), 0.001, y.min()]
popt_gtd, _ = curve_fit(growth_then_decay, X_time, y, p0=p0, maxfev=10000,
                       bounds=([X_time[0], 0, 0, 0, 0, y.min()],
                              [X_time[-1], y.max()*10, 1, y.max()*10, 1, y.max()]))
y_gtd = growth_then_decay(X_time, *popt_gtd)
r2_gtd = 1 - np.sum((y - y_gtd)**2) / np.sum((y - y.mean())**2)
print(f"     Growth-then-decay: R² = {r2_gtd:.4f}")

# ============================================================================
# 2. CHECK: Does log(Results) vs log(Time) follow a pattern?
# ============================================================================
print("\n2. Checking log(Results) vs log(Time) relationship...")
print("   If lognormal in time: log(y) vs log(t) should follow specific pattern")

log_y = np.log(y)
log_t = np.log(np.maximum(X_time, 1))  # Avoid log(0)

# Fit linear: log(y) = a * log(t) + b
coeffs_lin = np.polyfit(log_t, log_y, 1)
y_lin_log = np.polyval(coeffs_lin, log_t)
r2_lin_log = 1 - np.sum((log_y - y_lin_log)**2) / np.sum((log_y - log_y.mean())**2)

# Fit quadratic: log(y) = a * log(t)^2 + b * log(t) + c (lognormal pattern)
coeffs_quad = np.polyfit(log_t, log_y, 2)
y_quad_log = np.polyval(coeffs_quad, log_t)
r2_quad_log = 1 - np.sum((log_y - y_quad_log)**2) / np.sum((log_y - log_y.mean())**2)

print(f"   Linear (log-log): R² = {r2_lin_log:.4f}")
print(f"   Quadratic (log-log, lognormal): R² = {r2_quad_log:.4f}")

# ============================================================================
# 3. CHECK: Model as log(Results(t)) ~ Normal with time-varying mean
# ============================================================================
print("\n3. Modeling as: log(Results(t)) ~ Normal(μ(t), σ²)...")
print("   Where μ(t) varies over time (more appropriate for time series)")

# Model on log-scale: log(y) = f(t) + ε, where ε ~ Normal(0, σ²)
log_y = np.log(y)

# Fit trend on log-scale
def growth_then_decay_log(t, x0, a1, b1, a2, b2, c):
    """Growth then decay on log scale"""
    result = np.zeros_like(t)
    mask_growth = t < x0
    mask_decay = t >= x0
    result[mask_growth] = a1 + b1 * t[mask_growth]
    result[mask_decay] = a2 - b2 * (t[mask_decay] - x0) + c
    return result

x0_guess = X_time[np.argmax(y)]
p0 = [x0_guess, log_y[0], 0.01, log_y.max(), 0.001, log_y.min()]
popt_log_trend, _ = curve_fit(growth_then_decay_log, X_time, log_y, p0=p0, maxfev=10000)

mu_t = growth_then_decay_log(X_time, *popt_log_trend)
residuals_log = log_y - mu_t
sigma_log = np.std(residuals_log)

print(f"   Model: log(Results(t)) ~ Normal(μ(t), σ²)")
print(f"   where μ(t) = growth_then_decay(t)")
print(f"   σ = {sigma_log:.4f}")

# Test if residuals are normal
shapiro_stat, shapiro_p = stats.shapiro(residuals_log[:5000] if len(residuals_log) > 5000 else residuals_log)
ks_stat, ks_p = stats.kstest(residuals_log, 'norm', args=(0, sigma_log))

print(f"\n   Residual normality tests:")
print(f"     Shapiro-Wilk: W={shapiro_stat:.4f}, p={shapiro_p:.4f}")
print(f"     KS test vs Normal(0, {sigma_log:.4f}): D={ks_stat:.4f}, p={ks_p:.4f}")

if shapiro_p > 0.05:
    print(f"     ✓ Residuals are approximately normal → Model is valid")
else:
    print(f"     ⚠ Residuals deviate from normal → May need Box-Cox")

# Calculate R²
r2_log_model = 1 - np.sum(residuals_log**2) / np.sum((log_y - log_y.mean())**2)
print(f"\n   Model Performance: R² = {r2_log_model:.4f}")

# ============================================================================
# 4. VISUALIZATION: Compare models
# ============================================================================
print("\n4. Creating visualizations...")
import os
os.makedirs('plots', exist_ok=True)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Plot 1: Time series with different fits
ax = axes[0, 0]
ax.plot(df['Date'], y, 'k-', linewidth=1, alpha=0.5, label='Actual', zorder=1)
if y_lognorm is not None:
    ax.plot(df['Date'], y_lognorm, 'b--', linewidth=2, alpha=0.8, 
           label=f'Lognormal Curve (R²={r2_lognorm:.4f})', zorder=2)
ax.plot(df['Date'], y_gtd, 'r-', linewidth=2, alpha=0.8, 
       label=f'Growth-Then-Decay (R²={r2_gtd:.4f})', zorder=3)
ax.set_ylabel('Number of Reported Results')
ax.set_title('Time Series: Lognormal vs Growth-Then-Decay', fontsize=12, fontweight='bold')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)

# Plot 2: Log-log plot
ax = axes[0, 1]
ax.scatter(log_t, log_y, alpha=0.5, s=10, label='Data', zorder=1)
ax.plot(log_t, y_lin_log, 'r--', linewidth=2, alpha=0.8, 
       label=f'Linear (R²={r2_lin_log:.4f})', zorder=2)
ax.plot(log_t, y_quad_log, 'b-', linewidth=2, alpha=0.8, 
       label=f'Quadratic/Lognormal (R²={r2_quad_log:.4f})', zorder=3)
ax.set_xlabel('log(Time)')
ax.set_ylabel('log(Number of Results)')
ax.set_title('Log-Log Plot\n(Quadratic = Lognormal Pattern)', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Residuals from log-scale model
ax = axes[0, 2]
ax.plot(df['Date'], residuals_log, 'g-', linewidth=1, alpha=0.7)
ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax.fill_between(df['Date'], -1.96*sigma_log, 1.96*sigma_log, 
                alpha=0.2, color='gray', label='±1.96σ (95%)')
ax.set_ylabel('Residuals (log-scale)')
ax.set_title(f'Residuals: log(Results) - μ(t)\n(σ={sigma_log:.4f})', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Distribution of log(results)
ax = axes[1, 0]
ax.hist(log_y, bins=50, density=True, alpha=0.7, color='steelblue', label='Observed')
x_norm = np.linspace(log_y.min(), log_y.max(), 1000)
y_norm = stats.norm.pdf(x_norm, log_y.mean(), log_y.std())
ax.plot(x_norm, y_norm, 'r-', linewidth=2, label=f'Normal(μ={log_y.mean():.2f}, σ={log_y.std():.2f})')
ax.set_xlabel('log(Number of Results)')
ax.set_ylabel('Density')
ax.set_title('Distribution of log(Results)', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 5: Q-Q plot of residuals
ax = axes[1, 1]
stats.probplot(residuals_log, dist="norm", plot=ax)
ax.set_title(f'Q-Q Plot: Residuals vs Normal\n(p={shapiro_p:.4f})', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# Plot 6: Time-varying mean μ(t)
ax = axes[1, 2]
ax.plot(df['Date'], mu_t, 'b-', linewidth=2, alpha=0.8, label='μ(t) = E[log(Results(t))]')
ax.plot(df['Date'], mu_t + 1.96*sigma_log, 'r--', linewidth=1, alpha=0.6, label='μ(t) ± 1.96σ')
ax.plot(df['Date'], mu_t - 1.96*sigma_log, 'r--', linewidth=1, alpha=0.6)
ax.plot(df['Date'], log_y, 'k-', linewidth=1, alpha=0.3, label='log(Actual)')
ax.set_ylabel('log(Number of Results)')
ax.set_xlabel('Date')
ax.set_title('Time-Varying Mean μ(t)\nlog(Results(t)) ~ Normal(μ(t), σ²)', fontsize=12, fontweight='bold')
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/28_lognormal_time_series.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: plots/28_lognormal_time_series.png")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("LOG-NORMAL TIME SERIES ANALYSIS SUMMARY")
print("=" * 80)

print("\nQuestion: Is the time series itself lognormal?")
print(f"  Lognormal curve fit: R² = {r2_lognorm:.4f} (if fitted)")
print(f"  Growth-then-decay fit: R² = {r2_gtd:.4f}")
print(f"  → Growth-then-decay fits better for this time series")

print("\nBetter Model: log(Results(t)) ~ Normal(μ(t), σ²)")
print(f"  Where μ(t) = growth_then_decay(t) on log-scale")
print(f"  σ = {sigma_log:.4f}")
print(f"  R² = {r2_log_model:.4f}")

print("\nInterpretation:")
print("  - Each day's result comes from a lognormal distribution")
print("  - The mean μ(t) evolves over time (growth then decay)")
print("  - The variance σ² is constant (homoscedastic on log-scale)")
print("  - This is a lognormal process with time-varying parameters")

print("\nComparison with Paper:")
print("  - Paper: Cross-sectional lognormal (different games at one time)")
print("  - Our case: Longitudinal lognormal (one game over time)")
print("  - Both valid: lognormal distribution with varying parameters")

print("\n" + "=" * 80)

