"""
Modeling Variation in Wordle Results Using Log-Normal Distributions
====================================================================
Based on: "Popularity of Video Games and Collective Memory" 
(https://pmc.ncbi.nlm.nih.gov/articles/PMC9320117/)

Uses lognormal distributions to explain variation, as they emerge from 
multiplicative processes. May use Box-Cox transformations to capture memory effects.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
from scipy.stats import lognorm, norm, boxcox
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)

print("=" * 80)
print("MODELING VARIATION IN WORDLE RESULTS USING LOG-NORMAL DISTRIBUTIONS")
print("Based on: Mendes et al. (2022) - Popularity of Video Games and Collective Memory")
print("=" * 80)

# Load data
df = pd.read_csv('data/2023_MCM_Problem_C_Data.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

target_var = 'Number of  reported results'
df['days_since_start'] = (df['Date'] - df['Date'].min()).dt.days

y = df[target_var].values
print(f"\nData: {len(df)} observations")
print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
print(f"Mean: {y.mean():,.0f}, Std: {y.std():,.0f}")

# ============================================================================
# 1. TEST LOG-NORMAL DISTRIBUTION
# ============================================================================
print("\n1. Testing Log-Normal Distribution Hypothesis...")
print("   Lognormal emerges from multiplicative processes (word-of-mouth, social sharing)")

# Log transform the data
log_y = np.log(y)

# Test normality of log-transformed data
shapiro_stat, shapiro_p = stats.shapiro(log_y[:5000] if len(log_y) > 5000 else log_y)
ks_stat, ks_p = stats.kstest(log_y, 'norm', args=(log_y.mean(), log_y.std()))

print(f"   Log-transformed data:")
print(f"     Shapiro-Wilk test: W={shapiro_stat:.4f}, p={shapiro_p:.4f}")
print(f"     KS test vs normal: D={ks_stat:.4f}, p={ks_p:.4f}")

# Fit lognormal distribution
# Lognormal parameters: shape = std of log(data), scale = exp(mean of log(data))
shape = log_y.std()
scale = np.exp(log_y.mean())
location = 0

print(f"\n   Fitted Lognormal Distribution:")
print(f"     Shape (σ): {shape:.4f}")
print(f"     Scale (exp(μ)): {scale:,.0f}")
print(f"     Location: {location}")

# Test fit quality
theoretical_log = lognorm.rvs(shape, loc=location, scale=scale, size=10000, random_state=42)
observed_mean = y.mean()
observed_std = y.std()
theoretical_mean = lognorm.mean(shape, loc=location, scale=scale)
theoretical_std = lognorm.std(shape, loc=location, scale=scale)

print(f"\n   Distribution Moments:")
print(f"     Observed - Mean: {observed_mean:,.0f}, Std: {observed_std:,.0f}")
print(f"     Theoretical - Mean: {theoretical_mean:,.0f}, Std: {theoretical_std:,.0f}")

# ============================================================================
# 2. BOX-COX TRANSFORMATION (to improve lognormal fit)
# ============================================================================
print("\n2. Testing Box-Cox Transformation...")
print("   Box-Cox can improve lognormal fit and capture memory effects")

# Find optimal Box-Cox lambda
y_positive = y - y.min() + 1  # Ensure positive
bc_y, bc_lambda = stats.boxcox(y_positive)
bc_mean = bc_y.mean()
bc_std = bc_y.std()

print(f"   Optimal Box-Cox λ: {bc_lambda:.4f}")
print(f"   Interpretation: λ close to 0 ≈ log transform (multiplicative process)")
print(f"                   λ far from 0 ≈ memory effects")

# Test normality of Box-Cox transformed data
shapiro_bc, shapiro_p_bc = stats.shapiro(bc_y[:5000] if len(bc_y) > 5000 else bc_y)
ks_bc, ks_p_bc = stats.kstest(bc_y, 'norm', args=(bc_mean, bc_std))

print(f"\n   Box-Cox transformed data:")
print(f"     Shapiro-Wilk: W={shapiro_bc:.4f}, p={shapiro_p_bc:.4f}")
print(f"     KS test: D={ks_bc:.4f}, p={ks_p_bc:.4f}")

# ============================================================================
# 3. EXPLAIN VARIATION WITH FACTORS (on log-scale for multiplicative model)
# ============================================================================
print("\n3. Building Explanatory Model on Log-Scale...")
print("   Multiplicative model: Results = Trend × Day_Effect × Month_Effect × ...")
print("   Log-transform: log(Results) = log(Trend) + log(Day_Effect) + ...")

# Time factors
df['day_of_week'] = df['Date'].dt.dayofweek
df['day_name'] = df['Date'].dt.day_name()
df['month'] = df['Date'].dt.month
df['days_since_start'] = (df['Date'] - df['Date'].min()).dt.days

# Build multiplicative trend (growth-then-decay on log scale)
from scipy.optimize import curve_fit

def growth_then_decay_log(x, x0, a1, b1, a2, b2, c):
    """Growth then decay on log scale (multiplicative)"""
    result = np.zeros_like(x)
    mask_growth = x < x0
    mask_decay = x >= x0
    result[mask_growth] = a1 + b1 * x[mask_growth]
    result[mask_decay] = a2 - b2 * (x[mask_decay] - x0) + c
    return result

X_time = df['days_since_start'].values
peak_idx = np.argmax(y)
x0_guess = X_time[peak_idx]
p0 = [x0_guess, log_y[0], 0.01, log_y.max(), 0.001, log_y.min()]

popt_log, _ = curve_fit(growth_then_decay_log, X_time, log_y, p0=p0, maxfev=10000)

df['log_trend'] = growth_then_decay_log(X_time, *popt_log)
df['log_residuals'] = log_y - df['log_trend']

# Day-of-week effects on log scale (multiplicative)
day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_effects_log = df.groupby('day_name')['log_residuals'].mean().reindex(day_names)
day_effects_log_centered = day_effects_log - day_effects_log.mean()

print(f"\n   Day-of-week effects (multiplicative factors):")
for day, effect in zip(day_names, day_effects_log_centered):
    factor = np.exp(effect)
    pct = (factor - 1.0) * 100
    print(f"     {day:12s}: {factor:.4f} ({pct:+.2f}%)")

# Month effects on log scale
month_effects_log = df.groupby('month')['log_residuals'].mean()
month_effects_log_centered = month_effects_log - month_effects_log.mean()

print(f"\n   Month effects (multiplicative factors):")
for month, effect in zip(month_effects_log_centered.index, month_effects_log_centered):
    factor = np.exp(effect)
    pct = (factor - 1.0) * 100
    print(f"     Month {month:2d}: {factor:.4f} ({pct:+.2f}%)")

# Full model: log(results) = log(trend) + day_effect + month_effect + ...
df['day_factor_log'] = df['day_name'].map(day_effects_log_centered)
df['month_factor_log'] = df['month'].map(month_effects_log_centered)

df['log_prediction'] = df['log_trend'] + df['day_factor_log'] + df['month_factor_log']
df['prediction'] = np.exp(df['log_prediction'])
df['residuals_log'] = log_y - df['log_prediction']

r2_log = r2_score(log_y, df['log_prediction'])
r2_original = r2_score(y, df['prediction'])

print(f"\n   Model Performance:")
print(f"     R² on log-scale: {r2_log:.4f}")
print(f"     R² on original scale: {r2_original:.4f}")

# ============================================================================
# 4. INTERPRETATION: MULTIPLICATIVE PROCESSES
# ============================================================================
print("\n4. Interpretation: Multiplicative Processes...")
print("\n   Model Structure (Multiplicative):")
print("     Results = Trend × Day_Factor × Month_Factor × Random_Multiplicative_Noise")
print("\n   On log-scale (Additive):")
print("     log(Results) = log(Trend) + log(Day_Factor) + log(Month_Factor) + ε")
print("\n   Why Multiplicative?")
print("     - Social sharing: each person shares with N friends → N-fold growth")
print("     - Word-of-mouth: popularity multiplies through networks")
print("     - Media coverage: exposure multiplies engagement")
print("     - Daily effects: factors scale with current popularity (multiplicative)")

# Test if residuals on log-scale are normal (multiplicative noise)
resid_log_mean = df['residuals_log'].mean()
resid_log_std = df['residuals_log'].std()
shapiro_resid, shapiro_p_resid = stats.shapiro(df['residuals_log'][:5000] if len(df) > 5000 else df['residuals_log'])

print(f"\n   Residual Analysis (log-scale):")
print(f"     Mean: {resid_log_mean:.4f} (should be ~0)")
print(f"     Std: {resid_log_std:.4f}")
print(f"     Shapiro-Wilk: W={shapiro_resid:.4f}, p={shapiro_p_resid:.4f}")
if shapiro_p_resid > 0.05:
    print(f"     ✓ Residuals are approximately normal → Multiplicative model valid")
else:
    print(f"     ⚠ Residuals not normal → May need Box-Cox transformation")

# ============================================================================
# 5. PREDICT MARCH 1, 2023 WITH PREDICTION INTERVAL
# ============================================================================
print("\n5. Prediction for March 1, 2023...")

march_1 = pd.to_datetime('2023-03-01')
days_to_march_1 = (march_1 - df['Date'].min()).days
march_1_dow = march_1.weekday()  # 0=Monday
march_1_month = 3

# Log-scale prediction
log_trend_pred = growth_then_decay_log(np.array([days_to_march_1]), *popt_log)[0]
day_effect_log = day_effects_log_centered.iloc[march_1_dow]
month_effect_log = month_effects_log_centered.iloc[march_1_month-1]

log_prediction_march1 = log_trend_pred + day_effect_log + month_effect_log

# Transform back to original scale
prediction_march1 = np.exp(log_prediction_march1)

# Prediction interval (on log-scale, then transform)
# 95% interval: log_pred ± 1.96 * std(residuals)
log_pred_lower = log_prediction_march1 - 1.96 * resid_log_std
log_pred_upper = log_prediction_march1 + 1.96 * resid_log_std

pred_lower = np.exp(log_pred_lower)
pred_upper = np.exp(log_pred_upper)

print(f"\n   Prediction for March 1, 2023 (Wednesday):")
print(f"     Log-scale prediction: {log_prediction_march1:.2f}")
print(f"     Point estimate: {prediction_march1:,.0f}")
print(f"\n   Prediction Interval (95%):")
print(f"     Lower: {pred_lower:,.0f}")
print(f"     Upper: {pred_upper:,.0f}")
print(f"     Interpretation: Multiplicative interval (ratio: {pred_upper/pred_lower:.2f}x)")

# ============================================================================
# 6. VISUALIZATIONS
# ============================================================================
print("\n6. Creating visualizations...")
import os
os.makedirs('plots', exist_ok=True)

fig = plt.figure(figsize=(18, 14))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Plot 1: Distribution of results
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(y, bins=50, density=True, alpha=0.7, color='steelblue', label='Observed')
x_plot = np.linspace(y.min(), y.max(), 1000)
y_lognorm = lognorm.pdf(x_plot, shape, loc=location, scale=scale)
ax1.plot(x_plot, y_lognorm, 'r-', linewidth=2, label=f'Lognormal (σ={shape:.2f})')
ax1.set_xlabel('Number of Reported Results')
ax1.set_ylabel('Density')
ax1.set_title('Distribution of Results\n(Lognormal Fit)', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Log-transform distribution
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(log_y, bins=50, density=True, alpha=0.7, color='steelblue', label='Observed (log)')
x_log = np.linspace(log_y.min(), log_y.max(), 1000)
y_norm = norm.pdf(x_log, log_y.mean(), log_y.std())
ax2.plot(x_log, y_norm, 'r-', linewidth=2, label=f'Normal (μ={log_y.mean():.2f}, σ={log_y.std():.2f})')
ax2.set_xlabel('log(Number of Results)')
ax2.set_ylabel('Density')
ax2.set_title('Log-Transformed Distribution\n(Normal Fit)', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Q-Q plot (lognormal)
ax3 = fig.add_subplot(gs[0, 2])
stats.probplot(log_y, dist="norm", plot=ax3)
ax3.set_title('Q-Q Plot (Log-Transformed Data)\nvs Normal Distribution', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Plot 4: Time series with multiplicative model
ax4 = fig.add_subplot(gs[1, :])
ax4.plot(df['Date'], y, 'k-', linewidth=1, alpha=0.5, label='Actual', zorder=1)
ax4.plot(df['Date'], df['prediction'], 'r-', linewidth=2, alpha=0.8, label='Multiplicative Model', zorder=2)
ax4.plot(df['Date'], np.exp(df['log_trend']), 'b--', linewidth=1.5, alpha=0.7, label='Base Trend', zorder=3)
ax4.set_ylabel('Number of Reported Results')
ax4.set_title('Time Series: Actual vs Multiplicative Model\n(Log-Normal Framework)', fontsize=12, fontweight='bold')
ax4.legend(loc='upper left')
ax4.grid(True, alpha=0.3)

# Plot 5: Residuals on log-scale
ax5 = fig.add_subplot(gs[2, 0])
ax5.plot(df['Date'], df['residuals_log'], 'g-', linewidth=1, alpha=0.7)
ax5.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax5.set_ylabel('Residuals (log-scale)')
ax5.set_xlabel('Date')
ax5.set_title(f'Residuals on Log-Scale\n(std={resid_log_std:.3f})', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3)

# Plot 6: Residual distribution (log-scale)
ax6 = fig.add_subplot(gs[2, 1])
ax6.hist(df['residuals_log'], bins=30, density=True, alpha=0.7, color='steelblue', label='Residuals')
x_resid = np.linspace(df['residuals_log'].min(), df['residuals_log'].max(), 100)
y_resid_norm = norm.pdf(x_resid, resid_log_mean, resid_log_std)
ax6.plot(x_resid, y_resid_norm, 'r-', linewidth=2, label='Normal Fit')
ax6.set_xlabel('Residuals (log-scale)')
ax6.set_ylabel('Density')
ax6.set_title('Residual Distribution\n(Should be Normal)', fontsize=12, fontweight='bold')
ax6.legend()
ax6.grid(True, alpha=0.3)

# Plot 7: Day-of-week effects (multiplicative factors)
ax7 = fig.add_subplot(gs[2, 2])
day_factors = np.exp(day_effects_log_centered)
colors = ['steelblue' if d not in ['Saturday', 'Sunday'] else 'coral' for d in day_names]
x_pos = np.arange(len(day_names))
ax7.bar(x_pos, day_factors - 1.0, alpha=0.7, color=colors)
ax7.set_xticks(x_pos)
ax7.set_xticklabels([d[:3] for d in day_names])
ax7.set_ylabel('Multiplicative Factor - 1')
ax7.set_title('Day-of-Week Effects\n(Multiplicative Factors)', fontsize=12, fontweight='bold')
ax7.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax7.grid(True, alpha=0.3, axis='y')

plt.savefig('plots/27_lognormal_model.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: plots/27_lognormal_model.png")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("LOG-NORMAL MODEL SUMMARY")
print("=" * 80)
print(f"\nModel Framework: Multiplicative Process (Lognormal Distribution)")
print(f"  Based on: Mendes et al. (2022) - Popularity of Video Games and Collective Memory")
print(f"  https://pmc.ncbi.nlm.nih.gov/articles/PMC9320117/")
print(f"\nKey Insight: Variation explained by multiplicative processes")
print(f"  - Social sharing multiplies popularity")
print(f"  - Daily effects scale with current popularity")
print(f"  - Random factors multiply (lognormal) rather than add (normal)")

print(f"\nModel Structure:")
print(f"  Results = Trend × Day_Factor × Month_Factor × ε")
print(f"  log(Results) = log(Trend) + log(Day_Factor) + log(Month_Factor) + log(ε)")

print(f"\nModel Performance:")
print(f"  R² (log-scale): {r2_log:.4f}")
print(f"  R² (original scale): {r2_original:.4f}")
print(f"  Residual std (log): {resid_log_std:.4f}")

print(f"\nMarch 1, 2023 Prediction:")
print(f"  Point estimate: {prediction_march1:,.0f}")
print(f"  95% Interval: [{pred_lower:,.0f}, {pred_upper:,.0f}]")
print(f"  Interval is multiplicative: {pred_upper/pred_lower:.2f}x range")

print("\n" + "=" * 80)

