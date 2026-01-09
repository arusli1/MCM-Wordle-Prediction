"""
Testing Additive vs Multiplicative Day-of-Week Effects
======================================================
Compare different ways to model day-of-week effects:
1. Additive: prediction = trend + day_effect
2. Multiplicative: prediction = trend * day_factor
3. Mixed: prediction = trend * day_factor + day_effect
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import splrep, splev
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

print("=" * 80)
print("COMPARING ADDITIVE vs MULTIPLICATIVE DAY-OF-WEEK EFFECTS")
print("=" * 80)

# ============================================================================
# 1. LOAD DATA AND FIT BASE TREND
# ============================================================================
print("\n1. Loading data and fitting base trend...")
df = pd.read_csv('data/2023_MCM_Problem_C_Data.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

target_var = 'Number of  reported results'
df['day_of_week'] = df['Date'].dt.dayofweek
df['day_name'] = df['Date'].dt.day_name()
df['days_since_start'] = (df['Date'] - df['Date'].min()).dt.days

X_time = df['days_since_start'].values
y = df[target_var].values

# Fit cubic B-spline trend (40 knots)
n_knots = 40
knots = np.linspace(X_time.min(), X_time.max(), n_knots)
internal_knots = knots[1:-1]
tck = splrep(X_time, y, t=internal_knots, k=3, s=0)
trend_base = splev(X_time, tck)
df['trend_base'] = trend_base
df['residuals_base'] = y - trend_base

print(f"   Base trend R²: {r2_score(y, trend_base):.4f}")

# ============================================================================
# 2. ADDITIVE MODEL
# ============================================================================
print("\n2. Additive Model: prediction = trend + day_effect")
print("-" * 80)

# Calculate mean residual for each day (additive effect)
day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_effects_add = df.groupby('day_name')['residuals_base'].mean().reindex(day_names)
day_effects_add_centered = day_effects_add - day_effects_add.mean()

# Predictions
df['pred_additive'] = df['trend_base'] + df['day_name'].map(day_effects_add_centered)
df['residuals_additive'] = y - df['pred_additive']

r2_add = r2_score(y, df['pred_additive'])
rmse_add = np.sqrt(mean_squared_error(y, df['pred_additive']))
mae_add = mean_absolute_error(y, df['pred_additive'])

print(f"   R² = {r2_add:.4f}, RMSE = {rmse_add:,.0f}, MAE = {mae_add:,.0f}")
print(f"   Day effects (additive):")
for day, effect in zip(day_names, day_effects_add_centered):
    print(f"     {day:12s}: {effect:+.0f}")

# ============================================================================
# 3. MULTIPLICATIVE MODEL
# ============================================================================
print("\n3. Multiplicative Model: prediction = trend * day_factor")
print("-" * 80)

# Calculate multiplicative factors (ratio of actual to trend for each day)
# Factor = mean(actual / trend) for each day
df['actual_trend_ratio'] = y / df['trend_base']
day_factors = df.groupby('day_name')['actual_trend_ratio'].mean().reindex(day_names)

# Center factors so they average to 1.0 (preserves overall trend level)
day_factors_centered = day_factors / day_factors.mean()

# Predictions
df['pred_multiplicative'] = df['trend_base'] * df['day_name'].map(day_factors_centered)
df['residuals_multiplicative'] = y - df['pred_multiplicative']

r2_mult = r2_score(y, df['pred_multiplicative'])
rmse_mult = np.sqrt(mean_squared_error(y, df['pred_multiplicative']))
mae_mult = mean_absolute_error(y, df['pred_multiplicative'])

print(f"   R² = {r2_mult:.4f}, RMSE = {rmse_mult:,.0f}, MAE = {mae_mult:,.0f}")
print(f"   Day factors (multiplicative):")
for day, factor in zip(day_names, day_factors_centered):
    pct = (factor - 1.0) * 100
    print(f"     {day:12s}: {factor:.4f} ({pct:+.2f}%)")

# ============================================================================
# 4. MIXED MODEL (Multiplicative + Additive)
# ============================================================================
print("\n4. Mixed Model: prediction = trend * day_factor + day_effect")
print("-" * 80)

# Use multiplicative factors and then fit additive adjustments to residuals
df['pred_mult_temp'] = df['trend_base'] * df['day_name'].map(day_factors_centered)
df['residuals_after_mult'] = y - df['pred_mult_temp']
day_effects_mixed = df.groupby('day_name')['residuals_after_mult'].mean().reindex(day_names)
day_effects_mixed_centered = day_effects_mixed - day_effects_mixed.mean()

# Predictions
df['pred_mixed'] = df['pred_mult_temp'] + df['day_name'].map(day_effects_mixed_centered)
df['residuals_mixed'] = y - df['pred_mixed']

r2_mixed = r2_score(y, df['pred_mixed'])
rmse_mixed = np.sqrt(mean_squared_error(y, df['pred_mixed']))
mae_mixed = mean_absolute_error(y, df['pred_mixed'])

print(f"   R² = {r2_mixed:.4f}, RMSE = {rmse_mixed:,.0f}, MAE = {mae_mixed:,.0f}")

# ============================================================================
# 5. COMPARISON
# ============================================================================
print("\n5. Model Comparison:")
print("-" * 80)

comparison = pd.DataFrame({
    'Model': ['Base (B-Spline only)', 'Additive', 'Multiplicative', 'Mixed'],
    'R²': [
        r2_score(y, trend_base),
        r2_add,
        r2_mult,
        r2_mixed
    ],
    'RMSE': [
        np.sqrt(mean_squared_error(y, trend_base)),
        rmse_add,
        rmse_mult,
        rmse_mixed
    ],
    'MAE': [
        mean_absolute_error(y, trend_base),
        mae_add,
        mae_mult,
        mae_mixed
    ]
})

print(comparison.to_string(index=False))

# Improvement over base
print("\n   Improvement over base model:")
print(f"     Additive:       R² +{r2_add - r2_score(y, trend_base):.4f}, RMSE -{((np.sqrt(mean_squared_error(y, trend_base)) - rmse_add) / np.sqrt(mean_squared_error(y, trend_base)) * 100):.2f}%")
print(f"     Multiplicative: R² +{r2_mult - r2_score(y, trend_base):.4f}, RMSE -{((np.sqrt(mean_squared_error(y, trend_base)) - rmse_mult) / np.sqrt(mean_squared_error(y, trend_base)) * 100):.2f}%")
print(f"     Mixed:          R² +{r2_mixed - r2_score(y, trend_base):.4f}, RMSE -{((np.sqrt(mean_squared_error(y, trend_base)) - rmse_mixed) / np.sqrt(mean_squared_error(y, trend_base)) * 100):.2f}%")

# Residual statistics
print("\n   Residual standard deviation:")
print(f"     Base:           {df['residuals_base'].std():.2f}")
print(f"     Additive:       {df['residuals_additive'].std():.2f}")
print(f"     Multiplicative: {df['residuals_multiplicative'].std():.2f}")
print(f"     Mixed:          {df['residuals_mixed'].std():.2f}")

# Residual autocorrelation
print("\n   Residual autocorrelation (lag 1):")
print(f"     Base:           {df['residuals_base'].autocorr(lag=1):.4f}")
print(f"     Additive:       {df['residuals_additive'].autocorr(lag=1):.4f}")
print(f"     Multiplicative: {df['residuals_multiplicative'].autocorr(lag=1):.4f}")
print(f"     Mixed:          {df['residuals_mixed'].autocorr(lag=1):.4f}")

# ============================================================================
# 6. THEORETICAL JUSTIFICATION
# ============================================================================
print("\n6. Theoretical Considerations:")
print("-" * 80)
print("""
Additive Model:
  - Assumes day-of-week effect is constant regardless of overall popularity
  - Example: Weekends always have ~3,000 fewer results, whether trend is 50k or 200k
  - Good when: Day effects are absolute differences
  
Multiplicative Model:
  - Assumes day-of-week effect scales with overall popularity
  - Example: Weekends are always ~3% lower, so effect grows as popularity grows
  - Good when: Day effects are proportional/percentage-based
  
Mixed Model:
  - Combines both: percentage scaling + fixed adjustment
  - Most flexible but more complex
  - Good when: Both proportional and fixed effects exist
""")

# Check if effect scales with trend
print("7. Checking if day-of-week effect scales with trend level:")
print("-" * 80)

# Group by trend level (low/medium/high) and check day effects
df['trend_tertile'] = pd.qcut(df['trend_base'], q=3, labels=['Low', 'Medium', 'High'])
for tertile in ['Low', 'Medium', 'High']:
    subset = df[df['trend_tertile'] == tertile]
    weekend_effect = subset[subset['day_name'].isin(['Saturday', 'Sunday'])]['residuals_base'].mean() - \
                     subset[~subset['day_name'].isin(['Saturday', 'Sunday'])]['residuals_base'].mean()
    weekend_ratio = subset[subset['day_name'].isin(['Saturday', 'Sunday'])][target_var].mean() / \
                     subset[~subset['day_name'].isin(['Saturday', 'Sunday'])][target_var].mean()
    print(f"   {tertile} trend level:")
    print(f"     Weekend absolute effect: {weekend_effect:.0f}")
    print(f"     Weekend ratio: {weekend_ratio:.4f} ({weekend_ratio*100:.2f}%)")

# ============================================================================
# 8. VISUALIZATIONS
# ============================================================================
print("\n8. Creating visualizations...")
import os
os.makedirs('plots', exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Time series comparison
ax = axes[0, 0]
ax.plot(df['Date'], y, 'k-', linewidth=1, alpha=0.5, label='Actual', zorder=1)
ax.plot(df['Date'], df['pred_additive'], 'b-', linewidth=1.5, alpha=0.7, label='Additive', zorder=2)
ax.plot(df['Date'], df['pred_multiplicative'], 'r-', linewidth=1.5, alpha=0.7, label='Multiplicative', zorder=3)
ax.set_ylabel('Number of Reported Results')
ax.set_title('Model Predictions: Additive vs Multiplicative', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Residuals comparison
ax = axes[0, 1]
ax.plot(df['Date'], df['residuals_additive'], 'b-', linewidth=1, alpha=0.6, label='Additive Residuals')
ax.plot(df['Date'], df['residuals_multiplicative'], 'r-', linewidth=1, alpha=0.6, label='Multiplicative Residuals')
ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax.set_ylabel('Residual')
ax.set_title('Residuals Comparison', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Day effects comparison
ax = axes[1, 0]
x_pos = np.arange(len(day_names))
width = 0.35
# Additive effects (normalized to percentage for comparison)
additive_pct = (day_effects_add_centered / y.mean()) * 100
multiplicative_pct = (day_factors_centered - 1.0) * 100
ax.bar(x_pos - width/2, additive_pct, width, label='Additive (% of mean)', alpha=0.7, color='steelblue')
ax.bar(x_pos + width/2, multiplicative_pct, width, label='Multiplicative (% change)', alpha=0.7, color='coral')
ax.set_xticks(x_pos)
ax.set_xticklabels([d[:3] for d in day_names])
ax.set_ylabel('Effect (% of mean)')
ax.set_title('Day-of-Week Effects: Additive vs Multiplicative', fontsize=14, fontweight='bold')
ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Performance metrics
ax = axes[1, 1]
models = ['Additive', 'Multiplicative', 'Mixed']
r2_values = [r2_add, r2_mult, r2_mixed]
colors = ['steelblue', 'coral', 'green']
bars = ax.bar(models, r2_values, alpha=0.7, color=colors)
ax.set_ylabel('R²')
ax.set_title('Model Performance (R²)', fontsize=14, fontweight='bold')
ax.set_ylim([min(r2_values) - 0.001, max(r2_values) + 0.001])
for bar, val in zip(bars, r2_values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001,
            f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('plots/21_additive_vs_multiplicative.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: plots/21_additive_vs_multiplicative.png")

# Plot 2: Effect scaling analysis
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Weekend effect vs trend level
ax = axes[0]
for tertile in ['Low', 'Medium', 'High']:
    subset = df[df['trend_tertile'] == tertile]
    weekend_effect = subset[subset['day_name'].isin(['Saturday', 'Sunday'])]['residuals_base'].mean() - \
                     subset[~subset['day_name'].isin(['Saturday', 'Sunday'])]['residuals_base'].mean()
    trend_mean = subset['trend_base'].mean()
    ax.scatter(trend_mean, weekend_effect, s=200, alpha=0.7, label=f'{tertile} Trend Level')
ax.set_xlabel('Mean Trend Level')
ax.set_ylabel('Weekend Effect (Absolute)')
ax.set_title('Weekend Effect vs Trend Level\n(Additive: constant, Multiplicative: scales)', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Weekend ratio vs trend level
ax = axes[1]
for tertile in ['Low', 'Medium', 'High']:
    subset = df[df['trend_tertile'] == tertile]
    weekend_ratio = subset[subset['day_name'].isin(['Saturday', 'Sunday'])][target_var].mean() / \
                     subset[~subset['day_name'].isin(['Saturday', 'Sunday'])][target_var].mean()
    trend_mean = subset['trend_base'].mean()
    ax.scatter(trend_mean, weekend_ratio, s=200, alpha=0.7, label=f'{tertile} Trend Level')
ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax.set_xlabel('Mean Trend Level')
ax.set_ylabel('Weekend Ratio (Weekend/Weekday)')
ax.set_title('Weekend Ratio vs Trend Level\n(Constant ratio = Multiplicative)', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/22_effect_scaling_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: plots/22_effect_scaling_analysis.png")

print("\n" + "=" * 80)
print("Analysis complete!")
print("=" * 80)

