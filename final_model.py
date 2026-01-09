"""
Final Wordle Prediction Model
=============================
Base: Cubic B-Spline trend (40 knots)
Additive: Day of Week effects

This script:
1. Fits cubic B-spline trend to capture overall popularity pattern
2. Adds day-of-week effects to capture weekly patterns
3. Compares model with and without day-of-week effects
4. Evaluates model performance and effect sizes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import splrep, splev
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

print("=" * 80)
print("FINAL WORDLE PREDICTION MODEL")
print("=" * 80)

# ============================================================================
# 1. LOAD AND PREPARE DATA
# ============================================================================
print("\n1. Loading data...")
df = pd.read_csv('data/2023_MCM_Problem_C_Data.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

target_var = 'Number of  reported results'

# Extract time features
df['day_of_week'] = df['Date'].dt.dayofweek  # 0=Monday, 6=Sunday
df['day_name'] = df['Date'].dt.day_name()
df['days_since_start'] = (df['Date'] - df['Date'].min()).dt.days

X_time = df['days_since_start'].values
y = df[target_var].values

print(f"   Data loaded: {len(df)} rows")
print(f"   Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
print(f"   Target variable: {target_var}")
print(f"   Mean: {y.mean():.0f}, Std: {y.std():.0f}")

# ============================================================================
# 2. FIT BASE MODEL: CUBIC B-SPLINE TREND (MINIMAL PIECEWISE - 2-3 SEGMENTS)
# ============================================================================
print("\n2. Fitting base model: Cubic B-Spline with minimal piecewise segments...")
print("   Goal: 2-3 segments (as few as possible)")

# Use fixed knots with very few segments
# For cubic spline: n_knots segments = n_knots+1 pieces
# 3 knots = 2 segments, 4 knots = 3 segments, 5 knots = 4 segments
# Try 2-3 segments (3-4 knots total including endpoints)
n_knots_options = [3, 4, 5]  # 3 knots = 2 segments, 4 = 3 segments, 5 = 4 segments
best_n_knots = None
best_r2 = -np.inf
best_trend = None
best_resid_ac = 1.0

print("   Testing number of knots (fewer = fewer segments)...")
for n_knots in n_knots_options:
    try:
        # Create evenly spaced knots
        knots = np.linspace(X_time.min(), X_time.max(), n_knots)
        # For cubic spline, we need internal knots (excluding endpoints for natural spline)
        # Actually, for splrep with fixed knots, we specify internal knots
        if n_knots <= 2:
            internal_knots = knots
        else:
            internal_knots = knots[1:-1]  # Exclude endpoints
        
        # Fit cubic B-spline with fixed knots
        tck = splrep(X_time, y, t=internal_knots, k=3, s=0)  # k=3 for cubic, s=0 for interpolation
        trend_test = splev(X_time, tck)
        r2_test = r2_score(y, trend_test)
        rmse_test = np.sqrt(mean_squared_error(y, trend_test))
        
        # Check residual autocorrelation (want low)
        residuals_test = y - trend_test
        if len(residuals_test) > 10:
            resid_lag1 = residuals_test[1:]
            resid_today = residuals_test[:-1]
            resid_ac = np.corrcoef(resid_today, resid_lag1)[0,1] if len(resid_today) > 1 else 0
        else:
            resid_ac = 0
        
        # Calculate number of segments (n_knots - 1 for cubic with endpoints)
        n_segments = n_knots - 1
        
        # Score: prioritize good fit first, then prefer fewer segments
        # Only prefer fewer segments if R² is reasonable (>0.95)
        if r2_test < 0.95:
            # Poor fit - heavily penalize
            score = r2_test - 10
        else:
            # Good fit - prefer fewer segments but prioritize fit quality
            score = r2_test - abs(resid_ac) * 0.5 - (rmse_test / y.mean()) * 0.1 - n_segments * 0.0001
        
        if score > (best_r2 - abs(best_r2) * 0.01 if best_r2 > -np.inf else -np.inf):
            best_n_knots = n_knots
            best_r2 = r2_test
            best_trend = trend_test
            best_resid_ac = abs(resid_ac)
            print(f"     {n_knots} knots ({n_segments} segments): R²={r2_test:.4f}, RMSE={rmse_test:,.0f}, Resid_AC={abs(resid_ac):.4f} ✓")
    except Exception as e:
        print(f"     {n_knots} knots: Failed - {e}")
        pass

# Use best number of knots
if best_n_knots is None:
    # Fallback: use 4 knots (3 segments)
    best_n_knots = 4
    print(f"   Using default: {best_n_knots} knots")
else:
    n_segments = best_n_knots - 1
    print(f"   Selected: {best_n_knots} knots ({n_segments} piecewise segments)")

# Fit final spline with best number of knots
knots = np.linspace(X_time.min(), X_time.max(), best_n_knots)
if best_n_knots <= 2:
    internal_knots = knots
else:
    internal_knots = knots[1:-1]
tck = splrep(X_time, y, t=internal_knots, k=3, s=0)
trend_base = splev(X_time, tck)

df['trend_base'] = trend_base
df['residuals_base'] = y - trend_base

# Base model metrics
r2_base = r2_score(y, trend_base)
rmse_base = np.sqrt(mean_squared_error(y, trend_base))
mae_base = mean_absolute_error(y, trend_base)

n_segments = best_n_knots - 1
print(f"   Base model (Cubic B-Spline with {best_n_knots} knots = {n_segments} segments):")
print(f"     R² = {r2_base:.4f}")
print(f"     RMSE = {rmse_base:,.0f}")
print(f"     MAE = {mae_base:,.0f}")
print(f"     Note: Minimal piecewise - only {n_segments} cubic polynomial segments")

# ============================================================================
# 3. FIT DAY-OF-WEEK EFFECTS (MIXED MODEL: Multiplicative + Additive)
# ============================================================================
print("\n3. Fitting day-of-week effects (Mixed Model: Multiplicative + Additive)...")

day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# Step 1: Multiplicative factors (day factors that scale with trend)
# Calculate ratio of actual to trend for each day
df['actual_trend_ratio'] = y / df['trend_base']
day_factors = df.groupby('day_name')['actual_trend_ratio'].mean().reindex(day_names)

# Center factors so they average to 1.0 (preserves overall trend level)
day_factors_centered = day_factors / day_factors.mean()

print("\n   Day-of-week factors (multiplicative, scale with trend):")
for day, factor in zip(day_names, day_factors_centered):
    pct = (factor - 1.0) * 100
    print(f"     {day:12s}: {factor:.4f} ({pct:+.2f}%)")

# Step 2: Apply multiplicative factors first
df['pred_after_mult'] = df['trend_base'] * df['day_name'].map(day_factors_centered)
df['residuals_after_mult'] = y - df['pred_after_mult']

# Step 3: Additive adjustments to residuals after multiplicative
day_effects = df.groupby('day_name')['residuals_after_mult'].mean().reindex(day_names)
day_effects_centered = day_effects - day_effects.mean()

print("\n   Day-of-week effects (additive adjustments after multiplicative):")
for day, effect in zip(day_names, day_effects_centered):
    print(f"     {day:12s}: {effect:+.0f}")

# Store both for final prediction
df['day_factor'] = df['day_name'].map(day_factors_centered)
df['day_effect'] = df['day_name'].map(day_effects_centered)

# ============================================================================
# 4. FULL MODEL: TREND * DAY_FACTOR + DAY_EFFECT (MIXED MODEL)
# ============================================================================
print("\n4. Full model: Trend * Day-Factor + Day-Effect (Mixed Model)...")

# Full model prediction: Mixed = Multiplicative + Additive
df['prediction_full'] = df['trend_base'] * df['day_factor'] + df['day_effect']
df['residuals_full'] = y - df['prediction_full']

# Full model metrics
r2_full = r2_score(y, df['prediction_full'])
rmse_full = np.sqrt(mean_squared_error(y, df['prediction_full']))
mae_full = mean_absolute_error(y, df['prediction_full'])

print(f"   Full model (Cubic B-Spline + Day-of-Week Mixed):")
print(f"     R² = {r2_full:.4f}")
print(f"     RMSE = {rmse_full:,.0f}")
print(f"     MAE = {mae_full:,.0f}")
print(f"     Formula: prediction = trend * day_factor + day_effect")

# ============================================================================
# 5. MODEL COMPARISON
# ============================================================================
print("\n5. Model Comparison:")
print("-" * 80)

comparison = pd.DataFrame({
    'Model': ['Base (B-Spline only)', 'Full (B-Spline + Day-of-Week Mixed)'],
    'R²': [r2_base, r2_full],
    'RMSE': [rmse_base, rmse_full],
    'MAE': [mae_base, mae_full]
})

print(comparison.to_string(index=False))

# Improvement metrics
r2_improvement = r2_full - r2_base
rmse_improvement = ((rmse_base - rmse_full) / rmse_base) * 100
mae_improvement = ((mae_base - mae_full) / mae_base) * 100

print(f"\n   Improvement from adding Day-of-Week:")
print(f"     R² improvement: {r2_improvement:+.4f} ({r2_improvement/r2_base*100:+.2f}%)")
print(f"     RMSE reduction: {rmse_improvement:+.2f}%")
print(f"     MAE reduction: {mae_improvement:+.2f}%")

# Residual statistics
print(f"\n   Residual statistics:")
print(f"     Base model - Mean: {df['residuals_base'].mean():.2f}, Std: {df['residuals_base'].std():.2f}")
print(f"     Full model - Mean: {df['residuals_full'].mean():.2f}, Std: {df['residuals_full'].std():.2f}")
print(f"     Std reduction: {((df['residuals_base'].std() - df['residuals_full'].std()) / df['residuals_base'].std() * 100):+.2f}%")

# Residual autocorrelation
resid_base_ac = df['residuals_base'].autocorr(lag=1)
resid_full_ac = df['residuals_full'].autocorr(lag=1)
print(f"\n   Residual autocorrelation (lag 1):")
print(f"     Base model: {resid_base_ac:.4f}")
print(f"     Full model: {resid_full_ac:.4f}")

# ============================================================================
# 6. DAY-OF-WEEK EFFECT SIZE ANALYSIS
# ============================================================================
print("\n6. Day-of-Week Effect Size Analysis:")
print("-" * 80)

# Multiplicative factors
print("\n   Day-of-week factors (multiplicative):")
for day, factor in zip(day_names, day_factors_centered):
    pct = (factor - 1.0) * 100
    print(f"     {day:12s}: {factor:.4f} ({pct:+.2f}%)")

factor_range = (day_factors_centered.max() - day_factors_centered.min()) * 100
print(f"\n   Factor range: {factor_range:.2f} percentage points")

# Additive effects
mean_target = y.mean()
day_effects_pct = (day_effects_centered / mean_target) * 100

print("\n   Day-of-week effects (additive) as percentage of mean:")
for day, effect, effect_pct in zip(day_names, day_effects_centered, day_effects_pct):
    print(f"     {day:12s}: {effect:+.0f} ({effect_pct:+.2f}%)")

# Range of effects
effect_range = day_effects_centered.max() - day_effects_centered.min()
effect_range_pct = (effect_range / mean_target) * 100
print(f"\n   Additive effect range: {effect_range:.0f} ({effect_range_pct:.2f}% of mean)")

# Statistical significance (ANOVA on residuals)
from scipy.stats import f_oneway
day_groups = [df[df['day_name'] == day]['residuals_base'].values for day in day_names]
f_stat, p_value = f_oneway(*day_groups)
print(f"\n   ANOVA on base residuals by day-of-week:")
print(f"     F-statistic: {f_stat:.2f}")
print(f"     P-value: {p_value:.4f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'}")

# ============================================================================
# 7. VISUALIZATIONS
# ============================================================================
print("\n7. Creating visualizations...")

# Create plots directory
import os
os.makedirs('plots', exist_ok=True)

# Plot 1: Model comparison - time series
fig, axes = plt.subplots(3, 1, figsize=(16, 12))

# Actual vs predictions
ax = axes[0]
ax.plot(df['Date'], y, 'k-', linewidth=1.5, alpha=0.7, label='Actual', zorder=1)
ax.plot(df['Date'], df['trend_base'], 'b--', linewidth=2, alpha=0.8, label='Base Model (B-Spline)', zorder=2)
ax.plot(df['Date'], df['prediction_full'], 'r-', linewidth=2, alpha=0.8, label='Full Model (B-Spline + Day-of-Week)', zorder=3)
ax.set_ylabel('Number of Reported Results')
ax.set_title('Model Comparison: Actual vs Predictions', fontsize=14, fontweight='bold')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)

# Residuals comparison
ax = axes[1]
ax.plot(df['Date'], df['residuals_base'], 'b-', linewidth=1, alpha=0.6, label='Base Model Residuals')
ax.plot(df['Date'], df['residuals_full'], 'r-', linewidth=1, alpha=0.6, label='Full Model Residuals')
ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax.set_ylabel('Residual')
ax.set_title('Residuals Comparison', fontsize=14, fontweight='bold')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)

# Residuals by day of week
ax = axes[2]
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
base_resid_by_day = df.groupby('day_name')['residuals_base'].mean().reindex(day_order)
full_resid_by_day = df.groupby('day_name')['residuals_full'].mean().reindex(day_order)
x_pos = np.arange(len(day_order))
width = 0.35
ax.bar(x_pos - width/2, base_resid_by_day, width, label='Base Model', alpha=0.7, color='steelblue')
ax.bar(x_pos + width/2, full_resid_by_day, width, label='Full Model', alpha=0.7, color='coral')
ax.set_xticks(x_pos)
ax.set_xticklabels([d[:3] for d in day_order])
ax.set_ylabel('Mean Residual')
ax.set_title('Residuals by Day of Week', fontsize=14, fontweight='bold')
ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('plots/18_model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: plots/18_model_comparison.png")

# Plot 2: Day-of-week effects
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Day-of-week factors (multiplicative)
ax = axes[0, 0]
x_pos = np.arange(len(day_names))
colors = ['steelblue' if d not in ['Saturday', 'Sunday'] else 'coral' for d in day_names]
day_factors_pct = (day_factors_centered - 1.0) * 100
ax.bar(x_pos, day_factors_pct, alpha=0.7, color=colors)
ax.set_xticks(x_pos)
ax.set_xticklabels([d[:3] for d in day_names])
ax.set_ylabel('Factor (% change)')
ax.set_title('Day-of-Week Factors (Multiplicative)', fontsize=14, fontweight='bold')
ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax.grid(True, alpha=0.3, axis='y')

# Day-of-week effects (additive)
ax = axes[0, 1]
day_effects_pct = (day_effects_centered / y.mean()) * 100
ax.bar(x_pos, day_effects_pct, alpha=0.7, color=colors)
ax.set_xticks(x_pos)
ax.set_xticklabels([d[:3] for d in day_names])
ax.set_ylabel('Effect (% of mean)')
ax.set_title('Day-of-Week Effects (Additive)', fontsize=14, fontweight='bold')
ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax.grid(True, alpha=0.3, axis='y')

# Residuals distribution comparison
ax = axes[1, 0]
ax.hist(df['residuals_base'], bins=30, alpha=0.6, label='Base Model', color='steelblue', density=True)
ax.hist(df['residuals_full'], bins=30, alpha=0.6, label='Full Model', color='coral', density=True)
ax.set_xlabel('Residual')
ax.set_ylabel('Density')
ax.set_title('Residuals Distribution', fontsize=14, fontweight='bold')
ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax.legend()
ax.grid(True, alpha=0.3)

# Scatter: Actual vs Predicted (full model)
ax = axes[1, 1]
ax.scatter(y, df['prediction_full'], alpha=0.5, s=20)
# Perfect prediction line
min_val = min(y.min(), df['prediction_full'].min())
max_val = max(y.max(), df['prediction_full'].max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
ax.set_title(f'Actual vs Predicted (Full Model)\nR² = {r2_full:.4f}', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/19_day_of_week_effects.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: plots/19_day_of_week_effects.png")

# Plot 3: Model performance metrics
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# R² comparison
ax = axes[0]
models = ['Base\n(B-Spline)', 'Full\n(B-Spline+\nDay-of-Week)']
r2_values = [r2_base, r2_full]
colors_bar = ['steelblue', 'coral']
bars = ax.bar(models, r2_values, alpha=0.7, color=colors_bar)
ax.set_ylabel('R²')
ax.set_title('R² Comparison', fontsize=14, fontweight='bold')
ax.set_ylim([min(r2_values) - 0.01, max(r2_values) + 0.01])
for bar, val in zip(bars, r2_values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# RMSE comparison
ax = axes[1]
rmse_values = [rmse_base, rmse_full]
bars = ax.bar(models, rmse_values, alpha=0.7, color=colors_bar)
ax.set_ylabel('RMSE')
ax.set_title('RMSE Comparison', fontsize=14, fontweight='bold')
for bar, val in zip(bars, rmse_values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(rmse_values)*0.02,
            f'{val:,.0f}', ha='center', va='bottom', fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# MAE comparison
ax = axes[2]
mae_values = [mae_base, mae_full]
bars = ax.bar(models, mae_values, alpha=0.7, color=colors_bar)
ax.set_ylabel('MAE')
ax.set_title('MAE Comparison', fontsize=14, fontweight='bold')
for bar, val in zip(bars, mae_values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mae_values)*0.02,
            f'{val:,.0f}', ha='center', va='bottom', fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('plots/20_model_performance.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: plots/20_model_performance.png")

# Plot 4: Variance/Volatility Analysis (relative to trend level)
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Calculate rolling variance and coefficient of variation
window = 30  # 30-day rolling window
df['rolling_std'] = df[target_var].rolling(window=window, center=True).std()
df['rolling_mean'] = df[target_var].rolling(window=window, center=True).mean()
df['coefficient_of_variation'] = (df['rolling_std'] / df['rolling_mean']) * 100

# Variance relative to trend
df['variance_relative_to_trend'] = ((df[target_var] - df['trend_base']) / df['trend_base']) * 100
df['abs_variance_relative'] = np.abs(df['variance_relative_to_trend'])

# Rolling variance relative to trend
df['rolling_var_rel_trend'] = df['variance_relative_to_trend'].rolling(window=window, center=True).std()

# Plot 1: Coefficient of Variation over time
ax = axes[0, 0]
ax.plot(df['Date'], df['coefficient_of_variation'], 'b-', linewidth=2, alpha=0.7, label='Coefficient of Variation')
ax.axhline(y=df['coefficient_of_variation'].mean(), color='red', linestyle='--', linewidth=1.5, 
           label=f'Mean CV: {df["coefficient_of_variation"].mean():.2f}%', alpha=0.7)
ax.set_ylabel('Coefficient of Variation (%)')
ax.set_xlabel('Date')
ax.set_title(f'Volatility Over Time\n(30-day Rolling CV = Std/Mean)', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Variance relative to trend level
ax = axes[0, 1]
ax.plot(df['Date'], df['variance_relative_to_trend'], 'g-', linewidth=1.5, alpha=0.7, label='% Deviation from Trend')
ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
ax.fill_between(df['Date'], -df['abs_variance_relative'].rolling(window=window, center=True).mean(), 
                 df['abs_variance_relative'].rolling(window=window, center=True).mean(), 
                 alpha=0.2, color='gray', label='Rolling Mean Abs Deviation')
ax.set_ylabel('% Deviation from Trend')
ax.set_xlabel('Date')
ax.set_title('Variance Relative to Trend Level\n(% Deviation from B-Spline Trend)', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Rolling variance by trend level
ax = axes[1, 0]
# Create trend level bins
df['trend_bin'] = pd.qcut(df['trend_base'], q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
trend_bin_stats = df.groupby('trend_bin').agg({
    'variance_relative_to_trend': ['mean', 'std'],
    'abs_variance_relative': 'mean'
}).round(2)

x_pos = np.arange(len(trend_bin_stats))
ax.bar(x_pos, trend_bin_stats['abs_variance_relative']['mean'], alpha=0.7, color='steelblue')
ax.set_xticks(x_pos)
ax.set_xticklabels(trend_bin_stats.index, rotation=45)
ax.set_ylabel('Mean |% Deviation from Trend|')
ax.set_title('Variance by Trend Level\n(Higher trend = more absolute variance?)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Plot 4: Scatter: Variance vs Trend Level
ax = axes[1, 1]
scatter = ax.scatter(df['trend_base'], df['abs_variance_relative'], 
                     c=df['days_since_start'], cmap='viridis', alpha=0.6, s=30)
ax.set_xlabel('Trend Level (B-Spline Prediction)')
ax.set_ylabel('|% Deviation from Trend|')
ax.set_title('Variance vs Trend Level\n(Color = Days Since Start)', fontsize=14, fontweight='bold')
plt.colorbar(scatter, ax=ax, label='Days Since Start')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/21_variance_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: plots/21_variance_analysis.png")

# ============================================================================
# 8. SAVE MODEL RESULTS
# ============================================================================
print("\n8. Saving model results...")

# Save predictions
df_output = df[['Date', target_var, 'trend_base', 'day_factor', 'day_effect', 'prediction_full', 
                'residuals_base', 'residuals_full', 'day_name', 'day_of_week']].copy()
df_output.to_csv('data/model_predictions.csv', index=False)
print("   Saved: data/model_predictions.csv")

# Save day-of-week effects (both multiplicative and additive)
day_effects_df = pd.DataFrame({
    'day_name': day_names,
    'day_of_week': range(7),
    'factor_multiplicative': day_factors_centered.values,
    'factor_pct': ((day_factors_centered - 1.0) * 100).values,
    'effect_additive': day_effects_centered.values,
    'effect_pct': day_effects_pct.values
})
day_effects_df.to_csv('data/day_of_week_effects.csv', index=False)
print("   Saved: data/day_of_week_effects.csv")

# Save model summary
summary = {
    'base_model': {
        'type': 'Cubic B-Spline (minimal piecewise)',
        'n_knots': int(best_n_knots) if best_n_knots else None,
        'n_segments': int(best_n_knots - 1) if best_n_knots else None,
        'r2': float(r2_base),
        'rmse': float(rmse_base),
        'mae': float(mae_base),
        'residual_std': float(df['residuals_base'].std()),
        'residual_autocorr': float(resid_base_ac)
    },
    'full_model': {
        'type': 'Cubic B-Spline + Day-of-Week Mixed (Multiplicative + Additive)',
        'formula': 'prediction = trend * day_factor + day_effect',
        'n_knots': int(best_n_knots) if best_n_knots else None,
        'n_segments': int(best_n_knots - 1) if best_n_knots else None,
        'r2': float(r2_full),
        'rmse': float(rmse_full),
        'mae': float(mae_full),
        'residual_std': float(df['residuals_full'].std()),
        'residual_autocorr': float(resid_full_ac)
    },
    'improvement': {
        'r2_improvement': float(r2_improvement),
        'r2_improvement_pct': float(r2_improvement/r2_base*100),
        'rmse_reduction_pct': float(rmse_improvement),
        'mae_reduction_pct': float(mae_improvement),
        'residual_std_reduction_pct': float((df['residuals_base'].std() - df['residuals_full'].std()) / df['residuals_base'].std() * 100)
    },
    'day_of_week_factors': day_factors_centered.to_dict(),
    'day_of_week_effects': day_effects_centered.to_dict()
}

import json
with open('data/model_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print("   Saved: data/model_summary.json")

# ============================================================================
# 9. SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("MODEL SUMMARY")
print("=" * 80)
n_segments = best_n_knots - 1
print(f"\nBase Model: Cubic B-Spline ({best_n_knots} knots = {n_segments} piecewise segments)")
print(f"  R² = {r2_base:.4f}, RMSE = {rmse_base:,.0f}, MAE = {mae_base:,.0f}")
print(f"  Minimal piecewise: only {n_segments} cubic polynomial segments (1-2 knobs: knot positions)")

n_segments = best_n_knots - 1
print(f"\nFull Model: Cubic B-Spline ({best_n_knots} knots = {n_segments} segments) + Day-of-Week Mixed")
print(f"  Formula: prediction = trend * day_factor + day_effect")
print(f"  Parameters: {best_n_knots-2} knot positions + 7 day factors + 7 day effects")
print(f"  R² = {r2_full:.4f}, RMSE = {rmse_full:,.0f}, MAE = {mae_full:,.0f}")

print(f"\nImprovement from adding Day-of-Week:")
print(f"  R² improvement: {r2_improvement:+.4f} ({r2_improvement/r2_base*100:+.2f}%)")
print(f"  RMSE reduction: {rmse_improvement:+.2f}%")
print(f"  MAE reduction: {mae_improvement:+.2f}%")
print(f"  Residual std reduction: {((df['residuals_base'].std() - df['residuals_full'].std()) / df['residuals_base'].std() * 100):+.2f}%")

print(f"\nDay-of-Week Factors (multiplicative):")
for day, factor in zip(day_names, day_factors_centered):
    pct = (factor - 1.0) * 100
    print(f"  {day:12s}: {factor:.4f} ({pct:+.2f}%)")

print(f"\nDay-of-Week Effects (additive adjustments):")
for day, effect in zip(day_names, day_effects_centered):
    pct = (effect / y.mean()) * 100
    print(f"  {day:12s}: {effect:+.0f} ({pct:+.2f}% of mean)")

print("\n" + "=" * 80)
print("Model complete! All results saved to data/ and plots/ directories.")
print("=" * 80)

