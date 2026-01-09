"""
Final Paper Model: Exponential Growth-Then-Decay with Day-of-Week Effects
==========================================================================
Fits model on full dataset, extrapolates to March 1, 2023
Creates publication-quality plots
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Publication-quality style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams.update({
    'figure.figsize': (12, 8),
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'font.family': 'serif',
    'mathtext.fontset': 'stix'
})

print("=" * 80)
print("FINAL PAPER MODEL: EXPONENTIAL GROWTH-THEN-DECAY + DAY-OF-WEEK")
print("=" * 80)

# Load data
df = pd.read_csv('data/2023_MCM_Problem_C_Data.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

target_var = 'Number of  reported results'
df['days_since_start'] = (df['Date'] - df['Date'].min()).dt.days
df['day_of_week'] = df['Date'].dt.dayofweek
df['day_name'] = df['Date'].dt.day_name()

X_time = df['days_since_start'].values
y = df[target_var].values

print(f"\nData: {len(df)} observations")
print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
print(f"Peak value: {y.max():,.0f} at day {X_time[np.argmax(y)]}")

# Create paper plots folder
import os
paper_plots_dir = 'plots_paper'
os.makedirs(paper_plots_dir, exist_ok=True)

# ============================================================================
# 1. FIT BASE MODEL: EXPONENTIAL GROWTH-THEN-DECAY
# ============================================================================
print("\n1. Fitting Exponential Growth-Then-Decay Model...")

def growth_then_decay(x, x0, a1, b1, a2, b2, c):
    """
    Growth phase: a1 * exp(b1 * x) for x < x0
    Decay phase: a2 * exp(-b2 * (x - x0)) + c for x >= x0
    """
    result = np.zeros_like(x)
    mask_growth = x < x0
    mask_decay = x >= x0
    result[mask_growth] = a1 * np.exp(b1 * x[mask_growth])
    result[mask_decay] = a2 * np.exp(-b2 * (x[mask_decay] - x0)) + c
    return result

# Fit on full dataset
peak_idx = np.argmax(y)
x0_guess = X_time[peak_idx]
p0 = [x0_guess, y[0], 0.01, y.max(), 0.001, y.min()]

popt_base, _ = curve_fit(growth_then_decay, X_time, y, p0=p0, maxfev=10000,
                        bounds=([X_time[0], 0, 0, 0, 0, y.min()],
                               [X_time[-1], y.max()*10, 1, y.max()*10, 1, y.max()]))

df['trend_base'] = growth_then_decay(X_time, *popt_base)
df['residuals_base'] = y - df['trend_base']

r2_base = r2_score(y, df['trend_base'])
rmse_base = np.sqrt(mean_squared_error(y, df['trend_base']))
mae_base = mean_absolute_error(y, df['trend_base'])

print(f"   Base Model (Growth-Then-Decay):")
print(f"     Breakpoint (peak): Day {popt_base[0]:.1f}")
print(f"     R² = {r2_base:.4f}")
print(f"     RMSE = {rmse_base:,.0f}")
print(f"     MAE = {mae_base:,.0f}")

# ============================================================================
# 2. ADD DAY-OF-WEEK EFFECTS (Multiplicative)
# ============================================================================
print("\n2. Adding Day-of-Week Effects (Multiplicative)...")

# Calculate multiplicative factors on residuals
df['actual_trend_ratio'] = y / df['trend_base']
day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_factors = df.groupby('day_name')['actual_trend_ratio'].mean().reindex(day_names)
day_factors_centered = day_factors / day_factors.mean()

print(f"\n   Day-of-week factors (multiplicative):")
for day, factor in zip(day_names, day_factors_centered):
    pct = (factor - 1.0) * 100
    print(f"     {day:12s}: {factor:.4f} ({pct:+.2f}%)")

# Full model: trend * day_factor
df['day_factor'] = df['day_name'].map(day_factors_centered)
df['prediction_full'] = df['trend_base'] * df['day_factor']
df['residuals_full'] = y - df['prediction_full']

r2_full = r2_score(y, df['prediction_full'])
rmse_full = np.sqrt(mean_squared_error(y, df['prediction_full']))
mae_full = mean_absolute_error(y, df['prediction_full'])

print(f"\n   Full Model (Growth-Then-Decay + Day-of-Week):")
print(f"     R² = {r2_full:.4f}")
print(f"     RMSE = {rmse_full:,.0f}")
print(f"     MAE = {mae_full:,.0f}")
print(f"     Improvement: R² +{r2_full - r2_base:.4f}, RMSE -{((rmse_base - rmse_full)/rmse_base*100):.2f}%")

# ============================================================================
# 3. EXTRAPOLATE TO MARCH 1, 2023
# ============================================================================
print("\n3. Extrapolating to March 1, 2023...")

march_1 = pd.to_datetime('2023-03-01')
days_to_march_1 = (march_1 - df['Date'].min()).days
march_1_dow = march_1.weekday()  # 0=Monday
march_1_day_name = march_1.strftime('%A')

# Extend time range
X_extended = np.arange(X_time.min(), days_to_march_1 + 1)
dates_extended = pd.date_range(start=df['Date'].min(), end=march_1, freq='D')

# Base trend extrapolation
trend_extended = growth_then_decay(X_extended, *popt_base)

# Day-of-week factors for extended period
day_names_extended = [pd.Timestamp(d).strftime('%A') for d in dates_extended]
day_factors_extended = np.array([day_factors_centered[d] for d in day_names_extended])

# Full model extrapolation
prediction_extended = trend_extended * day_factors_extended

# Prediction for March 1 specifically
march_1_idx = len(X_extended) - 1
march_1_base_trend = trend_extended[march_1_idx]
march_1_day_factor = day_factors_centered[march_1_day_name]
march_1_prediction = march_1_base_trend * march_1_day_factor

# Prediction intervals (using residuals)
residual_std = np.std(df['residuals_full'])
march_1_pred_lower = march_1_prediction - 1.96 * residual_std
march_1_pred_upper = march_1_prediction + 1.96 * residual_std

print(f"\n   March 1, 2023 Prediction ({march_1_day_name}):")
print(f"     Base trend: {march_1_base_trend:,.0f}")
print(f"     Day factor: {march_1_day_factor:.4f} ({((march_1_day_factor - 1.0) * 100):+.2f}%)")
print(f"     Point estimate: {march_1_prediction:,.0f}")
print(f"     95% Prediction Interval: [{march_1_pred_lower:,.0f}, {march_1_pred_upper:,.0f}]")

# ============================================================================
# 4. CREATE PUBLICATION-QUALITY PLOTS
# ============================================================================
print("\n4. Creating publication-quality plots...")

# Plot 1: Time series with models and extrapolation
fig, ax = plt.subplots(figsize=(14, 8))

# Historical data
ax.scatter(df['Date'], y, s=30, alpha=0.6, color='black', label='Observed Data', zorder=3)

# Base model
ax.plot(df['Date'], df['trend_base'], '--', linewidth=2.5, color='#2E86AB', 
        alpha=0.8, label='Base Model (Growth-Then-Decay)', zorder=2)

# Full model
ax.plot(df['Date'], df['prediction_full'], '-', linewidth=2.5, color='#A23B72', 
        alpha=0.9, label='Full Model (+ Day-of-Week)', zorder=2)

# Extrapolation
ax.plot(dates_extended[len(df):], prediction_extended[len(df):], '-', 
        linewidth=2.5, color='#F18F01', alpha=0.9, 
        label='Extrapolation', zorder=2, linestyle='-')
ax.plot(dates_extended[len(df):], trend_extended[len(df):], '--', 
        linewidth=2, color='#2E86AB', alpha=0.6, 
        label='Base Trend Extrapolation', zorder=1)

# March 1 prediction point
ax.scatter([march_1], [march_1_prediction], s=200, color='red', 
          marker='*', edgecolors='darkred', linewidths=2, zorder=5,
          label=f'March 1, 2023 Prediction: {march_1_prediction:,.0f}')

# Prediction interval
ax.fill_between([march_1], march_1_pred_lower, march_1_pred_upper, 
                alpha=0.3, color='red', zorder=4)

# Vertical line at data end
ax.axvline(x=df['Date'].max(), color='gray', linestyle=':', linewidth=2, 
          alpha=0.7, label='End of Training Data', zorder=1)

ax.set_xlabel('Date', fontsize=13, fontweight='bold')
ax.set_ylabel('Number of Reported Results', fontsize=13, fontweight='bold')
ax.set_title('Wordle Popularity: Exponential Growth-Then-Decay Model\nWith Day-of-Week Effects and Extrapolation to March 1, 2023', 
            fontsize=15, fontweight='bold', pad=15)
ax.legend(loc='upper left', framealpha=0.95, fontsize=11)
ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

# Format x-axis dates
ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b %Y'))
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.savefig(f'{paper_plots_dir}/01_time_series_model_extrapolation.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{paper_plots_dir}/01_time_series_model_extrapolation.pdf', bbox_inches='tight')
plt.close()
print(f"   Saved: {paper_plots_dir}/01_time_series_model_extrapolation.png/pdf")

# Plot 2: Model comparison (base vs full)
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Residuals comparison
ax = axes[0, 0]
ax.plot(df['Date'], df['residuals_base'], 'o-', markersize=3, linewidth=1, 
        alpha=0.6, color='#2E86AB', label='Base Model Residuals', zorder=2)
ax.plot(df['Date'], df['residuals_full'], 'o-', markersize=3, linewidth=1, 
        alpha=0.6, color='#A23B72', label='Full Model Residuals', zorder=2)
ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7, zorder=1)
ax.set_ylabel('Residuals', fontsize=12, fontweight='bold')
ax.set_title('(a) Residuals Comparison', fontsize=13, fontweight='bold')
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)

# Actual vs Predicted (base)
ax = axes[0, 1]
ax.scatter(y, df['trend_base'], alpha=0.6, s=30, color='#2E86AB', zorder=2)
min_val = min(y.min(), df['trend_base'].min())
max_val = max(y.max(), df['trend_base'].max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.8, zorder=1)
ax.set_xlabel('Actual', fontsize=12, fontweight='bold')
ax.set_ylabel('Predicted', fontsize=12, fontweight='bold')
ax.set_title(f'(b) Base Model Fit\nR² = {r2_base:.4f}', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

# Actual vs Predicted (full)
ax = axes[1, 0]
ax.scatter(y, df['prediction_full'], alpha=0.6, s=30, color='#A23B72', zorder=2)
ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.8, zorder=1)
ax.set_xlabel('Actual', fontsize=12, fontweight='bold')
ax.set_ylabel('Predicted', fontsize=12, fontweight='bold')
ax.set_title(f'(c) Full Model Fit\nR² = {r2_full:.4f}', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

# Residual distribution comparison
ax = axes[1, 1]
ax.hist(df['residuals_base'], bins=40, alpha=0.6, color='#2E86AB', 
       density=True, label='Base Model', edgecolor='black', linewidth=0.5)
ax.hist(df['residuals_full'], bins=40, alpha=0.6, color='#A23B72', 
       density=True, label='Full Model', edgecolor='black', linewidth=0.5)
ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
ax.set_xlabel('Residuals', fontsize=12, fontweight='bold')
ax.set_ylabel('Density', fontsize=12, fontweight='bold')
ax.set_title('(d) Residual Distributions', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{paper_plots_dir}/02_model_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{paper_plots_dir}/02_model_comparison.pdf', bbox_inches='tight')
plt.close()
print(f"   Saved: {paper_plots_dir}/02_model_comparison.png/pdf")

# Plot 3: Day-of-week effects
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Day-of-week factors (bar chart)
ax = axes[0]
x_pos = np.arange(len(day_names))
colors = ['#2E86AB' if d not in ['Saturday', 'Sunday'] else '#F18F01' 
          for d in day_names]
bars = ax.bar(x_pos, day_factors_centered - 1.0, alpha=0.8, color=colors, 
             edgecolor='black', linewidth=1.5, zorder=2)
ax.set_xticks(x_pos)
ax.set_xticklabels([d[:3] for d in day_names], fontsize=11, fontweight='bold')
ax.set_ylabel('Multiplicative Factor - 1', fontsize=12, fontweight='bold')
ax.set_title('(a) Day-of-Week Effects\n(Multiplicative Factors)', 
            fontsize=13, fontweight='bold')
ax.axhline(y=0, color='black', linestyle='-', linewidth=2, zorder=1)
ax.grid(True, alpha=0.3, axis='y', zorder=0)

# Add value labels on bars
for i, (bar, factor) in enumerate(zip(bars, day_factors_centered)):
    height = bar.get_height()
    pct = (factor - 1.0) * 100
    ax.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height >= 0 else -0.015),
           f'{pct:+.1f}%', ha='center', va='bottom' if height >= 0 else 'top',
           fontsize=9, fontweight='bold')

# Day-of-week effects (time series view)
ax = axes[1]
for day_idx, day in enumerate(day_names):
    day_mask = df['day_name'] == day
    ax.scatter(df[day_mask]['Date'], df[day_mask]['residuals_base'], 
              alpha=0.6, s=40, label=day[:3], zorder=2)
ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7, zorder=1)
ax.set_xlabel('Date', fontsize=12, fontweight='bold')
ax.set_ylabel('Residuals from Base Model', fontsize=12, fontweight='bold')
ax.set_title('(b) Day-of-Week Patterns Over Time', fontsize=13, fontweight='bold')
ax.legend(ncol=4, loc='upper left', fontsize=9, framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b %Y'))
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.savefig(f'{paper_plots_dir}/03_day_of_week_effects.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{paper_plots_dir}/03_day_of_week_effects.pdf', bbox_inches='tight')
plt.close()
print(f"   Saved: {paper_plots_dir}/03_day_of_week_effects.png/pdf")

# Plot 4: Model performance metrics
fig, ax = plt.subplots(figsize=(10, 6))

models_compare = ['Base Model\n(Growth-Then-Decay)', 'Full Model\n(+ Day-of-Week)']
r2_values = [r2_base, r2_full]
rmse_values = [rmse_base, rmse_full]
mae_values = [mae_base, mae_full]

x_pos = np.arange(len(models_compare))
width = 0.25

bars1 = ax.bar(x_pos - width, r2_values, width, label='R²', alpha=0.8, 
              color='#2E86AB', edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x_pos, [rm/1000 for rm in rmse_values], width, label='RMSE (×1000)', 
              alpha=0.8, color='#A23B72', edgecolor='black', linewidth=1.5)
bars3 = ax.bar(x_pos + width, [ma/1000 for ma in mae_values], width, label='MAE (×1000)', 
              alpha=0.8, color='#F18F01', edgecolor='black', linewidth=1.5)

ax.set_ylabel('Metric Value', fontsize=12, fontweight='bold')
ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x_pos)
ax.set_xticklabels(models_compare, fontsize=11, fontweight='bold')
ax.legend(loc='upper left', fontsize=11, framealpha=0.95)
ax.grid(True, alpha=0.3, axis='y')

# Add value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        label = f'{height:.4f}' if bar.get_facecolor() == bars1[0].get_facecolor() else f'{height:.1f}'
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               label, ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{paper_plots_dir}/04_model_performance.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{paper_plots_dir}/04_model_performance.pdf', bbox_inches='tight')
plt.close()
print(f"   Saved: {paper_plots_dir}/04_model_performance.png/pdf")

# Plot 5: Zoomed extrapolation view
fig, ax = plt.subplots(figsize=(14, 7))

# Last 60 days of data + extrapolation
last_n_days = 60
start_idx = max(0, len(df) - last_n_days)

ax.plot(df['Date'].iloc[start_idx:], y[start_idx:], 'o-', markersize=6, 
       linewidth=2, alpha=0.8, color='black', label='Observed Data', zorder=3)
ax.plot(df['Date'].iloc[start_idx:], df['prediction_full'].iloc[start_idx:], 
       '-', linewidth=2.5, color='#A23B72', alpha=0.9, 
       label='Full Model Fit', zorder=2)

# Extrapolation
ax.plot(dates_extended[len(df):], prediction_extended[len(df):], '-', 
       linewidth=2.5, color='#F18F01', alpha=0.9, 
       label='Extrapolation to March 1, 2023', zorder=2)
ax.plot(dates_extended[len(df):], trend_extended[len(df):], '--', 
       linewidth=2, color='#2E86AB', alpha=0.6, 
       label='Base Trend Extrapolation', zorder=1)

# March 1 prediction
ax.scatter([march_1], [march_1_prediction], s=300, color='red', 
          marker='*', edgecolors='darkred', linewidths=2.5, zorder=5,
          label=f'March 1, 2023: {march_1_prediction:,.0f}')

# Prediction interval
ax.fill_between([march_1], march_1_pred_lower, march_1_pred_upper, 
                alpha=0.3, color='red', zorder=4)
ax.plot([march_1, march_1], [march_1_pred_lower, march_1_pred_upper], 
       'r-', linewidth=2, zorder=4)

# Vertical line
ax.axvline(x=df['Date'].max(), color='gray', linestyle=':', linewidth=2, 
          alpha=0.7, label='End of Training Data', zorder=1)

ax.set_xlabel('Date', fontsize=13, fontweight='bold')
ax.set_ylabel('Number of Reported Results', fontsize=13, fontweight='bold')
ax.set_title('Model Extrapolation: Last 60 Days + Future Prediction', 
            fontsize=15, fontweight='bold', pad=15)
ax.legend(loc='upper left', framealpha=0.95, fontsize=11)
ax.grid(True, alpha=0.3)

ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b %d, %Y'))
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.savefig(f'{paper_plots_dir}/05_extrapolation_zoomed.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{paper_plots_dir}/05_extrapolation_zoomed.pdf', bbox_inches='tight')
plt.close()
print(f"   Saved: {paper_plots_dir}/05_extrapolation_zoomed.png/pdf")

# Plot 6: Residual analysis
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Residuals over time (base)
ax = axes[0, 0]
ax.scatter(df['Date'], df['residuals_base'], s=20, alpha=0.6, color='#2E86AB', zorder=2)
ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7, zorder=1)
ax.fill_between(df['Date'], -1.96*np.std(df['residuals_base']), 
               1.96*np.std(df['residuals_base']), alpha=0.2, color='gray', zorder=0)
ax.set_ylabel('Residuals', fontsize=12, fontweight='bold')
ax.set_title('(a) Base Model Residuals Over Time', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

# Residuals over time (full)
ax = axes[0, 1]
ax.scatter(df['Date'], df['residuals_full'], s=20, alpha=0.6, color='#A23B72', zorder=2)
ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7, zorder=1)
ax.fill_between(df['Date'], -1.96*np.std(df['residuals_full']), 
               1.96*np.std(df['residuals_full']), alpha=0.2, color='gray', zorder=0)
ax.set_ylabel('Residuals', fontsize=12, fontweight='bold')
ax.set_title('(b) Full Model Residuals Over Time', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

# Q-Q plot (base)
ax = axes[1, 0]
stats.probplot(df['residuals_base'], dist="norm", plot=ax)
ax.set_title('(c) Q-Q Plot: Base Model Residuals', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

# Q-Q plot (full)
ax = axes[1, 1]
stats.probplot(df['residuals_full'], dist="norm", plot=ax)
ax.set_title('(d) Q-Q Plot: Full Model Residuals', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{paper_plots_dir}/06_residual_analysis.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{paper_plots_dir}/06_residual_analysis.pdf', bbox_inches='tight')
plt.close()
print(f"   Saved: {paper_plots_dir}/06_residual_analysis.png/pdf")

# ============================================================================
# 5. SAVE MODEL SUMMARY
# ============================================================================
print("\n5. Saving model summary...")

summary = {
    'base_model': {
        'type': 'Exponential Growth-Then-Decay',
        'breakpoint_day': float(popt_base[0]),
        'parameters': {
            'x0': float(popt_base[0]),
            'a1': float(popt_base[1]),
            'b1': float(popt_base[2]),
            'a2': float(popt_base[3]),
            'b2': float(popt_base[4]),
            'c': float(popt_base[5])
        },
        'r2': float(r2_base),
        'rmse': float(rmse_base),
        'mae': float(mae_base)
    },
    'full_model': {
        'type': 'Growth-Then-Decay + Day-of-Week (Multiplicative)',
        'r2': float(r2_full),
        'rmse': float(rmse_full),
        'mae': float(mae_full),
        'day_of_week_factors': {day: float(factor) for day, factor in zip(day_names, day_factors_centered)}
    },
    'march_1_2023_prediction': {
        'date': '2023-03-01',
        'day_of_week': march_1_day_name,
        'base_trend': float(march_1_base_trend),
        'day_factor': float(march_1_day_factor),
        'point_estimate': float(march_1_prediction),
        'prediction_interval_95': {
            'lower': float(march_1_pred_lower),
            'upper': float(march_1_pred_upper)
        },
        'residual_std': float(residual_std)
    },
    'improvement': {
        'r2_improvement': float(r2_full - r2_base),
        'r2_improvement_pct': float((r2_full - r2_base) / r2_base * 100),
        'rmse_reduction_pct': float((rmse_base - rmse_full) / rmse_base * 100),
        'mae_reduction_pct': float((mae_base - mae_full) / mae_base * 100)
    }
}

import json
with open(f'{paper_plots_dir}/model_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f"   Saved: {paper_plots_dir}/model_summary.json")

# Save predictions
df_predictions = pd.DataFrame({
    'Date': df['Date'],
    'Observed': y,
    'Base_Trend': df['trend_base'],
    'Day_Factor': df['day_factor'],
    'Full_Prediction': df['prediction_full'],
    'Residuals_Base': df['residuals_base'],
    'Residuals_Full': df['residuals_full']
})

# Add extrapolation
extrapolation_df = pd.DataFrame({
    'Date': dates_extended[len(df):],
    'Days_Since_Start': X_extended[len(df):],
    'Base_Trend': trend_extended[len(df):],
    'Day_Factor': day_factors_extended[len(df):],
    'Full_Prediction': prediction_extended[len(df):],
    'Is_Extrapolation': True
})

df_predictions['Is_Extrapolation'] = False
df_all = pd.concat([df_predictions, extrapolation_df], ignore_index=True)
df_all.to_csv(f'{paper_plots_dir}/predictions.csv', index=False)
print(f"   Saved: {paper_plots_dir}/predictions.csv")

# ============================================================================
# 6. COPY IMPORTANT EXISTING PLOTS TO PAPER FOLDER
# ============================================================================
print("\n6. Copying important analysis plots to paper folder...")

import shutil

important_plots = [
    ('plots/26_explanatory_model.png', '07_explanatory_factors.png'),
    ('plots/27_lognormal_model.png', '08_lognormal_framework.png'),
    ('plots/15_residual_patterns_basic.png', '09_residual_patterns.png'),
    ('plots/21_variance_analysis.png', '10_variance_analysis.png'),
]

for source, dest in important_plots:
    if os.path.exists(source):
        shutil.copy(source, f'{paper_plots_dir}/{dest}')
        print(f"   Copied: {source} → {paper_plots_dir}/{dest}")

print("\n" + "=" * 80)
print("FINAL PAPER MODEL SUMMARY")
print("=" * 80)
print(f"\nBase Model: Exponential Growth-Then-Decay")
print(f"  Breakpoint: Day {popt_base[0]:.1f}")
print(f"  R² = {r2_base:.4f}, RMSE = {rmse_base:,.0f}")

print(f"\nFull Model: Growth-Then-Decay + Day-of-Week (Multiplicative)")
print(f"  R² = {r2_full:.4f}, RMSE = {rmse_full:,.0f}")
print(f"  Improvement: R² +{r2_full - r2_base:.4f} ({((r2_full - r2_base)/r2_base*100):+.2f}%)")

print(f"\nMarch 1, 2023 Prediction ({march_1_day_name}):")
print(f"  Point estimate: {march_1_prediction:,.0f}")
print(f"  95% Prediction Interval: [{march_1_pred_lower:,.0f}, {march_1_pred_upper:,.0f}]")

print(f"\nAll plots saved to: {paper_plots_dir}/")
print("=" * 80)

