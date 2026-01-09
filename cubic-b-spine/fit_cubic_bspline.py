"""
Simple Cubic B-Spline Fitting
Fits a cubic B-spline with fewer than 10 knots to reported scores vs date
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Load enhanced dataset
print("Loading enhanced dataset...")
df = pd.read_csv('../data/2023_MCM_Problem_C_Data_enhanced.csv')

# Clean column names
df.columns = df.columns.str.strip()

# Convert Date to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Sort by date
df = df.sort_values('Date').reset_index(drop=True)

# Extract target variable
target_col = 'Number of  reported results'
df['reported_scores'] = df[target_col]

print(f"Data loaded: {len(df)} records")
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")

# Create numeric time index (days since start)
df['days_since_start'] = (df['Date'] - df['Date'].min()).dt.days
X = df['days_since_start'].values.reshape(-1, 1)
y = df['reported_scores'].values

print(f"\nTime range: {X.min()} to {X.max()} days")

# Try different numbers of knots (less than 10, fewer is better)
n_knots_options = [3, 4, 5, 6, 7, 8, 9]
best_r2 = -np.inf
best_model = None
best_transformer = None
best_predictions = None
best_n_knots = None
best_rmse = np.inf

print("\n" + "="*80)
print("FITTING CUBIC B-SPLINE WITH < 10 KNOTS")
print("="*80)
print("\nTesting different numbers of knots...")

for n_knots in n_knots_options:
    try:
        # Create cubic B-spline transformer
        spline_transformer = SplineTransformer(n_knots=n_knots, degree=3, include_bias=True)
        X_spline_basis = spline_transformer.fit_transform(X)
        
        # Fit linear regression on spline basis
        model = LinearRegression()
        model.fit(X_spline_basis, y)
        predictions = model.predict(X_spline_basis)
        
        # Calculate metrics
        r2 = r2_score(y, predictions)
        rmse = np.sqrt(mean_squared_error(y, predictions))
        mae = np.mean(np.abs(y - predictions))
        
        print(f"  {n_knots} knots: R² = {r2:.4f}, RMSE = {rmse:.2f}, MAE = {mae:.2f}")
        
        # Prefer fewer knots if R² is similar (within 0.01)
        if r2 > best_r2 or (r2 >= best_r2 - 0.01 and n_knots < best_n_knots):
            best_r2 = r2
            best_rmse = rmse
            best_model = model
            best_transformer = spline_transformer
            best_predictions = predictions
            best_n_knots = n_knots
    except Exception as e:
        print(f"  {n_knots} knots: Failed - {str(e)}")
        continue

print("\n" + "-"*80)
print("BEST MODEL")
print("-"*80)
print(f"Number of knots: {best_n_knots}")
print(f"R² = {best_r2:.4f}")
print(f"RMSE = {best_rmse:.2f}")
print(f"MAE = {np.mean(np.abs(y - best_predictions)):.2f}")

# Store base model results
df['cubic_bspline_fit'] = best_predictions
df['base_residuals'] = df['reported_scores'] - df['cubic_bspline_fit']
# Calculate percentage residuals: (actual - predicted) / predicted * 100
# Avoid division by zero
df['base_residuals_pct'] = np.where(df['cubic_bspline_fit'] != 0,
                                     (df['base_residuals'] / df['cubic_bspline_fit']) * 100,
                                     np.nan)

# ============================================================================
# ADD DAY-OF-WEEK EFFECTS
# ============================================================================

print("\n" + "="*80)
print("ADDING DAY-OF-WEEK EFFECTS")
print("="*80)

# Extract day of week (0=Monday, 6=Sunday)
if 'day_of_week' not in df.columns:
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['day_name'] = df['Date'].dt.day_name()

day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# 1. ADDITIVE MODEL: predicted = cubic_bspline_fit + day_of_week_effect
print("\n" + "-"*80)
print("1. ADDITIVE MODEL (Linear)")
print("-"*80)
print("Model: predicted = cubic_bspline_fit + day_of_week_effect")

# Calculate residuals from base model
base_residuals = df['base_residuals'].values

# Fit day-of-week effects by averaging residuals per day
dow_effects = df.groupby('day_name')['base_residuals'].mean()
dow_effects = dow_effects.reindex(day_order)

print("\nDay-of-week effects (to add to base prediction):")
for day in day_order:
    if day in dow_effects.index:
        print(f"  {day}: {dow_effects[day]:.2f}")

# Create additive predictions
df['dow_effect'] = df['day_name'].map(dow_effects)
df['additive_fit'] = df['cubic_bspline_fit'] + df['dow_effect']
df['additive_residuals'] = df['reported_scores'] - df['additive_fit']
# Calculate percentage residuals
df['additive_residuals_pct'] = np.where(df['additive_fit'] != 0,
                                        (df['additive_residuals'] / df['additive_fit']) * 100,
                                        np.nan)

# Calculate metrics for additive model
additive_r2 = r2_score(df['reported_scores'], df['additive_fit'])
additive_rmse = np.sqrt(mean_squared_error(df['reported_scores'], df['additive_fit']))
additive_mae = np.mean(np.abs(df['additive_residuals']))

print(f"\nAdditive Model Metrics:")
print(f"  R² = {additive_r2:.4f}")
print(f"  RMSE = {additive_rmse:.2f}")
print(f"  MAE = {additive_mae:.2f}")
print(f"  Improvement in R²: {additive_r2 - best_r2:.4f}")
print(f"  Improvement in RMSE: {best_rmse - additive_rmse:.2f}")

# 2. MULTIPLICATIVE MODEL: predicted = cubic_bspline_fit * day_of_week_factor
print("\n" + "-"*80)
print("2. MULTIPLICATIVE MODEL")
print("-"*80)
print("Model: predicted = cubic_bspline_fit * day_of_week_factor")

# Calculate multiplicative factors
# Factor = actual / cubic_bspline_fit (average per day)
df['ratio'] = df['reported_scores'] / df['cubic_bspline_fit']
dow_factors = df.groupby('day_name')['ratio'].mean()
dow_factors = dow_factors.reindex(day_order)

print("\nDay-of-week factors (to multiply base prediction):")
for day in day_order:
    if day in dow_factors.index:
        print(f"  {day}: {dow_factors[day]:.4f}")

# Create multiplicative predictions
df['dow_factor'] = df['day_name'].map(dow_factors)
df['multiplicative_fit'] = df['cubic_bspline_fit'] * df['dow_factor']
df['multiplicative_residuals'] = df['reported_scores'] - df['multiplicative_fit']
# Calculate percentage residuals
df['multiplicative_residuals_pct'] = np.where(df['multiplicative_fit'] != 0,
                                               (df['multiplicative_residuals'] / df['multiplicative_fit']) * 100,
                                               np.nan)

# Calculate metrics for multiplicative model
multiplicative_r2 = r2_score(df['reported_scores'], df['multiplicative_fit'])
multiplicative_rmse = np.sqrt(mean_squared_error(df['reported_scores'], df['multiplicative_fit']))
multiplicative_mae = np.mean(np.abs(df['multiplicative_residuals']))

print(f"\nMultiplicative Model Metrics:")
print(f"  R² = {multiplicative_r2:.4f}")
print(f"  RMSE = {multiplicative_rmse:.2f}")
print(f"  MAE = {multiplicative_mae:.2f}")
print(f"  Improvement in R²: {multiplicative_r2 - best_r2:.4f}")
print(f"  Improvement in RMSE: {best_rmse - multiplicative_rmse:.2f}")

# ============================================================================
# MODEL COMPARISON
# ============================================================================

print("\n" + "="*80)
print("MODEL COMPARISON")
print("="*80)

comparison = pd.DataFrame({
    'Model': [
        'Base (Cubic B-Spline only)',
        'Additive (Base + Day-of-Week)',
        'Multiplicative (Base * Day-of-Week)'
    ],
    'R²': [best_r2, additive_r2, multiplicative_r2],
    'RMSE': [best_rmse, additive_rmse, multiplicative_rmse],
    'MAE': [
        np.mean(np.abs(df['base_residuals'])),
        additive_mae,
        multiplicative_mae
    ]
})

comparison['R²_Improvement'] = comparison['R²'] - best_r2
comparison['RMSE_Improvement'] = best_rmse - comparison['RMSE']

print("\n" + comparison.to_string(index=False))

# Determine best model
best_model_idx = comparison['R²'].idxmax()
best_model_name = comparison.loc[best_model_idx, 'Model']
print(f"\nBest Model: {best_model_name}")
print(f"  R² = {comparison.loc[best_model_idx, 'R²']:.4f}")
print(f"  RMSE = {comparison.loc[best_model_idx, 'RMSE']:.2f}")
print(f"  MAE = {comparison.loc[best_model_idx, 'MAE']:.2f}")

# Visualizations
print("\n" + "="*80)
print("CREATING VISUALIZATIONS")
print("="*80)

fig = plt.figure(figsize=(20, 16))

# 1. All three models comparison
ax1 = plt.subplot(3, 3, 1)
ax1.plot(df['Date'], df['reported_scores'], 'ko', markersize=1.5, alpha=0.3, label='Data')
ax1.plot(df['Date'], df['cubic_bspline_fit'], 'b-', linewidth=2, 
         label=f'Base (R²={best_r2:.4f})', alpha=0.8)
ax1.plot(df['Date'], df['additive_fit'], 'g-', linewidth=2, 
         label=f'Additive (R²={additive_r2:.4f})', alpha=0.8)
ax1.plot(df['Date'], df['multiplicative_fit'], 'r-', linewidth=2, 
         label=f'Multiplicative (R²={multiplicative_r2:.4f})', alpha=0.8)
ax1.set_title('Model Comparison: All Fits', fontsize=14, fontweight='bold')
ax1.set_xlabel('Date')
ax1.set_ylabel('Number of Reported Scores')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='x', rotation=45)

# 2. Base model percentage residuals
ax2 = plt.subplot(3, 3, 2)
ax2.plot(df['Date'], df['base_residuals_pct'], 'b-', linewidth=1, alpha=0.7)
ax2.axhline(y=0, color='r', linestyle='--', linewidth=1, alpha=0.5)
ax2.set_title(f'Base Model Residuals % (R²={best_r2:.4f})', fontsize=14, fontweight='bold')
ax2.set_xlabel('Date')
ax2.set_ylabel('Residual (%)')
ax2.grid(True, alpha=0.3)
ax2.tick_params(axis='x', rotation=45)

# 3. Additive model percentage residuals
ax3 = plt.subplot(3, 3, 3)
ax3.plot(df['Date'], df['additive_residuals_pct'], 'g-', linewidth=1, alpha=0.7)
ax3.axhline(y=0, color='r', linestyle='--', linewidth=1, alpha=0.5)
ax3.set_title(f'Additive Model Residuals % (R²={additive_r2:.4f})', fontsize=14, fontweight='bold')
ax3.set_xlabel('Date')
ax3.set_ylabel('Residual (%)')
ax3.grid(True, alpha=0.3)
ax3.tick_params(axis='x', rotation=45)

# 4. Multiplicative model percentage residuals
ax4 = plt.subplot(3, 3, 4)
ax4.plot(df['Date'], df['multiplicative_residuals_pct'], 'r-', linewidth=1, alpha=0.7)
ax4.axhline(y=0, color='r', linestyle='--', linewidth=1, alpha=0.5)
ax4.set_title(f'Multiplicative Model Residuals % (R²={multiplicative_r2:.4f})', fontsize=14, fontweight='bold')
ax4.set_xlabel('Date')
ax4.set_ylabel('Residual (%)')
ax4.grid(True, alpha=0.3)
ax4.tick_params(axis='x', rotation=45)

# 5. Percentage residual distributions comparison
ax5 = plt.subplot(3, 3, 5)
ax5.hist(df['base_residuals_pct'], bins=30, alpha=0.5, label='Base', color='blue', density=True)
ax5.hist(df['additive_residuals_pct'], bins=30, alpha=0.5, label='Additive', color='green', density=True)
ax5.hist(df['multiplicative_residuals_pct'], bins=30, alpha=0.5, label='Multiplicative', color='red', density=True)
ax5.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax5.set_title('Percentage Residual Distributions', fontsize=14, fontweight='bold')
ax5.set_xlabel('Residual (%)')
ax5.set_ylabel('Density')
ax5.legend()
ax5.grid(True, alpha=0.3, axis='y')

# 6. Day-of-week effects (additive)
ax6 = plt.subplot(3, 3, 6)
colors = ['red' if x < 0 else 'green' for x in dow_effects.values]
bars = ax6.bar(range(len(day_order)), dow_effects.values, color=colors, alpha=0.7)
ax6.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax6.set_title('Day-of-Week Effects (Additive)', fontsize=14, fontweight='bold')
ax6.set_xlabel('Day of Week')
ax6.set_ylabel('Effect (to add)')
ax6.set_xticks(range(len(day_order)))
ax6.set_xticklabels(day_order, rotation=45)
ax6.grid(True, alpha=0.3, axis='y')
for i, (bar, val) in enumerate(zip(bars, dow_effects.values)):
    ax6.text(bar.get_x() + bar.get_width()/2, val + (max(dow_effects.values) * 0.02 if val > 0 else -max(abs(dow_effects.values)) * 0.02), 
             f'{val:.0f}', ha='center', va='bottom' if val > 0 else 'top', fontsize=8)

# 7. Day-of-week factors (multiplicative)
ax7 = plt.subplot(3, 3, 7)
colors = ['red' if x < 1 else 'green' for x in dow_factors.values]
bars = ax7.bar(range(len(day_order)), dow_factors.values, color=colors, alpha=0.7)
ax7.axhline(y=1, color='black', linestyle='-', linewidth=1)
ax7.set_title('Day-of-Week Factors (Multiplicative)', fontsize=14, fontweight='bold')
ax7.set_xlabel('Day of Week')
ax7.set_ylabel('Factor (to multiply)')
ax7.set_xticks(range(len(day_order)))
ax7.set_xticklabels(day_order, rotation=45)
ax7.grid(True, alpha=0.3, axis='y')
for i, (bar, val) in enumerate(zip(bars, dow_factors.values)):
    ax7.text(bar.get_x() + bar.get_width()/2, val + (max(dow_factors.values) - 1) * 0.02 if val > 1 else -(1 - min(dow_factors.values)) * 0.02, 
             f'{val:.3f}', ha='center', va='bottom' if val > 1 else 'top', fontsize=8)

# 8. Metrics comparison bar chart
ax8 = plt.subplot(3, 3, 8)
x = np.arange(3)
width = 0.25
models = ['Base', 'Additive', 'Multiplicative']
r2_values = [best_r2, additive_r2, multiplicative_r2]
rmse_values = [best_rmse, additive_rmse, multiplicative_rmse]
# Normalize RMSE for visualization (divide by max)
rmse_norm = [v / max(rmse_values) for v in rmse_values]

bars1 = ax8.bar(x - width, r2_values, width, label='R²', alpha=0.8)
bars2 = ax8.bar(x, rmse_norm, width, label='RMSE (normalized)', alpha=0.8)
ax8.set_title('Model Metrics Comparison', fontsize=14, fontweight='bold')
ax8.set_xlabel('Model')
ax8.set_ylabel('Value (normalized)')
ax8.set_xticks(x)
ax8.set_xticklabels(models)
ax8.legend()
ax8.grid(True, alpha=0.3, axis='y')
# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=7)

# 9. Zoomed view of fits (first 100 points)
ax9 = plt.subplot(3, 3, 9)
subset_idx = slice(0, 100)
ax9.plot(df['Date'].iloc[subset_idx], df['reported_scores'].iloc[subset_idx], 
         'ko', markersize=3, alpha=0.5, label='Data')
ax9.plot(df['Date'].iloc[subset_idx], df['cubic_bspline_fit'].iloc[subset_idx], 
         'b-', linewidth=2, label='Base', alpha=0.8)
ax9.plot(df['Date'].iloc[subset_idx], df['additive_fit'].iloc[subset_idx], 
         'g-', linewidth=2, label='Additive', alpha=0.8)
ax9.plot(df['Date'].iloc[subset_idx], df['multiplicative_fit'].iloc[subset_idx], 
         'r-', linewidth=2, label='Multiplicative', alpha=0.8)
ax9.set_title('Zoomed View: First 100 Points', fontsize=14, fontweight='bold')
ax9.set_xlabel('Date')
ax9.set_ylabel('Number of Reported Scores')
ax9.legend(fontsize=8)
ax9.grid(True, alpha=0.3)
ax9.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('cubic_bspline_with_dow_comparison.png', dpi=300, bbox_inches='tight')
print("\nVisualization saved to 'cubic_bspline_with_dow_comparison.png'")

# Save results
output_df = df[['Date', 'day_name', 'reported_scores', 
                'cubic_bspline_fit', 'base_residuals', 'base_residuals_pct',
                'additive_fit', 'additive_residuals', 'additive_residuals_pct',
                'multiplicative_fit', 'multiplicative_residuals', 'multiplicative_residuals_pct',
                'dow_effect', 'dow_factor']].copy()
output_df.to_csv('cubic_bspline_with_dow_results.csv', index=False)
print("Results saved to 'cubic_bspline_with_dow_results.csv'")

# Save day-of-week parameters
dow_params = pd.DataFrame({
    'Day': day_order,
    'Additive_Effect': [dow_effects.get(day, 0) for day in day_order],
    'Multiplicative_Factor': [dow_factors.get(day, 1.0) for day in day_order]
})
dow_params.to_csv('day_of_week_parameters.csv', index=False)
print("Day-of-week parameters saved to 'day_of_week_parameters.csv'")

# Summary statistics
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

print("\nBase Model Residuals:")
print(f"  Mean: {df['base_residuals'].mean():.2f}")
print(f"  Std Dev: {df['base_residuals'].std():.2f}")
print(f"  Min: {df['base_residuals'].min():.2f}")
print(f"  Max: {df['base_residuals'].max():.2f}")
print(f"\nBase Model Percentage Residuals:")
print(f"  Mean: {df['base_residuals_pct'].mean():.2f}%")
print(f"  Std Dev: {df['base_residuals_pct'].std():.2f}%")
print(f"  Min: {df['base_residuals_pct'].min():.2f}%")
print(f"  Max: {df['base_residuals_pct'].max():.2f}%")

print("\nAdditive Model Residuals:")
print(f"  Mean: {df['additive_residuals'].mean():.2f}")
print(f"  Std Dev: {df['additive_residuals'].std():.2f}")
print(f"  Min: {df['additive_residuals'].min():.2f}")
print(f"  Max: {df['additive_residuals'].max():.2f}")
print(f"\nAdditive Model Percentage Residuals:")
print(f"  Mean: {df['additive_residuals_pct'].mean():.2f}%")
print(f"  Std Dev: {df['additive_residuals_pct'].std():.2f}%")
print(f"  Min: {df['additive_residuals_pct'].min():.2f}%")
print(f"  Max: {df['additive_residuals_pct'].max():.2f}%")

print("\nMultiplicative Model Residuals:")
print(f"  Mean: {df['multiplicative_residuals'].mean():.2f}")
print(f"  Std Dev: {df['multiplicative_residuals'].std():.2f}")
print(f"  Min: {df['multiplicative_residuals'].min():.2f}")
print(f"  Max: {df['multiplicative_residuals'].max():.2f}")
print(f"\nMultiplicative Model Percentage Residuals:")
print(f"  Mean: {df['multiplicative_residuals_pct'].mean():.2f}%")
print(f"  Std Dev: {df['multiplicative_residuals_pct'].std():.2f}%")
print(f"  Min: {df['multiplicative_residuals_pct'].min():.2f}%")
print(f"  Max: {df['multiplicative_residuals_pct'].max():.2f}%")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"\nBase Model: Cubic B-Spline with {best_n_knots} knots")
print(f"  R² = {best_r2:.4f}")
print(f"  RMSE = {best_rmse:.2f}")

print(f"\nBest Overall Model: {best_model_name}")
print(f"  R² = {comparison.loc[best_model_idx, 'R²']:.4f}")
print(f"  RMSE = {comparison.loc[best_model_idx, 'RMSE']:.2f}")
print(f"  Improvement in R²: {comparison.loc[best_model_idx, 'R²_Improvement']:.4f}")
print(f"  Improvement in RMSE: {comparison.loc[best_model_idx, 'RMSE_Improvement']:.2f}")

