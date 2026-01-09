"""
Visualize Cubic B-Spline with Different Numbers of Segments
===========================================================
Shows how the model looks with 1, 2, 3, 4, 5 piecewise segments
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
plt.rcParams['figure.figsize'] = (18, 12)

print("Creating visualization of different segment counts...")

# Load data
df = pd.read_csv('data/2023_MCM_Problem_C_Data.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

target_var = 'Number of  reported results'
df['days_since_start'] = (df['Date'] - df['Date'].min()).dt.days

X_time = df['days_since_start'].values
y = df[target_var].values

# Test different numbers of segments (1-5)
# For cubic spline: n_knots = n_segments + 1
# 1 segment = 2 knots, 2 segments = 3 knots, etc.
n_segments_list = [1, 2, 3, 4, 5]
results = []

fig, axes = plt.subplots(3, 2, figsize=(18, 14))
axes = axes.flatten()

for idx, n_segments in enumerate(n_segments_list):
    n_knots = n_segments + 1
    ax = axes[idx]
    
    try:
        # Create evenly spaced knots
        knots = np.linspace(X_time.min(), X_time.max(), n_knots)
        
        # For cubic spline with fixed knots
        if n_knots <= 2:
            internal_knots = knots
        else:
            internal_knots = knots[1:-1]  # Exclude endpoints for natural spline
        
        # Fit cubic B-spline
        tck = splrep(X_time, y, t=internal_knots, k=3, s=0)
        trend = splev(X_time, tck)
        
        # Calculate metrics
        r2 = r2_score(y, trend)
        rmse = np.sqrt(mean_squared_error(y, trend))
        mae = mean_absolute_error(y, trend)
        
        # Residual autocorrelation
        residuals = y - trend
        if len(residuals) > 10:
            resid_lag1 = residuals[1:]
            resid_today = residuals[:-1]
            resid_ac = np.corrcoef(resid_today, resid_lag1)[0,1] if len(resid_today) > 1 else 0
        else:
            resid_ac = 0
        
        results.append({
            'n_segments': n_segments,
            'n_knots': n_knots,
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'resid_ac': abs(resid_ac)
        })
        
        # Plot
        ax.plot(df['Date'], y, 'k-', linewidth=1.5, alpha=0.5, label='Actual Data', zorder=1)
        ax.plot(df['Date'], trend, 'r-', linewidth=2.5, alpha=0.9, label=f'B-Spline Fit', zorder=2)
        
        # Mark knot positions
        knot_dates = [df[df['days_since_start'] == int(k)]['Date'].values[0] if len(df[df['days_since_start'] == int(k)]) > 0 
                     else df['Date'].iloc[np.argmin(np.abs(df['days_since_start'].values - k))] 
                     for k in knots]
        for i, (knot_date, knot_val) in enumerate(zip(knot_dates, splev(knots, tck))):
            ax.scatter([knot_date], [knot_val], s=150, c='blue', marker='o', 
                      edgecolors='darkblue', linewidths=2, zorder=3, 
                      label='Knots' if i == 0 else '')
        
        # Add vertical lines at knot positions
        for knot_date in knot_dates:
            ax.axvline(x=knot_date, color='blue', linestyle='--', linewidth=1, alpha=0.4, zorder=0)
        
        ax.set_title(f'{n_segments} Segment{"s" if n_segments > 1 else ""} ({n_knots} knots)\n'
                    f'R²={r2:.4f}, RMSE={rmse:,.0f}, Resid_AC={abs(resid_ac):.4f}', 
                    fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Reported Results')
        if idx >= 4:  # Bottom row
            ax.set_xlabel('Date')
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        
    except Exception as e:
        ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'{n_segments} Segment{"s" if n_segments > 1 else ""} - Failed', fontsize=12)

# Remove the last subplot (6th one) if we only have 5 segments
axes[5].axis('off')

plt.tight_layout()
plt.savefig('plots/23_segment_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# Print summary
print("\n" + "=" * 80)
print("SUMMARY: Different Numbers of Segments")
print("=" * 80)
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

print("\n" + "=" * 80)
print("Saved: plots/23_segment_comparison.png")
print("=" * 80)

