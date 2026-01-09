"""
Testing Models for Forecasting (March 1st)
==========================================
Compare different models for their ability to forecast/extrapolate:
- Splines (interpolation - may not extrapolate well)
- Parametric models (polynomial, exponential, logistic) - better for extrapolation
- Simple smooth curves
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import splrep, splev
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (18, 12)

print("=" * 80)
print("TESTING MODELS FOR FORECASTING (Extrapolation Ability)")
print("=" * 80)

# Load data
df = pd.read_csv('data/2023_MCM_Problem_C_Data.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

target_var = 'Number of  reported results'
df['days_since_start'] = (df['Date'] - df['Date'].min()).dt.days

X_time = df['days_since_start'].values
y = df[target_var].values

# Split: Use most data for training, last 30 days to test extrapolation
split_idx = len(df) - 30
X_train = X_time[:split_idx]
y_train = y[:split_idx]
X_test = X_time[split_idx:]
y_test = y[split_idx:]

print(f"\nTraining: {len(X_train)} points ({df['Date'].iloc[0].date()} to {df['Date'].iloc[split_idx-1].date()})")
print(f"Test (extrapolation): {len(X_test)} points ({df['Date'].iloc[split_idx].date()} to {df['Date'].iloc[-1].date()})")

models = {}
results = []

# ============================================================================
# 1. CUBIC B-SPLINE (2 segments - minimal piecewise)
# ============================================================================
print("\n1. Testing Cubic B-Spline (2 segments)...")
try:
    n_segments = 2
    n_knots = n_segments + 1
    knots = np.linspace(X_train.min(), X_train.max(), n_knots)
    internal_knots = knots[1:-1] if n_knots > 2 else knots
    
    tck = splrep(X_train, y_train, t=internal_knots, k=3, s=0)
    trend_train = splev(X_train, tck)
    trend_test = splev(X_test, tck)  # Extrapolation
    
    r2_train = r2_score(y_train, trend_train)
    r2_test = r2_score(y_test, trend_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, trend_test))
    
    models['B-Spline (2 seg)'] = {
        'train': trend_train,
        'test': trend_test,
        'full': splev(X_time, tck),
        'r2_train': r2_train,
        'r2_test': r2_test,
        'rmse_test': rmse_test
    }
    results.append({
        'Model': 'B-Spline (2 segments)',
        'R²_train': r2_train,
        'R²_test': r2_test,
        'RMSE_test': rmse_test,
        'Type': 'Spline'
    })
    print(f"   Train R²={r2_train:.4f}, Test R²={r2_test:.4f}, Test RMSE={rmse_test:,.0f}")
except Exception as e:
    print(f"   Failed: {e}")

# ============================================================================
# 2. CUBIC B-SPLINE (3 segments)
# ============================================================================
print("\n2. Testing Cubic B-Spline (3 segments)...")
try:
    n_segments = 3
    n_knots = n_segments + 1
    knots = np.linspace(X_train.min(), X_train.max(), n_knots)
    internal_knots = knots[1:-1]
    
    tck = splrep(X_train, y_train, t=internal_knots, k=3, s=0)
    trend_train = splev(X_train, tck)
    trend_test = splev(X_test, tck)
    
    r2_train = r2_score(y_train, trend_train)
    r2_test = r2_score(y_test, trend_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, trend_test))
    
    models['B-Spline (3 seg)'] = {
        'train': trend_train,
        'test': trend_test,
        'full': splev(X_time, tck),
        'r2_train': r2_train,
        'r2_test': r2_test,
        'rmse_test': rmse_test
    }
    results.append({
        'Model': 'B-Spline (3 segments)',
        'R²_train': r2_train,
        'R²_test': r2_test,
        'RMSE_test': rmse_test,
        'Type': 'Spline'
    })
    print(f"   Train R²={r2_train:.4f}, Test R²={r2_test:.4f}, Test RMSE={rmse_test:,.0f}")
except Exception as e:
    print(f"   Failed: {e}")

# ============================================================================
# 3. POLYNOMIAL (Degree 3 - single smooth curve)
# ============================================================================
print("\n3. Testing Polynomial (Degree 3)...")
try:
    coeffs = np.polyfit(X_train, y_train, 3)
    poly_func = np.poly1d(coeffs)
    
    trend_train = poly_func(X_train)
    trend_test = poly_func(X_test)
    
    r2_train = r2_score(y_train, trend_train)
    r2_test = r2_score(y_test, trend_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, trend_test))
    
    models['Polynomial (deg 3)'] = {
        'train': trend_train,
        'test': trend_test,
        'full': poly_func(X_time),
        'r2_train': r2_train,
        'r2_test': r2_test,
        'rmse_test': rmse_test
    }
    results.append({
        'Model': 'Polynomial (deg 3)',
        'R²_train': r2_train,
        'R²_test': r2_test,
        'RMSE_test': rmse_test,
        'Type': 'Parametric'
    })
    print(f"   Train R²={r2_train:.4f}, Test R²={r2_test:.4f}, Test RMSE={rmse_test:,.0f}")
except Exception as e:
    print(f"   Failed: {e}")

# ============================================================================
# 4. POLYNOMIAL (Degree 4)
# ============================================================================
print("\n4. Testing Polynomial (Degree 4)...")
try:
    coeffs = np.polyfit(X_train, y_train, 4)
    poly_func = np.poly1d(coeffs)
    
    trend_train = poly_func(X_train)
    trend_test = poly_func(X_test)
    
    r2_train = r2_score(y_train, trend_train)
    r2_test = r2_score(y_test, trend_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, trend_test))
    
    models['Polynomial (deg 4)'] = {
        'train': trend_train,
        'test': trend_test,
        'full': poly_func(X_time),
        'r2_train': r2_train,
        'r2_test': r2_test,
        'rmse_test': rmse_test
    }
    results.append({
        'Model': 'Polynomial (deg 4)',
        'R²_train': r2_train,
        'R²_test': r2_test,
        'RMSE_test': rmse_test,
        'Type': 'Parametric'
    })
    print(f"   Train R²={r2_train:.4f}, Test R²={r2_test:.4f}, Test RMSE={rmse_test:,.0f}")
except Exception as e:
    print(f"   Failed: {e}")

# ============================================================================
# 5. LOGISTIC/SIGMOID (Good for growth patterns)
# ============================================================================
print("\n5. Testing Logistic/Sigmoid Curve...")
def logistic(x, a, b, c, d):
    """Logistic/S-curve: a / (1 + exp(-b*(x-c))) + d"""
    return a / (1 + np.exp(-b * (x - c))) + d

try:
    # Initial guess
    p0 = [y_train.max() - y_train.min(), 0.01, X_train.mean(), y_train.min()]
    popt, _ = curve_fit(logistic, X_train, y_train, p0=p0, maxfev=5000)
    
    trend_train = logistic(X_train, *popt)
    trend_test = logistic(X_test, *popt)
    
    r2_train = r2_score(y_train, trend_train)
    r2_test = r2_score(y_test, trend_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, trend_test))
    
    models['Logistic/Sigmoid'] = {
        'train': trend_train,
        'test': trend_test,
        'full': logistic(X_time, *popt),
        'r2_train': r2_train,
        'r2_test': r2_test,
        'rmse_test': rmse_test
    }
    results.append({
        'Model': 'Logistic/Sigmoid',
        'R²_train': r2_train,
        'R²_test': r2_test,
        'RMSE_test': rmse_test,
        'Type': 'Parametric'
    })
    print(f"   Train R²={r2_train:.4f}, Test R²={r2_test:.4f}, Test RMSE={rmse_test:,.0f}")
except Exception as e:
    print(f"   Failed: {e}")

# ============================================================================
# 6. PIECEWISE LINEAR (2 segments - simplest)
# ============================================================================
print("\n6. Testing Piecewise Linear (2 segments)...")
def piecewise_linear(x, x0, y0, k1, k2):
    """Two linear segments with breakpoint at x0"""
    return np.piecewise(x, [x < x0], 
                       [lambda x: k1*x + y0, 
                        lambda x: k2*x + (k1*x0 + y0 - k2*x0)])

try:
    # Initial guess: breakpoint at middle, slopes from data
    x0_guess = X_train.mean()
    y0_guess = y_train[0]
    k1_guess = (y_train[len(y_train)//2] - y_train[0]) / (X_train[len(X_train)//2] - X_train[0])
    k2_guess = (y_train[-1] - y_train[len(y_train)//2]) / (X_train[-1] - X_train[len(X_train)//2])
    
    popt, _ = curve_fit(piecewise_linear, X_train, y_train, 
                       p0=[x0_guess, y0_guess, k1_guess, k2_guess],
                       maxfev=5000)
    
    trend_train = piecewise_linear(X_train, *popt)
    trend_test = piecewise_linear(X_test, *popt)
    
    r2_train = r2_score(y_train, trend_train)
    r2_test = r2_score(y_test, trend_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, trend_test))
    
    models['Piecewise Linear (2)'] = {
        'train': trend_train,
        'test': trend_test,
        'full': piecewise_linear(X_time, *popt),
        'r2_train': r2_train,
        'r2_test': r2_test,
        'rmse_test': rmse_test
    }
    results.append({
        'Model': 'Piecewise Linear (2)',
        'R²_train': r2_train,
        'R²_test': r2_test,
        'RMSE_test': rmse_test,
        'Type': 'Parametric'
    })
    print(f"   Train R²={r2_train:.4f}, Test R²={r2_test:.4f}, Test RMSE={rmse_test:,.0f}")
except Exception as e:
    print(f"   Failed: {e}")

# ============================================================================
# 7. EXPONENTIAL DECAY (if popularity is declining)
# ============================================================================
print("\n7. Testing Exponential Decay...")
def exp_decay(x, a, b, c):
    """Exponential decay: a * exp(-b*x) + c"""
    return a * np.exp(-b * x) + c

try:
    p0 = [y_train.max() - y_train.min(), 0.001, y_train.min()]
    popt, _ = curve_fit(exp_decay, X_train, y_train, p0=p0, maxfev=5000)
    
    trend_train = exp_decay(X_train, *popt)
    trend_test = exp_decay(X_test, *popt)
    
    r2_train = r2_score(y_train, trend_train)
    r2_test = r2_score(y_test, trend_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, trend_test))
    
    models['Exponential Decay'] = {
        'train': trend_train,
        'test': trend_test,
        'full': exp_decay(X_time, *popt),
        'r2_train': r2_train,
        'r2_test': r2_test,
        'rmse_test': rmse_test
    }
    results.append({
        'Model': 'Exponential Decay',
        'R²_train': r2_train,
        'R²_test': r2_test,
        'RMSE_test': rmse_test,
        'Type': 'Parametric'
    })
    print(f"   Train R²={r2_train:.4f}, Test R²={r2_test:.4f}, Test RMSE={rmse_test:,.0f}")
except Exception as e:
    print(f"   Failed: {e}")

# ============================================================================
# SUMMARY AND VISUALIZATION
# ============================================================================
print("\n" + "=" * 80)
print("FORECASTING PERFORMANCE SUMMARY")
print("=" * 80)
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('R²_test', ascending=False)
print(results_df.to_string(index=False))

# Create visualization
fig, axes = plt.subplots(3, 3, figsize=(20, 16))
axes = axes.flatten()

for idx, (name, model_data) in enumerate(models.items()):
    if idx >= 9:
        break
    
    ax = axes[idx]
    
    # Plot training data
    ax.scatter(df['Date'].iloc[:split_idx], y_train, alpha=0.3, s=10, 
              color='gray', label='Training Data', zorder=1)
    
    # Plot test data
    ax.scatter(df['Date'].iloc[split_idx:], y_test, alpha=0.5, s=15, 
              color='red', marker='x', label='Test Data (Extrapolation)', zorder=2)
    
    # Plot full prediction
    ax.plot(df['Date'], model_data['full'], 'b-', linewidth=2.5, 
           alpha=0.8, label='Model Prediction', zorder=3)
    
    # Highlight extrapolation region
    ax.axvline(x=df['Date'].iloc[split_idx], color='red', linestyle='--', 
              linewidth=1.5, alpha=0.7, label='Extrapolation Start', zorder=0)
    
    ax.set_title(f'{name}\nTrain R²={model_data["r2_train"]:.3f}, '
                f'Test R²={model_data["r2_test"]:.3f}\nTest RMSE={model_data["rmse_test"]:,.0f}',
                fontsize=10, fontweight='bold')
    ax.set_ylabel('Number of Results')
    if idx >= 6:
        ax.set_xlabel('Date')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

# Remove unused subplots
for idx in range(len(models), 9):
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig('plots/24_forecasting_models.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n" + "=" * 80)
print("Best model for forecasting (highest Test R²):")
print(f"  {results_df.iloc[0]['Model']}")
print(f"  Train R² = {results_df.iloc[0]['R²_train']:.4f}")
print(f"  Test R² = {results_df.iloc[0]['R²_test']:.4f}")
print(f"  Test RMSE = {results_df.iloc[0]['RMSE_test']:,.0f}")
print("\n" + "=" * 80)
print("Saved: plots/24_forecasting_models.png")
print("=" * 80)

