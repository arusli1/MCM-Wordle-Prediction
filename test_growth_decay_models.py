"""
Testing Growth-Then-Decay Models
=================================
Wordle likely had rapid growth initially, then decayed.
Test models that capture this right-skewed pattern.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (18, 12)

print("=" * 80)
print("TESTING GROWTH-THEN-DECAY MODELS (Right-Skewed Pattern)")
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

print(f"\nTraining: {len(X_train)} points")
print(f"Test (extrapolation): {len(X_test)} points")
print(f"Peak value: {y.max():,.0f} at day {X_time[np.argmax(y)]}")

models = {}
results = []

# ============================================================================
# 1. GROWTH-THEN-DECAY (Piecewise: Exponential Growth + Exponential Decay)
# ============================================================================
print("\n1. Testing Growth-Then-Decay (Exponential Growth + Exponential Decay)...")

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

try:
    # Find peak to estimate breakpoint
    peak_idx = np.argmax(y_train)
    x0_guess = X_train[peak_idx]
    
    # Growth phase: fit exponential to first half
    growth_data = y_train[:peak_idx+1]
    growth_x = X_train[:peak_idx+1]
    if len(growth_data) > 3:
        try:
            popt_growth, _ = curve_fit(lambda x, a, b: a * np.exp(b * x), 
                                       growth_x, growth_data, maxfev=5000,
                                       p0=[growth_data[0], 0.01])
            a1_guess, b1_guess = popt_growth
        except:
            a1_guess = growth_data[0]
            b1_guess = 0.01
    else:
        a1_guess = growth_data[0] if len(growth_data) > 0 else y_train[0]
        b1_guess = 0.01
    
    # Decay phase: fit exponential decay to second half
    decay_data = y_train[peak_idx:]
    decay_x = X_train[peak_idx:]
    if len(decay_data) > 3:
        try:
            popt_decay, _ = curve_fit(lambda x, a, b, c: a * np.exp(-b * (x - x0_guess)) + c,
                                     decay_x, decay_data, maxfev=5000,
                                     p0=[decay_data[0], 0.001, decay_data[-1]])
            a2_guess, b2_guess, c_guess = popt_decay
        except:
            a2_guess = decay_data[0] if len(decay_data) > 0 else y_train.max()
            b2_guess = 0.001
            c_guess = decay_data[-1] if len(decay_data) > 0 else y_train.min()
    else:
        a2_guess = y_train.max()
        b2_guess = 0.001
        c_guess = y_train.min()
    
    # Fit full model
    p0 = [x0_guess, a1_guess, b1_guess, a2_guess, b2_guess, c_guess]
    popt, _ = curve_fit(growth_then_decay, X_train, y_train, p0=p0, maxfev=10000,
                       bounds=([X_train[0], 0, 0, 0, 0, y_train.min()],
                              [X_train[-1], y_train.max()*10, 1, y_train.max()*10, 1, y_train.max()]))
    
    trend_train = growth_then_decay(X_train, *popt)
    trend_test = growth_then_decay(X_test, *popt)
    trend_full = growth_then_decay(X_time, *popt)
    
    r2_train = r2_score(y_train, trend_train)
    r2_test = r2_score(y_test, trend_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, trend_test))
    
    models['Growth-Then-Decay (Exp)'] = {
        'train': trend_train,
        'test': trend_test,
        'full': trend_full,
        'r2_train': r2_train,
        'r2_test': r2_test,
        'rmse_test': rmse_test,
        'breakpoint': popt[0]
    }
    results.append({
        'Model': 'Growth-Then-Decay (Exp)',
        'R²_train': r2_train,
        'R²_test': r2_test,
        'RMSE_test': rmse_test,
        'Type': 'Piecewise Parametric'
    })
    print(f"   Train R²={r2_train:.4f}, Test R²={r2_test:.4f}, Test RMSE={rmse_test:,.0f}")
    print(f"   Breakpoint at day {popt[0]:.0f}")
except Exception as e:
    print(f"   Failed: {e}")

# ============================================================================
# 2. GROWTH-THEN-LINEAR DECAY (Simpler)
# ============================================================================
print("\n2. Testing Growth-Then-Linear Decay...")

def growth_then_linear(x, x0, a, b, k, c):
    """
    Growth: a * exp(b * x) for x < x0
    Linear decay: k * (x - x0) + c for x >= x0
    """
    result = np.zeros_like(x)
    mask_growth = x < x0
    mask_decay = x >= x0
    result[mask_growth] = a * np.exp(b * x[mask_growth])
    result[mask_decay] = k * (x[mask_decay] - x0) + c
    return result

try:
    peak_idx = np.argmax(y_train)
    x0_guess = X_train[peak_idx]
    
    # Growth phase
    growth_data = y_train[:peak_idx+1]
    growth_x = X_train[:peak_idx+1]
    if len(growth_data) > 3:
        try:
            popt_growth, _ = curve_fit(lambda x, a, b: a * np.exp(b * x),
                                      growth_x, growth_data, maxfev=5000,
                                      p0=[growth_data[0], 0.01])
            a_guess, b_guess = popt_growth
        except:
            a_guess = growth_data[0]
            b_guess = 0.01
    else:
        a_guess = growth_data[0] if len(growth_data) > 0 else y_train[0]
        b_guess = 0.01
    
    # Linear decay
    decay_data = y_train[peak_idx:]
    decay_x = X_train[peak_idx:]
    if len(decay_data) > 2:
        k_guess = (decay_data[-1] - decay_data[0]) / (decay_x[-1] - decay_x[0]) if decay_x[-1] != decay_x[0] else -1
        c_guess = decay_data[0]
    else:
        k_guess = -1
        c_guess = y_train.max()
    
    p0 = [x0_guess, a_guess, b_guess, k_guess, c_guess]
    popt, _ = curve_fit(growth_then_linear, X_train, y_train, p0=p0, maxfev=10000,
                       bounds=([X_train[0], 0, 0, -np.inf, y_train.min()],
                              [X_train[-1], y_train.max()*10, 1, 0, y_train.max()]))
    
    trend_train = growth_then_linear(X_train, *popt)
    trend_test = growth_then_linear(X_test, *popt)
    trend_full = growth_then_linear(X_time, *popt)
    
    r2_train = r2_score(y_train, trend_train)
    r2_test = r2_score(y_test, trend_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, trend_test))
    
    models['Growth-Then-Linear'] = {
        'train': trend_train,
        'test': trend_test,
        'full': trend_full,
        'r2_train': r2_train,
        'r2_test': r2_test,
        'rmse_test': rmse_test,
        'breakpoint': popt[0]
    }
    results.append({
        'Model': 'Growth-Then-Linear',
        'R²_train': r2_train,
        'R²_test': r2_test,
        'RMSE_test': rmse_test,
        'Type': 'Piecewise Parametric'
    })
    print(f"   Train R²={r2_train:.4f}, Test R²={r2_test:.4f}, Test RMSE={rmse_test:,.0f}")
    print(f"   Breakpoint at day {popt[0]:.0f}")
except Exception as e:
    print(f"   Failed: {e}")

# ============================================================================
# 3. ASYMMETRIC LOGISTIC (Right-Skewed S-Curve)
# ============================================================================
print("\n3. Testing Asymmetric Logistic (Right-Skewed)...")

def asymmetric_logistic(x, a, b, c, d, e):
    """
    Asymmetric logistic: a / (1 + exp(-b*(x-c)))^d + e
    d controls asymmetry (right-skewed if d < 1)
    """
    return a / ((1 + np.exp(-b * (x - c)))**d) + e

try:
    p0 = [y_train.max() - y_train.min(), 0.01, X_train.mean(), 0.5, y_train.min()]
    popt, _ = curve_fit(asymmetric_logistic, X_train, y_train, p0=p0, maxfev=10000,
                       bounds=([0, 0, X_train.min(), 0.1, y_train.min()],
                              [y_train.max()*2, 1, X_train.max(), 2, y_train.max()]))
    
    trend_train = asymmetric_logistic(X_train, *popt)
    trend_test = asymmetric_logistic(X_test, *popt)
    trend_full = asymmetric_logistic(X_time, *popt)
    
    r2_train = r2_score(y_train, trend_train)
    r2_test = r2_score(y_test, trend_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, trend_test))
    
    models['Asymmetric Logistic'] = {
        'train': trend_train,
        'test': trend_test,
        'full': trend_full,
        'r2_train': r2_train,
        'r2_test': r2_test,
        'rmse_test': rmse_test
    }
    results.append({
        'Model': 'Asymmetric Logistic',
        'R²_train': r2_train,
        'R²_test': r2_test,
        'RMSE_test': rmse_test,
        'Type': 'Parametric'
    })
    print(f"   Train R²={r2_train:.4f}, Test R²={r2_test:.4f}, Test RMSE={rmse_test:,.0f}")
except Exception as e:
    print(f"   Failed: {e}")

# ============================================================================
# 4. GAMMA DISTRIBUTION SHAPE (Right-Skewed)
# ============================================================================
print("\n4. Testing Gamma Distribution Shape...")

def gamma_shape(x, a, b, c, d):
    """
    Gamma-like shape: a * (x - c)^b * exp(-d*(x - c)) + e
    Right-skewed distribution shape
    """
    x_shifted = x - c
    x_shifted = np.maximum(x_shifted, 0.1)  # Avoid negative/zero
    return a * (x_shifted ** b) * np.exp(-d * x_shifted)

try:
    peak_idx = np.argmax(y_train)
    c_guess = X_train[0]  # Start point
    a_guess = y_train.max()
    b_guess = 2.0
    d_guess = 0.01
    
    p0 = [a_guess, b_guess, c_guess, d_guess]
    popt, _ = curve_fit(gamma_shape, X_train, y_train, p0=p0, maxfev=10000,
                       bounds=([0, 0, X_train.min() - 10, 0],
                              [y_train.max()*10, 10, X_train.max(), 1]))
    
    trend_train = gamma_shape(X_train, *popt)
    trend_test = gamma_shape(X_test, *popt)
    trend_full = gamma_shape(X_time, *popt)
    
    r2_train = r2_score(y_train, trend_train)
    r2_test = r2_score(y_test, trend_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, trend_test))
    
    models['Gamma Shape'] = {
        'train': trend_train,
        'test': trend_test,
        'full': trend_full,
        'r2_train': r2_train,
        'r2_test': r2_test,
        'rmse_test': rmse_test
    }
    results.append({
        'Model': 'Gamma Shape',
        'R²_train': r2_train,
        'R²_test': r2_test,
        'RMSE_test': rmse_test,
        'Type': 'Parametric'
    })
    print(f"   Train R²={r2_train:.4f}, Test R²={r2_test:.4f}, Test RMSE={rmse_test:,.0f}")
except Exception as e:
    print(f"   Failed: {e}")

# ============================================================================
# 5. PIECEWISE CUBIC (2 segments - growth then decay)
# ============================================================================
print("\n5. Testing Piecewise Cubic (Growth then Decay)...")

def piecewise_cubic(x, x0, a0, a1, a2, a3, b0, b1, b2, b3):
    """
    Two cubic polynomials with smooth transition at x0
    Growth: a0 + a1*x + a2*x^2 + a3*x^3 for x < x0
    Decay: b0 + b1*x + b2*x^2 + b3*x^3 for x >= x0
    """
    result = np.zeros_like(x)
    mask_growth = x < x0
    mask_decay = x >= x0
    result[mask_growth] = a0 + a1*x[mask_growth] + a2*x[mask_growth]**2 + a3*x[mask_growth]**3
    result[mask_decay] = b0 + b1*x[mask_decay] + b2*x[mask_decay]**2 + b3*x[mask_decay]**3
    return result

try:
    peak_idx = np.argmax(y_train)
    x0_guess = X_train[peak_idx]
    
    # Fit cubic to growth phase
    growth_data = y_train[:peak_idx+1]
    growth_x = X_train[:peak_idx+1]
    if len(growth_data) > 4:
        coeffs_growth = np.polyfit(growth_x, growth_data, 3)
        a0_guess, a1_guess, a2_guess, a3_guess = coeffs_growth[::-1]
    else:
        a0_guess, a1_guess, a2_guess, a3_guess = growth_data[0], 0, 0, 0
    
    # Fit cubic to decay phase
    decay_data = y_train[peak_idx:]
    decay_x = X_train[peak_idx:]
    if len(decay_data) > 4:
        coeffs_decay = np.polyfit(decay_x, decay_data, 3)
        b0_guess, b1_guess, b2_guess, b3_guess = coeffs_decay[::-1]
    else:
        b0_guess, b1_guess, b2_guess, b3_guess = decay_data[0], -1, 0, 0
    
    p0 = [x0_guess, a0_guess, a1_guess, a2_guess, a3_guess, b0_guess, b1_guess, b2_guess, b3_guess]
    popt, _ = curve_fit(piecewise_cubic, X_train, y_train, p0=p0, maxfev=10000)
    
    trend_train = piecewise_cubic(X_train, *popt)
    trend_test = piecewise_cubic(X_test, *popt)
    trend_full = piecewise_cubic(X_time, *popt)
    
    r2_train = r2_score(y_train, trend_train)
    r2_test = r2_score(y_test, trend_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, trend_test))
    
    models['Piecewise Cubic'] = {
        'train': trend_train,
        'test': trend_test,
        'full': trend_full,
        'r2_train': r2_train,
        'r2_test': r2_test,
        'rmse_test': rmse_test,
        'breakpoint': popt[0]
    }
    results.append({
        'Model': 'Piecewise Cubic',
        'R²_train': r2_train,
        'R²_test': r2_test,
        'RMSE_test': rmse_test,
        'Type': 'Piecewise Parametric'
    })
    print(f"   Train R²={r2_train:.4f}, Test R²={r2_test:.4f}, Test RMSE={rmse_test:,.0f}")
    print(f"   Breakpoint at day {popt[0]:.0f}")
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
n_models = len(models)
n_cols = 3
n_rows = (n_models + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 6*n_rows))
if n_models == 1:
    axes = [axes]
else:
    axes = axes.flatten()

for idx, (name, model_data) in enumerate(models.items()):
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
    
    # Mark breakpoint if exists
    if 'breakpoint' in model_data:
        breakpoint_date = df[df['days_since_start'] == int(model_data['breakpoint'])]['Date'].values
        if len(breakpoint_date) > 0:
            ax.axvline(x=breakpoint_date[0], color='green', linestyle='--', 
                      linewidth=1.5, alpha=0.7, label='Breakpoint', zorder=0)
    
    # Highlight extrapolation region
    ax.axvline(x=df['Date'].iloc[split_idx], color='red', linestyle='--', 
              linewidth=1.5, alpha=0.7, label='Extrapolation Start', zorder=0)
    
    ax.set_title(f'{name}\nTrain R²={model_data["r2_train"]:.3f}, '
                f'Test R²={model_data["r2_test"]:.3f}\nTest RMSE={model_data["rmse_test"]:,.0f}',
                fontsize=11, fontweight='bold')
    ax.set_ylabel('Number of Results')
    if idx >= n_models - n_cols:
        ax.set_xlabel('Date')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

# Remove unused subplots
for idx in range(n_models, len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig('plots/25_growth_decay_models.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n" + "=" * 80)
if len(results_df) > 0:
    print("Best model for forecasting (highest Test R²):")
    print(f"  {results_df.iloc[0]['Model']}")
    print(f"  Train R² = {results_df.iloc[0]['R²_train']:.4f}")
    print(f"  Test R² = {results_df.iloc[0]['R²_test']:.4f}")
    print(f"  Test RMSE = {results_df.iloc[0]['RMSE_test']:,.0f}")
print("\n" + "=" * 80)
print("Saved: plots/25_growth_decay_models.png")
print("=" * 80)

