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

# Initialize storage for all model attempts
all_model_attempts = []

def growth_then_decay_smooth(x, x0, a1, b1, a2, b2, c, k=1.0):
    """
    Growth phase: a1 * exp(b1 * x) for x < x0
    Decay phase: a2 * exp(-b2 * (x - x0)) + c for x >= x0
    With smooth transition parameter k (higher = smoother transition)
    """
    result = np.zeros_like(x)
    mask_growth = x < x0
    mask_decay = x >= x0
    
    # Growth phase
    result[mask_growth] = a1 * np.exp(b1 * x[mask_growth])
    
    # Decay phase - ensure continuity at x0
    y0 = a1 * np.exp(b1 * x0)  # Value at breakpoint
    result[mask_decay] = (y0 - c) * np.exp(-b2 * (x[mask_decay] - x0)) + c
    
    return result

def growth_then_decay_optimized(x, x0, a1, b1, b2, c):
    """
    Optimized version with continuity constraint at x0
    Growth: a1 * exp(b1 * x)
    Decay: (a1*exp(b1*x0) - c) * exp(-b2*(x-x0)) + c
    """
    result = np.zeros_like(x)
    mask_growth = x < x0
    mask_decay = x >= x0
    
    result[mask_growth] = a1 * np.exp(b1 * x[mask_growth])
    y0 = a1 * np.exp(b1 * x0)  # Ensure continuity
    result[mask_decay] = (y0 - c) * np.exp(-b2 * (x[mask_decay] - x0)) + c
    
    return result

def growth_then_decay_flexible(x, x0, a1, b1, b2, c, d):
    """
    More flexible: Growth with initial offset, smoother transition
    Growth: a1 * exp(b1 * x) + d
    Decay: (y0 - c) * exp(-b2*(x-x0)) + c where y0 = a1*exp(b1*x0) + d
    """
    result = np.zeros_like(x)
    mask_growth = x < x0
    mask_decay = x >= x0
    
    result[mask_growth] = a1 * np.exp(b1 * x[mask_growth]) + d
    y0 = a1 * np.exp(b1 * x0) + d  # Value at breakpoint
    result[mask_decay] = (y0 - c) * np.exp(-b2 * (x[mask_decay] - x0)) + c
    
    return result

def growth_then_decay_with_oscillation(x, x0, a1, b1, b2, c, d, A, omega, phi):
    """
    Growth then decay with oscillatory component to capture up-down pattern
    Growth: a1 * exp(b1 * x) + d
    Decay: (y0 - c) * exp(-b2*(x-x0)) + c + A * sin(omega*(x-x0) + phi)
    """
    result = np.zeros_like(x)
    mask_growth = x < x0
    mask_decay = x >= x0
    
    result[mask_growth] = a1 * np.exp(b1 * x[mask_growth]) + d
    y0 = a1 * np.exp(b1 * x0) + d
    decay_base = (y0 - c) * np.exp(-b2 * (x[mask_decay] - x0)) + c
    oscillation = A * np.sin(omega * (x[mask_decay] - x0) + phi)
    result[mask_decay] = decay_base + oscillation
    
    return result

def growth_then_decay_polynomial_decay(x, x0, a1, b1, b2, c, d, p1, p2):
    """
    Growth then decay with polynomial adjustment in decay phase
    Growth: a1 * exp(b1 * x) + d
    Decay: (y0 - c) * exp(-b2*(x-x0)) + c + p1*(x-x0) + p2*(x-x0)^2
    """
    result = np.zeros_like(x)
    mask_growth = x < x0
    mask_decay = x >= x0
    
    result[mask_growth] = a1 * np.exp(b1 * x[mask_growth]) + d
    y0 = a1 * np.exp(b1 * x0) + d
    x_decay = x[mask_decay] - x0
    decay_base = (y0 - c) * np.exp(-b2 * x_decay) + c
    polynomial = p1 * x_decay + p2 * x_decay**2
    result[mask_decay] = decay_base + polynomial
    
    return result

def growth_then_decay_quadratic_decay(x, x0, a1, b1, b2, c, d, q1, q2, q3):
    """
    Growth then decay with quadratic polynomial decay (more flexible)
    Growth: a1 * exp(b1 * x) + d
    Decay: q1 + q2*(x-x0) + q3*(x-x0)^2 (fully quadratic, ensuring continuity)
    """
    result = np.zeros_like(x)
    mask_growth = x < x0
    mask_decay = x >= x0
    
    result[mask_growth] = a1 * np.exp(b1 * x[mask_growth]) + d
    y0 = a1 * np.exp(b1 * x0) + d  # Ensure continuity
    q1_constrained = y0  # q1 must equal y0 at x=x0
    x_decay = x[mask_decay] - x0
    result[mask_decay] = q1_constrained + q2 * x_decay + q3 * x_decay**2
    
    return result

def growth_then_decay_cubic_decay(x, x0, a1, b1, c, d, q1, q2, q3, q4):
    """
    Growth then cubic decay (very flexible for up-down patterns)
    Growth: a1 * exp(b1 * x) + d
    Decay: q1 + q2*(x-x0) + q3*(x-x0)^2 + q4*(x-x0)^3 (ensuring continuity)
    """
    result = np.zeros_like(x)
    mask_growth = x < x0
    mask_decay = x >= x0
    
    result[mask_growth] = a1 * np.exp(b1 * x[mask_growth]) + d
    y0 = a1 * np.exp(b1 * x0) + d  # Continuity constraint
    x_decay = x[mask_decay] - x0
    result[mask_decay] = y0 + q1 * x_decay + q2 * x_decay**2 + q3 * x_decay**3
    
    return result

def growth_then_decay_adaptive(x, x0, a1, b1, b2, c, d, A, omega, phi, p1):
    """
    Combined: oscillation + polynomial adjustment
    """
    result = np.zeros_like(x)
    mask_growth = x < x0
    mask_decay = x >= x0
    
    result[mask_growth] = a1 * np.exp(b1 * x[mask_growth]) + d
    y0 = a1 * np.exp(b1 * x0) + d
    x_decay = x[mask_decay] - x0
    decay_base = (y0 - c) * np.exp(-b2 * x_decay) + c
    oscillation = A * np.sin(omega * x_decay + phi)
    polynomial = p1 * x_decay
    result[mask_decay] = decay_base + oscillation + polynomial
    
    return result

def growth_then_decay_weighted(x, x0, a1, b1, b2, c):
    """Same as optimized but will use weighted fitting"""
    result = np.zeros_like(x)
    mask_growth = x < x0
    mask_decay = x >= x0
    
    result[mask_growth] = a1 * np.exp(b1 * x[mask_growth])
    y0 = a1 * np.exp(b1 * x0)
    result[mask_decay] = (y0 - c) * np.exp(-b2 * (x[mask_decay] - x0)) + c
    
    return result

# Fit on full dataset with better initial guesses
peak_idx = np.argmax(y)
x0_guess = X_time[peak_idx]
y0_actual = y[peak_idx]

# Better initial parameters
# Growth: fit to early data
early_data = y[:min(peak_idx+5, len(y)//3)]
early_x = X_time[:len(early_data)]
if len(early_data) > 3:
    try:
        log_early = np.log(np.maximum(early_data, 1))
        b1_guess, a1_log = np.polyfit(early_x, log_early, 1)
        a1_guess = np.exp(a1_log)
    except:
        a1_guess = y[0]
        b1_guess = 0.01
else:
    a1_guess = y[0]
    b1_guess = 0.01

# Decay: fit to later data
late_data = y[peak_idx:]
late_x = X_time[peak_idx:] - x0_guess
if len(late_data) > 10:
    try:
        # Fit exponential decay: y = A*exp(-b*x) + c
        decay_fit = lambda x, A, b, c: A * np.exp(-b * x) + c
        p0_decay = [y0_actual, 0.001, y[-1]]
        popt_decay, _ = curve_fit(decay_fit, late_x, late_data, p0=p0_decay, maxfev=5000,
                                  bounds=([0, 0, y.min()], [y.max()*2, 1, y.max()]))
        b2_guess = popt_decay[1]
        c_guess = popt_decay[2]
    except:
        b2_guess = 0.001
        c_guess = y[-1]
else:
    b2_guess = 0.001
    c_guess = y[-1]

p0 = [x0_guess, a1_guess, b1_guess, b2_guess, c_guess]

print(f"   Initial parameter guesses:")
print(f"     Breakpoint: {x0_guess:.1f}")
print(f"     Growth: a1={a1_guess:.0f}, b1={b1_guess:.4f}")
print(f"     Decay: b2={b2_guess:.4f}, c={c_guess:.0f}")

# Try both models and pick the best
print(f"\n   Testing different model variants...")

# Model 1: Optimized (continuity constraint)
try:
    popt1, _ = curve_fit(growth_then_decay_optimized, X_time, y, p0=p0, maxfev=20000,
                         bounds=([X_time[0], 0, 0, 0, y.min()],
                                [X_time[-1], y.max()*10, 1, 1, y.max()]))
    y_pred1 = growth_then_decay_optimized(X_time, *popt1)
    r2_1 = r2_score(y, y_pred1)
    rmse_1 = np.sqrt(mean_squared_error(y, y_pred1))
    print(f"     Model 1 (Optimized): R²={r2_1:.4f}, RMSE={rmse_1:,.0f}")
    all_model_attempts.append(('Model 1: Optimized', y_pred1, r2_1, rmse_1))
except Exception as e:
    print(f"     Model 1 failed: {e}")
    r2_1 = -np.inf
    popt1 = None

# Model 2: Flexible (with offset parameter)
try:
    p0_flex = p0 + [0]  # Add d parameter
    popt2, _ = curve_fit(growth_then_decay_flexible, X_time, y, p0=p0_flex, maxfev=20000,
                         bounds=([X_time[0], 0, 0, 0, y.min(), -y.max()],
                                [X_time[-1], y.max()*10, 1, 1, y.max(), y.max()]))
    y_pred2 = growth_then_decay_flexible(X_time, *popt2)
    r2_2 = r2_score(y, y_pred2)
    rmse_2 = np.sqrt(mean_squared_error(y, y_pred2))
    
    # Check residuals in decay phase for patterns
    residuals2 = y - y_pred2
    decay_mask = X_time >= popt2[0]
    decay_residuals = residuals2[decay_mask]
    print(f"     Model 2 (Flexible): R²={r2_2:.4f}, RMSE={rmse_2:,.0f}")
    print(f"       Decay phase residual std: {decay_residuals.std():,.0f}")
except Exception as e:
    print(f"     Model 2 failed: {e}")
    r2_2 = -np.inf
    popt2 = None

# Model 2b: Flexible with polynomial decay adjustment (to capture up-down pattern)
try:
    p0_poly = p0 + [0, 0, 0]  # Add d, p1, p2 parameters
    popt2b, _ = curve_fit(growth_then_decay_polynomial_decay, X_time, y, p0=p0_poly, maxfev=30000,
                          bounds=([X_time[0], 0, 0, 0, y.min(), -y.max(), -y.max()/10, -y.max()/100],
                                 [X_time[-1], y.max()*10, 1, 1, y.max(), y.max(), y.max()/10, y.max()/100]))
    y_pred2b = growth_then_decay_polynomial_decay(X_time, *popt2b)
    r2_2b = r2_score(y, y_pred2b)
    rmse_2b = np.sqrt(mean_squared_error(y, y_pred2b))
    
    residuals2b = y - y_pred2b
    decay_mask = X_time >= popt2b[0]
    decay_residuals = residuals2b[decay_mask]
    print(f"     Model 2b (Flexible + Polynomial Decay): R²={r2_2b:.4f}, RMSE={rmse_2b:,.0f}")
    print(f"       Decay phase residual std: {decay_residuals.std():,.0f}")
    all_model_attempts.append(('Model 2b: Flexible+Poly', y_pred2b, r2_2b, rmse_2b))
except Exception as e:
    print(f"     Model 2b failed: {e}")
    r2_2b = -np.inf
    popt2b = None

# Model 2c: Flexible with oscillatory component
try:
    # Estimate oscillation from decay phase residuals
    if popt2 is not None:
        temp_pred = growth_then_decay_flexible(X_time, *popt2)
        temp_resid = y - temp_pred
        decay_mask = X_time >= popt2[0]
        decay_resid = temp_resid[decay_mask]
        decay_x = X_time[decay_mask] - popt2[0]
        
        # Try to fit oscillation
        if len(decay_resid) > 20:
            # Estimate period from autocorrelation or use reasonable guess
            period_guess = 30  # ~monthly cycle
            omega_guess = 2 * np.pi / period_guess
            A_guess = np.std(decay_resid) * 0.5
            phi_guess = 0
            
            p0_osc = p0_flex + [A_guess, omega_guess, phi_guess]
            popt2c, _ = curve_fit(growth_then_decay_with_oscillation, X_time, y, p0=p0_osc, maxfev=30000,
                                  bounds=([X_time[0], 0, 0, 0, y.min(), -y.max(), 0, 0, -np.pi],
                                         [X_time[-1], y.max()*10, 1, 1, y.max(), y.max(), y.max()/2, 2*np.pi/7, np.pi]))
            y_pred2c = growth_then_decay_with_oscillation(X_time, *popt2c)
            r2_2c = r2_score(y, y_pred2c)
            rmse_2c = np.sqrt(mean_squared_error(y, y_pred2c))
            
            residuals2c = y - y_pred2c
            decay_residuals = residuals2c[decay_mask]
            print(f"     Model 2c (Flexible + Oscillation): R²={r2_2c:.4f}, RMSE={rmse_2c:,.0f}")
            print(f"       Decay phase residual std: {decay_residuals.std():,.0f}")
        else:
            r2_2c = -np.inf
            popt2c = None
    else:
        r2_2c = -np.inf
        popt2c = None
except Exception as e:
    print(f"     Model 2c failed: {e}")
    r2_2c = -np.inf
    popt2c = None

# Model 2d: Quadratic decay (more flexible for up-down pattern)
try:
    # Need to fit growth phase first to get y0 constraint
    if popt2 is not None:
        y0_at_break = growth_then_decay_flexible(np.array([popt2[0]]), *popt2)[0]
        p0_quad = [popt2[0], popt2[1], popt2[2], popt2[4], popt2[5], -10, -1, 0.1]
        # q1 is constrained by continuity, so we only fit q2, q3
        def growth_then_decay_quad_constrained(x, x0, a1, b1, c, d, q2, q3):
            result = np.zeros_like(x)
            mask_growth = x < x0
            mask_decay = x >= x0
            result[mask_growth] = a1 * np.exp(b1 * x[mask_growth]) + d
            y0 = a1 * np.exp(b1 * x0) + d
            x_decay = x[mask_decay] - x0
            result[mask_decay] = y0 + q2 * x_decay + q3 * x_decay**2
            return result
        
        p0_quad_const = [popt2[0], popt2[1], popt2[2], popt2[4], popt2[5], -10, 0.1]
        popt2d, _ = curve_fit(growth_then_decay_quad_constrained, X_time, y, p0=p0_quad_const, maxfev=30000,
                              bounds=([X_time[0], 0, 0, y.min(), -y.max(), -y.max(), -y.max()],
                                     [X_time[-1], y.max()*10, 1, y.max(), y.max(), y.max(), y.max()]))
        y_pred2d = growth_then_decay_quad_constrained(X_time, *popt2d)
        r2_2d = r2_score(y, y_pred2d)
        rmse_2d = np.sqrt(mean_squared_error(y, y_pred2d))
        residuals2d = y - y_pred2d
        decay_mask = X_time >= popt2d[0]
        decay_residuals = residuals2d[decay_mask]
        
        # Check residual autocorrelation in decay phase
        if len(decay_residuals) > 10:
            decay_resid_lag1 = decay_residuals[1:]
            decay_resid_today = decay_residuals[:-1]
            decay_resid_ac = np.corrcoef(decay_resid_today, decay_resid_lag1)[0,1] if len(decay_resid_today) > 1 else 0
        else:
            decay_resid_ac = 0
        
        print(f"     Model 2d (Quadratic Decay): R²={r2_2d:.4f}, RMSE={rmse_2d:,.0f}")
        print(f"       Decay phase residual std: {decay_residuals.std():,.0f}, AC: {abs(decay_resid_ac):.4f}")
        all_model_attempts.append(('Model 2d: Quadratic Decay', y_pred2d, r2_2d, rmse_2d))
        
        # Store function for later
        growth_then_decay_quad_constrained_global = growth_then_decay_quad_constrained
    else:
        r2_2d = -np.inf
        popt2d = None
except Exception as e:
    print(f"     Model 2d failed: {e}")
    r2_2d = -np.inf
    popt2d = None

# Model 2e: Cubic decay (very flexible)
try:
    if popt2 is not None:
        def growth_then_decay_cubic_constrained(x, x0, a1, b1, c, d, q1, q2, q3):
            result = np.zeros_like(x)
            mask_growth = x < x0
            mask_decay = x >= x0
            result[mask_growth] = a1 * np.exp(b1 * x[mask_growth]) + d
            y0 = a1 * np.exp(b1 * x0) + d
            x_decay = x[mask_decay] - x0
            result[mask_decay] = y0 + q1 * x_decay + q2 * x_decay**2 + q3 * x_decay**3
            return result
        
        p0_cubic = [popt2[0], popt2[1], popt2[2], popt2[4], popt2[5], -10, 0.1, -0.001]
        popt2e, _ = curve_fit(growth_then_decay_cubic_constrained, X_time, y, p0=p0_cubic, maxfev=30000,
                              bounds=([X_time[0], 0, 0, y.min(), -y.max(), -y.max(), -y.max(), -y.max()],
                                     [X_time[-1], y.max()*10, 1, y.max(), y.max(), y.max(), y.max(), y.max()]))
        y_pred2e = growth_then_decay_cubic_constrained(X_time, *popt2e)
        r2_2e = r2_score(y, y_pred2e)
        rmse_2e = np.sqrt(mean_squared_error(y, y_pred2e))
        residuals2e = y - y_pred2e
        decay_mask = X_time >= popt2e[0]
        decay_residuals = residuals2e[decay_mask]
        
        if len(decay_residuals) > 10:
            decay_resid_lag1 = decay_residuals[1:]
            decay_resid_today = decay_residuals[:-1]
            decay_resid_ac = np.corrcoef(decay_resid_today, decay_resid_lag1)[0,1] if len(decay_resid_today) > 1 else 0
        else:
            decay_resid_ac = 0
        
        print(f"     Model 2e (Cubic Decay): R²={r2_2e:.4f}, RMSE={rmse_2e:,.0f}")
        print(f"       Decay phase residual std: {decay_residuals.std():,.0f}, AC: {abs(decay_resid_ac):.4f}")
        all_model_attempts.append(('Model 2e: Cubic Decay', y_pred2e, r2_2e, rmse_2e))
        
        growth_then_decay_cubic_constrained_global = growth_then_decay_cubic_constrained
    else:
        r2_2e = -np.inf
        popt2e = None
except Exception as e:
    print(f"     Model 2e failed: {e}")
    r2_2e = -np.inf
    popt2e = None

# Model 2f: Combined oscillation + polynomial (most flexible)
try:
    if popt2c is not None:
        # Add polynomial term to oscillation model
        def growth_then_decay_combined(x, x0, a1, b1, b2, c, d, A, omega, phi, p1):
            result = np.zeros_like(x)
            mask_growth = x < x0
            mask_decay = x >= x0
            result[mask_growth] = a1 * np.exp(b1 * x[mask_growth]) + d
            y0 = a1 * np.exp(b1 * x0) + d
            x_decay = x[mask_decay] - x0
            decay_base = (y0 - c) * np.exp(-b2 * x_decay) + c
            oscillation = A * np.sin(omega * x_decay + phi)
            polynomial = p1 * x_decay
            result[mask_decay] = decay_base + oscillation + polynomial
            return result
        
        p0_comb = list(popt2c) + [0]  # Add polynomial term
        popt2f, _ = curve_fit(growth_then_decay_combined, X_time, y, p0=p0_comb, maxfev=40000,
                              bounds=([X_time[0], 0, 0, 0, y.min(), -y.max(), 0, 0, -np.pi, -y.max()/10],
                                     [X_time[-1], y.max()*10, 1, 1, y.max(), y.max(), y.max()/2, 2*np.pi/7, np.pi, y.max()/10]))
        y_pred2f = growth_then_decay_combined(X_time, *popt2f)
        r2_2f = r2_score(y, y_pred2f)
        rmse_2f = np.sqrt(mean_squared_error(y, y_pred2f))
        residuals2f = y - y_pred2f
        decay_mask = X_time >= popt2f[0]
        decay_residuals = residuals2f[decay_mask]
        
        if len(decay_residuals) > 10:
            decay_resid_lag1 = decay_residuals[1:]
            decay_resid_today = decay_residuals[:-1]
            decay_resid_ac = np.corrcoef(decay_resid_today, decay_resid_lag1)[0,1] if len(decay_resid_today) > 1 else 0
        else:
            decay_resid_ac = 0
        
        print(f"     Model 2f (Oscillation + Polynomial): R²={r2_2f:.4f}, RMSE={rmse_2f:,.0f}")
        print(f"       Decay phase residual std: {decay_residuals.std():,.0f}, AC: {abs(decay_resid_ac):.4f}")
        all_model_attempts.append(('Model 2f: Osc+Poly', y_pred2f, r2_2f, rmse_2f))
        
        growth_then_decay_combined_global = growth_then_decay_combined
    else:
        r2_2f = -np.inf
        popt2f = None
except Exception as e:
    print(f"     Model 2f failed: {e}")
    r2_2f = -np.inf
    popt2f = None

# Initialize additional model variables (will be set if models succeed)
popt2g = None
r2_2g = -np.inf
rmse_2g = np.inf
decay_resid_std_2g = np.inf
growth_then_decay_combined_optimized_global = None

popt2h = None
r2_2h = -np.inf
rmse_2h = np.inf
decay_resid_std_2h = np.inf
growth_then_decay_multi_osc_global = None

# Model 3: Weighted fit (more weight on start to improve initial fit)
try:
    # Weight function: higher weight on early days
    weights = np.exp(-0.01 * X_time)  # Decay weight over time
    weights = weights / weights.max()  # Normalize
    
    # Use weighted least squares approach
    def weighted_residuals(params):
        pred = growth_then_decay_weighted(X_time, *params)
        return np.sqrt(weights) * (y - pred)
    
    from scipy.optimize import least_squares
    result = least_squares(weighted_residuals, p0, bounds=([X_time[0], 0, 0, 0, y.min()],
                                                            [X_time[-1], y.max()*10, 1, 1, y.max()]),
                          max_nfev=20000)
    popt3 = result.x
    y_pred3 = growth_then_decay_weighted(X_time, *popt3)
    r2_3 = r2_score(y, y_pred3)
    rmse_3 = np.sqrt(mean_squared_error(y, y_pred3))
    print(f"     Model 3 (Weighted): R²={r2_3:.4f}, RMSE={rmse_3:,.0f}")
    all_model_attempts.append(('Model 3: Weighted', y_pred3, r2_3, rmse_3))
except Exception as e:
    print(f"     Model 3 failed: {e}")
    r2_3 = -np.inf
    popt3 = None

# Select best model (prioritize low residual patterns in decay phase)
models = []
if popt1 is not None:
    temp_resid = y - growth_then_decay_optimized(X_time, *popt1)
    if len(temp_resid) > 10:
        resid_lag1_all = temp_resid[1:]
        resid_today_all = temp_resid[:-1]
        resid_ac_all = abs(np.corrcoef(resid_today_all, resid_lag1_all)[0,1] if len(resid_today_all) > 1 else 1.0)
    else:
        resid_ac_all = 1.0
    decay_mask = X_time >= popt1[0]
    decay_resid = temp_resid[decay_mask]
    decay_resid_std = decay_resid.std()
    if len(decay_resid) > 10:
        decay_resid_lag1 = decay_resid[1:]
        decay_resid_today = decay_resid[:-1]
        decay_resid_ac = abs(np.corrcoef(decay_resid_today, decay_resid_lag1)[0,1] if len(decay_resid_today) > 1 else 1.0)
    else:
        decay_resid_ac = 1.0
    models.append(('optimized', popt1, r2_1, rmse_1, growth_then_decay_optimized, decay_resid_std, decay_resid_ac, resid_ac_all))
if popt2 is not None:
    temp_resid = y - growth_then_decay_flexible(X_time, *popt2)
    if len(temp_resid) > 10:
        resid_lag1_all = temp_resid[1:]
        resid_today_all = temp_resid[:-1]
        resid_ac_all = abs(np.corrcoef(resid_today_all, resid_lag1_all)[0,1] if len(resid_today_all) > 1 else 1.0)
    else:
        resid_ac_all = 1.0
    decay_mask = X_time >= popt2[0]
    decay_resid = temp_resid[decay_mask]
    decay_resid_std = decay_resid.std()
    if len(decay_resid) > 10:
        decay_resid_lag1 = decay_resid[1:]
        decay_resid_today = decay_resid[:-1]
        decay_resid_ac = abs(np.corrcoef(decay_resid_today, decay_resid_lag1)[0,1] if len(decay_resid_today) > 1 else 1.0)
    else:
        decay_resid_ac = 1.0
    models.append(('flexible', popt2, r2_2, rmse_2, growth_then_decay_flexible, decay_resid_std, decay_resid_ac, resid_ac_all))
if popt2b is not None:
    temp_resid = y - growth_then_decay_polynomial_decay(X_time, *popt2b)
    if len(temp_resid) > 10:
        resid_lag1_all = temp_resid[1:]
        resid_today_all = temp_resid[:-1]
        resid_ac_all = abs(np.corrcoef(resid_today_all, resid_lag1_all)[0,1] if len(resid_today_all) > 1 else 1.0)
    else:
        resid_ac_all = 1.0
    decay_mask = X_time >= popt2b[0]
    decay_resid = temp_resid[decay_mask]
    decay_resid_std = decay_resid.std()
    if len(decay_resid) > 10:
        decay_resid_lag1 = decay_resid[1:]
        decay_resid_today = decay_resid[:-1]
        decay_resid_ac = abs(np.corrcoef(decay_resid_today, decay_resid_lag1)[0,1] if len(decay_resid_today) > 1 else 1.0)
    else:
        decay_resid_ac = 1.0
    models.append(('flexible_poly', popt2b, r2_2b, rmse_2b, growth_then_decay_polynomial_decay, decay_resid_std, decay_resid_ac, resid_ac_all))
if popt2c is not None:
    temp_resid = y - growth_then_decay_with_oscillation(X_time, *popt2c)
    if len(temp_resid) > 10:
        resid_lag1_all = temp_resid[1:]
        resid_today_all = temp_resid[:-1]
        resid_ac_all = abs(np.corrcoef(resid_today_all, resid_lag1_all)[0,1] if len(resid_today_all) > 1 else 1.0)
    else:
        resid_ac_all = 1.0
    decay_mask = X_time >= popt2c[0]
    decay_resid = temp_resid[decay_mask]
    decay_resid_std = decay_resid.std()
    if len(decay_resid) > 10:
        decay_resid_lag1 = decay_resid[1:]
        decay_resid_today = decay_resid[:-1]
        decay_resid_ac = abs(np.corrcoef(decay_resid_today, decay_resid_lag1)[0,1] if len(decay_resid_today) > 1 else 1.0)
    else:
        decay_resid_ac = 1.0
    models.append(('flexible_osc', popt2c, r2_2c, rmse_2c, growth_then_decay_with_oscillation, decay_resid_std, decay_resid_ac, resid_ac_all))
if popt2d is not None:
    temp_resid = y - growth_then_decay_quad_constrained_global(X_time, *popt2d)
    if len(temp_resid) > 10:
        resid_lag1_all = temp_resid[1:]
        resid_today_all = temp_resid[:-1]
        resid_ac_all = abs(np.corrcoef(resid_today_all, resid_lag1_all)[0,1] if len(resid_today_all) > 1 else 1.0)
    else:
        resid_ac_all = 1.0
    decay_mask = X_time >= popt2d[0]
    decay_resid = temp_resid[decay_mask]
    decay_resid_std = decay_resid.std()
    if len(decay_resid) > 10:
        decay_resid_lag1 = decay_resid[1:]
        decay_resid_today = decay_resid[:-1]
        decay_resid_ac = abs(np.corrcoef(decay_resid_today, decay_resid_lag1)[0,1] if len(decay_resid_today) > 1 else 1.0)
    else:
        decay_resid_ac = 1.0
    models.append(('quadratic_decay', popt2d, r2_2d, rmse_2d, growth_then_decay_quad_constrained_global, decay_resid_std, decay_resid_ac, resid_ac_all))
if popt2e is not None:
    temp_resid = y - growth_then_decay_cubic_constrained_global(X_time, *popt2e)
    if len(temp_resid) > 10:
        resid_lag1_all = temp_resid[1:]
        resid_today_all = temp_resid[:-1]
        resid_ac_all = abs(np.corrcoef(resid_today_all, resid_lag1_all)[0,1] if len(resid_today_all) > 1 else 1.0)
    else:
        resid_ac_all = 1.0
    decay_mask = X_time >= popt2e[0]
    decay_resid = temp_resid[decay_mask]
    decay_resid_std = decay_resid.std()
    if len(decay_resid) > 10:
        decay_resid_lag1 = decay_resid[1:]
        decay_resid_today = decay_resid[:-1]
        decay_resid_ac = abs(np.corrcoef(decay_resid_today, decay_resid_lag1)[0,1] if len(decay_resid_today) > 1 else 1.0)
    else:
        decay_resid_ac = 1.0
    models.append(('cubic_decay', popt2e, r2_2e, rmse_2e, growth_then_decay_cubic_constrained_global, decay_resid_std, decay_resid_ac, resid_ac_all))
if popt2f is not None:
    temp_resid = y - growth_then_decay_combined_global(X_time, *popt2f)
    if len(temp_resid) > 10:
        resid_lag1_all = temp_resid[1:]
        resid_today_all = temp_resid[:-1]
        resid_ac_all = abs(np.corrcoef(resid_today_all, resid_lag1_all)[0,1] if len(resid_today_all) > 1 else 1.0)
    else:
        resid_ac_all = 1.0
    decay_mask = X_time >= popt2f[0]
    decay_resid = temp_resid[decay_mask]
    decay_resid_std = decay_resid.std()
    if len(decay_resid) > 10:
        decay_resid_lag1 = decay_resid[1:]
        decay_resid_today = decay_resid[:-1]
        decay_resid_ac = abs(np.corrcoef(decay_resid_today, decay_resid_lag1)[0,1] if len(decay_resid_today) > 1 else 1.0)
    else:
        decay_resid_ac = 1.0
    models.append(('combined_osc_poly', popt2f, r2_2f, rmse_2f, growth_then_decay_combined_global, decay_resid_std, decay_resid_ac, resid_ac_all))
if popt2g is not None:
    temp_resid = y - growth_then_decay_combined_optimized_global(X_time, *popt2g)
    if len(temp_resid) > 10:
        resid_lag1_all = temp_resid[1:]
        resid_today_all = temp_resid[:-1]
        resid_ac_all = abs(np.corrcoef(resid_today_all, resid_lag1_all)[0,1] if len(resid_today_all) > 1 else 1.0)
    else:
        resid_ac_all = 1.0
    decay_mask = X_time >= popt2g[0]
    decay_resid = temp_resid[decay_mask]
    if len(decay_resid) > 10:
        decay_resid_lag1 = decay_resid[1:]
        decay_resid_today = decay_resid[:-1]
        decay_resid_ac = abs(np.corrcoef(decay_resid_today, decay_resid_lag1)[0,1] if len(decay_resid_today) > 1 else 1.0)
    else:
        decay_resid_ac = 1.0
    models.append(('pattern_optimized', popt2g, r2_2g, rmse_2g, growth_then_decay_combined_optimized_global, decay_resid_std_2g, decay_resid_ac, resid_ac_all))
if popt2h is not None:
    temp_resid = y - growth_then_decay_multi_osc_global(X_time, *popt2h)
    if len(temp_resid) > 10:
        resid_lag1_all = temp_resid[1:]
        resid_today_all = temp_resid[:-1]
        resid_ac_all = abs(np.corrcoef(resid_today_all, resid_lag1_all)[0,1] if len(resid_today_all) > 1 else 1.0)
    else:
        resid_ac_all = 1.0
    decay_mask = X_time >= popt2h[0]
    decay_resid = temp_resid[decay_mask]
    if len(decay_resid) > 10:
        decay_resid_lag1 = decay_resid[1:]
        decay_resid_today = decay_resid[:-1]
        decay_resid_ac = abs(np.corrcoef(decay_resid_today, decay_resid_lag1)[0,1] if len(decay_resid_today) > 1 else 1.0)
    else:
        decay_resid_ac = 1.0
    models.append(('multi_osc', popt2h, r2_2h, rmse_2h, growth_then_decay_multi_osc_global, decay_resid_std_2h, decay_resid_ac, resid_ac_all))
if popt3 is not None:
    temp_resid = y - growth_then_decay_weighted(X_time, *popt3)
    if len(temp_resid) > 10:
        resid_lag1_all = temp_resid[1:]
        resid_today_all = temp_resid[:-1]
        resid_ac_all = abs(np.corrcoef(resid_today_all, resid_lag1_all)[0,1] if len(resid_today_all) > 1 else 1.0)
    else:
        resid_ac_all = 1.0
    decay_mask = X_time >= popt3[0]
    decay_resid = temp_resid[decay_mask]
    decay_resid_std = decay_resid.std()
    if len(decay_resid) > 10:
        decay_resid_lag1 = decay_resid[1:]
        decay_resid_today = decay_resid[:-1]
        decay_resid_ac = abs(np.corrcoef(decay_resid_today, decay_resid_lag1)[0,1] if len(decay_resid_today) > 1 else 1.0)
    else:
        decay_resid_ac = 1.0
    models.append(('weighted', popt3, r2_3, rmse_3, growth_then_decay_weighted, decay_resid_std, decay_resid_ac, resid_ac_all))

if len(models) == 0:
    # Fallback to basic model
    popt_base = p0
    model_func = growth_then_decay_optimized
    print(f"     Using fallback model")
else:
    # Select by combination of R² and low residual patterns
    # Prioritize: low residual autocorrelation (overall) > low residual std > high R²
    for i, model in enumerate(models):
        # Get overall residual autocorrelation (prefer models with 8 elements)
        if len(model) >= 8:
            overall_ac = model[7]
        else:
            # Calculate it if missing
            temp_pred = model[4](X_time, *model[1])
            temp_resid = y - temp_pred
            if len(temp_resid) > 10:
                resid_lag1 = temp_resid[1:]
                resid_today = temp_resid[:-1]
                overall_ac = abs(np.corrcoef(resid_today, resid_lag1)[0,1] if len(resid_today) > 1 else 1.0)
            else:
                overall_ac = 1.0
        
        # Score: prioritize low autocorrelation VERY heavily, then low std, then R²
        ac_penalty = overall_ac * 0.5  # Heavily penalize overall autocorrelation
        decay_ac_penalty = model[6] * 0.3  # Also penalize decay phase autocorrelation
        std_penalty = (model[5] / y.mean()) * 0.05  # Smaller penalty for std
        score = model[2] - ac_penalty - decay_ac_penalty - std_penalty
        
        if len(model) < 8:
            model = list(model) + [overall_ac]
        models[i] = list(model) + [score]
    
    best_model = max(models, key=lambda x: x[8] if len(x) > 8 else x[2])
    model_name = best_model[0]
    popt_base = best_model[1]
    r2_best = best_model[2]
    rmse_best = best_model[3]
    model_func = best_model[4]
    decay_std = best_model[5]
    decay_ac = best_model[6]
    overall_ac = best_model[7] if len(best_model) > 7 else 1.0
    
    print(f"\n   Selected: {model_name} model")
    print(f"     R² = {r2_best:.4f}, RMSE = {rmse_best:,.0f}")
    print(f"     Overall residual autocorrelation: {overall_ac:.4f}")
    print(f"     Decay phase: Residual std = {decay_std:,.0f}, AC = {decay_ac:.4f}")

df['trend_base'] = model_func(X_time, *popt_base)
df['residuals_base'] = y - df['trend_base']

r2_base = r2_score(y, df['trend_base'])
rmse_base = np.sqrt(mean_squared_error(y, df['trend_base']))
mae_base = mean_absolute_error(y, df['trend_base'])

print(f"   Base Model (Growth-Then-Decay):")
print(f"     Breakpoint (peak): Day {popt_base[0]:.1f}")
print(f"     R² = {r2_base:.4f}")
print(f"     RMSE = {rmse_base:,.0f}")
print(f"     MAE = {mae_base:,.0f}")

# Create plot of all model attempts
if len(all_model_attempts) > 0:
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    
    # Top: All model fits
    ax = axes[0]
    ax.scatter(df['Date'], y, s=40, alpha=0.7, color='black', label='Observed Data', zorder=10)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_model_attempts)))
    for i, (name, pred, r2, rmse) in enumerate(all_model_attempts):
        ax.plot(df['Date'], pred, linewidth=2, alpha=0.8, color=colors[i], 
               label=f'{name} (R²={r2:.4f}, RMSE={rmse:,.0f})', zorder=5)
    
    # Highlight selected model
    ax.plot(df['Date'], df['trend_base'], linewidth=3.5, alpha=1.0, color='red', 
           linestyle='--', label=f'SELECTED: {model_name} (R²={r2_base:.4f})', zorder=8)
    
    ax.set_xlabel('Date', fontsize=13, fontweight='bold')
    ax.set_ylabel('Number of Reported Results', fontsize=13, fontweight='bold')
    ax.set_title('All Model Attempts - Fits Comparison', fontsize=15, fontweight='bold', pad=15)
    ax.legend(loc='upper left', framealpha=0.95, fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b %Y'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Bottom: Residuals comparison
    ax = axes[1]
    for i, (name, pred, r2, rmse) in enumerate(all_model_attempts):
        residuals = y - pred
        ax.plot(df['Date'], residuals, linewidth=1.5, alpha=0.7, color=colors[i], 
               label=f'{name}', zorder=5)
    
    # Highlight selected model residuals
    ax.plot(df['Date'], df['residuals_base'], linewidth=3, alpha=1.0, color='red', 
           linestyle='--', label=f'SELECTED: {model_name}', zorder=8)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.5, zorder=1)
    ax.set_xlabel('Date', fontsize=13, fontweight='bold')
    ax.set_ylabel('Residuals', fontsize=13, fontweight='bold')
    ax.set_title('All Model Attempts - Residuals Comparison', fontsize=15, fontweight='bold', pad=15)
    ax.legend(loc='upper left', framealpha=0.95, fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b %Y'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(f'{paper_plots_dir}/00_all_model_attempts.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{paper_plots_dir}/00_all_model_attempts.pdf', bbox_inches='tight')
    plt.close()
    print(f"   Saved: {paper_plots_dir}/00_all_model_attempts.png/pdf")

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
trend_extended = model_func(X_extended, *popt_base)

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
        'type': 'Exponential Growth-Then-Decay (Optimized)',
        'breakpoint_day': float(popt_base[0]),
        'parameters': {
            'x0': float(popt_base[0]),
            'a1': float(popt_base[1]),
            'b1': float(popt_base[2]),
            'b2': float(popt_base[3]),
            'c': float(popt_base[4]),
            **({'d': float(popt_base[5])} if len(popt_base) > 5 else {})
        },
        'model_type': 'flexible_osc' if len(popt_base) > 7 else ('flexible' if len(popt_base) > 5 else 'optimized'),
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

