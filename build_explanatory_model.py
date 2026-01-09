"""
Explanatory Model for Wordle Results Variation
==============================================
Goal: EXPLAIN the variation, not just forecast
- What factors cause daily variation?
- Day of week, time factors, word attributes
- Make it interpretable and explanatory
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)

print("=" * 80)
print("BUILDING EXPLANATORY MODEL FOR WORDLE VARIATION")
print("=" * 80)

# Load data
df = pd.read_csv('data/2023_MCM_Problem_C_Data.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

target_var = 'Number of  reported results'
df['days_since_start'] = (df['Date'] - df['Date'].min()).dt.days

print(f"\nData: {len(df)} observations")
print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
print(f"Target: {target_var}")

# ============================================================================
# 1. EXTRACT EXPLANATORY VARIABLES
# ============================================================================
print("\n1. Extracting explanatory variables...")

# Time-based factors (found to be significant)
df['day_of_week'] = df['Date'].dt.dayofweek  # 0=Monday, 6=Sunday
df['day_name'] = df['Date'].dt.day_name()
df['is_weekend'] = df['day_of_week'].isin([5, 6])
df['month'] = df['Date'].dt.month
df['month_name'] = df['Date'].dt.month_name()
df['day_of_month'] = df['Date'].dt.day

# Season
def get_season(date):
    month = date.month
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'
df['season'] = df['Date'].apply(get_season)

# Word attributes (potential explanatory factors)
df['word_length'] = df['Word'].str.len()
df['has_repeated_letters'] = df['Word'].apply(lambda w: len(set(w)) < len(w))

# Difficulty indicators (from distribution of tries)
df['pct_1_try'] = df['1 try'] / df[target_var] * 100
df['pct_6_or_more'] = (df['6 tries'] + df['7 or more tries (X)']) / df[target_var] * 100
df['avg_tries'] = (
    df['1 try'] * 1 +
    df['2 tries'] * 2 +
    df['3 tries'] * 3 +
    df['4 tries'] * 4 +
    df['5 tries'] * 5 +
    df['6 tries'] * 6 +
    df['7 or more tries (X)'] * 6.5  # approximate
) / df[target_var]

# Hard mode percentage
df['hard_mode_pct'] = df['Number in hard mode'] / df[target_var] * 100

# Days since peak popularity (capture decay phase)
peak_day = df.loc[df[target_var].idxmax(), 'days_since_start']
df['days_since_peak'] = df['days_since_start'] - peak_day
df['is_after_peak'] = df['days_since_peak'] > 0

print("   Extracted variables:")
print("     - Time: day_of_week, month, season, day_of_month")
print("     - Word: word_length, has_repeated_letters, avg_tries, pct_1_try")
print("     - Hard mode: hard_mode_pct")

# ============================================================================
# 2. FIT BASE TREND (Growth-Then-Decay) - Overall Popularity Lifecycle
# ============================================================================
print("\n2. Fitting base trend (Growth-Then-Decay)...")
print("   This captures the overall 'fad lifecycle' - not explanatory but necessary baseline")

X_time = df['days_since_start'].values
y = df[target_var].values

def growth_then_decay(x, x0, a1, b1, a2, b2, c):
    """Growth phase then decay phase - captures overall popularity trend"""
    result = np.zeros_like(x)
    mask_growth = x < x0
    mask_decay = x >= x0
    result[mask_growth] = a1 * np.exp(b1 * x[mask_growth])
    result[mask_decay] = a2 * np.exp(-b2 * (x[mask_decay] - x0)) + c
    return result

# Fit with initial guess at peak
peak_idx = np.argmax(y)
x0_guess = X_time[peak_idx]
p0 = [x0_guess, y[0], 0.01, y.max(), 0.001, y.min()]

popt, _ = curve_fit(growth_then_decay, X_time, y, p0=p0, maxfev=10000,
                   bounds=([X_time[0], 0, 0, 0, 0, y.min()],
                          [X_time[-1], y.max()*10, 1, y.max()*10, 1, y.max()]))

df['trend_base'] = growth_then_decay(X_time, *popt)
df['residuals_base'] = y - df['trend_base']

r2_trend = r2_score(y, df['trend_base'])
print(f"   Base trend R² = {r2_trend:.4f}")
print(f"   Breakpoint at day {popt[0]:.0f} (peak popularity)")

# ============================================================================
# 3. EXPLAIN RESIDUALS WITH FACTORS
# ============================================================================
print("\n3. Building explanatory model for residuals...")
print("   Goal: Explain why actual differs from trend")

# Prepare features for regression
features = []

# Day of week (categorical - use dummy variables or factors)
day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
for i, day in enumerate(day_names):
    df[f'is_{day[:3].lower()}'] = (df['day_of_week'] == i).astype(int)
    if i > 0:  # Use Monday as reference
        features.append(f'is_{day[:3].lower()}')

# Month (categorical)
for month in range(2, 13):  # Use January as reference
    df[f'is_month_{month}'] = (df['month'] == month).astype(int)
    features.append(f'is_month_{month}')

# Time factors (continuous)
features.extend(['days_since_peak', 'day_of_month'])

# Word attributes
features.extend(['word_length', 'has_repeated_letters', 'avg_tries', 'pct_1_try'])

# Hard mode percentage (might affect engagement)
features.append('hard_mode_pct')

print(f"   Features ({len(features)}): {features[:10]}...")

# Build regression model on residuals
X_features = df[features].values
y_residuals = df['residuals_base'].values

# Fit model
model = LinearRegression()
model.fit(X_features, y_residuals)

# Predictions
df['residuals_predicted'] = model.predict(X_features)
df['prediction_full'] = df['trend_base'] + df['residuals_predicted']
df['residuals_final'] = y - df['prediction_full']

# Model performance
r2_residuals = r2_score(y_residuals, df['residuals_predicted'])
r2_full = r2_score(y, df['prediction_full'])

print(f"\n   Model Performance:")
print(f"     Base trend R² = {r2_trend:.4f}")
print(f"     Residuals explained R² = {r2_residuals:.4f}")
print(f"     Full model R² = {r2_full:.4f}")
print(f"     Improvement from adding factors: {r2_full - r2_trend:.4f}")

# ============================================================================
# 4. ANALYZE FEATURE IMPORTANCE (What explains variation?)
# ============================================================================
print("\n4. Analyzing what explains the variation...")

# Feature coefficients
feature_importance = pd.DataFrame({
    'feature': features,
    'coefficient': model.coef_,
    'abs_coefficient': np.abs(model.coef_)
}).sort_values('abs_coefficient', ascending=False)

print("\n   Top explanatory factors (by coefficient magnitude):")
print(feature_importance.head(15).to_string(index=False))

# Calculate contribution of each factor type
day_features = [f for f in features if f.startswith('is_') and any(d in f for d in ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun'])]
month_features = [f for f in features if f.startswith('is_month_')]
time_features = ['days_since_peak', 'day_of_month']
word_features = ['word_length', 'has_repeated_letters', 'avg_tries', 'pct_1_try', 'hard_mode_pct']

day_importance = feature_importance[feature_importance['feature'].isin(day_features)]['abs_coefficient'].sum()
month_importance = feature_importance[feature_importance['feature'].isin(month_features)]['abs_coefficient'].sum()
time_importance = feature_importance[feature_importance['feature'].isin(time_features)]['abs_coefficient'].sum()
word_importance = feature_importance[feature_importance['feature'].isin(word_features)]['abs_coefficient'].sum()

print(f"\n   Contribution by factor type:")
print(f"     Day of week effects: {day_importance:.0f}")
print(f"     Month effects: {month_importance:.0f}")
print(f"     Time effects: {time_importance:.0f}")
print(f"     Word attributes: {word_importance:.0f}")

# ============================================================================
# 5. PREDICT MARCH 1, 2023
# ============================================================================
print("\n5. Creating prediction for March 1, 2023...")

march_1 = pd.to_datetime('2023-03-01')
days_to_march_1 = (march_1 - df['Date'].min()).days

# Base trend prediction
trend_pred = growth_then_decay(np.array([days_to_march_1]), *popt)[0]

# Factor predictions (we need to know what day of week March 1 is)
march_1_dow = march_1.weekday()  # 0=Monday
march_1_month = 3  # March

# Create feature vector for March 1
march_1_features = np.zeros(len(features))
for i, feat in enumerate(features):
    if feat.startswith('is_') and any(d in feat for d in ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']):
        day_idx = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun'].index(feat.split('_')[1][:3])
        march_1_features[i] = 1 if day_idx == march_1_dow else 0
    elif feat.startswith('is_month_'):
        month_num = int(feat.split('_')[2])
        march_1_features[i] = 1 if month_num == march_1_month else 0
    elif feat == 'days_since_peak':
        march_1_features[i] = days_to_march_1 - popt[0]
    elif feat == 'day_of_month':
        march_1_features[i] = 1
    # For word attributes, use mean values (we don't know the word yet)
    elif feat in word_features:
        march_1_features[i] = df[feat].mean()

# Residual prediction
residual_pred = model.predict(march_1_features.reshape(1, -1))[0]

# Full prediction
march_1_prediction = trend_pred + residual_pred

print(f"\n   Prediction for March 1, 2023 (Wednesday):")
print(f"     Base trend: {trend_pred:,.0f}")
print(f"     Factor adjustment: {residual_pred:+,.0f}")
print(f"     Total prediction: {march_1_prediction:,.0f}")

# Prediction interval (using residuals standard error)
residual_std = np.std(df['residuals_final'])
prediction_interval_95 = 1.96 * residual_std  # 95% confidence interval

print(f"\n   Prediction Interval (95%):")
print(f"     Lower: {march_1_prediction - prediction_interval_95:,.0f}")
print(f"     Upper: {march_1_prediction + prediction_interval_95:,.0f}")
print(f"     Standard Error: ±{residual_std:,.0f}")

# ============================================================================
# 6. ANALYZE HARD MODE PERCENTAGE
# ============================================================================
print("\n6. Analyzing Hard Mode Percentage...")
print("   Question: Do word attributes affect hard mode percentage?")

# Test correlations between word attributes and hard mode percentage
from scipy.stats import pearsonr

print("\n   Correlations with Hard Mode Percentage:")
correlations = []

# Word attributes
for attr in ['word_length', 'avg_tries', 'pct_1_try', 'pct_6_or_more', 'has_repeated_letters']:
    corr, p_val = pearsonr(df[attr], df['hard_mode_pct'])
    correlations.append({
        'Attribute': attr,
        'Correlation': corr,
        'P-value': p_val,
        'Significant': p_val < 0.05
    })
    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
    print(f"     {attr:25s}: r={corr:7.4f}, p={p_val:.4f} {sig}")

# Day of week effect on hard mode
print("\n   Hard Mode Percentage by Day of Week:")
hard_mode_by_dow = df.groupby('day_name')['hard_mode_pct'].agg(['mean', 'std']).reindex(day_names)
print(hard_mode_by_dow.round(2))

from scipy.stats import f_oneway
dow_groups = [df[df['day_name'] == day]['hard_mode_pct'].values for day in day_names]
f_stat, p_val = f_oneway(*dow_groups)
print(f"\n   ANOVA (day of week): F={f_stat:.2f}, p={p_val:.4f} {'***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'}")

# Build model for hard mode percentage
print("\n   Building explanatory model for Hard Mode Percentage:")
hard_mode_features = ['word_length', 'avg_tries', 'pct_1_try', 'has_repeated_letters']
for i, day in enumerate(day_names[1:], 1):  # Skip Monday as reference
    df[f'hm_dow_{day[:3].lower()}'] = (df['day_of_week'] == i).astype(int)
    hard_mode_features.append(f'hm_dow_{day[:3].lower()}')

X_hm = df[hard_mode_features].values
y_hm = df['hard_mode_pct'].values

model_hm = LinearRegression()
model_hm.fit(X_hm, y_hm)

r2_hm = r2_score(y_hm, model_hm.predict(X_hm))
print(f"     R² = {r2_hm:.4f}")

hm_feature_importance = pd.DataFrame({
    'feature': hard_mode_features,
    'coefficient': model_hm.coef_,
    'abs_coefficient': np.abs(model_hm.coef_)
}).sort_values('abs_coefficient', ascending=False)

print("\n     Feature importance:")
print(hm_feature_importance.to_string(index=False))

# ============================================================================
# 7. VISUALIZATIONS
# ============================================================================
print("\n7. Creating visualizations...")
import os
os.makedirs('plots', exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Model fit
ax = axes[0, 0]
ax.scatter(y, df['prediction_full'], alpha=0.5, s=20)
min_val = min(y.min(), df['prediction_full'].min())
max_val = max(y.max(), df['prediction_full'].max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
ax.set_title(f'Explanatory Model Fit\nR² = {r2_full:.4f}', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Feature importance
ax = axes[0, 1]
top_features = feature_importance.head(10)
colors = ['steelblue' if 'day' in f.lower() or 'month' in f.lower() else 
          'coral' if 'word' in f.lower() else 'green' 
          for f in top_features['feature']]
ax.barh(range(len(top_features)), top_features['abs_coefficient'], color=colors)
ax.set_yticks(range(len(top_features)))
ax.set_yticklabels(top_features['feature'], fontsize=9)
ax.set_xlabel('|Coefficient| (Importance)')
ax.set_title('Top 10 Explanatory Factors', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

# Hard mode by word attributes
ax = axes[1, 0]
if abs(pearsonr(df['avg_tries'], df['hard_mode_pct'])[0]) > 0.3:
    ax.scatter(df['avg_tries'], df['hard_mode_pct'], alpha=0.5, s=20)
    ax.set_xlabel('Average Tries (Difficulty)')
    ax.set_ylabel('Hard Mode %')
    ax.set_title('Hard Mode % vs Word Difficulty', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

# Residuals over time
ax = axes[1, 1]
ax.plot(df['Date'], df['residuals_base'], 'b-', alpha=0.5, linewidth=1, label='Residuals from Trend')
ax.plot(df['Date'], df['residuals_final'], 'r-', alpha=0.7, linewidth=1, label='Final Residuals')
ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax.set_xlabel('Date')
ax.set_ylabel('Residual')
ax.set_title('Residuals: Before vs After Adding Explanatory Factors', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/26_explanatory_model.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: plots/26_explanatory_model.png")

print("\n" + "=" * 80)
print("EXPLANATORY MODEL SUMMARY")
print("=" * 80)
print(f"\nModel Structure:")
print(f"   Prediction = Base Trend (Growth-Then-Decay) + Factor Adjustments")
print(f"\nBase Trend: Captures overall popularity lifecycle")
print(f"  - Growth phase: Days 0-{int(popt[0])}")
print(f"  - Decay phase: Days {int(popt[0])}+")
print(f"  - R² = {r2_trend:.4f}")

print(f"\nExplanatory Factors: {len(features)} factors explain daily variation")
print(f"  - Day of week effects (weekly patterns)")
print(f"  - Month effects (seasonal patterns)")
print(f"  - Time effects (days since peak, day of month)")
print(f"  - Word attributes (difficulty, length, etc.)")
print(f"  - Residual R² = {r2_residuals:.4f}")

print(f"\nMarch 1, 2023 Prediction:")
print(f"  Point estimate: {march_1_prediction:,.0f}")
print(f"  95% Interval: [{march_1_prediction - prediction_interval_95:,.0f}, {march_1_prediction + prediction_interval_95:,.0f}]")

print(f"\nHard Mode Analysis:")
print(f"  Word attributes explain {r2_hm:.2%} of hard mode percentage variation")
print("=" * 80)

