"""
Exploratory Data Analysis: Time Variables and Wordle Results
============================================================
This script explores how time-related variables affect the number of reported Wordle results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from scipy.ndimage import gaussian_filter1d
from scipy import stats
from scipy.stats import ttest_ind, f_oneway, mannwhitneyu
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Load data
print("Loading data...")
df = pd.read_csv('data/2023_MCM_Problem_C_Data.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

print(f"Data loaded: {len(df)} rows from {df['Date'].min().date()} to {df['Date'].max().date()}\n")

# ============================================================================
# 1. EXTRACT TIME-RELATED FEATURES
# ============================================================================
print("=" * 80)
print("EXTRACTING TIME-RELATED FEATURES")
print("=" * 80)

# Basic date features
df['year'] = df['Date'].dt.year
df['month'] = df['Date'].dt.month
df['day'] = df['Date'].dt.day
df['day_of_week'] = df['Date'].dt.dayofweek  # 0=Monday, 6=Sunday
df['day_name'] = df['Date'].dt.day_name()
df['week_of_year'] = df['Date'].dt.isocalendar().week
df['day_of_year'] = df['Date'].dt.dayofyear
df['quarter'] = df['Date'].dt.quarter

# Weekend indicator
df['is_weekend'] = df['day_of_week'].isin([5, 6])

# First/last day of month
df['is_first_day'] = df['Date'].dt.day == 1
df['is_last_day'] = df['Date'].dt.is_month_end

# Season (Northern Hemisphere)
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

# School year periods (US academic calendar)
def get_school_period(date):
    month = date.month
    day = date.day
    
    # Summer break (June-August)
    if month in [6, 7, 8]:
        return 'Summer Break'
    # Fall semester (September-December)
    elif month in [9, 10, 11, 12]:
        return 'Fall Semester'
    # Spring semester (January-May)
    else:
        return 'Spring Semester'

df['school_period'] = df['Date'].apply(get_school_period)

# School days vs non-school days
# School day = weekday during school semester (not summer break, not weekends, not major holidays)
def is_school_day(row):
    # Weekends are never school days
    if row['is_weekend']:
        return False
    
    # Summer break (June-August) - not school days
    if row['school_period'] == 'Summer Break':
        return False
    
    # Major holidays during school periods - typically not school days
    # (Thanksgiving week, Christmas week, New Year's week, Spring Break in March/April)
    month = row['Date'].month
    day = row['Date'].day
    
    # Thanksgiving week (late November)
    if month == 11 and day >= 20 and day <= 25:
        return False
    
    # Christmas/New Year break (late Dec - early Jan)
    if (month == 12 and day >= 20) or (month == 1 and day <= 5):
        return False
    
    # Spring break (typically mid-March to mid-April)
    if month == 3 and day >= 10 and day <= 25:
        return False
    if month == 4 and day <= 10:
        return False
    
    # Easter week (varies, but typically March-April)
    # We'll use a simple approximation
    
    # If it's a weekday during school semester and not a major holiday period, it's a school day
    if row['school_period'] in ['Fall Semester', 'Spring Semester']:
        return True
    
    return False

df['is_school_day'] = df.apply(is_school_day, axis=1)

# Days since start of dataset
df['days_since_start'] = (df['Date'] - df['Date'].min()).dt.days

# ============================================================================
# 2. IDENTIFY HOLIDAYS AND SPECIAL DATES
# ============================================================================
print("\nIdentifying holidays and special dates...")

# US Holidays (approximate dates for 2022-2023)
holidays_dict = {
    'New Year\'s Day': [(2022, 1, 1), (2023, 1, 1)],
    'Martin Luther King Jr. Day': [(2022, 1, 17), (2023, 1, 16)],  # 3rd Monday
    'Valentine\'s Day': [(2022, 2, 14), (2023, 2, 14)],
    'Presidents\' Day': [(2022, 2, 21), (2023, 2, 20)],  # 3rd Monday
    'St. Patrick\'s Day': [(2022, 3, 17), (2023, 3, 17)],
    'Easter': [(2022, 4, 17), (2023, 4, 9)],
    'Memorial Day': [(2022, 5, 30), (2023, 5, 29)],  # Last Monday
    'Independence Day': [(2022, 7, 4), (2023, 7, 4)],
    'Labor Day': [(2022, 9, 5), (2023, 9, 4)],  # 1st Monday
    'Halloween': [(2022, 10, 31), (2023, 10, 31)],
    'Thanksgiving': [(2022, 11, 24), (2023, 11, 23)],  # 4th Thursday
    'Black Friday': [(2022, 11, 25), (2023, 11, 24)],
    'Christmas Eve': [(2022, 12, 24), (2023, 12, 24)],
    'Christmas': [(2022, 12, 25), (2023, 12, 25)],
    'New Year\'s Eve': [(2022, 12, 31), (2023, 12, 31)],
}

# Create holiday columns
df['is_holiday'] = False
df['holiday_name'] = ''

for holiday, dates in holidays_dict.items():
    for year, month, day in dates:
        mask = (df['Date'].dt.year == year) & \
               (df['Date'].dt.month == month) & \
               (df['Date'].dt.day == day)
        df.loc[mask, 'is_holiday'] = True
        df.loc[mask, 'holiday_name'] = holiday

# Special date categories
df['is_major_holiday'] = df['holiday_name'].isin([
    'New Year\'s Day', 'Independence Day', 'Thanksgiving', 
    'Christmas', 'New Year\'s Eve'
])

# Days around holidays (within 2 days)
df['days_from_holiday'] = np.nan
for holiday, dates in holidays_dict.items():
    for year, month, day in dates:
        holiday_date = pd.Timestamp(year, month, day)
        mask = (df['Date'] >= holiday_date - timedelta(days=2)) & \
               (df['Date'] <= holiday_date + timedelta(days=2))
        df.loc[mask, 'days_from_holiday'] = (df.loc[mask, 'Date'] - holiday_date).dt.days

# ============================================================================
# 3. OPTIONAL: DETREND DATA (Remove overall time trend)
# ============================================================================
print("\n" + "=" * 80)
print("OPTIONAL DETRENDING")
print("=" * 80)

target_var = 'Number of  reported results'

# Note: The "trend" might actually be the real Wordle popularity pattern
# Detrending is optional - you can analyze raw data and just control for time
print("\nNote: The overall time pattern (trend) might be the real Wordle popularity")
print("      pattern, not noise to remove. Detrending is optional.")
print("\nOptions:")
print("  1. Analyze raw data (trend is part of the story)")
print("  2. Detrend to see time variable effects after removing trend")
print("\nProceeding with detrending for comparison, but raw analysis is also valid...")

# Use Gaussian filter for trend (smooth and handles edges well)
print("\nFitting trend model using Gaussian filter...")

# Gaussian filter with sigma=10 (smooth but not overfitting)
sigma = 10
df['trend'] = gaussian_filter1d(df[target_var].values, sigma=sigma)

# Calculate residuals (detrended data)
df['residuals'] = df[target_var] - df['trend']
df['residuals_pct'] = (df['residuals'] / df['trend'] * 100)  # Percentage deviation from trend

# Calculate momentum/autocorrelation features
df['yesterday_raw'] = df[target_var].shift(1)
df['yesterday_residual'] = df['residuals'].shift(1)
df['change_from_yesterday'] = df[target_var] - df['yesterday_raw']
df['residual_change'] = df['residuals'] - df['yesterday_residual']

# Multi-day momentum (rolling averages of past days)
for lag in [2, 3, 7]:
    df[f'residual_lag_{lag}'] = df['residuals'].shift(lag)
    df[f'past_{lag}day_avg_residual'] = df['residuals'].shift(1).rolling(window=lag, min_periods=1).mean()

# Calculate R² for trend
trend_r2 = np.corrcoef(df[target_var], df['trend'])[0,1]**2
print(f"Trend model R²: {trend_r2:.4f}")
print(f"Trend explains {trend_r2*100:.2f}% of variance in reported results")
print(f"Residual std: {df['residuals'].std():,.0f}")
print(f"Using Gaussian filter with sigma={sigma} for smooth trend")
print("\n⚠️  Remember: The 'trend' might be the real pattern, not noise!")

# ============================================================================
# 4. EXPLORATORY ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("EXPLORATORY ANALYSIS")
print("=" * 80)

# Summary statistics
print(f"\n1. SUMMARY STATISTICS FOR '{target_var}'")
print("-" * 80)
print(df[target_var].describe())
print(f"\nTotal range: {df[target_var].min():,} to {df[target_var].max():,}")
print(f"Mean: {df[target_var].mean():,.0f}")
print(f"Median: {df[target_var].median():,.0f}")
print(f"Std Dev: {df[target_var].std():,.0f}")

print(f"\n2. SUMMARY STATISTICS FOR RESIDUALS (Detrended)")
print("-" * 80)
print(df['residuals'].describe())
print(f"\nMean residual: {df['residuals'].mean():,.0f}")
print(f"Std Dev: {df['residuals'].std():,.0f}")

# ============================================================================
# 5. VISUALIZATIONS
# ============================================================================
print("\n" + "=" * 80)
print("CREATING VISUALIZATIONS")
print("=" * 80)

# Create output directory for plots
import os
os.makedirs('plots', exist_ok=True)

# Helper functions for statistical tests
def test_significance(group1, group2, name="Comparison"):
    """Perform t-test and return results"""
    try:
        # T-test (assumes normal distribution)
        t_stat, p_value = ttest_ind(group1, group2, equal_var=False)
        
        # Also try Mann-Whitney U test (non-parametric, more robust)
        u_stat, p_value_mw = mannwhitneyu(group1, group2, alternative='two-sided')
        
        # Use the more conservative p-value
        p_value_final = max(p_value, p_value_mw)
        
        # Determine significance
        if p_value_final < 0.001:
            sig = "***"
            sig_text = "Highly Significant (p<0.001)"
        elif p_value_final < 0.01:
            sig = "**"
            sig_text = "Very Significant (p<0.01)"
        elif p_value_final < 0.05:
            sig = "*"
            sig_text = "Significant (p<0.05)"
        elif p_value_final < 0.1:
            sig = "."
            sig_text = "Marginally Significant (p<0.1)"
        else:
            sig = "ns"
            sig_text = "Not Significant (p≥0.1)"
        
        return {
            't_stat': t_stat,
            'p_value': p_value_final,
            'sig': sig,
            'sig_text': sig_text,
            'mean_diff': group1.mean() - group2.mean()
        }
    except Exception as e:
        return {
            't_stat': np.nan,
            'p_value': np.nan,
            'sig': 'error',
            'sig_text': f'Error: {str(e)}',
            'mean_diff': group1.mean() - group2.mean()
        }

# 5.1 Time series plot - show both raw and detrended
print("\n1. Creating time series plot...")
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# Raw data only (no detrending)
axes[0].plot(df['Date'], df[target_var], linewidth=2, alpha=0.8, label='Actual', color='steelblue')
axes[0].set_xlabel('Date', fontsize=12)
axes[0].set_ylabel('Number of Reported Results', fontsize=12)
axes[0].set_title('Raw Data (No Detrending - Trend is Part of the Story)', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# Highlight holidays
holiday_dates = df[df['is_holiday']]['Date']
holiday_values = df[df['is_holiday']][target_var]
axes[0].scatter(holiday_dates, holiday_values, color='orange', s=50, alpha=0.6, 
                label='Holidays', zorder=5)

# Raw data with trend overlay
axes[1].plot(df['Date'], df[target_var], linewidth=1.5, alpha=0.7, label='Actual', color='steelblue')
axes[1].plot(df['Date'], df['trend'], linewidth=2, alpha=0.8, label='Estimated Trend', color='red', linestyle='--')
axes[1].set_xlabel('Date', fontsize=12)
axes[1].set_ylabel('Number of Reported Results', fontsize=12)
axes[1].set_title('Raw Data with Estimated Trend Overlay', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].legend()

# Detrended data (residuals) - optional view
axes[2].plot(df['Date'], df['residuals'], linewidth=1.5, alpha=0.7, color='green')
axes[2].axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Zero line')
axes[2].set_xlabel('Date', fontsize=12)
axes[2].set_ylabel('Residuals (Actual - Trend)', fontsize=12)
axes[2].set_title('Detrended Data (Residuals) - Optional View', fontsize=14, fontweight='bold')
axes[2].grid(True, alpha=0.3)
axes[2].legend()

# Highlight holidays on residuals
holiday_residuals = df[df['is_holiday']]['residuals']
axes[2].scatter(holiday_dates, holiday_residuals, color='orange', s=50, alpha=0.6, 
                label='Holidays', zorder=5)

plt.tight_layout()
plt.savefig('plots/01_time_series.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: plots/01_time_series.png")

# 5.2 Day of week analysis (raw vs detrended)
print("\n2. Creating day of week analysis...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
df_day = df.copy()
df_day['day_name'] = pd.Categorical(df_day['day_name'], categories=day_order, ordered=True)

# Raw data
day_means_raw = df_day.groupby('day_name')[target_var].mean().reindex(day_order)
day_stds_raw = df_day.groupby('day_name')[target_var].std().reindex(day_order)
bars0 = axes[0].bar(range(len(day_means_raw)), day_means_raw.values, color='steelblue', alpha=0.7, 
                    yerr=day_stds_raw.values, capsize=5, edgecolor='black')
axes[0].set_xticks(range(len(day_means_raw)))
axes[0].set_xticklabels(day_means_raw.index, rotation=45)
axes[0].set_title('Average Results by Day of Week (Raw) ±1 SD', fontweight='bold')
axes[0].set_ylabel('Average Number of Reported Results')
axes[0].grid(axis='y', alpha=0.3)

# Detrended data with significance
day_means_resid = df_day.groupby('day_name')['residuals'].mean().reindex(day_order)
day_stds_resid = df_day.groupby('day_name')['residuals'].std().reindex(day_order)
colors = ['green' if x > 0 else 'red' for x in day_means_resid.values]
bars1 = axes[1].bar(range(len(day_means_resid)), day_means_resid.values, color=colors, alpha=0.7,
                    yerr=day_stds_resid.values, capsize=5, edgecolor='black')
axes[1].axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
axes[1].set_xticks(range(len(day_means_resid)))
axes[1].set_xticklabels(day_means_resid.index, rotation=45)
axes[1].set_title('Average Residuals by Day of Week (Detrended) ±1 SD', fontweight='bold')
axes[1].set_ylabel('Average Residuals')
axes[1].grid(axis='y', alpha=0.3)

# Add significance indicators (compare each day to overall mean)
overall_mean_resid = df['residuals'].mean()
for i, day in enumerate(day_order):
    day_data = df_day[df_day['day_name'] == day]['residuals']
    if len(day_data) > 0:
        # T-test against zero (or overall mean)
        t_stat, p_val = stats.ttest_1samp(day_data, 0)
        if p_val < 0.001:
            sig = '***'
        elif p_val < 0.01:
            sig = '**'
        elif p_val < 0.05:
            sig = '*'
        elif p_val < 0.1:
            sig = '.'
        else:
            sig = ''
        
        if sig:
            height = day_means_resid.iloc[i]
            axes[1].text(i, height + day_stds_resid.iloc[i] + 2000, sig, 
                        ha='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('plots/02_day_of_week.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: plots/02_day_of_week.png")

# 5.3 Month and season analysis (raw vs detrended)
print("\n3. Creating month and season analysis...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

month_order = ['January', 'February', 'March', 'April', 'May', 'June',
               'July', 'August', 'September', 'October', 'November', 'December']
df['month_name'] = df['Date'].dt.month_name()
df['month_name'] = pd.Categorical(df['month_name'], categories=month_order, ordered=True)

# Month - Raw
month_means_raw = df.groupby('month_name')[target_var].mean()
axes[0, 0].bar(range(len(month_means_raw)), month_means_raw.values, color='coral', alpha=0.7)
axes[0, 0].set_xticks(range(len(month_means_raw)))
axes[0, 0].set_xticklabels(month_means_raw.index, rotation=45, ha='right')
axes[0, 0].set_title('Average Results by Month (Raw)', fontweight='bold')
axes[0, 0].set_ylabel('Average Number of Reported Results')
axes[0, 0].grid(axis='y', alpha=0.3)

# Month - Detrended
month_means_resid = df.groupby('month_name')['residuals'].mean()
colors = ['green' if x > 0 else 'red' for x in month_means_resid.values]
axes[0, 1].bar(range(len(month_means_resid)), month_means_resid.values, color=colors, alpha=0.7)
axes[0, 1].axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
axes[0, 1].set_xticks(range(len(month_means_resid)))
axes[0, 1].set_xticklabels(month_means_resid.index, rotation=45, ha='right')
axes[0, 1].set_title('Average Residuals by Month (Detrended)', fontweight='bold')
axes[0, 1].set_ylabel('Average Residuals')
axes[0, 1].grid(axis='y', alpha=0.3)

# Season - Raw
season_order = ['Spring', 'Summer', 'Fall', 'Winter']
df['season'] = pd.Categorical(df['season'], categories=season_order, ordered=True)
season_means_raw = df.groupby('season')[target_var].mean()
axes[1, 0].bar(range(len(season_means_raw)), season_means_raw.values, color='lightgreen', alpha=0.7)
axes[1, 0].set_xticks(range(len(season_means_raw)))
axes[1, 0].set_xticklabels(season_means_raw.index)
axes[1, 0].set_title('Average Results by Season (Raw)', fontweight='bold')
axes[1, 0].set_ylabel('Average Number of Reported Results')
axes[1, 0].grid(axis='y', alpha=0.3)

# Season - Detrended
season_means_resid = df.groupby('season')['residuals'].mean()
colors = ['green' if x > 0 else 'red' for x in season_means_resid.values]
axes[1, 1].bar(range(len(season_means_resid)), season_means_resid.values, color=colors, alpha=0.7)
axes[1, 1].axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
axes[1, 1].set_xticks(range(len(season_means_resid)))
axes[1, 1].set_xticklabels(season_means_resid.index)
axes[1, 1].set_title('Average Residuals by Season (Detrended)', fontweight='bold')
axes[1, 1].set_ylabel('Average Residuals')
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('plots/03_month_season.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: plots/03_month_season.png")

# 5.4 Holiday analysis (raw vs detrended)
print("\n4. Creating holiday analysis...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Holiday vs non-holiday - Raw
holiday_means_raw = df.groupby('is_holiday')[target_var].mean()
holiday_stds_raw = df.groupby('is_holiday')[target_var].std()
bars0 = axes[0].bar(['Non-Holiday', 'Holiday'], holiday_means_raw.values, 
                    color='indianred', alpha=0.7, yerr=holiday_stds_raw.values, 
                    capsize=5, edgecolor='black')
axes[0].set_title('Average Results: Holiday vs Non-Holiday (Raw) ±1 SD', fontweight='bold')
axes[0].set_ylabel('Average Number of Reported Results')
axes[0].grid(axis='y', alpha=0.3)

# Test significance and add marker
test = test_significance(df[~df['is_holiday']][target_var], df[df['is_holiday']][target_var])
if test['sig'] != 'ns':
    axes[0].text(0.5, max(holiday_means_raw.values) + holiday_stds_raw.max() * 0.1, 
                test['sig'], ha='center', fontsize=14, fontweight='bold')

# Holiday vs non-holiday - Detrended
holiday_means_resid = df.groupby('is_holiday')['residuals'].mean()
holiday_stds_resid = df.groupby('is_holiday')['residuals'].std()
colors = ['green' if x > 0 else 'red' for x in holiday_means_resid.values]
bars1 = axes[1].bar(['Non-Holiday', 'Holiday'], holiday_means_resid.values, 
                    color=colors, alpha=0.7, yerr=holiday_stds_resid.values,
                    capsize=5, edgecolor='black')
axes[1].axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
axes[1].set_title('Average Residuals: Holiday vs Non-Holiday (Detrended) ±1 SD', fontweight='bold')
axes[1].set_ylabel('Average Residuals')
axes[1].grid(axis='y', alpha=0.3)

# Test significance and add marker
test = test_significance(df[~df['is_holiday']]['residuals'], df[df['is_holiday']]['residuals'])
if test['sig'] != 'ns':
    axes[1].text(0.5, max(holiday_means_resid.values) + holiday_stds_resid.max() * 0.1, 
                test['sig'], ha='center', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('plots/04_holidays.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: plots/04_holidays.png")

# 4.5 Day of month and special days
print("\n5. Creating day of month analysis...")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Day of month
day_of_month_means = df.groupby('day')[target_var].mean()
axes[0].plot(day_of_month_means.index, day_of_month_means.values, marker='o', linewidth=2, markersize=4)
axes[0].set_title('Average Results by Day of Month', fontweight='bold')
axes[0].set_xlabel('Day of Month')
axes[0].set_ylabel('Average Number of Reported Results')
axes[0].grid(True, alpha=0.3)

# First vs last day
special_days = pd.DataFrame({
    'First Day': [df[df['is_first_day']][target_var].mean()],
    'Last Day': [df[df['is_last_day']][target_var].mean()],
    'Other Days': [df[~(df['is_first_day'] | df['is_last_day'])][target_var].mean()]
}).T
axes[1].bar(special_days.index, special_days[0].values, color='teal', alpha=0.7)
axes[1].set_title('Average Results: Special Days', fontweight='bold')
axes[1].set_ylabel('Average Number of Reported Results')
axes[1].tick_params(axis='x', rotation=15)
axes[1].grid(axis='y', alpha=0.3)

# Week of year
week_means = df.groupby('week_of_year')[target_var].mean()
axes[2].plot(week_means.index, week_means.values, linewidth=2, color='darkblue')
axes[2].set_title('Average Results by Week of Year', fontweight='bold')
axes[2].set_xlabel('Week of Year')
axes[2].set_ylabel('Average Number of Reported Results')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/05_day_of_month.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: plots/05_day_of_month.png")

# 5.6 Correlation heatmap (raw vs detrended)
print("\n6. Creating correlation heatmap...")
time_vars = [
    'day_of_week', 'month', 'quarter', 'week_of_year', 'day_of_year',
    'is_weekend', 'is_holiday', 'is_major_holiday'
]

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Raw data correlations
corr_raw = df[time_vars + [target_var]].corr()
sns.heatmap(corr_raw, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=axes[0])
axes[0].set_title('Correlation: Time Variables vs Results (Raw)', fontweight='bold', fontsize=12)

# Detrended data correlations
corr_resid = df[time_vars + ['residuals']].corr()
sns.heatmap(corr_resid, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=axes[1])
axes[1].set_title('Correlation: Time Variables vs Residuals (Detrended)', fontweight='bold', fontsize=12)

plt.tight_layout()
plt.savefig('plots/06_correlation.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: plots/06_correlation.png")

# ============================================================================
# 6. STATISTICAL SIGNIFICANCE TESTS
# ============================================================================
print("\n" + "=" * 80)
print("STATISTICAL SIGNIFICANCE TESTS")
print("=" * 80)

def test_significance(group1, group2, name="Comparison"):
    """Perform t-test and return results"""
    try:
        # T-test (assumes normal distribution)
        t_stat, p_value = ttest_ind(group1, group2, equal_var=False)
        
        # Also try Mann-Whitney U test (non-parametric, more robust)
        u_stat, p_value_mw = mannwhitneyu(group1, group2, alternative='two-sided')
        
        # Use the more conservative p-value
        p_value_final = max(p_value, p_value_mw)
        
        # Determine significance
        if p_value_final < 0.001:
            sig = "***"
            sig_text = "Highly Significant (p<0.001)"
        elif p_value_final < 0.01:
            sig = "**"
            sig_text = "Very Significant (p<0.01)"
        elif p_value_final < 0.05:
            sig = "*"
            sig_text = "Significant (p<0.05)"
        elif p_value_final < 0.1:
            sig = "."
            sig_text = "Marginally Significant (p<0.1)"
        else:
            sig = "ns"
            sig_text = "Not Significant (p≥0.1)"
        
        return {
            't_stat': t_stat,
            'p_value': p_value_final,
            'sig': sig,
            'sig_text': sig_text,
            'mean_diff': group1.mean() - group2.mean()
        }
    except Exception as e:
        return {
            't_stat': np.nan,
            'p_value': np.nan,
            'sig': 'error',
            'sig_text': f'Error: {str(e)}',
            'mean_diff': group1.mean() - group2.mean()
        }

def anova_test(groups_dict, name="ANOVA"):
    """Perform one-way ANOVA test"""
    try:
        groups = [g for g in groups_dict.values() if len(g) > 0]
        if len(groups) < 2:
            return {'f_stat': np.nan, 'p_value': np.nan, 'sig': 'ns', 'sig_text': 'Not enough groups'}
        
        f_stat, p_value = f_oneway(*groups)
        
        if p_value < 0.001:
            sig = "***"
            sig_text = "Highly Significant (p<0.001)"
        elif p_value < 0.01:
            sig = "**"
            sig_text = "Very Significant (p<0.01)"
        elif p_value < 0.05:
            sig = "*"
            sig_text = "Significant (p<0.05)"
        elif p_value < 0.1:
            sig = "."
            sig_text = "Marginally Significant (p<0.1)"
        else:
            sig = "ns"
            sig_text = "Not Significant (p≥0.1)"
        
        return {
            'f_stat': f_stat,
            'p_value': p_value,
            'sig': sig,
            'sig_text': sig_text
        }
    except Exception as e:
        return {'f_stat': np.nan, 'p_value': np.nan, 'sig': 'error', 'sig_text': f'Error: {str(e)}'}

print("\n1. WEEKEND VS WEEKDAY:")
print("-" * 80)
weekday_raw = df[~df['is_weekend']][target_var]
weekend_raw = df[df['is_weekend']][target_var]
weekday_resid = df[~df['is_weekend']]['residuals']
weekend_resid = df[df['is_weekend']]['residuals']

test_raw = test_significance(weekday_raw, weekend_raw, "Weekday vs Weekend (Raw)")
test_resid = test_significance(weekday_resid, weekend_resid, "Weekday vs Weekend (Detrended)")

print(f"Raw Data:")
print(f"  Mean difference: {test_raw['mean_diff']:,.0f}")
print(f"  T-statistic: {test_raw['t_stat']:.3f}")
print(f"  P-value: {test_raw['p_value']:.4f} {test_raw['sig']}")
print(f"  {test_raw['sig_text']}")

print(f"\nDetrended Data:")
print(f"  Mean difference: {test_resid['mean_diff']:,.0f}")
print(f"  T-statistic: {test_resid['t_stat']:.3f}")
print(f"  P-value: {test_resid['p_value']:.4f} {test_resid['sig']}")
print(f"  {test_resid['sig_text']}")

print("\n2. HOLIDAY VS NON-HOLIDAY:")
print("-" * 80)
non_holiday_raw = df[~df['is_holiday']][target_var]
holiday_raw = df[df['is_holiday']][target_var]
non_holiday_resid = df[~df['is_holiday']]['residuals']
holiday_resid = df[df['is_holiday']]['residuals']

test_raw = test_significance(non_holiday_raw, holiday_raw, "Non-Holiday vs Holiday (Raw)")
test_resid = test_significance(non_holiday_resid, holiday_resid, "Non-Holiday vs Holiday (Detrended)")

print(f"Raw Data:")
print(f"  Mean difference: {test_raw['mean_diff']:,.0f}")
print(f"  T-statistic: {test_raw['t_stat']:.3f}")
print(f"  P-value: {test_raw['p_value']:.4f} {test_raw['sig']}")
print(f"  {test_raw['sig_text']}")

print(f"\nDetrended Data:")
print(f"  Mean difference: {test_resid['mean_diff']:,.0f}")
print(f"  T-statistic: {test_resid['t_stat']:.3f}")
print(f"  P-value: {test_resid['p_value']:.4f} {test_resid['sig']}")
print(f"  {test_resid['sig_text']}")

print("\n3. DAY OF WEEK (ANOVA):")
print("-" * 80)
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_groups_raw = {day: df[df['day_name'] == day][target_var] for day in day_order}
day_groups_resid = {day: df[df['day_name'] == day]['residuals'] for day in day_order}

anova_raw = anova_test(day_groups_raw, "Day of Week (Raw)")
anova_resid = anova_test(day_groups_resid, "Day of Week (Detrended)")

print(f"Raw Data:")
print(f"  F-statistic: {anova_raw['f_stat']:.3f}")
print(f"  P-value: {anova_raw['p_value']:.4f} {anova_raw['sig']}")
print(f"  {anova_raw['sig_text']}")

print(f"\nDetrended Data:")
print(f"  F-statistic: {anova_resid['f_stat']:.3f}")
print(f"  P-value: {anova_resid['p_value']:.4f} {anova_resid['sig']}")
print(f"  {anova_resid['sig_text']}")

# Pairwise comparisons for day of week (if ANOVA is significant)
if anova_resid['p_value'] < 0.05:
    print(f"\n  Pairwise comparisons (Detrended) - significant differences:")
    from itertools import combinations
    for day1, day2 in combinations(day_order, 2):
        test = test_significance(day_groups_resid[day1], day_groups_resid[day2], f"{day1} vs {day2}")
        if test['p_value'] < 0.05:
            print(f"    {day1} vs {day2}: p={test['p_value']:.4f} {test['sig']} (diff={test['mean_diff']:,.0f})")

print("\n4. MONTH (ANOVA):")
print("-" * 80)
month_order = ['January', 'February', 'March', 'April', 'May', 'June',
               'July', 'August', 'September', 'October', 'November', 'December']
month_groups_raw = {month: df[df['month_name'] == month][target_var] for month in month_order if month in df['month_name'].values}
month_groups_resid = {month: df[df['month_name'] == month]['residuals'] for month in month_order if month in df['month_name'].values}

anova_raw = anova_test(month_groups_raw, "Month (Raw)")
anova_resid = anova_test(month_groups_resid, "Month (Detrended)")

print(f"Raw Data:")
print(f"  F-statistic: {anova_raw['f_stat']:.3f}")
print(f"  P-value: {anova_raw['p_value']:.4f} {anova_raw['sig']}")
print(f"  {anova_raw['sig_text']}")

print(f"\nDetrended Data:")
print(f"  F-statistic: {anova_resid['f_stat']:.3f}")
print(f"  P-value: {anova_resid['p_value']:.4f} {anova_resid['sig']}")
print(f"  {anova_resid['sig_text']}")

print("\n5. SEASON (ANOVA):")
print("-" * 80)
season_order = ['Spring', 'Summer', 'Fall', 'Winter']
season_groups_raw = {season: df[df['season'] == season][target_var] for season in season_order}
season_groups_resid = {season: df[df['season'] == season]['residuals'] for season in season_order}

anova_raw = anova_test(season_groups_raw, "Season (Raw)")
anova_resid = anova_test(season_groups_resid, "Season (Detrended)")

print(f"Raw Data:")
print(f"  F-statistic: {anova_raw['f_stat']:.3f}")
print(f"  P-value: {anova_raw['p_value']:.4f} {anova_raw['sig']}")
print(f"  {anova_raw['sig_text']}")

print(f"\nDetrended Data:")
print(f"  F-statistic: {anova_resid['f_stat']:.3f}")
print(f"  P-value: {anova_resid['p_value']:.4f} {anova_resid['sig']}")
print(f"  {anova_resid['sig_text']}")

print("\n6. SCHOOL DAYS VS NON-SCHOOL DAYS:")
print("-" * 80)
school_day_raw = df[df['is_school_day']][target_var]
non_school_day_raw = df[~df['is_school_day']][target_var]
school_day_resid = df[df['is_school_day']]['residuals']
non_school_day_resid = df[~df['is_school_day']]['residuals']

test_raw = test_significance(school_day_raw, non_school_day_raw, "School Day vs Non-School Day (Raw)")
test_resid = test_significance(school_day_resid, non_school_day_resid, "School Day vs Non-School Day (Detrended)")

print(f"Raw Data:")
print(f"  School days: {len(school_day_raw)} days, Mean: {school_day_raw.mean():,.0f}")
print(f"  Non-school days: {len(non_school_day_raw)} days, Mean: {non_school_day_raw.mean():,.0f}")
print(f"  Mean difference: {test_raw['mean_diff']:,.0f}")
print(f"  T-statistic: {test_raw['t_stat']:.3f}")
print(f"  P-value: {test_raw['p_value']:.4f} {test_raw['sig']}")
print(f"  {test_raw['sig_text']}")

print(f"\nDetrended Data:")
print(f"  School days residual mean: {school_day_resid.mean():,.0f}")
print(f"  Non-school days residual mean: {non_school_day_resid.mean():,.0f}")
print(f"  Mean difference: {test_resid['mean_diff']:,.0f}")
print(f"  T-statistic: {test_resid['t_stat']:.3f}")
print(f"  P-value: {test_resid['p_value']:.4f} {test_resid['sig']}")
print(f"  {test_resid['sig_text']}")

print("\n" + "=" * 80)
print("NOTE: Significance levels: *** p<0.001, ** p<0.01, * p<0.05, . p<0.1, ns p≥0.1")
print("=" * 80)

# ============================================================================
# 6.5 MOMENTUM/AUTOCORRELATION ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("MOMENTUM/AUTOCORRELATION ANALYSIS")
print("=" * 80)
print("\nQuestion: Does yesterday/past influence today (after removing trend)?")

# Calculate autocorrelations for residuals
from scipy.stats import pearsonr

print("\n1. AUTOCORRELATION IN RESIDUALS (Detrended):")
print("-" * 80)

# Remove NaN values for correlation
valid_mask = ~(df['residuals'].isna() | df['yesterday_residual'].isna())
residuals_today = df.loc[valid_mask, 'residuals']
residuals_yesterday = df.loc[valid_mask, 'yesterday_residual']

if len(residuals_today) > 1:
    corr_yesterday, p_yesterday = pearsonr(residuals_today, residuals_yesterday)
    print(f"Correlation with yesterday's residual: {corr_yesterday:.4f}")
    if p_yesterday < 0.001:
        sig = "***"
    elif p_yesterday < 0.01:
        sig = "**"
    elif p_yesterday < 0.05:
        sig = "*"
    elif p_yesterday < 0.1:
        sig = "."
    else:
        sig = "ns"
    print(f"P-value: {p_yesterday:.4f} {sig}")
    print(f"Interpretation: {'Strong momentum' if abs(corr_yesterday) > 0.3 else 'Weak momentum' if abs(corr_yesterday) > 0.1 else 'No momentum'}")

# Check multiple lags
print("\n2. AUTOCORRELATION AT DIFFERENT LAGS:")
print("-" * 80)
lags = [1, 2, 3, 7, 14]
autocorr_results = []

for lag in lags:
    lagged_residual = df['residuals'].shift(lag)
    valid_mask = ~(df['residuals'].isna() | lagged_residual.isna())
    if valid_mask.sum() > 10:
        corr, p_val = pearsonr(df.loc[valid_mask, 'residuals'], lagged_residual[valid_mask])
        autocorr_results.append({'lag': lag, 'correlation': corr, 'p_value': p_val})
        
        if p_val < 0.05:
            sig = "*" if p_val >= 0.01 else "**" if p_val >= 0.001 else "***"
        elif p_val < 0.1:
            sig = "."
        else:
            sig = "ns"
        
        print(f"  Lag {lag:2d} days: r={corr:6.3f}, p={p_val:.4f} {sig}")

# Regression: today's residual vs yesterday's residual
print("\n3. REGRESSION: TODAY'S RESIDUAL vs YESTERDAY'S RESIDUAL:")
print("-" * 80)
from sklearn.linear_model import LinearRegression

valid_mask = ~(df['residuals'].isna() | df['yesterday_residual'].isna())
X_momentum = df.loc[valid_mask, 'yesterday_residual'].values.reshape(-1, 1)
y_momentum = df.loc[valid_mask, 'residuals'].values

if len(X_momentum) > 1:
    momentum_model = LinearRegression()
    momentum_model.fit(X_momentum, y_momentum)
    r2_momentum = momentum_model.score(X_momentum, y_momentum)
    
    print(f"R²: {r2_momentum:.4f}")
    print(f"Yesterday's residual explains {r2_momentum*100:.2f}% of variance in today's residual")
    print(f"Coefficient: {momentum_model.coef_[0]:.4f}")
    print(f"  (For every 1 unit increase in yesterday's residual, today's residual changes by {momentum_model.coef_[0]:.4f})")
    
    if abs(momentum_model.coef_[0]) > 0.3:
        print("  → Strong momentum effect")
    elif abs(momentum_model.coef_[0]) > 0.1:
        print("  → Moderate momentum effect")
    else:
        print("  → Weak/no momentum effect")

# Visualize momentum
print("\n4. Creating momentum visualizations...")
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Scatter: today vs yesterday residuals
valid_mask = ~(df['residuals'].isna() | df['yesterday_residual'].isna())
axes[0, 0].scatter(df.loc[valid_mask, 'yesterday_residual'], 
                   df.loc[valid_mask, 'residuals'], 
                   alpha=0.6, s=30, color='steelblue')
axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
axes[0, 0].axvline(x=0, color='red', linestyle='--', linewidth=1, alpha=0.5)

# Add regression line
if len(X_momentum) > 1:
    x_line = np.linspace(df.loc[valid_mask, 'yesterday_residual'].min(), 
                        df.loc[valid_mask, 'yesterday_residual'].max(), 100)
    y_line = momentum_model.predict(x_line.reshape(-1, 1))
    axes[0, 0].plot(x_line, y_line, 'r--', linewidth=2, label=f'Regression (R²={r2_momentum:.3f})')
    axes[0, 0].legend()

axes[0, 0].set_xlabel("Yesterday's Residual", fontsize=12)
axes[0, 0].set_ylabel("Today's Residual", fontsize=12)
axes[0, 0].set_title('Momentum: Today vs Yesterday Residuals', fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# Autocorrelation function plot
lags_plot = list(range(1, 15))
autocorr_values = []
for lag in lags_plot:
    lagged = df['residuals'].shift(lag)
    valid = ~(df['residuals'].isna() | lagged.isna())
    if valid.sum() > 10:
        corr, _ = pearsonr(df.loc[valid, 'residuals'], lagged[valid])
        autocorr_values.append(corr)
    else:
        autocorr_values.append(np.nan)

axes[0, 1].plot(lags_plot, autocorr_values, marker='o', linewidth=2, markersize=6, color='green')
axes[0, 1].axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
axes[0, 1].axhline(y=0.1, color='orange', linestyle=':', linewidth=1, alpha=0.5, label='Weak threshold')
axes[0, 1].axhline(y=-0.1, color='orange', linestyle=':', linewidth=1, alpha=0.5)
axes[0, 1].set_xlabel('Lag (days)', fontsize=12)
axes[0, 1].set_ylabel('Autocorrelation', fontsize=12)
axes[0, 1].set_title('Autocorrelation Function (Residuals)', fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Time series of residuals showing momentum
axes[1, 0].plot(df['Date'], df['residuals'], linewidth=1.5, alpha=0.7, color='green', label='Residuals')
axes[1, 0].axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
axes[1, 0].set_xlabel('Date', fontsize=12)
axes[1, 0].set_ylabel('Residuals', fontsize=12)
axes[1, 0].set_title('Residuals Over Time (Momentum Pattern)', fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Change from yesterday
valid_mask = ~df['residual_change'].isna()
axes[1, 1].plot(df.loc[valid_mask, 'Date'], df.loc[valid_mask, 'residual_change'], 
                linewidth=1.5, alpha=0.7, color='purple')
axes[1, 1].axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
axes[1, 1].set_xlabel('Date', fontsize=12)
axes[1, 1].set_ylabel('Change in Residuals from Yesterday', fontsize=12)
axes[1, 1].set_title('Day-to-Day Change in Residuals', fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/12_momentum_autocorrelation.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: plots/12_momentum_autocorrelation.png")

print("\n5. SUMMARY:")
print("-" * 80)
if len(autocorr_results) > 0:
    max_corr = max([abs(r['correlation']) for r in autocorr_results])
    max_lag = next(r['lag'] for r in autocorr_results if abs(r['correlation']) == max_corr)
    print(f"Strongest autocorrelation: Lag {max_lag} days (r={max_corr:.3f})")
    
    if max_corr > 0.3:
        print("→ Strong momentum: Past days significantly influence present")
    elif max_corr > 0.1:
        print("→ Moderate momentum: Some influence from past days")
    else:
        print("→ Weak/no momentum: Little influence from past days")
    
    if autocorr_results[0]['p_value'] < 0.05:
        print(f"→ Yesterday's residual is a significant predictor (p={autocorr_results[0]['p_value']:.4f})")
    else:
        print(f"→ Yesterday's residual is NOT a significant predictor (p={autocorr_results[0]['p_value']:.4f})")

# ============================================================================
# 6.6 SCHOOL DAYS VS NON-SCHOOL DAYS ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("SCHOOL DAYS VS NON-SCHOOL DAYS ANALYSIS")
print("=" * 80)

print(f"\nData breakdown:")
print(f"  School days: {df['is_school_day'].sum()} days ({df['is_school_day'].sum()/len(df)*100:.1f}%)")
print(f"  Non-school days: {(~df['is_school_day']).sum()} days ({(~df['is_school_day']).sum()/len(df)*100:.1f}%)")

print("\n1. Creating school days visualizations...")
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Time series with school days highlighted
axes[0, 0].plot(df['Date'], df[target_var], linewidth=1.5, alpha=0.5, color='gray', label='All days')
school_days = df[df['is_school_day']]
non_school_days = df[~df['is_school_day']]
axes[0, 0].scatter(school_days['Date'], school_days[target_var], 
                   color='blue', s=20, alpha=0.6, label='School days', zorder=3)
axes[0, 0].scatter(non_school_days['Date'], non_school_days[target_var], 
                   color='orange', s=20, alpha=0.6, label='Non-school days', zorder=3)
axes[0, 0].set_title('Time Series: School Days vs Non-School Days (Raw)', fontweight='bold')
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Number of Reported Results')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Box plot comparison (raw)
school_vs_non = pd.DataFrame({
    'Type': ['School Day'] * len(school_days) + ['Non-School Day'] * len(non_school_days),
    'Value': list(school_days[target_var]) + list(non_school_days[target_var])
})
sns.boxplot(data=school_vs_non, x='Type', y='Value', ax=axes[0, 1])
axes[0, 1].set_title('Distribution: School Days vs Non-School Days (Raw)', fontweight='bold')
axes[0, 1].set_ylabel('Number of Reported Results')
axes[0, 1].grid(axis='y', alpha=0.3)

# Add significance marker
test = test_significance(school_days[target_var], non_school_days[target_var])
if test['sig'] != 'ns':
    axes[0, 1].text(0.5, max(school_vs_non['Value']) * 1.05, test['sig'], 
                   ha='center', fontsize=14, fontweight='bold')

# Residuals comparison (detrended)
school_vs_non_resid = pd.DataFrame({
    'Type': ['School Day'] * len(school_days) + ['Non-School Day'] * len(non_school_days),
    'Value': list(school_days['residuals']) + list(non_school_days['residuals'])
})
sns.boxplot(data=school_vs_non_resid, x='Type', y='Value', ax=axes[1, 0])
axes[1, 0].axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
axes[1, 0].set_title('Distribution: School Days vs Non-School Days (Detrended)', fontweight='bold')
axes[1, 0].set_ylabel('Residuals')
axes[1, 0].grid(axis='y', alpha=0.3)

# Add significance marker
test = test_significance(school_days['residuals'], non_school_days['residuals'])
if test['sig'] != 'ns':
    axes[1, 0].text(0.5, max(school_vs_non_resid['Value']) * 1.05, test['sig'], 
                   ha='center', fontsize=14, fontweight='bold')

# Bar chart with error bars
school_means_raw = df.groupby('is_school_day')[target_var].agg(['mean', 'std'])
school_means_resid = df.groupby('is_school_day')['residuals'].agg(['mean', 'std'])

x_pos = [0, 1]
labels = ['Non-School Day', 'School Day']

axes[1, 1].bar(x_pos, school_means_resid['mean'], 
               yerr=school_means_resid['std'], 
               color=['orange', 'blue'], alpha=0.7, capsize=5, edgecolor='black')
axes[1, 1].axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
axes[1, 1].set_xticks(x_pos)
axes[1, 1].set_xticklabels(labels)
axes[1, 1].set_title('Average Residuals: School Days vs Non-School Days (Detrended)', fontweight='bold')
axes[1, 1].set_ylabel('Average Residuals')
axes[1, 1].grid(axis='y', alpha=0.3)

# Add significance marker
if test['sig'] != 'ns':
    axes[1, 1].text(0.5, max(school_means_resid['mean']) + school_means_resid['std'].max() * 0.2, 
                   test['sig'], ha='center', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('plots/13_school_days.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: plots/13_school_days.png")

# Detailed breakdown by school period
print("\n2. BREAKDOWN BY SCHOOL PERIOD:")
print("-" * 80)
for period in ['Fall Semester', 'Spring Semester', 'Summer Break']:
    period_data = df[df['school_period'] == period]
    if len(period_data) > 0:
        school_in_period = period_data[period_data['is_school_day']]
        non_school_in_period = period_data[~period_data['is_school_day']]
        
        print(f"\n{period}:")
        print(f"  School days: {len(school_in_period)} ({len(school_in_period)/len(period_data)*100:.1f}%)")
        if len(school_in_period) > 0:
            print(f"    Mean (raw): {school_in_period[target_var].mean():,.0f}")
            print(f"    Mean (residual): {school_in_period['residuals'].mean():,.0f}")
        print(f"  Non-school days: {len(non_school_in_period)} ({len(non_school_in_period)/len(period_data)*100:.1f}%)")
        if len(non_school_in_period) > 0:
            print(f"    Mean (raw): {non_school_in_period[target_var].mean():,.0f}")
            print(f"    Mean (residual): {non_school_in_period['residuals'].mean():,.0f}")
        
        if len(school_in_period) > 5 and len(non_school_in_period) > 5:
            test_period = test_significance(school_in_period['residuals'], non_school_in_period['residuals'])
            print(f"  Difference (detrended): {test_period['mean_diff']:,.0f}, p={test_period['p_value']:.4f} {test_period['sig']}")

# Day of week breakdown by school days
print("\n3. DAY OF WEEK BREAKDOWN: SCHOOL DAYS VS NON-SCHOOL DAYS:")
print("-" * 80)

day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
df['day_name'] = pd.Categorical(df['day_name'], categories=day_order, ordered=True)

# Create comparison table
day_school_comparison = []
for day in day_order:
    day_data = df[df['day_name'] == day]
    school_day_data = day_data[day_data['is_school_day']]
    non_school_day_data = day_data[~day_data['is_school_day']]
    
    if len(school_day_data) > 0 and len(non_school_day_data) > 0:
        day_school_comparison.append({
            'Day': day,
            'School_Days_Count': len(school_day_data),
            'School_Days_Mean_Raw': school_day_data[target_var].mean(),
            'School_Days_Mean_Resid': school_day_data['residuals'].mean(),
            'Non_School_Days_Count': len(non_school_day_data),
            'Non_School_Days_Mean_Raw': non_school_day_data[target_var].mean(),
            'Non_School_Days_Mean_Resid': non_school_day_data['residuals'].mean(),
            'Diff_Raw': school_day_data[target_var].mean() - non_school_day_data[target_var].mean(),
            'Diff_Resid': school_day_data['residuals'].mean() - non_school_day_data['residuals'].mean()
        })

day_school_df = pd.DataFrame(day_school_comparison)
print("\nRaw Data:")
print(day_school_df[['Day', 'School_Days_Count', 'School_Days_Mean_Raw', 
                     'Non_School_Days_Count', 'Non_School_Days_Mean_Raw', 'Diff_Raw']].round(0))
print("\nDetrended Data:")
print(day_school_df[['Day', 'School_Days_Mean_Resid', 'Non_School_Days_Mean_Resid', 'Diff_Resid']].round(0))

# Create visualization
print("\n4. Creating day of week by school days visualization...")
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Raw data - grouped bar chart
x = np.arange(len(day_order))
width = 0.35
school_means_raw = [day_school_df[day_school_df['Day'] == day]['School_Days_Mean_Raw'].values[0] 
                    if len(day_school_df[day_school_df['Day'] == day]) > 0 else 0 for day in day_order]
non_school_means_raw = [day_school_df[day_school_df['Day'] == day]['Non_School_Days_Mean_Raw'].values[0] 
                        if len(day_school_df[day_school_df['Day'] == day]) > 0 else 0 for day in day_order]

axes[0, 0].bar(x - width/2, school_means_raw, width, label='School Days', color='blue', alpha=0.7, edgecolor='black')
axes[0, 0].bar(x + width/2, non_school_means_raw, width, label='Non-School Days', color='orange', alpha=0.7, edgecolor='black')
axes[0, 0].set_xlabel('Day of Week')
axes[0, 0].set_ylabel('Average Number of Reported Results')
axes[0, 0].set_title('Average by Day of Week: School Days vs Non-School Days (Raw)', fontweight='bold')
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels(day_order, rotation=45)
axes[0, 0].legend()
axes[0, 0].grid(axis='y', alpha=0.3)

# Detrended data - grouped bar chart
school_means_resid = [day_school_df[day_school_df['Day'] == day]['School_Days_Mean_Resid'].values[0] 
                      if len(day_school_df[day_school_df['Day'] == day]) > 0 else 0 for day in day_order]
non_school_means_resid = [day_school_df[day_school_df['Day'] == day]['Non_School_Days_Mean_Resid'].values[0] 
                          if len(day_school_df[day_school_df['Day'] == day]) > 0 else 0 for day in day_order]

axes[0, 1].bar(x - width/2, school_means_resid, width, label='School Days', color='blue', alpha=0.7, edgecolor='black')
axes[0, 1].bar(x + width/2, non_school_means_resid, width, label='Non-School Days', color='orange', alpha=0.7, edgecolor='black')
axes[0, 1].axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
axes[0, 1].set_xlabel('Day of Week')
axes[0, 1].set_ylabel('Average Residuals')
axes[0, 1].set_title('Average by Day of Week: School Days vs Non-School Days (Detrended)', fontweight='bold')
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels(day_order, rotation=45)
axes[0, 1].legend()
axes[0, 1].grid(axis='y', alpha=0.3)

# Difference plot (raw)
diffs_raw = [day_school_df[day_school_df['Day'] == day]['Diff_Raw'].values[0] 
             if len(day_school_df[day_school_df['Day'] == day]) > 0 else 0 for day in day_order]
colors_raw = ['green' if d > 0 else 'red' for d in diffs_raw]
axes[1, 0].bar(x, diffs_raw, color=colors_raw, alpha=0.7, edgecolor='black')
axes[1, 0].axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
axes[1, 0].set_xlabel('Day of Week')
axes[1, 0].set_ylabel('Difference (School Days - Non-School Days)')
axes[1, 0].set_title('Difference by Day of Week (Raw)', fontweight='bold')
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(day_order, rotation=45)
axes[1, 0].grid(axis='y', alpha=0.3)

# Difference plot (detrended)
diffs_resid = [day_school_df[day_school_df['Day'] == day]['Diff_Resid'].values[0] 
               if len(day_school_df[day_school_df['Day'] == day]) > 0 else 0 for day in day_order]
colors_resid = ['green' if d > 0 else 'red' for d in diffs_resid]
axes[1, 1].bar(x, diffs_resid, color=colors_resid, alpha=0.7, edgecolor='black')
axes[1, 1].axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
axes[1, 1].set_xlabel('Day of Week')
axes[1, 1].set_ylabel('Difference in Residuals (School Days - Non-School Days)')
axes[1, 1].set_title('Difference by Day of Week (Detrended)', fontweight='bold')
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(day_order, rotation=45)
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('plots/14_day_of_week_school_days.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: plots/14_day_of_week_school_days.png")

# Statistical tests for each day
print("\n5. STATISTICAL SIGNIFICANCE BY DAY OF WEEK:")
print("-" * 80)
for day in day_order:
    day_data = df[df['day_name'] == day]
    school_day_data = day_data[day_data['is_school_day']]
    non_school_day_data = day_data[~day_data['is_school_day']]
    
    if len(school_day_data) > 2 and len(non_school_day_data) > 2:
        test = test_significance(school_day_data['residuals'], non_school_day_data['residuals'])
        print(f"{day:12s}: School days={len(school_day_data):2d}, Non-school={len(non_school_day_data):2d}, "
              f"Diff={test['mean_diff']:7,.0f}, p={test['p_value']:.4f} {test['sig']}")

# ============================================================================
# 7. STATISTICAL SUMMARY (Raw vs Detrended)
# ============================================================================
print("\n" + "=" * 80)
print("STATISTICAL SUMMARY: RAW vs DETRENDED")
print("=" * 80)

# Group by different time variables
print("\n1. BY DAY OF WEEK (Raw vs Detrended):")
print("-" * 80)
day_stats_raw = df.groupby('day_name')[target_var].agg(['mean', 'std']).reindex(day_order)
day_stats_resid = df.groupby('day_name')['residuals'].agg(['mean', 'std']).reindex(day_order)
day_comparison = pd.DataFrame({
    'Raw_Mean': day_stats_raw['mean'],
    'Raw_Std': day_stats_raw['std'],
    'Residual_Mean': day_stats_resid['mean'],
    'Residual_Std': day_stats_resid['std']
})
print(day_comparison.round(0))

print("\n2. BY MONTH:")
print("-" * 80)
month_stats = df.groupby('month_name')[target_var].agg(['mean', 'std', 'count']).reindex(month_order)
print(month_stats.round(0))

print("\n3. BY SEASON:")
print("-" * 80)
season_stats = df.groupby('season')[target_var].agg(['mean', 'std', 'count']).reindex(season_order)
print(season_stats.round(0))

print("\n4. WEEKEND VS WEEKDAY (Raw vs Detrended):")
print("-" * 80)
weekend_raw = df.groupby('is_weekend')[target_var].agg(['mean', 'std'])
weekend_resid = df.groupby('is_weekend')['residuals'].agg(['mean', 'std'])
weekend_comparison = pd.DataFrame({
    'Raw_Mean': weekend_raw['mean'],
    'Raw_Std': weekend_raw['std'],
    'Residual_Mean': weekend_resid['mean'],
    'Residual_Std': weekend_resid['std']
})
weekend_comparison.index = ['Weekday', 'Weekend']
print(weekend_comparison.round(0))

print("\n5. HOLIDAY VS NON-HOLIDAY (Raw vs Detrended):")
print("-" * 80)
holiday_raw = df.groupby('is_holiday')[target_var].agg(['mean', 'std'])
holiday_resid = df.groupby('is_holiday')['residuals'].agg(['mean', 'std'])
holiday_comparison = pd.DataFrame({
    'Raw_Mean': holiday_raw['mean'],
    'Raw_Std': holiday_raw['std'],
    'Residual_Mean': holiday_resid['mean'],
    'Residual_Std': holiday_resid['std']
})
holiday_comparison.index = ['Non-Holiday', 'Holiday']
print(holiday_comparison.round(0))

print("\n6. TOP 10 HOLIDAYS BY AVERAGE RESULTS (Raw vs Detrended):")
print("-" * 80)
holiday_raw_means = df[df['is_holiday']].groupby('holiday_name')[target_var].mean()
holiday_resid_means = df[df['is_holiday']].groupby('holiday_name')['residuals'].mean()
holiday_combined = pd.DataFrame({
    'Raw': holiday_raw_means,
    'Residual': holiday_resid_means
}).sort_values('Raw', ascending=False).head(10)
print(holiday_combined.round(0))

print("\n7. SCHOOL PERIOD (Raw vs Detrended):")
print("-" * 80)
school_order = ['Fall Semester', 'Spring Semester', 'Summer Break']
df['school_period'] = pd.Categorical(df['school_period'], categories=school_order, ordered=True)
school_raw = df.groupby('school_period')[target_var].agg(['mean', 'std'])
school_resid = df.groupby('school_period')['residuals'].agg(['mean', 'std'])
school_comparison = pd.DataFrame({
    'Raw_Mean': school_raw['mean'],
    'Raw_Std': school_raw['std'],
    'Residual_Mean': school_resid['mean'],
    'Residual_Std': school_resid['std']
})
print(school_comparison.round(0))

# ============================================================================
# 7. ADDITIONAL COMPREHENSIVE ANALYSES
# ============================================================================
print("\n" + "=" * 80)
print("ADDITIONAL COMPREHENSIVE ANALYSES")
print("=" * 80)

# 7.1 Distribution analysis
print("\n7.1 Creating distribution analysis...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Raw data distribution
axes[0, 0].hist(df[target_var], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
axes[0, 0].axvline(df[target_var].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df[target_var].mean():,.0f}')
axes[0, 0].axvline(df[target_var].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {df[target_var].median():,.0f}')
axes[0, 0].set_title('Distribution of Reported Results (Raw)', fontweight='bold')
axes[0, 0].set_xlabel('Number of Reported Results')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Residuals distribution
axes[0, 1].hist(df['residuals'], bins=50, alpha=0.7, color='green', edgecolor='black')
axes[0, 1].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero')
axes[0, 1].axvline(df['residuals'].mean(), color='orange', linestyle='--', linewidth=2, label=f'Mean: {df["residuals"].mean():.0f}')
axes[0, 1].set_title('Distribution of Residuals (Detrended)', fontweight='bold')
axes[0, 1].set_xlabel('Residuals')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Box plot by day of week (raw)
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
df_day = df.copy()
df_day['day_name'] = pd.Categorical(df_day['day_name'], categories=day_order, ordered=True)
sns.boxplot(data=df_day, x='day_name', y=target_var, ax=axes[1, 0])
axes[1, 0].set_title('Distribution by Day of Week (Raw)', fontweight='bold')
axes[1, 0].set_xlabel('Day of Week')
axes[1, 0].set_ylabel('Number of Reported Results')
axes[1, 0].tick_params(axis='x', rotation=45)

# Box plot by day of week (detrended)
sns.boxplot(data=df_day, x='day_name', y='residuals', ax=axes[1, 1])
axes[1, 1].axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
axes[1, 1].set_title('Distribution by Day of Week (Detrended)', fontweight='bold')
axes[1, 1].set_xlabel('Day of Week')
axes[1, 1].set_ylabel('Residuals')
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('plots/07_distributions.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: plots/07_distributions.png")

# 7.2 Weekday vs Weekend detailed comparison
print("\n7.2 Creating weekday/weekend detailed analysis...")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Time series split
weekday_data = df[~df['is_weekend']]
weekend_data = df[df['is_weekend']]

axes[0].plot(weekday_data['Date'], weekday_data[target_var], label='Weekday', alpha=0.7, linewidth=1.5, color='blue')
axes[0].plot(weekend_data['Date'], weekend_data[target_var], label='Weekend', alpha=0.7, linewidth=1.5, color='orange')
axes[0].set_title('Time Series: Weekday vs Weekend (Raw)', fontweight='bold')
axes[0].set_xlabel('Date')
axes[0].set_ylabel('Number of Reported Results')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Violin plot
weekend_violin = pd.DataFrame({
    'Type': ['Weekday'] * len(weekday_data) + ['Weekend'] * len(weekend_data),
    'Value': list(weekday_data[target_var]) + list(weekend_data[target_var])
})
sns.violinplot(data=weekend_violin, x='Type', y='Value', ax=axes[1])
axes[1].set_title('Distribution: Weekday vs Weekend (Raw)', fontweight='bold')
axes[1].set_ylabel('Number of Reported Results')
axes[1].grid(axis='y', alpha=0.3)

# Detrended comparison
weekend_violin_resid = pd.DataFrame({
    'Type': ['Weekday'] * len(weekday_data) + ['Weekend'] * len(weekend_data),
    'Value': list(weekday_data['residuals']) + list(weekend_data['residuals'])
})
sns.violinplot(data=weekend_violin_resid, x='Type', y='Value', ax=axes[2])
axes[2].axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
axes[2].set_title('Distribution: Weekday vs Weekend (Detrended)', fontweight='bold')
axes[2].set_ylabel('Residuals')
axes[2].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('plots/08_weekday_weekend.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: plots/08_weekday_weekend.png")

# 7.3 Month-by-month trend analysis
print("\n7.3 Creating month-by-month analysis...")
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

month_order = ['January', 'February', 'March', 'April', 'May', 'June',
               'July', 'August', 'September', 'October', 'November', 'December']
df['month_name'] = df['Date'].dt.month_name()
df['month_name'] = pd.Categorical(df['month_name'], categories=month_order, ordered=True)

# Monthly averages with error bars (raw)
month_stats = df.groupby('month_name')[target_var].agg(['mean', 'std', 'count'])
month_stats['se'] = month_stats['std'] / np.sqrt(month_stats['count'])
axes[0].bar(range(len(month_stats)), month_stats['mean'], yerr=month_stats['se'], 
            color='steelblue', alpha=0.7, capsize=5, edgecolor='black')
axes[0].set_xticks(range(len(month_stats)))
axes[0].set_xticklabels(month_stats.index, rotation=45, ha='right')
axes[0].set_title('Monthly Averages with Standard Error (Raw)', fontweight='bold', fontsize=14)
axes[0].set_ylabel('Average Number of Reported Results')
axes[0].grid(axis='y', alpha=0.3)

# Monthly averages (detrended)
month_stats_resid = df.groupby('month_name')['residuals'].agg(['mean', 'std', 'count'])
month_stats_resid['se'] = month_stats_resid['std'] / np.sqrt(month_stats_resid['count'])
colors = ['green' if x > 0 else 'red' for x in month_stats_resid['mean'].values]
axes[1].bar(range(len(month_stats_resid)), month_stats_resid['mean'], 
            yerr=month_stats_resid['se'], color=colors, alpha=0.7, capsize=5, edgecolor='black')
axes[1].axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
axes[1].set_xticks(range(len(month_stats_resid)))
axes[1].set_xticklabels(month_stats_resid.index, rotation=45, ha='right')
axes[1].set_title('Monthly Averages with Standard Error (Detrended)', fontweight='bold', fontsize=14)
axes[1].set_ylabel('Average Residuals')
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('plots/09_monthly_detailed.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: plots/09_monthly_detailed.png")

# 7.4 Holiday impact analysis
print("\n7.4 Creating detailed holiday impact analysis...")
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Holiday vs surrounding days
holiday_impact = []
for holiday in df[df['is_holiday']]['holiday_name'].unique():
    if holiday:
        holiday_rows = df[df['holiday_name'] == holiday]
        if len(holiday_rows) > 0:
            holiday_date = holiday_rows.iloc[0]['Date']
            # Get surrounding days (±3 days)
            mask = (df['Date'] >= holiday_date - timedelta(days=3)) & \
                   (df['Date'] <= holiday_date + timedelta(days=3))
            surrounding = df[mask].copy()
            surrounding['days_from_holiday'] = (surrounding['Date'] - holiday_date).dt.days
            holiday_impact.append(surrounding[['days_from_holiday', target_var, 'residuals', 'holiday_name']])

if holiday_impact:
    holiday_df = pd.concat(holiday_impact, ignore_index=True)
    
    # Raw data
    holiday_means = holiday_df.groupby('days_from_holiday')[target_var].mean()
    axes[0, 0].plot(holiday_means.index, holiday_means.values, marker='o', linewidth=2, markersize=8, color='steelblue')
    axes[0, 0].axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Holiday')
    axes[0, 0].set_title('Holiday Impact: Days Around Holiday (Raw)', fontweight='bold')
    axes[0, 0].set_xlabel('Days from Holiday')
    axes[0, 0].set_ylabel('Average Number of Reported Results')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Detrended
    holiday_means_resid = holiday_df.groupby('days_from_holiday')['residuals'].mean()
    axes[0, 1].plot(holiday_means_resid.index, holiday_means_resid.values, marker='o', linewidth=2, markersize=8, color='green')
    axes[0, 1].axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Holiday')
    axes[0, 1].axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.3)
    axes[0, 1].set_title('Holiday Impact: Days Around Holiday (Detrended)', fontweight='bold')
    axes[0, 1].set_xlabel('Days from Holiday')
    axes[0, 1].set_ylabel('Average Residuals')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

# Top holidays comparison
top_holidays = df[df['is_holiday']].groupby('holiday_name')[target_var].mean().sort_values(ascending=False).head(8)
top_holidays_resid = df[df['is_holiday']].groupby('holiday_name')['residuals'].mean().reindex(top_holidays.index)

axes[1, 0].barh(range(len(top_holidays)), top_holidays.values, color='steelblue', alpha=0.7)
axes[1, 0].set_yticks(range(len(top_holidays)))
axes[1, 0].set_yticklabels(top_holidays.index)
axes[1, 0].set_title('Top Holidays by Average Results (Raw)', fontweight='bold')
axes[1, 0].set_xlabel('Average Number of Reported Results')
axes[1, 0].grid(axis='x', alpha=0.3)

colors = ['green' if x > 0 else 'red' for x in top_holidays_resid.values]
axes[1, 1].barh(range(len(top_holidays_resid)), top_holidays_resid.values, color=colors, alpha=0.7)
axes[1, 1].axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
axes[1, 1].set_yticks(range(len(top_holidays_resid)))
axes[1, 1].set_yticklabels(top_holidays_resid.index)
axes[1, 1].set_title('Top Holidays by Average Residuals (Detrended)', fontweight='bold')
axes[1, 1].set_xlabel('Average Residuals')
axes[1, 1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('plots/10_holiday_impact.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: plots/10_holiday_impact.png")

# 7.5 Pattern detection: day of month effects
print("\n7.5 Creating day of month pattern analysis...")
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Day of month (raw)
day_of_month = df.groupby('day')[target_var].agg(['mean', 'std', 'count'])
day_of_month['se'] = day_of_month['std'] / np.sqrt(day_of_month['count'])
axes[0].plot(day_of_month.index, day_of_month['mean'], marker='o', linewidth=2, markersize=4, color='steelblue')
axes[0].fill_between(day_of_month.index, 
                     day_of_month['mean'] - day_of_month['se'],
                     day_of_month['mean'] + day_of_month['se'],
                     alpha=0.3, color='steelblue')
axes[0].set_title('Average Results by Day of Month (Raw) with Confidence Intervals', fontweight='bold')
axes[0].set_xlabel('Day of Month')
axes[0].set_ylabel('Average Number of Reported Results')
axes[0].grid(True, alpha=0.3)

# Day of month (detrended)
day_of_month_resid = df.groupby('day')['residuals'].agg(['mean', 'std', 'count'])
day_of_month_resid['se'] = day_of_month_resid['std'] / np.sqrt(day_of_month_resid['count'])
axes[1].plot(day_of_month_resid.index, day_of_month_resid['mean'], marker='o', linewidth=2, markersize=4, color='green')
axes[1].fill_between(day_of_month_resid.index,
                    day_of_month_resid['mean'] - day_of_month_resid['se'],
                    day_of_month_resid['mean'] + day_of_month_resid['se'],
                    alpha=0.3, color='green')
axes[1].axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
axes[1].set_title('Average Residuals by Day of Month (Detrended) with Confidence Intervals', fontweight='bold')
axes[1].set_xlabel('Day of Month')
axes[1].set_ylabel('Average Residuals')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/11_day_of_month_patterns.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: plots/11_day_of_month_patterns.png")

# 7.6 Summary insights
print("\n" + "=" * 80)
print("KEY INSIGHTS AND PATTERNS")
print("=" * 80)

print("\n📊 RAW DATA INSIGHTS:")
print("-" * 80)
print(f"1. Overall Trend: Results range from {df[target_var].min():,.0f} to {df[target_var].max():,.0f}")
print(f"   Peak popularity: {df.loc[df[target_var].idxmax(), 'Date'].strftime('%B %d, %Y')} ({df[target_var].max():,.0f} results)")
print(f"   Lowest: {df.loc[df[target_var].idxmin(), 'Date'].strftime('%B %d, %Y')} ({df[target_var].min():,.0f} results)")

best_day = df.groupby('day_name')[target_var].mean().idxmax()
worst_day = df.groupby('day_name')[target_var].mean().idxmin()
print(f"\n2. Day of Week Effect (Raw):")
print(f"   Highest average: {best_day} ({df.groupby('day_name')[target_var].mean()[best_day]:,.0f})")
print(f"   Lowest average: {worst_day} ({df.groupby('day_name')[target_var].mean()[worst_day]:,.0f})")

best_month = df.groupby('month_name')[target_var].mean().idxmax()
worst_month = df.groupby('month_name')[target_var].mean().idxmin()
print(f"\n3. Month Effect (Raw):")
print(f"   Highest average: {best_month} ({df.groupby('month_name')[target_var].mean()[best_month]:,.0f})")
print(f"   Lowest average: {worst_month} ({df.groupby('month_name')[target_var].mean()[worst_month]:,.0f})")

weekend_diff = df[df['is_weekend']][target_var].mean() - df[~df['is_weekend']][target_var].mean()
print(f"\n4. Weekend Effect (Raw):")
print(f"   Weekend average: {df[df['is_weekend']][target_var].mean():,.0f}")
print(f"   Weekday average: {df[~df['is_weekend']][target_var].mean():,.0f}")
print(f"   Difference: {weekend_diff:,.0f} ({weekend_diff/df[~df['is_weekend']][target_var].mean()*100:.1f}%)")

print("\n📈 DETRENDED DATA INSIGHTS:")
print("-" * 80)
best_day_resid = df.groupby('day_name')['residuals'].mean().idxmax()
worst_day_resid = df.groupby('day_name')['residuals'].mean().idxmin()
print(f"\n1. Day of Week Effect (After Removing Trend):")
print(f"   Highest residual: {best_day_resid} ({df.groupby('day_name')['residuals'].mean()[best_day_resid]:,.0f})")
print(f"   Lowest residual: {worst_day_resid} ({df.groupby('day_name')['residuals'].mean()[worst_day_resid]:,.0f})")

weekend_diff_resid = df[df['is_weekend']]['residuals'].mean() - df[~df['is_weekend']]['residuals'].mean()
print(f"\n2. Weekend Effect (After Removing Trend):")
print(f"   Weekend residual: {df[df['is_weekend']]['residuals'].mean():,.0f}")
print(f"   Weekday residual: {df[~df['is_weekend']]['residuals'].mean():,.0f}")
print(f"   Difference: {weekend_diff_resid:,.0f}")

holiday_diff_resid = df[df['is_holiday']]['residuals'].mean() - df[~df['is_holiday']]['residuals'].mean()
print(f"\n3. Holiday Effect (After Removing Trend):")
print(f"   Holiday residual: {df[df['is_holiday']]['residuals'].mean():,.0f}")
print(f"   Non-holiday residual: {df[~df['is_holiday']]['residuals'].mean():,.0f}")
print(f"   Difference: {holiday_diff_resid:,.0f}")

print("\n💡 INTERPRETATION NOTES:")
print("-" * 80)
print("• Raw data shows the full story including Wordle's popularity trend")
print("• Detrended data isolates time variable effects from overall popularity")
print("• Both perspectives are valid - choose based on your research question")
print("• The 'trend' itself (Wordle's popularity over time) is part of the story")

# ============================================================================
# 8. SAVE ENHANCED DATASET
# ============================================================================
print("\n" + "=" * 80)
print("SAVING ENHANCED DATASET")
print("=" * 80)

output_file = 'data/2023_MCM_Problem_C_Data_enhanced.csv'
df.to_csv(output_file, index=False)
print(f"Enhanced dataset saved to: {output_file}")

original_cols = ['Date', 'Contest number', 'Word', 'Number of  reported results', 
                 'Number in hard mode', '1 try', '2 tries', '3 tries', '4 tries', 
                 '5 tries', '6 tries', '7 or more tries (X)']
new_features = [c for c in df.columns if c not in original_cols]
print(f"Added {len(new_features)} new time-related features")

print("\n" + "=" * 80)
print("COMPREHENSIVE EDA COMPLETE!")
print("=" * 80)
print(f"\n📊 Generated {14} visualization plots:")
print("   01_time_series.png - Time series with trend")
print("   02_day_of_week.png - Day of week analysis")
print("   03_month_season.png - Month and season analysis")
print("   04_holidays.png - Holiday analysis")
print("   05_day_of_month.png - Day of month patterns")
print("   06_correlation.png - Correlation heatmaps")
print("   07_distributions.png - Distribution analysis")
print("   08_weekday_weekend.png - Weekday/weekend detailed")
print("   09_monthly_detailed.png - Monthly detailed analysis")
print("   10_holiday_impact.png - Holiday impact analysis")
print("   11_day_of_month_patterns.png - Day of month patterns")
print("   12_momentum_autocorrelation.png - Momentum/autocorrelation analysis")
print("   13_school_days.png - School days vs non-school days analysis")
print("   14_day_of_week_school_days.png - Day of week by school days")
print(f"\n📁 All plots saved to: plots/")
print(f"📁 Enhanced dataset saved to: {output_file}")
print("\n✅ Analysis complete! Review plots and insights above.")

