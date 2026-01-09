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
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
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
# 3. DETREND DATA (Remove overall time trend)
# ============================================================================
print("\n" + "=" * 80)
print("DETRENDING DATA")
print("=" * 80)

target_var = 'Number of  reported results'

# Fit polynomial trend (degree 3 for flexibility)
print("\nFitting trend model...")

# Create numeric time variable
X_trend = df['days_since_start'].values.reshape(-1, 1)
y = df[target_var].values

# Fit polynomial trend
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X_trend)
trend_model = LinearRegression()
trend_model.fit(X_poly, y)
df['trend'] = trend_model.predict(X_poly)

# Calculate residuals (detrended data)
df['residuals'] = df[target_var] - df['trend']
df['residuals_pct'] = (df['residuals'] / df['trend'] * 100)  # Percentage deviation from trend

# Calculate R² for trend
trend_r2 = r2_score(y, df['trend'])
print(f"Trend model R²: {trend_r2:.4f}")
print(f"Trend explains {trend_r2*100:.2f}% of variance in reported results")

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

# 5.1 Time series plot with trend
print("\n1. Creating time series plot with trend...")
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Raw data with trend
axes[0].plot(df['Date'], df[target_var], linewidth=1.5, alpha=0.7, label='Actual', color='steelblue')
axes[0].plot(df['Date'], df['trend'], linewidth=2, alpha=0.8, label='Trend', color='red', linestyle='--')
axes[0].set_xlabel('Date', fontsize=12)
axes[0].set_ylabel('Number of Reported Results', fontsize=12)
axes[0].set_title('Wordle Results Over Time (with Trend)', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# Highlight holidays
holiday_dates = df[df['is_holiday']]['Date']
holiday_values = df[df['is_holiday']][target_var]
axes[0].scatter(holiday_dates, holiday_values, color='orange', s=50, alpha=0.6, 
                label='Holidays', zorder=5)

# Detrended data (residuals)
axes[1].plot(df['Date'], df['residuals'], linewidth=1.5, alpha=0.7, color='green')
axes[1].axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Zero line')
axes[1].set_xlabel('Date', fontsize=12)
axes[1].set_ylabel('Residuals (Actual - Trend)', fontsize=12)
axes[1].set_title('Detrended Data (Residuals)', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].legend()

# Highlight holidays on residuals
holiday_residuals = df[df['is_holiday']]['residuals']
axes[1].scatter(holiday_dates, holiday_residuals, color='orange', s=50, alpha=0.6, 
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
axes[0].bar(range(len(day_means_raw)), day_means_raw.values, color='steelblue', alpha=0.7)
axes[0].set_xticks(range(len(day_means_raw)))
axes[0].set_xticklabels(day_means_raw.index, rotation=45)
axes[0].set_title('Average Results by Day of Week (Raw)', fontweight='bold')
axes[0].set_ylabel('Average Number of Reported Results')
axes[0].grid(axis='y', alpha=0.3)

# Detrended data
day_means_resid = df_day.groupby('day_name')['residuals'].mean().reindex(day_order)
colors = ['green' if x > 0 else 'red' for x in day_means_resid.values]
axes[1].bar(range(len(day_means_resid)), day_means_resid.values, color=colors, alpha=0.7)
axes[1].axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
axes[1].set_xticks(range(len(day_means_resid)))
axes[1].set_xticklabels(day_means_resid.index, rotation=45)
axes[1].set_title('Average Residuals by Day of Week (Detrended)', fontweight='bold')
axes[1].set_ylabel('Average Residuals')
axes[1].grid(axis='y', alpha=0.3)

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
axes[0].bar(['Non-Holiday', 'Holiday'], holiday_means_raw.values, color='indianred', alpha=0.7)
axes[0].set_title('Average Results: Holiday vs Non-Holiday (Raw)', fontweight='bold')
axes[0].set_ylabel('Average Number of Reported Results')
axes[0].grid(axis='y', alpha=0.3)

# Holiday vs non-holiday - Detrended
holiday_means_resid = df.groupby('is_holiday')['residuals'].mean()
colors = ['green' if x > 0 else 'red' for x in holiday_means_resid.values]
axes[1].bar(['Non-Holiday', 'Holiday'], holiday_means_resid.values, color=colors, alpha=0.7)
axes[1].axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
axes[1].set_title('Average Residuals: Holiday vs Non-Holiday (Detrended)', fontweight='bold')
axes[1].set_ylabel('Average Residuals')
axes[1].grid(axis='y', alpha=0.3)

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
# 6. STATISTICAL SUMMARY (Raw vs Detrended)
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
# 7. SAVE ENHANCED DATASET
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
print("ANALYSIS COMPLETE!")
print("=" * 80)
print(f"\nAll plots saved to: plots/")
print(f"Enhanced dataset saved to: {output_file}")

