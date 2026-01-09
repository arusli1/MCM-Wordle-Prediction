"""
Residual Pattern Analysis
=========================
Analyze residuals from cubic B-spline trend model to identify time-based patterns
that should be added to the base model.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ttest_ind, f_oneway, mannwhitneyu
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# Load enhanced dataset
print("Loading enhanced dataset with residuals...")
df = pd.read_csv('data/2023_MCM_Problem_C_Data_enhanced.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

target_var = 'Number of  reported results'

print(f"Data loaded: {len(df)} rows")
print(f"Residual statistics:")
print(df['residuals'].describe())
print(f"\nResidual autocorrelation (lag 1): {df['residuals'].autocorr(lag=1):.4f}")

# ============================================================================
# STATISTICAL TESTS FOR RESIDUAL PATTERNS
# ============================================================================
print("\n" + "=" * 80)
print("STATISTICAL ANALYSIS OF RESIDUAL PATTERNS")
print("=" * 80)

results = []

# 1. DAY OF WEEK
print("\n1. DAY OF WEEK:")
print("-" * 80)
day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_stats = df.groupby('day_name')['residuals'].agg(['mean', 'std', 'count']).reindex(day_names)
print(day_stats.round(2))

# ANOVA test
day_groups = [df[df['day_name'] == day]['residuals'].values for day in day_names]
f_stat, p_value = f_oneway(*day_groups)
results.append({
    'Variable': 'Day of Week',
    'Test': 'ANOVA',
    'Statistic': f_stat,
    'P-value': p_value,
    'Significant': p_value < 0.05,
    'Effect_Size': day_stats['mean'].max() - day_stats['mean'].min()
})

print(f"\nANOVA: F={f_stat:.2f}, p={p_value:.4f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'}")

# 2. WEEKEND VS WEEKDAY
print("\n2. WEEKEND VS WEEKDAY:")
print("-" * 80)
weekend_stats = df.groupby('is_weekend')['residuals'].agg(['mean', 'std', 'count'])
print(weekend_stats.round(2))

weekend_resid = df[df['is_weekend']]['residuals'].values
weekday_resid = df[~df['is_weekend']]['residuals'].values
t_stat, p_value = ttest_ind(weekday_resid, weekend_resid)
results.append({
    'Variable': 'Weekend vs Weekday',
    'Test': 'T-test',
    'Statistic': t_stat,
    'P-value': p_value,
    'Significant': p_value < 0.05,
    'Effect_Size': weekend_stats.loc[True, 'mean'] - weekend_stats.loc[False, 'mean']
})
print(f"T-test: t={t_stat:.2f}, p={p_value:.4f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'}")

# 3. MONTH
print("\n3. MONTH:")
print("-" * 80)
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
               'July', 'August', 'September', 'October', 'November', 'December']
month_stats = df.groupby('month_name')['residuals'].agg(['mean', 'std', 'count']).reindex(month_order)
print(month_stats.round(2))

month_groups = [df[df['month_name'] == month]['residuals'].values for month in month_order if month in df['month_name'].values]
f_stat, p_value = f_oneway(*month_groups)
results.append({
    'Variable': 'Month',
    'Test': 'ANOVA',
    'Statistic': f_stat,
    'P-value': p_value,
    'Significant': p_value < 0.05,
    'Effect_Size': month_stats['mean'].max() - month_stats['mean'].min()
})
print(f"\nANOVA: F={f_stat:.2f}, p={p_value:.4f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'}")

# 4. SEASON
print("\n4. SEASON:")
print("-" * 80)
season_order = ['Spring', 'Summer', 'Fall', 'Winter']
season_stats = df.groupby('season')['residuals'].agg(['mean', 'std', 'count']).reindex(season_order)
print(season_stats.round(2))

season_groups = [df[df['season'] == season]['residuals'].values for season in season_order]
f_stat, p_value = f_oneway(*season_groups)
results.append({
    'Variable': 'Season',
    'Test': 'ANOVA',
    'Statistic': f_stat,
    'P-value': p_value,
    'Significant': p_value < 0.05,
    'Effect_Size': season_stats['mean'].max() - season_stats['mean'].min()
})
print(f"\nANOVA: F={f_stat:.2f}, p={p_value:.4f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'}")

# 5. HOLIDAYS
print("\n5. HOLIDAY VS NON-HOLIDAY:")
print("-" * 80)
holiday_stats = df.groupby('is_holiday')['residuals'].agg(['mean', 'std', 'count'])
print(holiday_stats.round(2))

holiday_resid = df[df['is_holiday']]['residuals'].values
non_holiday_resid = df[~df['is_holiday']]['residuals'].values
t_stat, p_value = ttest_ind(non_holiday_resid, holiday_resid)
results.append({
    'Variable': 'Holiday vs Non-Holiday',
    'Test': 'T-test',
    'Statistic': t_stat,
    'P-value': p_value,
    'Significant': p_value < 0.05,
    'Effect_Size': holiday_stats.loc[True, 'mean'] - holiday_stats.loc[False, 'mean']
})
print(f"T-test: t={t_stat:.2f}, p={p_value:.4f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'}")

# 6. SCHOOL DAYS
print("\n6. SCHOOL DAY VS NON-SCHOOL DAY:")
print("-" * 80)
school_stats = df.groupby('is_school_day')['residuals'].agg(['mean', 'std', 'count'])
print(school_stats.round(2))

school_resid = df[df['is_school_day']]['residuals'].values
non_school_resid = df[~df['is_school_day']]['residuals'].values
t_stat, p_value = ttest_ind(school_resid, non_school_resid)
results.append({
    'Variable': 'School Day vs Non-School Day',
    'Test': 'T-test',
    'Statistic': t_stat,
    'P-value': p_value,
    'Significant': p_value < 0.05,
    'Effect_Size': school_stats.loc[True, 'mean'] - school_stats.loc[False, 'mean']
})
print(f"T-test: t={t_stat:.2f}, p={p_value:.4f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'}")

# 7. FIRST/LAST DAY OF MONTH
print("\n7. FIRST DAY OF MONTH:")
print("-" * 80)
first_day_stats = df.groupby('is_first_day')['residuals'].agg(['mean', 'std', 'count'])
print(first_day_stats.round(2))

first_day_resid = df[df['is_first_day']]['residuals'].values
not_first_day_resid = df[~df['is_first_day']]['residuals'].values
t_stat, p_value = ttest_ind(not_first_day_resid, first_day_resid)
results.append({
    'Variable': 'First Day of Month',
    'Test': 'T-test',
    'Statistic': t_stat,
    'P-value': p_value,
    'Significant': p_value < 0.05,
    'Effect_Size': first_day_stats.loc[True, 'mean'] - first_day_stats.loc[False, 'mean']
})
print(f"T-test: t={t_stat:.2f}, p={p_value:.4f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'}")

# 8. DAY OF MONTH (linear trend)
print("\n8. DAY OF MONTH (Linear Trend):")
print("-" * 80)
from scipy.stats import pearsonr
corr, p_value = pearsonr(df['day'], df['residuals'])
results.append({
    'Variable': 'Day of Month (linear)',
    'Test': 'Pearson Correlation',
    'Statistic': corr,
    'P-value': p_value,
    'Significant': p_value < 0.05,
    'Effect_Size': abs(corr)
})
print(f"Correlation: r={corr:.4f}, p={p_value:.4f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'}")

# 9. QUARTER
print("\n9. QUARTER:")
print("-" * 80)
quarter_stats = df.groupby('quarter')['residuals'].agg(['mean', 'std', 'count'])
print(quarter_stats.round(2))

quarter_groups = [df[df['quarter'] == q]['residuals'].values for q in [1, 2, 3, 4]]
f_stat, p_value = f_oneway(*quarter_groups)
results.append({
    'Variable': 'Quarter',
    'Test': 'ANOVA',
    'Statistic': f_stat,
    'P-value': p_value,
    'Significant': p_value < 0.05,
    'Effect_Size': quarter_stats['mean'].max() - quarter_stats['mean'].min()
})
print(f"\nANOVA: F={f_stat:.2f}, p={p_value:.4f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'}")

# Summary of significant patterns
print("\n" + "=" * 80)
print("SUMMARY OF SIGNIFICANT PATTERNS")
print("=" * 80)
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('P-value')
print("\nAll tests (sorted by p-value):")
print(results_df.to_string(index=False))

significant = results_df[results_df['Significant'] == True].copy()
if len(significant) > 0:
    print(f"\n✓ {len(significant)} SIGNIFICANT PATTERNS FOUND (p < 0.05):")
    print("-" * 80)
    for _, row in significant.iterrows():
        sig_marker = '***' if row['P-value'] < 0.001 else '**' if row['P-value'] < 0.01 else '*'
        print(f"{sig_marker} {row['Variable']:30s}: p={row['P-value']:.4f}, Effect Size={row['Effect_Size']:.2f}")
else:
    print("\n⚠ No significant patterns found (p < 0.05)")

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("\n" + "=" * 80)
print("CREATING VISUALIZATIONS")
print("=" * 80)

# Create plots directory if it doesn't exist
import os
os.makedirs('plots', exist_ok=True)

# 1. Residuals by Day of Week
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Day of week
ax = axes[0, 0]
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_means = df.groupby('day_name')['residuals'].mean().reindex(day_order)
day_stds = df.groupby('day_name')['residuals'].std().reindex(day_order)
x_pos = np.arange(len(day_order))
ax.bar(x_pos, day_means, yerr=day_stds, capsize=5, alpha=0.7, color='steelblue')
ax.set_xticks(x_pos)
ax.set_xticklabels([d[:3] for d in day_order], rotation=45)
ax.set_ylabel('Mean Residual')
ax.set_title('Residuals by Day of Week', fontsize=14, fontweight='bold')
ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax.grid(True, alpha=0.3)

# Weekend vs Weekday
ax = axes[0, 1]
weekend_means = df.groupby('is_weekend')['residuals'].mean()
weekend_stds = df.groupby('is_weekend')['residuals'].std()
x_pos = np.arange(2)
ax.bar(x_pos, weekend_means, yerr=weekend_stds, capsize=5, alpha=0.7, color=['steelblue', 'coral'])
ax.set_xticks(x_pos)
ax.set_xticklabels(['Weekday', 'Weekend'])
ax.set_ylabel('Mean Residual')
ax.set_title('Residuals: Weekday vs Weekend', fontsize=14, fontweight='bold')
ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax.grid(True, alpha=0.3)

# Month
ax = axes[1, 0]
month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
month_means = df.groupby('month_name')['residuals'].mean().reindex([m + 'uary' if m == 'Jan' else m + 'uary' if m == 'Feb' else m + 'ch' if m == 'Mar' else m + 'il' if m == 'Apr' else m + 'y' if m == 'May' else m + 'e' if m == 'Jun' else m + 'y' if m == 'Jul' else m + 'ust' if m == 'Aug' else m + 'tember' if m == 'Sep' else m + 'ober' if m == 'Oct' else m + 'ember' if m == 'Nov' else m + 'ember' for m in month_order])
month_means = df.groupby('month_name')['residuals'].mean()
month_stds = df.groupby('month_name')['residuals'].std()
x_pos = np.arange(len(month_means))
ax.bar(x_pos, month_means, yerr=month_stds, capsize=5, alpha=0.7, color='steelblue')
ax.set_xticks(x_pos)
ax.set_xticklabels([m[:3] for m in month_means.index], rotation=45)
ax.set_ylabel('Mean Residual')
ax.set_title('Residuals by Month', fontsize=14, fontweight='bold')
ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax.grid(True, alpha=0.3)

# Season
ax = axes[1, 1]
season_order = ['Spring', 'Summer', 'Fall', 'Winter']
season_means = df.groupby('season')['residuals'].mean().reindex(season_order)
season_stds = df.groupby('season')['residuals'].std().reindex(season_order)
x_pos = np.arange(len(season_order))
ax.bar(x_pos, season_means, yerr=season_stds, capsize=5, alpha=0.7, color=['green', 'orange', 'brown', 'lightblue'])
ax.set_xticks(x_pos)
ax.set_xticklabels(season_order)
ax.set_ylabel('Mean Residual')
ax.set_title('Residuals by Season', fontsize=14, fontweight='bold')
ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/15_residual_patterns_basic.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: plots/15_residual_patterns_basic.png")

# 2. Detailed residual patterns
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Holiday vs Non-Holiday
ax = axes[0, 0]
holiday_means = df.groupby('is_holiday')['residuals'].mean()
holiday_stds = df.groupby('is_holiday')['residuals'].std()
x_pos = np.arange(2)
ax.bar(x_pos, holiday_means, yerr=holiday_stds, capsize=5, alpha=0.7, color=['steelblue', 'coral'])
ax.set_xticks(x_pos)
ax.set_xticklabels(['Non-Holiday', 'Holiday'])
ax.set_ylabel('Mean Residual')
ax.set_title('Residuals: Holiday vs Non-Holiday', fontsize=14, fontweight='bold')
ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax.grid(True, alpha=0.3)

# School Day vs Non-School Day
ax = axes[0, 1]
school_means = df.groupby('is_school_day')['residuals'].mean()
school_stds = df.groupby('is_school_day')['residuals'].std()
x_pos = np.arange(2)
ax.bar(x_pos, school_means, yerr=school_stds, capsize=5, alpha=0.7, color=['steelblue', 'coral'])
ax.set_xticks(x_pos)
ax.set_xticklabels(['Non-School Day', 'School Day'])
ax.set_ylabel('Mean Residual')
ax.set_title('Residuals: School Day vs Non-School Day', fontsize=14, fontweight='bold')
ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax.grid(True, alpha=0.3)

# First Day of Month
ax = axes[1, 0]
first_day_means = df.groupby('is_first_day')['residuals'].mean()
first_day_stds = df.groupby('is_first_day')['residuals'].std()
x_pos = np.arange(2)
ax.bar(x_pos, first_day_means, yerr=first_day_stds, capsize=5, alpha=0.7, color=['steelblue', 'coral'])
ax.set_xticks(x_pos)
ax.set_xticklabels(['Not First Day', 'First Day'])
ax.set_ylabel('Mean Residual')
ax.set_title('Residuals: First Day of Month', fontsize=14, fontweight='bold')
ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax.grid(True, alpha=0.3)

# Day of Month scatter
ax = axes[1, 1]
ax.scatter(df['day'], df['residuals'], alpha=0.3, s=20)
# Add trend line
z = np.polyfit(df['day'], df['residuals'], 1)
p = np.poly1d(z)
ax.plot(df['day'].sort_values(), p(df['day'].sort_values()), "r--", alpha=0.8, linewidth=2, label=f'Trend (r={corr:.3f})')
ax.set_xlabel('Day of Month')
ax.set_ylabel('Residual')
ax.set_title('Residuals by Day of Month', fontsize=14, fontweight='bold')
ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/16_residual_patterns_detailed.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: plots/16_residual_patterns_detailed.png")

# 3. Time series of residuals with patterns highlighted
fig, axes = plt.subplots(3, 1, figsize=(16, 12))

# Raw residuals over time
ax = axes[0]
ax.plot(df['Date'], df['residuals'], 'b-', linewidth=1, alpha=0.7, label='Residuals')
ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax.set_ylabel('Residual')
ax.set_title('Residuals Over Time', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend()

# Residuals colored by weekend
ax = axes[1]
weekend_mask = df['is_weekend']
ax.scatter(df[~weekend_mask]['Date'], df[~weekend_mask]['residuals'], 
           alpha=0.5, s=10, color='steelblue', label='Weekday')
ax.scatter(df[weekend_mask]['Date'], df[weekend_mask]['residuals'], 
           alpha=0.5, s=10, color='coral', label='Weekend')
ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax.set_ylabel('Residual')
ax.set_title('Residuals: Weekday (blue) vs Weekend (orange)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend()

# Residuals colored by holiday
ax = axes[2]
holiday_mask = df['is_holiday']
ax.scatter(df[~holiday_mask]['Date'], df[~holiday_mask]['residuals'], 
           alpha=0.5, s=10, color='steelblue', label='Non-Holiday')
ax.scatter(df[holiday_mask]['Date'], df[holiday_mask]['residuals'], 
           alpha=0.5, s=10, color='coral', label='Holiday')
ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax.set_xlabel('Date')
ax.set_ylabel('Residual')
ax.set_title('Residuals: Non-Holiday (blue) vs Holiday (orange)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig('plots/17_residuals_time_series.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: plots/17_residuals_time_series.png")

# ============================================================================
# RECOMMENDATIONS
# ============================================================================
print("\n" + "=" * 80)
print("RECOMMENDATIONS FOR MODEL ADDITIONS")
print("=" * 80)

print("\nBased on the residual analysis, consider adding the following to your base cubic B-spline model:\n")

recommendations = []

for _, row in significant.iterrows():
    var = row['Variable']
    effect = row['Effect_Size']
    p_val = row['P-value']
    
    if 'Day of Week' in var:
        recommendations.append({
            'Variable': 'Day of Week',
            'Type': 'Categorical (7 levels)',
            'Effect Size': f"{effect:.2f}",
            'Recommendation': 'Add day_of_week as categorical variable (one-hot encoded or as day indicators)',
            'Priority': 'HIGH' if abs(effect) > 5000 else 'MEDIUM'
        })
    elif 'Weekend' in var:
        recommendations.append({
            'Variable': 'is_weekend',
            'Type': 'Binary',
            'Effect Size': f"{effect:.2f}",
            'Recommendation': 'Add is_weekend binary indicator',
            'Priority': 'HIGH' if abs(effect) > 5000 else 'MEDIUM'
        })
    elif var == 'Month':
        recommendations.append({
            'Variable': 'Month',
            'Type': 'Categorical (12 levels)',
            'Effect Size': f"{effect:.2f}",
            'Recommendation': 'Add month as categorical variable or use seasonal indicators',
            'Priority': 'MEDIUM'
        })
    elif var == 'Season':
        recommendations.append({
            'Variable': 'Season',
            'Type': 'Categorical (4 levels)',
            'Effect Size': f"{effect:.2f}",
            'Recommendation': 'Add season as categorical variable',
            'Priority': 'MEDIUM' if abs(effect) > 3000 else 'LOW'
        })
    elif 'Holiday' in var:
        recommendations.append({
            'Variable': 'is_holiday',
            'Type': 'Binary',
            'Effect Size': f"{effect:.2f}",
            'Recommendation': 'Add is_holiday binary indicator',
            'Priority': 'HIGH' if abs(effect) > 5000 else 'MEDIUM'
        })
    elif 'School' in var:
        recommendations.append({
            'Variable': 'is_school_day',
            'Type': 'Binary',
            'Effect Size': f"{effect:.2f}",
            'Recommendation': 'Add is_school_day binary indicator',
            'Priority': 'MEDIUM' if abs(effect) > 3000 else 'LOW'
        })
    elif 'First Day' in var:
        recommendations.append({
            'Variable': 'is_first_day',
            'Type': 'Binary',
            'Effect Size': f"{effect:.2f}",
            'Recommendation': 'Add is_first_day binary indicator',
            'Priority': 'LOW'
        })
    elif 'Day of Month' in var:
        recommendations.append({
            'Variable': 'day (of month)',
            'Type': 'Continuous or Categorical',
            'Effect Size': f"{effect:.4f}",
            'Recommendation': 'Add day of month (linear or as bins) if effect is meaningful',
            'Priority': 'LOW'
        })

if recommendations:
    rec_df = pd.DataFrame(recommendations)
    # Remove duplicates
    rec_df = rec_df.drop_duplicates(subset=['Variable'])
    print(rec_df.to_string(index=False))
    
    print("\n\nSuggested model structure:")
    print("-" * 80)
    print("Base Model: Cubic B-Spline trend (40 knots)")
    print("\nAdd to residuals:")
    high_priority = rec_df[rec_df['Priority'] == 'HIGH']
    medium_priority = rec_df[rec_df['Priority'] == 'MEDIUM']
    
    if len(high_priority) > 0:
        print("\nHIGH PRIORITY (add these first):")
        for _, row in high_priority.iterrows():
            print(f"  - {row['Variable']}: {row['Recommendation']}")
    
    if len(medium_priority) > 0:
        print("\nMEDIUM PRIORITY:")
        for _, row in medium_priority.iterrows():
            print(f"  - {row['Variable']}: {row['Recommendation']}")
else:
    print("\n⚠ No significant patterns found. The cubic B-spline trend model appears to capture")
    print("  most of the time-based variation. Consider checking for non-linear interactions")
    print("  or other features not captured in this analysis.")

print("\n" + "=" * 80)
print("Analysis complete!")
print("=" * 80)

