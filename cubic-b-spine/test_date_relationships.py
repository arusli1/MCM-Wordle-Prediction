"""
Test file to analyze relationships between date/time factors and 
the number of reported scores over time.

Factors analyzed:
- Day of week
- Time of month (day number, week of month)
- Time of year (month, day of year, season)
- School year (academic calendar)
- Holidays and special dates
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from scipy import stats
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.optimize import curve_fit
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Try to import holidays library, if not available, we'll define common holidays manually
try:
    import holidays
    HAS_HOLIDAYS_LIB = True
except ImportError:
    HAS_HOLIDAYS_LIB = False
    print("Note: 'holidays' library not found. Using manual holiday definitions.")

# Set style for better plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# Load the data
print("Loading data...")
df = pd.read_csv('data/2023_MCM_Problem_C_Data.csv')

# Clean column names (remove extra spaces)
df.columns = df.columns.str.strip()

# Convert Date to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Extract the target variable
target_col = 'Number of  reported results'
if target_col not in df.columns:
    # Try alternative names
    for col in df.columns:
        if 'reported' in col.lower() and 'result' in col.lower():
            target_col = col
            break

df['reported_scores'] = df[target_col]

print(f"Data loaded: {len(df)} records")
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
print(f"\nFirst few rows:")
print(df[['Date', 'reported_scores']].head())
print(f"\nBasic statistics:")
print(df['reported_scores'].describe())

# ============================================================================
# EXTRACT DATE/TIME FEATURES
# ============================================================================

print("\n" + "="*80)
print("EXTRACTING DATE/TIME FEATURES")
print("="*80)

# Day of week (0=Monday, 6=Sunday)
df['day_of_week'] = df['Date'].dt.dayofweek
df['day_name'] = df['Date'].dt.day_name()

# Day of month (1-31)
df['day_of_month'] = df['Date'].dt.day

# Week of month (1-5)
df['week_of_month'] = ((df['Date'].dt.day - 1) // 7) + 1

# Month (1-12)
df['month'] = df['Date'].dt.month
df['month_name'] = df['Date'].dt.month_name()

# Day of year (1-365/366)
df['day_of_year'] = df['Date'].dt.dayofyear

# Quarter
df['quarter'] = df['Date'].dt.quarter

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

# School year (Academic calendar: typically Aug/Sept to May/June)
# Assuming school year starts in August/September
def get_school_year(date):
    month = date.month
    year = date.year
    if month >= 8:  # August onwards belongs to next school year
        return f"{year}-{year+1}"
    else:  # Jan-July belongs to previous school year
        return f"{year-1}-{year}"

df['school_year'] = df['Date'].apply(get_school_year)

# Is weekend?
df['is_weekend'] = df['day_of_week'].isin([5, 6])

# Is start of month (first 3 days)?
df['is_month_start'] = df['day_of_month'] <= 3

# Is end of month (last 3 days)?
df['is_month_end'] = df['day_of_month'] >= df['Date'].dt.days_in_month - 2

# Is Monday? (often lower engagement after weekends)
df['is_monday'] = df['day_of_week'] == 0

# Is Friday? (often higher engagement before weekends)
df['is_friday'] = df['day_of_week'] == 4

# ============================================================================
# HOLIDAYS AND SPECIAL DATES
# ============================================================================

print("\nIdentifying holidays and special dates...")

# Create holiday flags
df['is_holiday'] = False
df['holiday_name'] = ''

# Define major US holidays manually
holiday_dates = {}

# New Year's Day
for year in range(2021, 2024):
    holiday_dates[f"{year}-01-01"] = "New Year's Day"

# Independence Day
for year in range(2021, 2024):
    holiday_dates[f"{year}-07-04"] = "Independence Day"

# Christmas
for year in range(2021, 2024):
    holiday_dates[f"{year}-12-25"] = "Christmas"

# Thanksgiving (4th Thursday of November)
for year in range(2021, 2024):
    nov_1 = datetime(year, 11, 1)
    first_thursday = nov_1 + timedelta(days=(3 - nov_1.weekday()) % 7)
    thanksgiving = first_thursday + timedelta(days=21)
    holiday_dates[thanksgiving.strftime("%Y-%m-%d")] = "Thanksgiving"

# Memorial Day (last Monday of May)
for year in range(2021, 2024):
    may_31 = datetime(year, 5, 31)
    memorial_day = may_31 - timedelta(days=may_31.weekday())
    holiday_dates[memorial_day.strftime("%Y-%m-%d")] = "Memorial Day"

# Labor Day (first Monday of September)
for year in range(2021, 2024):
    sep_1 = datetime(year, 9, 1)
    labor_day = sep_1 + timedelta(days=(7 - sep_1.weekday()) % 7)
    holiday_dates[labor_day.strftime("%Y-%m-%d")] = "Labor Day"

# Valentine's Day
for year in range(2021, 2024):
    holiday_dates[f"{year}-02-14"] = "Valentine's Day"

# Halloween
for year in range(2021, 2024):
    holiday_dates[f"{year}-10-31"] = "Halloween"

# Easter (simplified - using common dates)
easter_dates = {
    "2021-04-04": "Easter",
    "2022-04-17": "Easter",
    "2023-04-09": "Easter"
}
holiday_dates.update(easter_dates)

# If holidays library is available, use it for more comprehensive holiday detection
if HAS_HOLIDAYS_LIB:
    us_holidays = holidays.UnitedStates(years=[2021, 2022, 2023])
    for date_str, holiday_name in us_holidays.items():
        holiday_dates[date_str.strftime("%Y-%m-%d")] = holiday_name

# Mark holidays in dataframe
for date_str, holiday_name in holiday_dates.items():
    date_obj = pd.to_datetime(date_str)
    mask = df['Date'].dt.date == date_obj.date()
    df.loc[mask, 'is_holiday'] = True
    df.loc[mask, 'holiday_name'] = holiday_name

# Days around holidays (holiday effect)
df['days_from_holiday'] = 999  # Initialize with large number
for idx, row in df.iterrows():
    date = row['Date']
    min_dist = 999
    for holiday_date_str in holiday_dates.keys():
        holiday_date = pd.to_datetime(holiday_date_str)
        dist = abs((date - holiday_date).days)
        if dist < min_dist:
            min_dist = dist
    df.at[idx, 'days_from_holiday'] = min_dist

df['near_holiday'] = df['days_from_holiday'] <= 2  # Within 2 days of a holiday

# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("STATISTICAL ANALYSIS")
print("="*80)

# 1. Day of Week Analysis
print("\n1. DAY OF WEEK ANALYSIS")
print("-" * 40)
dow_stats = df.groupby('day_name')['reported_scores'].agg(['mean', 'median', 'std', 'count'])
dow_stats = dow_stats.reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
print(dow_stats)

# 2. Month Analysis
print("\n2. MONTH ANALYSIS")
print("-" * 40)
month_stats = df.groupby('month_name')['reported_scores'].agg(['mean', 'median', 'std', 'count'])
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
               'July', 'August', 'September', 'October', 'November', 'December']
month_stats = month_stats.reindex([m for m in month_order if m in month_stats.index])
print(month_stats)

# 3. Season Analysis
print("\n3. SEASON ANALYSIS")
print("-" * 40)
season_stats = df.groupby('season')['reported_scores'].agg(['mean', 'median', 'std', 'count'])
season_order = ['Spring', 'Summer', 'Fall', 'Winter']
season_stats = season_stats.reindex([s for s in season_order if s in season_stats.index])
print(season_stats)

# 4. School Year Analysis
print("\n4. SCHOOL YEAR ANALYSIS")
print("-" * 40)
school_year_stats = df.groupby('school_year')['reported_scores'].agg(['mean', 'median', 'std', 'count'])
print(school_year_stats)

# 5. Weekend vs Weekday
print("\n5. WEEKEND VS WEEKDAY")
print("-" * 40)
weekend_stats = df.groupby('is_weekend')['reported_scores'].agg(['mean', 'median', 'std', 'count'])
print(weekend_stats)

# 6. Holiday Analysis
print("\n6. HOLIDAY ANALYSIS")
print("-" * 40)
holiday_stats = df.groupby('is_holiday')['reported_scores'].agg(['mean', 'median', 'std', 'count'])
print(holiday_stats)

if df['is_holiday'].sum() > 0:
    print("\nHoliday-specific statistics:")
    holiday_detail = df[df['is_holiday']].groupby('holiday_name')['reported_scores'].agg(['mean', 'count'])
    print(holiday_detail)

# 7. Month Start/End Analysis
print("\n7. MONTH START/END ANALYSIS")
print("-" * 40)
month_start_stats = df.groupby('is_month_start')['reported_scores'].agg(['mean', 'median', 'std', 'count'])
print("Month Start (first 3 days):")
print(month_start_stats)

month_end_stats = df.groupby('is_month_end')['reported_scores'].agg(['mean', 'median', 'std', 'count'])
print("\nMonth End (last 3 days):")
print(month_end_stats)

# 8. Correlation Analysis
print("\n8. CORRELATION ANALYSIS")
print("-" * 40)
numeric_features = ['day_of_week', 'day_of_month', 'week_of_month', 'month', 
                   'day_of_year', 'quarter', 'is_weekend', 'is_holiday', 
                   'is_month_start', 'is_month_end', 'days_from_holiday']
correlations = df[numeric_features + ['reported_scores']].corr()['reported_scores'].sort_values(ascending=False)
print("\nCorrelations with reported_scores:")
print(correlations)

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("\n" + "="*80)
print("CREATING VISUALIZATIONS")
print("="*80)

# Create a figure with multiple subplots
fig = plt.figure(figsize=(20, 24))

# 1. Time series plot
ax1 = plt.subplot(6, 2, 1)
ax1.plot(df['Date'], df['reported_scores'], linewidth=1.5, alpha=0.7)
ax1.set_title('Reported Scores Over Time', fontsize=14, fontweight='bold')
ax1.set_xlabel('Date')
ax1.set_ylabel('Number of Reported Scores')
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='x', rotation=45)

# 2. Day of week box plot
ax2 = plt.subplot(6, 2, 2)
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
df_plot = df[df['day_name'].isin(day_order)]
sns.boxplot(data=df_plot, x='day_name', y='reported_scores', order=day_order, ax=ax2)
ax2.set_title('Reported Scores by Day of Week', fontsize=14, fontweight='bold')
ax2.set_xlabel('Day of Week')
ax2.set_ylabel('Number of Reported Scores')
ax2.tick_params(axis='x', rotation=45)

# 3. Month box plot
ax3 = plt.subplot(6, 2, 3)
month_order_plot = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
df['month_abbr'] = df['Date'].dt.strftime('%b')
sns.boxplot(data=df, x='month_abbr', y='reported_scores', order=month_order_plot, ax=ax3)
ax3.set_title('Reported Scores by Month', fontsize=14, fontweight='bold')
ax3.set_xlabel('Month')
ax3.set_ylabel('Number of Reported Scores')
ax3.tick_params(axis='x', rotation=45)

# 4. Season box plot
ax4 = plt.subplot(6, 2, 4)
season_order_plot = ['Spring', 'Summer', 'Fall', 'Winter']
sns.boxplot(data=df, x='season', y='reported_scores', order=season_order_plot, ax=ax4)
ax4.set_title('Reported Scores by Season', fontsize=14, fontweight='bold')
ax4.set_xlabel('Season')
ax4.set_ylabel('Number of Reported Scores')

# 5. Weekend vs Weekday
ax5 = plt.subplot(6, 2, 5)
sns.boxplot(data=df, x='is_weekend', y='reported_scores', ax=ax5)
ax5.set_title('Weekend vs Weekday', fontsize=14, fontweight='bold')
ax5.set_xlabel('Is Weekend')
ax5.set_ylabel('Number of Reported Scores')
ax5.set_xticklabels(['Weekday', 'Weekend'])

# 6. Holiday vs Non-Holiday
ax6 = plt.subplot(6, 2, 6)
sns.boxplot(data=df, x='is_holiday', y='reported_scores', ax=ax6)
ax6.set_title('Holiday vs Non-Holiday', fontsize=14, fontweight='bold')
ax6.set_xlabel('Is Holiday')
ax6.set_ylabel('Number of Reported Scores')
ax6.set_xticklabels(['Non-Holiday', 'Holiday'])

# 7. Day of month trend
ax7 = plt.subplot(6, 2, 7)
day_of_month_avg = df.groupby('day_of_month')['reported_scores'].mean()
ax7.plot(day_of_month_avg.index, day_of_month_avg.values, marker='o', linewidth=2, markersize=4)
ax7.set_title('Average Reported Scores by Day of Month', fontsize=14, fontweight='bold')
ax7.set_xlabel('Day of Month')
ax7.set_ylabel('Average Number of Reported Scores')
ax7.grid(True, alpha=0.3)

# 8. Week of month
ax8 = plt.subplot(6, 2, 8)
week_of_month_avg = df.groupby('week_of_month')['reported_scores'].mean()
sns.barplot(x=week_of_month_avg.index, y=week_of_month_avg.values, ax=ax8)
ax8.set_title('Average Reported Scores by Week of Month', fontsize=14, fontweight='bold')
ax8.set_xlabel('Week of Month')
ax8.set_ylabel('Average Number of Reported Scores')

# 9. School year comparison
ax9 = plt.subplot(6, 2, 9)
school_year_avg = df.groupby('school_year')['reported_scores'].mean().sort_index()
sns.barplot(x=school_year_avg.index, y=school_year_avg.values, ax=ax9)
ax9.set_title('Average Reported Scores by School Year', fontsize=14, fontweight='bold')
ax9.set_xlabel('School Year')
ax9.set_ylabel('Average Number of Reported Scores')
ax9.tick_params(axis='x', rotation=45)

# 10. Days from holiday
ax10 = plt.subplot(6, 2, 10)
df_near_holiday = df[df['days_from_holiday'] <= 7].copy()
if len(df_near_holiday) > 0:
    days_from_holiday_avg = df_near_holiday.groupby('days_from_holiday')['reported_scores'].mean()
    ax10.plot(days_from_holiday_avg.index, days_from_holiday_avg.values, marker='o', linewidth=2, markersize=6)
    ax10.set_title('Average Reported Scores by Days from Holiday', fontsize=14, fontweight='bold')
    ax10.set_xlabel('Days from Nearest Holiday')
    ax10.set_ylabel('Average Number of Reported Scores')
    ax10.grid(True, alpha=0.3)
    ax10.axvline(x=0, color='r', linestyle='--', alpha=0.5, label='Holiday')
    ax10.legend()

# 11. Month start/end comparison
ax11 = plt.subplot(6, 2, 11)
month_timing = pd.DataFrame({
    'Month Start': [df[df['is_month_start']]['reported_scores'].mean()],
    'Month Middle': [df[~(df['is_month_start'] | df['is_month_end'])]['reported_scores'].mean()],
    'Month End': [df[df['is_month_end']]['reported_scores'].mean()]
})
month_timing.T.plot(kind='bar', ax=ax11, legend=False)
ax11.set_title('Average Reported Scores: Month Start vs Middle vs End', fontsize=14, fontweight='bold')
ax11.set_xlabel('Month Timing')
ax11.set_ylabel('Average Number of Reported Scores')
ax11.tick_params(axis='x', rotation=0)

# 12. Correlation heatmap
ax12 = plt.subplot(6, 2, 12)
corr_matrix = df[numeric_features + ['reported_scores']].corr()
sns.heatmap(corr_matrix[['reported_scores']].sort_values('reported_scores', ascending=False), 
            annot=True, fmt='.3f', cmap='coolwarm', center=0, ax=ax12, cbar_kws={'label': 'Correlation'})
ax12.set_title('Correlation with Reported Scores', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('date_relationships_analysis.png', dpi=300, bbox_inches='tight')
print("\nVisualizations saved to 'date_relationships_analysis.png'")

# ============================================================================
# SUMMARY INSIGHTS
# ============================================================================

print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)

# Find the day of week with highest average
best_dow = dow_stats['mean'].idxmax()
worst_dow = dow_stats['mean'].idxmin()
print(f"\n1. Day of Week:")
print(f"   - Highest average: {best_dow} ({dow_stats.loc[best_dow, 'mean']:.0f} scores)")
print(f"   - Lowest average: {worst_dow} ({dow_stats.loc[worst_dow, 'mean']:.0f} scores)")
print(f"   - Difference: {dow_stats.loc[best_dow, 'mean'] - dow_stats.loc[worst_dow, 'mean']:.0f} scores")

# Find the month with highest average
best_month = month_stats['mean'].idxmax()
worst_month = month_stats['mean'].idxmin()
print(f"\n2. Month:")
print(f"   - Highest average: {best_month} ({month_stats.loc[best_month, 'mean']:.0f} scores)")
print(f"   - Lowest average: {worst_month} ({month_stats.loc[worst_month, 'mean']:.0f} scores)")

# Weekend effect
weekend_avg = df[df['is_weekend']]['reported_scores'].mean()
weekday_avg = df[~df['is_weekend']]['reported_scores'].mean()
print(f"\n3. Weekend Effect:")
print(f"   - Weekday average: {weekday_avg:.0f} scores")
print(f"   - Weekend average: {weekend_avg:.0f} scores")
print(f"   - Difference: {abs(weekday_avg - weekend_avg):.0f} scores ({'Weekend' if weekend_avg > weekday_avg else 'Weekday'} is higher)")

# Holiday effect
if df['is_holiday'].sum() > 0:
    holiday_avg = df[df['is_holiday']]['reported_scores'].mean()
    non_holiday_avg = df[~df['is_holiday']]['reported_scores'].mean()
    print(f"\n4. Holiday Effect:")
    print(f"   - Non-holiday average: {non_holiday_avg:.0f} scores")
    print(f"   - Holiday average: {holiday_avg:.0f} scores")
    print(f"   - Difference: {abs(holiday_avg - non_holiday_avg):.0f} scores ({'Holiday' if holiday_avg > non_holiday_avg else 'Non-holiday'} is higher)")

# Season effect
best_season = season_stats['mean'].idxmax()
worst_season = season_stats['mean'].idxmin()
print(f"\n5. Season:")
print(f"   - Highest average: {best_season} ({season_stats.loc[best_season, 'mean']:.0f} scores)")
print(f"   - Lowest average: {worst_season} ({season_stats.loc[worst_season, 'mean']:.0f} scores)")

# Top correlations
print(f"\n6. Top Correlations:")
top_corr = correlations.abs().sort_values(ascending=False).head(6)
for feature, corr in top_corr.items():
    if feature != 'reported_scores':
        print(f"   - {feature}: {correlations[feature]:.3f}")

# ============================================================================
# DETRENDING ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("DETRENDING ANALYSIS")
print("="*80)

print("\nDETRENDING METHOD EXPLANATION:")
print("-" * 80)
print("""
To remove the strong time trend from the data, we use Holt's Exponential Smoothing:
1. Holt's method (double exponential smoothing) captures both level and trend components
2. It's designed for time series with trends and adapts to changes over time
3. The method uses exponential smoothing with two parameters:
   - Alpha (α): controls the level smoothing
   - Beta (β): controls the trend smoothing
4. Calculate residuals = original_data - fitted_trend
5. The residuals represent the variation around the trend
6. Analyze day-of-week effects on these detrended residuals

This allows us to see if day-of-week patterns exist independently of the 
overall popularity decline over time. Holt's method is better than polynomial
regression for time series as it adapts to the data's natural trend.
""")

# Sort by date to ensure proper ordering
df_sorted = df.sort_values('Date').copy()

# Set Date as index for time series analysis
df_sorted_indexed = df_sorted.set_index('Date')

# Fit Holt's exponential smoothing model
print("\nFitting Holt's exponential smoothing model...")
y = df_sorted_indexed['reported_scores']

# Try different parameter combinations for Holt's method
# We'll use additive trend since the data shows a declining trend
best_r2 = -np.inf
best_model = None
best_predictions = None
best_params = None

# Try different combinations of smoothing parameters
alphas = [0.1, 0.3, 0.5, 0.7, 0.9]
betas = [0.1, 0.3, 0.5, 0.7, 0.9]

print("Testing different smoothing parameter combinations...")
for alpha in alphas:
    for beta in betas:
        try:
            # Fit Holt's method with additive trend
            model = ExponentialSmoothing(
                y, 
                trend='add', 
                seasonal=None,
                initialization_method='estimated'
            )
            fitted_model = model.fit(smoothing_level=alpha, smoothing_trend=beta, optimized=False)
            predictions = fitted_model.fittedvalues
            
            # Calculate R²
            r2 = r2_score(y, predictions)
            
            if r2 > best_r2:
                best_r2 = r2
                best_model = fitted_model
                best_predictions = predictions
                best_params = (alpha, beta)
        except:
            continue

# If optimization didn't work well, try with optimized parameters
if best_r2 < 0.5:
    print("Trying with optimized parameters...")
    try:
        model_opt = ExponentialSmoothing(
            y, 
            trend='add', 
            seasonal=None,
            initialization_method='estimated'
        )
        fitted_model_opt = model_opt.fit(optimized=True)
        predictions_opt = fitted_model_opt.fittedvalues
        r2_opt = r2_score(y, predictions_opt)
        
        if r2_opt > best_r2:
            best_r2 = r2_opt
            best_model = fitted_model_opt
            best_predictions = predictions_opt
            best_params = "optimized"
            print(f"Using optimized parameters (R² = {r2_opt:.4f})")
    except:
        pass

if best_params == "optimized":
    print(f"Holt's method with optimized parameters (R² = {best_r2:.4f})")
    if hasattr(best_model, 'params'):
        print(f"  Optimized α = {best_model.params.get('smoothing_level', 'N/A'):.4f}")
        print(f"  Optimized β = {best_model.params.get('smoothing_trend', 'N/A'):.4f}")
else:
    print(f"Holt's method with α={best_params[0]:.2f}, β={best_params[1]:.2f} (R² = {best_r2:.4f})")

# Calculate additional metrics
mse = mean_squared_error(y, best_predictions)
rmse = np.sqrt(mse)
print(f"  RMSE: {rmse:.2f}")
print(f"  Mean Absolute Error: {np.mean(np.abs(y - best_predictions)):.2f}")

# Calculate detrended residuals
# Convert predictions back to series aligned with df_sorted
df_sorted['trend'] = best_predictions.values
df_sorted['detrended_scores'] = df_sorted['reported_scores'] - df_sorted['trend']

# Also calculate percentage deviation from trend
df_sorted['pct_deviation'] = (df_sorted['detrended_scores'] / df_sorted['trend']) * 100

print(f"\nDetrended data statistics:")
print(f"  Mean residual: {df_sorted['detrended_scores'].mean():.2f}")
print(f"  Std deviation: {df_sorted['detrended_scores'].std():.2f}")
print(f"  Min residual: {df_sorted['detrended_scores'].min():.2f}")
print(f"  Max residual: {df_sorted['detrended_scores'].max():.2f}")

# Analyze day of week on detrended data
print("\n" + "-"*80)
print("DAY OF WEEK ANALYSIS ON DETRENDED DATA")
print("-"*80)

dow_detrended_stats = df_sorted.groupby('day_name')['detrended_scores'].agg(['mean', 'median', 'std', 'count'])
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
dow_detrended_stats = dow_detrended_stats.reindex(day_order)
print("\nDetrended scores by day of week:")
print(dow_detrended_stats)

# Statistical test for day-of-week effect
# One-way ANOVA to test if day-of-week means are significantly different
dow_groups = [df_sorted[df_sorted['day_name'] == day]['detrended_scores'].values 
              for day in day_order if day in df_sorted['day_name'].values]
f_stat, p_value = stats.f_oneway(*dow_groups)

print(f"\nStatistical Test (ANOVA):")
print(f"  F-statistic: {f_stat:.4f}")
print(f"  p-value: {p_value:.6f}")
if p_value < 0.05:
    print(f"  Result: Day-of-week effect is statistically significant (p < 0.05)")
else:
    print(f"  Result: Day-of-week effect is NOT statistically significant (p >= 0.05)")

# Correlation between day of week and detrended scores
dow_correlation = df_sorted['day_of_week'].corr(df_sorted['detrended_scores'])
print(f"\nCorrelation between day_of_week and detrended_scores: {dow_correlation:.4f}")

# Weekend vs weekday on detrended data
print("\n" + "-"*80)
print("WEEKEND VS WEEKDAY ON DETRENDED DATA")
print("-"*80)
weekend_detrended = df_sorted.groupby('is_weekend')['detrended_scores'].agg(['mean', 'median', 'std', 'count'])
print(weekend_detrended)

# T-test for weekend vs weekday
weekday_residuals = df_sorted[~df_sorted['is_weekend']]['detrended_scores']
weekend_residuals = df_sorted[df_sorted['is_weekend']]['detrended_scores']
t_stat, t_pvalue = stats.ttest_ind(weekday_residuals, weekend_residuals)

print(f"\nT-test (Weekday vs Weekend):")
print(f"  T-statistic: {t_stat:.4f}")
print(f"  p-value: {t_pvalue:.6f}")
if t_pvalue < 0.05:
    print(f"  Result: Weekend effect is statistically significant (p < 0.05)")
else:
    print(f"  Result: Weekend effect is NOT statistically significant (p >= 0.05)")

# Visualizations for detrended analysis
print("\n" + "-"*80)
print("CREATING DETRENDING VISUALIZATIONS")
print("-"*80)

fig2 = plt.figure(figsize=(20, 16))

# 1. Original data with trend line
ax1 = plt.subplot(4, 2, 1)
ax1.plot(df_sorted['Date'], df_sorted['reported_scores'], 'b-', linewidth=1.5, alpha=0.7, label='Original Data')
ax1.plot(df_sorted['Date'], df_sorted['trend'], 'r--', linewidth=2, label="Holt's Trend")
ax1.set_title('Original Data with Fitted Trend (Holt\'s Method)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Date')
ax1.set_ylabel('Number of Reported Scores')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='x', rotation=45)

# 2. Detrended residuals over time
ax2 = plt.subplot(4, 2, 2)
ax2.plot(df_sorted['Date'], df_sorted['detrended_scores'], 'g-', linewidth=1.5, alpha=0.7)
ax2.axhline(y=0, color='r', linestyle='--', linewidth=1, alpha=0.5)
ax2.set_title('Detrended Residuals Over Time', fontsize=14, fontweight='bold')
ax2.set_xlabel('Date')
ax2.set_ylabel('Residual (Actual - Trend)')
ax2.grid(True, alpha=0.3)
ax2.tick_params(axis='x', rotation=45)

# 3. Day of week box plot - Original data
ax3 = plt.subplot(4, 2, 3)
sns.boxplot(data=df_sorted, x='day_name', y='reported_scores', order=day_order, ax=ax3)
ax3.set_title('Original Scores by Day of Week', fontsize=14, fontweight='bold')
ax3.set_xlabel('Day of Week')
ax3.set_ylabel('Number of Reported Scores')
ax3.tick_params(axis='x', rotation=45)

# 4. Day of week box plot - Detrended data
ax4 = plt.subplot(4, 2, 4)
sns.boxplot(data=df_sorted, x='day_name', y='detrended_scores', order=day_order, ax=ax4)
ax4.axhline(y=0, color='r', linestyle='--', linewidth=1, alpha=0.5)
ax4.set_title('Detrended Residuals by Day of Week', fontsize=14, fontweight='bold')
ax4.set_xlabel('Day of Week')
ax4.set_ylabel('Residual (Actual - Trend)')
ax4.tick_params(axis='x', rotation=45)

# 5. Weekend vs Weekday - Original
ax5 = plt.subplot(4, 2, 5)
sns.boxplot(data=df_sorted, x='is_weekend', y='reported_scores', ax=ax5)
ax5.set_title('Original Scores: Weekend vs Weekday', fontsize=14, fontweight='bold')
ax5.set_xlabel('Is Weekend')
ax5.set_ylabel('Number of Reported Scores')
ax5.set_xticklabels(['Weekday', 'Weekend'])

# 6. Weekend vs Weekday - Detrended
ax6 = plt.subplot(4, 2, 6)
sns.boxplot(data=df_sorted, x='is_weekend', y='detrended_scores', ax=ax6)
ax6.axhline(y=0, color='r', linestyle='--', linewidth=1, alpha=0.5)
ax6.set_title('Detrended Residuals: Weekend vs Weekday', fontsize=14, fontweight='bold')
ax6.set_xlabel('Is Weekend')
ax6.set_ylabel('Residual (Actual - Trend)')
ax6.set_xticklabels(['Weekday', 'Weekend'])

# 7. Average detrended scores by day of week
ax7 = plt.subplot(4, 2, 7)
dow_avg_detrended = df_sorted.groupby('day_name')['detrended_scores'].mean().reindex(day_order)
colors = ['red' if x < 0 else 'green' for x in dow_avg_detrended.values]
bars = ax7.bar(range(len(dow_avg_detrended)), dow_avg_detrended.values, color=colors, alpha=0.7)
ax7.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax7.set_title('Average Detrended Residuals by Day of Week', fontsize=14, fontweight='bold')
ax7.set_xlabel('Day of Week')
ax7.set_ylabel('Average Residual')
ax7.set_xticks(range(len(dow_avg_detrended)))
ax7.set_xticklabels(dow_avg_detrended.index, rotation=45)
ax7.grid(True, alpha=0.3, axis='y')
# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, dow_avg_detrended.values)):
    ax7.text(bar.get_x() + bar.get_width()/2, val + (50 if val > 0 else -200), 
             f'{val:.0f}', ha='center', va='bottom' if val > 0 else 'top', fontsize=9)

# 8. Comparison: Original vs Detrended day-of-week means
ax8 = plt.subplot(4, 2, 8)
dow_avg_original = df_sorted.groupby('day_name')['reported_scores'].mean().reindex(day_order)
dow_avg_detrended_norm = df_sorted.groupby('day_name')['detrended_scores'].mean().reindex(day_order)
x = np.arange(len(day_order))
width = 0.35
ax8.bar(x - width/2, dow_avg_original.values, width, label='Original', alpha=0.7)
ax8.bar(x + width/2, dow_avg_detrended_norm.values, width, label='Detrended', alpha=0.7)
ax8.set_title('Day-of-Week Effect: Original vs Detrended', fontsize=14, fontweight='bold')
ax8.set_xlabel('Day of Week')
ax8.set_ylabel('Average Value')
ax8.set_xticks(x)
ax8.set_xticklabels(day_order, rotation=45)
ax8.legend()
ax8.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('detrended_day_of_week_analysis.png', dpi=300, bbox_inches='tight')
print("\nDetrending visualizations saved to 'detrended_day_of_week_analysis.png'")

# Summary of detrended analysis
print("\n" + "="*80)
print("DETRENDING ANALYSIS SUMMARY")
print("="*80)

best_dow_detrended = dow_detrended_stats['mean'].idxmax()
worst_dow_detrended = dow_detrended_stats['mean'].idxmin()
print(f"\nDay of Week on Detrended Data:")
print(f"  - Highest average residual: {best_dow_detrended} ({dow_detrended_stats.loc[best_dow_detrended, 'mean']:.0f})")
print(f"  - Lowest average residual: {worst_dow_detrended} ({dow_detrended_stats.loc[worst_dow_detrended, 'mean']:.0f})")
print(f"  - Range: {dow_detrended_stats.loc[best_dow_detrended, 'mean'] - dow_detrended_stats.loc[worst_dow_detrended, 'mean']:.0f}")

weekday_detrended_avg = df_sorted[~df_sorted['is_weekend']]['detrended_scores'].mean()
weekend_detrended_avg = df_sorted[df_sorted['is_weekend']]['detrended_scores'].mean()
print(f"\nWeekend vs Weekday on Detrended Data:")
print(f"  - Weekday average residual: {weekday_detrended_avg:.0f}")
print(f"  - Weekend average residual: {weekend_detrended_avg:.0f}")
print(f"  - Difference: {abs(weekday_detrended_avg - weekend_detrended_avg):.0f}")

# ============================================================================
# DAY-OF-WEEK AVERAGED DATASET ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("DAY-OF-WEEK AVERAGED DATASET ANALYSIS")
print("="*80)

print("\nMETHOD EXPLANATION:")
print("-" * 80)
print("""
1. Calculate the average reported scores for each day of week (across all dates)
2. Create a new dataset where each date's score is replaced with its day-of-week average
3. Apply Holt's method to detrend this averaged dataset
4. Calculate variance of detrended residuals by day of week

This helps us understand the variance in the day-of-week pattern after removing
the overall time trend.
""")

# Calculate day-of-week averages
print("\nCalculating day-of-week averages...")
dow_averages = df_sorted.groupby('day_name')['reported_scores'].mean()
print("\nDay-of-week averages:")
for day in day_order:
    if day in dow_averages.index:
        print(f"  {day}: {dow_averages[day]:.2f}")

# Create new dataset with day-of-week averaged scores
df_dow_avg = df_sorted.copy()
df_dow_avg['dow_averaged_scores'] = df_dow_avg['day_name'].map(dow_averages)

print(f"\nCreated new dataset with {len(df_dow_avg)} records")
print(f"Original scores range: {df_dow_avg['reported_scores'].min():.0f} to {df_dow_avg['reported_scores'].max():.0f}")
print(f"Day-of-week averaged scores range: {df_dow_avg['dow_averaged_scores'].min():.0f} to {df_dow_avg['dow_averaged_scores'].max():.0f}")

# Set Date as index for time series analysis
df_dow_avg_indexed = df_dow_avg.set_index('Date').sort_index()

# Fit Holt's exponential smoothing model on the averaged dataset
print("\nFitting Holt's method on day-of-week averaged dataset...")
y_avg = df_dow_avg_indexed['dow_averaged_scores']

# Try different parameter combinations
best_r2_avg = -np.inf
best_model_avg = None
best_predictions_avg = None
best_params_avg = None

alphas = [0.1, 0.3, 0.5, 0.7, 0.9]
betas = [0.1, 0.3, 0.5, 0.7, 0.9]

print("Testing different smoothing parameter combinations...")
for alpha in alphas:
    for beta in betas:
        try:
            model = ExponentialSmoothing(
                y_avg, 
                trend='add', 
                seasonal=None,
                initialization_method='estimated'
            )
            fitted_model = model.fit(smoothing_level=alpha, smoothing_trend=beta, optimized=False)
            predictions = fitted_model.fittedvalues
            r2 = r2_score(y_avg, predictions)
            
            if r2 > best_r2_avg:
                best_r2_avg = r2
                best_model_avg = fitted_model
                best_predictions_avg = predictions
                best_params_avg = (alpha, beta)
        except:
            continue

# Try with optimized parameters
if best_r2_avg < 0.5:
    print("Trying with optimized parameters...")
    try:
        model_opt = ExponentialSmoothing(
            y_avg, 
            trend='add', 
            seasonal=None,
            initialization_method='estimated'
        )
        fitted_model_opt = model_opt.fit(optimized=True)
        predictions_opt = fitted_model_opt.fittedvalues
        r2_opt = r2_score(y_avg, predictions_opt)
        
        if r2_opt > best_r2_avg:
            best_r2_avg = r2_opt
            best_model_avg = fitted_model_opt
            best_predictions_avg = predictions_opt
            best_params_avg = "optimized"
    except:
        pass

if best_params_avg == "optimized":
    print(f"Holt's method with optimized parameters (R² = {best_r2_avg:.4f})")
else:
    print(f"Holt's method with α={best_params_avg[0]:.2f}, β={best_params_avg[1]:.2f} (R² = {best_r2_avg:.4f})")

# Calculate detrended residuals for averaged dataset
df_dow_avg['dow_avg_trend'] = best_predictions_avg.values
df_dow_avg['dow_avg_detrended'] = df_dow_avg['dow_averaged_scores'] - df_dow_avg['dow_avg_trend']

print(f"\nDetrended averaged data statistics:")
print(f"  Mean residual: {df_dow_avg['dow_avg_detrended'].mean():.2f}")
print(f"  Std deviation: {df_dow_avg['dow_avg_detrended'].std():.2f}")

# Calculate variance of detrended residuals by day of week
print("\n" + "-"*80)
print("VARIANCE OF DETRENDED RESIDUALS BY DAY OF WEEK")
print("-"*80)

dow_variance_stats = df_dow_avg.groupby('day_name')['dow_avg_detrended'].agg(['var', 'std', 'mean', 'count'])
dow_variance_stats = dow_variance_stats.reindex(day_order)
dow_variance_stats.columns = ['Variance', 'Std_Dev', 'Mean', 'Count']

print("\nVariance of detrended residuals by day of week:")
print(dow_variance_stats)

# Also calculate variance for the original detrended residuals (from first analysis)
print("\n" + "-"*80)
print("VARIANCE OF ORIGINAL DETRENDED RESIDUALS BY DAY OF WEEK")
print("-"*80)

original_dow_variance = df_sorted.groupby('day_name')['detrended_scores'].agg(['var', 'std', 'mean', 'count'])
original_dow_variance = original_dow_variance.reindex(day_order)
original_dow_variance.columns = ['Variance', 'Std_Dev', 'Mean', 'Count']

print("\nVariance of original detrended residuals by day of week:")
print(original_dow_variance)

# Comparison
print("\n" + "-"*80)
print("COMPARISON: AVERAGED vs ORIGINAL DETRENDED RESIDUALS")
print("-"*80)

comparison_df = pd.DataFrame({
    'Day_of_Week': day_order,
    'Avg_Dataset_Variance': dow_variance_stats['Variance'].values,
    'Original_Dataset_Variance': original_dow_variance['Variance'].values,
    'Avg_Dataset_StdDev': dow_variance_stats['Std_Dev'].values,
    'Original_Dataset_StdDev': original_dow_variance['Std_Dev'].values
})
comparison_df['Variance_Reduction'] = ((original_dow_variance['Variance'].values - dow_variance_stats['Variance'].values) / original_dow_variance['Variance'].values) * 100

print("\nVariance comparison:")
print(comparison_df.to_string(index=False))

# Save the averaged dataset
df_dow_avg_output = df_dow_avg[['Date', 'day_name', 'reported_scores', 'dow_averaged_scores', 
                                 'dow_avg_trend', 'dow_avg_detrended']].copy()
df_dow_avg_output.to_csv('day_of_week_averaged_dataset.csv', index=False)
print("\nDay-of-week averaged dataset saved to 'day_of_week_averaged_dataset.csv'")

# Visualization
print("\n" + "-"*80)
print("CREATING VISUALIZATIONS FOR AVERAGED DATASET")
print("-"*80)

fig3 = plt.figure(figsize=(20, 12))

# 1. Original vs Day-of-week averaged scores
ax1 = plt.subplot(3, 2, 1)
ax1.plot(df_dow_avg['Date'], df_dow_avg['reported_scores'], 'b-', linewidth=1, alpha=0.5, label='Original Scores')
ax1.plot(df_dow_avg['Date'], df_dow_avg['dow_averaged_scores'], 'r-', linewidth=2, label='Day-of-Week Averaged')
ax1.set_title('Original vs Day-of-Week Averaged Scores', fontsize=14, fontweight='bold')
ax1.set_xlabel('Date')
ax1.set_ylabel('Number of Reported Scores')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='x', rotation=45)

# 2. Averaged scores with Holt's trend
ax2 = plt.subplot(3, 2, 2)
ax2.plot(df_dow_avg['Date'], df_dow_avg['dow_averaged_scores'], 'b-', linewidth=1.5, alpha=0.7, label='Averaged Scores')
ax2.plot(df_dow_avg['Date'], df_dow_avg['dow_avg_trend'], 'r--', linewidth=2, label="Holt's Trend")
ax2.set_title('Day-of-Week Averaged Scores with Fitted Trend', fontsize=14, fontweight='bold')
ax2.set_xlabel('Date')
ax2.set_ylabel('Number of Reported Scores')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.tick_params(axis='x', rotation=45)

# 3. Detrended residuals from averaged dataset
ax3 = plt.subplot(3, 2, 3)
ax3.plot(df_dow_avg['Date'], df_dow_avg['dow_avg_detrended'], 'g-', linewidth=1.5, alpha=0.7)
ax3.axhline(y=0, color='r', linestyle='--', linewidth=1, alpha=0.5)
ax3.set_title('Detrended Residuals (Averaged Dataset)', fontsize=14, fontweight='bold')
ax3.set_xlabel('Date')
ax3.set_ylabel('Residual (Averaged - Trend)')
ax3.grid(True, alpha=0.3)
ax3.tick_params(axis='x', rotation=45)

# 4. Variance by day of week (averaged dataset)
ax4 = plt.subplot(3, 2, 4)
bars = ax4.bar(range(len(dow_variance_stats)), dow_variance_stats['Variance'].values, alpha=0.7)
ax4.set_title('Variance of Detrended Residuals by Day of Week\n(Averaged Dataset)', fontsize=14, fontweight='bold')
ax4.set_xlabel('Day of Week')
ax4.set_ylabel('Variance')
ax4.set_xticks(range(len(dow_variance_stats)))
ax4.set_xticklabels(dow_variance_stats.index, rotation=45)
ax4.grid(True, alpha=0.3, axis='y')
for i, (bar, val) in enumerate(zip(bars, dow_variance_stats['Variance'].values)):
    ax4.text(bar.get_x() + bar.get_width()/2, val + max(dow_variance_stats['Variance'].values) * 0.01, 
             f'{val:.0f}', ha='center', va='bottom', fontsize=9)

# 5. Variance comparison: Averaged vs Original
ax5 = plt.subplot(3, 2, 5)
x = np.arange(len(day_order))
width = 0.35
bars1 = ax5.bar(x - width/2, dow_variance_stats['Variance'].values, width, label='Averaged Dataset', alpha=0.7)
bars2 = ax5.bar(x + width/2, original_dow_variance['Variance'].values, width, label='Original Dataset', alpha=0.7)
ax5.set_title('Variance Comparison: Averaged vs Original', fontsize=14, fontweight='bold')
ax5.set_xlabel('Day of Week')
ax5.set_ylabel('Variance')
ax5.set_xticks(x)
ax5.set_xticklabels(day_order, rotation=45)
ax5.legend()
ax5.grid(True, alpha=0.3, axis='y')

# 6. Standard deviation comparison
ax6 = plt.subplot(3, 2, 6)
bars1 = ax6.bar(x - width/2, dow_variance_stats['Std_Dev'].values, width, label='Averaged Dataset', alpha=0.7)
bars2 = ax6.bar(x + width/2, original_dow_variance['Std_Dev'].values, width, label='Original Dataset', alpha=0.7)
ax6.set_title('Standard Deviation Comparison: Averaged vs Original', fontsize=14, fontweight='bold')
ax6.set_xlabel('Day of Week')
ax6.set_ylabel('Standard Deviation')
ax6.set_xticks(x)
ax6.set_xticklabels(day_order, rotation=45)
ax6.legend()
ax6.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('day_of_week_averaged_analysis.png', dpi=300, bbox_inches='tight')
print("\nVisualizations saved to 'day_of_week_averaged_analysis.png'")

# Summary
print("\n" + "="*80)
print("DAY-OF-WEEK AVERAGED DATASET SUMMARY")
print("="*80)

print(f"\nOverall Statistics:")
print(f"  Average variance (averaged dataset): {dow_variance_stats['Variance'].mean():.2f}")
print(f"  Average variance (original dataset): {original_dow_variance['Variance'].mean():.2f}")
print(f"  Variance reduction: {((original_dow_variance['Variance'].mean() - dow_variance_stats['Variance'].mean()) / original_dow_variance['Variance'].mean() * 100):.2f}%")

max_var_day = dow_variance_stats['Variance'].idxmax()
min_var_day = dow_variance_stats['Variance'].idxmin()
print(f"\nDay with highest variance (averaged): {max_var_day} ({dow_variance_stats.loc[max_var_day, 'Variance']:.2f})")
print(f"Day with lowest variance (averaged): {min_var_day} ({dow_variance_stats.loc[min_var_day, 'Variance']:.2f})")

# ============================================================================
# SMOOTH PIECEWISE FITTING
# ============================================================================

print("\n" + "="*80)
print("SMOOTH PIECEWISE FITTING")
print("="*80)

print("\nMETHOD EXPLANATION:")
print("-" * 80)
print("""
Fitting smooth piecewise functions to the data using splines:
1. Cubic Splines: Piecewise cubic polynomials with smooth transitions
2. B-Splines: Basis splines for flexible piecewise fitting
3. UnivariateSpline: Smoothing spline that balances fit and smoothness
4. Piecewise linear with smooth transitions

Splines provide smooth transitions between segments, making them ideal for
capturing trends with multiple phases or breakpoints.
""")

# Prepare data for spline fitting
df_spline = df.sort_values('Date').copy()
df_spline['days_since_start'] = (df_spline['Date'] - df_spline['Date'].min()).dt.days
X_spline = df_spline['days_since_start'].values
y_spline = df_spline['reported_scores'].values

print(f"\nData prepared: {len(df_spline)} points")
print(f"Time range: {X_spline.min()} to {X_spline.max()} days")

# 1. Cubic Spline Interpolation
print("\n" + "-"*80)
print("1. CUBIC SPLINE INTERPOLATION")
print("-"*80)

# Try different numbers of knots for piecewise fitting
n_knots_options = [5, 7, 10, 15, 20]
best_spline_r2 = -np.inf
best_spline_model = None
best_spline_pred = None
best_n_knots = None

for n_knots in n_knots_options:
    try:
        # Create evenly spaced knots
        knot_positions = np.linspace(X_spline.min(), X_spline.max(), n_knots)
        
        # Use B-spline basis with cubic splines
        spline_transformer = SplineTransformer(n_knots=n_knots, degree=3, include_bias=True)
        X_spline_basis = spline_transformer.fit_transform(X_spline.reshape(-1, 1))
        
        # Fit linear regression on spline basis
        spline_model = LinearRegression()
        spline_model.fit(X_spline_basis, y_spline)
        spline_pred = spline_model.predict(X_spline_basis)
        
        r2 = r2_score(y_spline, spline_pred)
        
        if r2 > best_spline_r2:
            best_spline_r2 = r2
            best_spline_model = spline_model
            best_spline_transformer = spline_transformer
            best_spline_pred = spline_pred
            best_n_knots = n_knots
    except:
        continue

print(f"Best cubic B-spline: {best_n_knots} knots (R² = {best_spline_r2:.4f})")
mse_spline = mean_squared_error(y_spline, best_spline_pred)
print(f"  RMSE: {np.sqrt(mse_spline):.2f}")
print(f"  MAE: {np.mean(np.abs(y_spline - best_spline_pred)):.2f}")

# 2. UnivariateSpline (Smoothing Spline)
print("\n" + "-"*80)
print("2. UNIVARIATESPLINE (SMOOTHING SPLINE)")
print("-"*80)

# Try different smoothing factors
smoothing_factors = [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]
best_uspline_r2 = -np.inf
best_uspline_model = None
best_uspline_pred = None
best_smoothing = None

for s in smoothing_factors:
    try:
        uspline = UnivariateSpline(X_spline, y_spline, s=s, k=3)  # k=3 for cubic
        uspline_pred = uspline(X_spline)
        r2 = r2_score(y_spline, uspline_pred)
        
        if r2 > best_uspline_r2:
            best_uspline_r2 = r2
            best_uspline_model = uspline
            best_uspline_pred = uspline_pred
            best_smoothing = s
    except:
        continue

print(f"Best UnivariateSpline: smoothing factor = {best_smoothing} (R² = {best_uspline_r2:.4f})")
mse_uspline = mean_squared_error(y_spline, best_uspline_pred)
print(f"  RMSE: {np.sqrt(mse_uspline):.2f}")
print(f"  MAE: {np.mean(np.abs(y_spline - best_uspline_pred)):.2f}")

# 3. Piecewise Linear with Smooth Transitions (using multiple linear segments)
print("\n" + "-"*80)
print("3. PIECEWISE LINEAR WITH SMOOTH TRANSITIONS")
print("-"*80)

# Define a piecewise function with smooth transitions
def piecewise_linear_smooth(x, *params):
    """
    Piecewise linear function with smooth transitions using sigmoid functions
    """
    n_segments = len(params) // 2
    result = params[0] + params[1] * x  # First segment
    
    for i in range(1, n_segments):
        # Breakpoint position
        breakpoint = params[2*i]
        # Slope change
        slope_change = params[2*i + 1]
        # Smooth transition using sigmoid
        transition = 1 / (1 + np.exp(-10 * (x - breakpoint)))
        result += slope_change * (x - breakpoint) * transition
    
    return result

# Try different numbers of segments
n_segments_options = [2, 3, 4, 5]
best_piecewise_r2 = -np.inf
best_piecewise_params = None
best_piecewise_pred = None
best_n_segments = None

for n_segments in n_segments_options:
    try:
        # Initial parameter guess
        breakpoints = np.linspace(X_spline.min() + 50, X_spline.max() - 50, n_segments - 1)
        initial_params = [y_spline[0], (y_spline[-1] - y_spline[0]) / (X_spline.max() - X_spline.min())]
        for bp in breakpoints:
            initial_params.extend([bp, 0])
        
        # Fit the piecewise function
        popt, _ = curve_fit(piecewise_linear_smooth, X_spline, y_spline, 
                           p0=initial_params, maxfev=5000)
        piecewise_pred = piecewise_linear_smooth(X_spline, *popt)
        r2 = r2_score(y_spline, piecewise_pred)
        
        if r2 > best_piecewise_r2:
            best_piecewise_r2 = r2
            best_piecewise_params = popt
            best_piecewise_pred = piecewise_pred
            best_n_segments = n_segments
    except:
        continue

if best_piecewise_r2 > -np.inf:
    print(f"Best piecewise linear: {best_n_segments} segments (R² = {best_piecewise_r2:.4f})")
    mse_piecewise = mean_squared_error(y_spline, best_piecewise_pred)
    print(f"  RMSE: {np.sqrt(mse_piecewise):.2f}")
    print(f"  MAE: {np.mean(np.abs(y_spline - best_piecewise_pred)):.2f}")
else:
    print("Piecewise linear fitting failed")

# Store results
df_spline['cubic_spline_fit'] = best_spline_pred
df_spline['uspline_fit'] = best_uspline_pred
if best_piecewise_r2 > -np.inf:
    df_spline['piecewise_fit'] = best_piecewise_pred

# Calculate residuals
df_spline['cubic_spline_residual'] = df_spline['reported_scores'] - df_spline['cubic_spline_fit']
df_spline['uspline_residual'] = df_spline['reported_scores'] - df_spline['uspline_fit']
if best_piecewise_r2 > -np.inf:
    df_spline['piecewise_residual'] = df_spline['reported_scores'] - df_spline['piecewise_fit']

# Comparison
print("\n" + "-"*80)
print("MODEL COMPARISON")
print("-"*80)

comparison = pd.DataFrame({
    'Model': ['Cubic B-Spline', 'UnivariateSpline', 'Piecewise Linear'],
    'R²': [best_spline_r2, best_uspline_r2, best_piecewise_r2 if best_piecewise_r2 > -np.inf else np.nan],
    'RMSE': [
        np.sqrt(mean_squared_error(y_spline, best_spline_pred)),
        np.sqrt(mean_squared_error(y_spline, best_uspline_pred)),
        np.sqrt(mean_squared_error(y_spline, best_piecewise_pred)) if best_piecewise_r2 > -np.inf else np.nan
    ],
    'MAE': [
        np.mean(np.abs(y_spline - best_spline_pred)),
        np.mean(np.abs(y_spline - best_uspline_pred)),
        np.mean(np.abs(y_spline - best_piecewise_pred)) if best_piecewise_r2 > -np.inf else np.nan
    ]
})
print(comparison.to_string(index=False))

# Visualizations
print("\n" + "-"*80)
print("CREATING SMOOTH PIECEWISE FITTING VISUALIZATIONS")
print("-"*80)

fig4 = plt.figure(figsize=(20, 16))

# 1. All models comparison
ax1 = plt.subplot(3, 2, 1)
ax1.plot(df_spline['Date'], df_spline['reported_scores'], 'ko', markersize=2, alpha=0.3, label='Data')
ax1.plot(df_spline['Date'], df_spline['cubic_spline_fit'], 'b-', linewidth=2, label=f'Cubic B-Spline (R²={best_spline_r2:.4f})')
ax1.plot(df_spline['Date'], df_spline['uspline_fit'], 'r-', linewidth=2, label=f'UnivariateSpline (R²={best_uspline_r2:.4f})')
if best_piecewise_r2 > -np.inf:
    ax1.plot(df_spline['Date'], df_spline['piecewise_fit'], 'g-', linewidth=2, label=f'Piecewise Linear (R²={best_piecewise_r2:.4f})')
ax1.set_title('Smooth Piecewise Fits Comparison', fontsize=14, fontweight='bold')
ax1.set_xlabel('Date')
ax1.set_ylabel('Number of Reported Scores')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='x', rotation=45)

# 2. Cubic B-Spline residuals
ax2 = plt.subplot(3, 2, 2)
ax2.plot(df_spline['Date'], df_spline['cubic_spline_residual'], 'b-', linewidth=1, alpha=0.7)
ax2.axhline(y=0, color='r', linestyle='--', linewidth=1, alpha=0.5)
ax2.set_title(f'Cubic B-Spline Residuals (R²={best_spline_r2:.4f})', fontsize=14, fontweight='bold')
ax2.set_xlabel('Date')
ax2.set_ylabel('Residual')
ax2.grid(True, alpha=0.3)
ax2.tick_params(axis='x', rotation=45)

# 3. UnivariateSpline residuals
ax3 = plt.subplot(3, 2, 3)
ax3.plot(df_spline['Date'], df_spline['uspline_residual'], 'r-', linewidth=1, alpha=0.7)
ax3.axhline(y=0, color='r', linestyle='--', linewidth=1, alpha=0.5)
ax3.set_title(f'UnivariateSpline Residuals (R²={best_uspline_r2:.4f})', fontsize=14, fontweight='bold')
ax3.set_xlabel('Date')
ax3.set_ylabel('Residual')
ax3.grid(True, alpha=0.3)
ax3.tick_params(axis='x', rotation=45)

# 4. Piecewise Linear residuals (if available)
ax4 = plt.subplot(3, 2, 4)
if best_piecewise_r2 > -np.inf:
    ax4.plot(df_spline['Date'], df_spline['piecewise_residual'], 'g-', linewidth=1, alpha=0.7)
    ax4.axhline(y=0, color='r', linestyle='--', linewidth=1, alpha=0.5)
    ax4.set_title(f'Piecewise Linear Residuals (R²={best_piecewise_r2:.4f})', fontsize=14, fontweight='bold')
else:
    ax4.text(0.5, 0.5, 'Piecewise Linear\nFitting Failed', ha='center', va='center', transform=ax4.transAxes)
    ax4.set_title('Piecewise Linear Residuals', fontsize=14, fontweight='bold')
ax4.set_xlabel('Date')
ax4.set_ylabel('Residual')
ax4.grid(True, alpha=0.3)
ax4.tick_params(axis='x', rotation=45)

# 5. Residual distributions
ax5 = plt.subplot(3, 2, 5)
ax5.hist(df_spline['cubic_spline_residual'], bins=30, alpha=0.5, label='Cubic B-Spline', density=True)
ax5.hist(df_spline['uspline_residual'], bins=30, alpha=0.5, label='UnivariateSpline', density=True)
if best_piecewise_r2 > -np.inf:
    ax5.hist(df_spline['piecewise_residual'], bins=30, alpha=0.5, label='Piecewise Linear', density=True)
ax5.set_title('Residual Distributions', fontsize=14, fontweight='bold')
ax5.set_xlabel('Residual')
ax5.set_ylabel('Density')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6. Zoomed view of fit quality
ax6 = plt.subplot(3, 2, 6)
# Show a subset of data for detail
subset_idx = slice(0, 100)  # First 100 points
ax6.plot(df_spline['Date'].iloc[subset_idx], df_spline['reported_scores'].iloc[subset_idx], 
         'ko', markersize=4, alpha=0.5, label='Data')
ax6.plot(df_spline['Date'].iloc[subset_idx], df_spline['cubic_spline_fit'].iloc[subset_idx], 
         'b-', linewidth=2, label='Cubic B-Spline')
ax6.plot(df_spline['Date'].iloc[subset_idx], df_spline['uspline_fit'].iloc[subset_idx], 
         'r-', linewidth=2, label='UnivariateSpline')
if best_piecewise_r2 > -np.inf:
    ax6.plot(df_spline['Date'].iloc[subset_idx], df_spline['piecewise_fit'].iloc[subset_idx], 
             'g-', linewidth=2, label='Piecewise Linear')
ax6.set_title('Zoomed View: First 100 Points', fontsize=14, fontweight='bold')
ax6.set_xlabel('Date')
ax6.set_ylabel('Number of Reported Scores')
ax6.legend()
ax6.grid(True, alpha=0.3)
ax6.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('smooth_piecewise_fitting.png', dpi=300, bbox_inches='tight')
print("\nVisualizations saved to 'smooth_piecewise_fitting.png'")

# Save the fitted data
df_spline_output = df_spline[['Date', 'reported_scores', 'cubic_spline_fit', 'uspline_fit']].copy()
if best_piecewise_r2 > -np.inf:
    df_spline_output['piecewise_fit'] = df_spline['piecewise_fit']
df_spline_output.to_csv('smooth_piecewise_fits.csv', index=False)
print("Fitted data saved to 'smooth_piecewise_fits.csv'")

# Summary
print("\n" + "="*80)
print("SMOOTH PIECEWISE FITTING SUMMARY")
print("="*80)

best_model_idx = comparison['R²'].idxmax()
best_model_name = comparison.loc[best_model_idx, 'Model']
print(f"\nBest fitting model: {best_model_name}")
print(f"  R² = {comparison.loc[best_model_idx, 'R²']:.4f}")
print(f"  RMSE = {comparison.loc[best_model_idx, 'RMSE']:.2f}")
print(f"  MAE = {comparison.loc[best_model_idx, 'MAE']:.2f}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)

