"""
Word Structure Analysis: Exploring how word characteristics influence hard mode percentage
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# Load data
print("Loading data...")
df = pd.read_csv('../data/2023_MCM_Problem_C_Data.csv')

# Clean column names
df.columns = df.columns.str.strip()

# Convert Word to uppercase for consistency
df['Word'] = df['Word'].str.upper()

# Calculate hard mode percentage
df['hard_mode_pct'] = (df['Number in hard mode'] / df['Number of  reported results']) * 100

print(f"Data loaded: {len(df)} words")
print(f"\nFirst few rows:")
print(df[['Word', 'Number of  reported results', 'Number in hard mode', 'hard_mode_pct']].head())

# ============================================================================
# EXTRACT WORD STRUCTURE FEATURES
# ============================================================================

print("\n" + "="*80)
print("EXTRACTING WORD STRUCTURE FEATURES")
print("="*80)

# Vowels
VOWELS = set('AEIOU')

def extract_word_features(word):
    """Extract all word structure features"""
    word = word.upper()
    features = {}
    
    # 1. Number of unique letters
    features['n_unique_letters'] = len(set(word))
    
    # 2. Repeat structure (which letters repeat and how many times)
    letter_counts = Counter(word)
    features['n_repeated_letters'] = sum(1 for count in letter_counts.values() if count > 1)
    features['max_repeats'] = max(letter_counts.values()) if letter_counts else 1
    features['has_repeats'] = 1 if features['n_repeated_letters'] > 0 else 0
    
    # Create repeat pattern string (e.g., "A2B1C1" for "AABC")
    repeat_pattern = ''.join(f"{letter}{count}" for letter, count in sorted(letter_counts.items()))
    features['repeat_pattern'] = repeat_pattern
    
    # 3. Number of vowels
    features['n_vowels'] = sum(1 for char in word if char in VOWELS)
    features['n_consonants'] = len(word) - features['n_vowels']
    features['vowel_ratio'] = features['n_vowels'] / len(word) if len(word) > 0 else 0
    
    # 4. Vowel/consonant pattern (e.g., "VCVCV" for "ALONE")
    features['vc_pattern'] = ''.join('V' if char in VOWELS else 'C' for char in word)
    
    # 5. Unique vowel/consonant pattern (e.g., "VCVCV" -> "VC")
    unique_vc = []
    for char in features['vc_pattern']:
        if not unique_vc or unique_vc[-1] != char:
            unique_vc.append(char)
    features['unique_vc_pattern'] = ''.join(unique_vc)
    
    return features

# Extract features for all words
print("\nExtracting word features...")
feature_list = []
for word in df['Word']:
    features = extract_word_features(word)
    feature_list.append(features)

features_df = pd.DataFrame(feature_list)
df = pd.concat([df, features_df], axis=1)

print(f"\nExtracted features:")
print(features_df.head())

# Get all words to calculate rarity metrics
all_words = df['Word'].tolist()

# 6. Letter rarity in solution list
print("\nCalculating letter rarity...")
all_letters = ''.join(all_words)
letter_freq = Counter(all_letters)
total_letters = len(all_letters)

def get_letter_rarity(word):
    """Calculate average letter rarity in word"""
    letter_rarities = []
    for char in word:
        freq = letter_freq[char]
        rarity = 1 - (freq / total_letters)  # Higher = rarer
        letter_rarities.append(rarity)
    return np.mean(letter_rarities) if letter_rarities else 0

df['letter_rarity'] = df['Word'].apply(get_letter_rarity)

# 7. Positional rarity (how common each letter is in each position)
print("\nCalculating positional rarity...")
position_letter_freq = {i: Counter() for i in range(5)}
for word in all_words:
    for i, char in enumerate(word):
        position_letter_freq[i][char] += 1

def get_positional_rarity(word):
    """Calculate average positional rarity"""
    positional_rarities = []
    for i, char in enumerate(word):
        if i < 5:
            freq = position_letter_freq[i][char]
            total_in_position = sum(position_letter_freq[i].values())
            rarity = 1 - (freq / total_in_position) if total_in_position > 0 else 0
            positional_rarities.append(rarity)
    return np.mean(positional_rarities) if positional_rarities else 0

df['positional_rarity'] = df['Word'].apply(get_positional_rarity)

# 8. Word rarity (how common the word appears - all words appear once, so this is 0)
# Instead, calculate based on letter combinations
def get_word_rarity_score(word):
    """Calculate word rarity based on letter combinations"""
    # Use letter rarity and positional rarity
    letter_rarity = get_letter_rarity(word)
    positional_rarity = get_positional_rarity(word)
    return (letter_rarity + positional_rarity) / 2

df['word_rarity'] = df['Word'].apply(get_word_rarity_score)

# 9. Number of common anagrams
print("\nCalculating anagrams...")
# Group words by sorted letters (anagrams will have same sorted letters)
anagram_groups = {}
for word in all_words:
    sorted_letters = ''.join(sorted(word))
    if sorted_letters not in anagram_groups:
        anagram_groups[sorted_letters] = []
    anagram_groups[sorted_letters].append(word)

def get_n_anagrams(word):
    """Get number of anagrams for this word"""
    sorted_letters = ''.join(sorted(word))
    anagram_list = anagram_groups.get(sorted_letters, [])
    return len(anagram_list) - 1  # Exclude the word itself

df['n_anagrams'] = df['Word'].apply(get_n_anagrams)
df['has_anagrams'] = (df['n_anagrams'] > 0).astype(int)

print("\n" + "="*80)
print("FEATURE SUMMARY STATISTICS")
print("="*80)
print(df[['n_unique_letters', 'n_repeated_letters', 'n_vowels', 'letter_rarity', 
         'positional_rarity', 'word_rarity', 'n_anagrams', 'hard_mode_pct']].describe())

# ============================================================================
# EXPLORATORY DATA ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("EXPLORATORY DATA ANALYSIS")
print("="*80)

# Correlation analysis
print("\nCorrelation with Hard Mode Percentage:")
numeric_features = ['n_unique_letters', 'n_repeated_letters', 'n_vowels', 'n_consonants',
                   'vowel_ratio', 'letter_rarity', 'positional_rarity', 'word_rarity', 'n_anagrams']
correlations = df[numeric_features + ['hard_mode_pct']].corr()['hard_mode_pct'].sort_values(ascending=False)
print(correlations)

# Statistical tests
print("\n" + "-"*80)
print("STATISTICAL TESTS")
print("-"*80)

for feature in numeric_features:
    corr, p_value = pearsonr(df[feature], df['hard_mode_pct'])
    print(f"{feature:25s}: r={corr:7.4f}, p={p_value:.6f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else ''}")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("\n" + "="*80)
print("CREATING VISUALIZATIONS")
print("="*80)

# Create comprehensive visualization figure
fig = plt.figure(figsize=(24, 20))

# 1. Hard mode percentage distribution
ax1 = plt.subplot(4, 4, 1)
ax1.hist(df['hard_mode_pct'], bins=30, alpha=0.7, edgecolor='black')
ax1.set_title('Hard Mode Percentage Distribution', fontsize=12, fontweight='bold')
ax1.set_xlabel('Hard Mode %')
ax1.set_ylabel('Frequency')
ax1.grid(True, alpha=0.3, axis='y')

# 2. Unique letters vs hard mode %
ax2 = plt.subplot(4, 4, 2)
sns.boxplot(data=df, x='n_unique_letters', y='hard_mode_pct', ax=ax2)
ax2.set_title('Hard Mode % by Number of Unique Letters', fontsize=12, fontweight='bold')
ax2.set_xlabel('Number of Unique Letters')
ax2.set_ylabel('Hard Mode %')
ax2.grid(True, alpha=0.3, axis='y')

# 3. Repeated letters vs hard mode %
ax3 = plt.subplot(4, 4, 3)
sns.boxplot(data=df, x='n_repeated_letters', y='hard_mode_pct', ax=ax3)
ax3.set_title('Hard Mode % by Number of Repeated Letters', fontsize=12, fontweight='bold')
ax3.set_xlabel('Number of Repeated Letters')
ax3.set_ylabel('Hard Mode %')
ax3.grid(True, alpha=0.3, axis='y')

# 4. Has repeats vs hard mode %
ax4 = plt.subplot(4, 4, 4)
sns.boxplot(data=df, x='has_repeats', y='hard_mode_pct', ax=ax4)
ax4.set_title('Hard Mode %: Words with/without Repeats', fontsize=12, fontweight='bold')
ax4.set_xlabel('Has Repeated Letters')
ax4.set_ylabel('Hard Mode %')
ax4.set_xticklabels(['No Repeats', 'Has Repeats'])
ax4.grid(True, alpha=0.3, axis='y')

# 5. Number of vowels vs hard mode %
ax5 = plt.subplot(4, 4, 5)
sns.boxplot(data=df, x='n_vowels', y='hard_mode_pct', ax=ax5)
ax5.set_title('Hard Mode % by Number of Vowels', fontsize=12, fontweight='bold')
ax5.set_xlabel('Number of Vowels')
ax5.set_ylabel('Hard Mode %')
ax5.grid(True, alpha=0.3, axis='y')

# 6. Vowel ratio vs hard mode %
ax6 = plt.subplot(4, 4, 6)
ax6.scatter(df['vowel_ratio'], df['hard_mode_pct'], alpha=0.5)
z = np.polyfit(df['vowel_ratio'], df['hard_mode_pct'], 1)
p = np.poly1d(z)
ax6.plot(df['vowel_ratio'].sort_values(), p(df['vowel_ratio'].sort_values()), "r--", alpha=0.8)
corr, p_val = pearsonr(df['vowel_ratio'], df['hard_mode_pct'])
ax6.set_title(f'Vowel Ratio vs Hard Mode % (r={corr:.3f})', fontsize=12, fontweight='bold')
ax6.set_xlabel('Vowel Ratio')
ax6.set_ylabel('Hard Mode %')
ax6.grid(True, alpha=0.3)

# 7. Letter rarity vs hard mode %
ax7 = plt.subplot(4, 4, 7)
ax7.scatter(df['letter_rarity'], df['hard_mode_pct'], alpha=0.5)
z = np.polyfit(df['letter_rarity'], df['hard_mode_pct'], 1)
p = np.poly1d(z)
ax7.plot(df['letter_rarity'].sort_values(), p(df['letter_rarity'].sort_values()), "r--", alpha=0.8)
corr, p_val = pearsonr(df['letter_rarity'], df['hard_mode_pct'])
ax7.set_title(f'Letter Rarity vs Hard Mode % (r={corr:.3f})', fontsize=12, fontweight='bold')
ax7.set_xlabel('Letter Rarity')
ax7.set_ylabel('Hard Mode %')
ax7.grid(True, alpha=0.3)

# 8. Positional rarity vs hard mode %
ax8 = plt.subplot(4, 4, 8)
ax8.scatter(df['positional_rarity'], df['hard_mode_pct'], alpha=0.5)
z = np.polyfit(df['positional_rarity'], df['hard_mode_pct'], 1)
p = np.poly1d(z)
ax8.plot(df['positional_rarity'].sort_values(), p(df['positional_rarity'].sort_values()), "r--", alpha=0.8)
corr, p_val = pearsonr(df['positional_rarity'], df['hard_mode_pct'])
ax8.set_title(f'Positional Rarity vs Hard Mode % (r={corr:.3f})', fontsize=12, fontweight='bold')
ax8.set_xlabel('Positional Rarity')
ax8.set_ylabel('Hard Mode %')
ax8.grid(True, alpha=0.3)

# 9. Word rarity vs hard mode %
ax9 = plt.subplot(4, 4, 9)
ax9.scatter(df['word_rarity'], df['hard_mode_pct'], alpha=0.5)
z = np.polyfit(df['word_rarity'], df['hard_mode_pct'], 1)
p = np.poly1d(z)
ax9.plot(df['word_rarity'].sort_values(), p(df['word_rarity'].sort_values()), "r--", alpha=0.8)
corr, p_val = pearsonr(df['word_rarity'], df['hard_mode_pct'])
ax9.set_title(f'Word Rarity vs Hard Mode % (r={corr:.3f})', fontsize=12, fontweight='bold')
ax9.set_xlabel('Word Rarity')
ax9.set_ylabel('Hard Mode %')
ax9.grid(True, alpha=0.3)

# 10. Number of anagrams vs hard mode %
ax10 = plt.subplot(4, 4, 10)
sns.boxplot(data=df, x='n_anagrams', y='hard_mode_pct', ax=ax10)
ax10.set_title('Hard Mode % by Number of Anagrams', fontsize=12, fontweight='bold')
ax10.set_xlabel('Number of Anagrams')
ax10.set_ylabel('Hard Mode %')
ax10.grid(True, alpha=0.3, axis='y')

# 11. Has anagrams vs hard mode %
ax11 = plt.subplot(4, 4, 11)
sns.boxplot(data=df, x='has_anagrams', y='hard_mode_pct', ax=ax11)
ax11.set_title('Hard Mode %: Words with/without Anagrams', fontsize=12, fontweight='bold')
ax11.set_xlabel('Has Anagrams')
ax11.set_ylabel('Hard Mode %')
ax11.set_xticklabels(['No Anagrams', 'Has Anagrams'])
ax11.grid(True, alpha=0.3, axis='y')

# 12. Correlation heatmap
ax12 = plt.subplot(4, 4, 12)
corr_matrix = df[numeric_features + ['hard_mode_pct']].corr()
sns.heatmap(corr_matrix[['hard_mode_pct']].sort_values('hard_mode_pct', ascending=False), 
            annot=True, fmt='.3f', cmap='coolwarm', center=0, ax=ax12, cbar_kws={'label': 'Correlation'})
ax12.set_title('Correlation with Hard Mode %', fontsize=12, fontweight='bold')

# 13. Unique letters scatter
ax13 = plt.subplot(4, 4, 13)
ax13.scatter(df['n_unique_letters'], df['hard_mode_pct'], alpha=0.5)
z = np.polyfit(df['n_unique_letters'], df['hard_mode_pct'], 1)
p = np.poly1d(z)
ax13.plot(df['n_unique_letters'].sort_values(), p(df['n_unique_letters'].sort_values()), "r--", alpha=0.8)
corr, p_val = pearsonr(df['n_unique_letters'], df['hard_mode_pct'])
ax13.set_title(f'Unique Letters vs Hard Mode % (r={corr:.3f})', fontsize=12, fontweight='bold')
ax13.set_xlabel('Number of Unique Letters')
ax13.set_ylabel('Hard Mode %')
ax13.grid(True, alpha=0.3)

# 14. Max repeats vs hard mode %
ax14 = plt.subplot(4, 4, 14)
sns.boxplot(data=df, x='max_repeats', y='hard_mode_pct', ax=ax14)
ax14.set_title('Hard Mode % by Max Letter Repeats', fontsize=12, fontweight='bold')
ax14.set_xlabel('Max Repeats')
ax14.set_ylabel('Hard Mode %')
ax14.grid(True, alpha=0.3, axis='y')

# 15. Vowel/Consonant pattern analysis
ax15 = plt.subplot(4, 4, 15)
vc_pattern_counts = df['unique_vc_pattern'].value_counts().head(10)
vc_pattern_avg = df.groupby('unique_vc_pattern')['hard_mode_pct'].mean().loc[vc_pattern_counts.index]
bars = ax15.bar(range(len(vc_pattern_avg)), vc_pattern_avg.values, alpha=0.7)
ax15.set_title('Hard Mode % by Vowel/Consonant Pattern', fontsize=12, fontweight='bold')
ax15.set_xlabel('VC Pattern')
ax15.set_ylabel('Avg Hard Mode %')
ax15.set_xticks(range(len(vc_pattern_avg)))
ax15.set_xticklabels(vc_pattern_avg.index, rotation=45, ha='right')
ax15.grid(True, alpha=0.3, axis='y')
for i, (bar, val) in enumerate(zip(bars, vc_pattern_avg.values)):
    ax15.text(bar.get_x() + bar.get_width()/2, val + max(vc_pattern_avg.values) * 0.01, 
             f'{val:.1f}', ha='center', va='bottom', fontsize=8)

# 16. Top and bottom words by hard mode %
ax16 = plt.subplot(4, 4, 16)
top_words = df.nlargest(10, 'hard_mode_pct')[['Word', 'hard_mode_pct']]
bottom_words = df.nsmallest(10, 'hard_mode_pct')[['Word', 'hard_mode_pct']]
x_pos = np.arange(10)
width = 0.35
ax16.barh(x_pos - width/2, top_words['hard_mode_pct'].values, width, label='Top 10', alpha=0.7)
ax16.barh(x_pos + width/2, bottom_words['hard_mode_pct'].values, width, label='Bottom 10', alpha=0.7)
ax16.set_yticks(x_pos)
ax16.set_yticklabels([f"{top_words.iloc[i]['Word']}/{bottom_words.iloc[i]['Word']}" for i in range(10)], fontsize=7)
ax16.set_title('Top vs Bottom 10 Words by Hard Mode %', fontsize=12, fontweight='bold')
ax16.set_xlabel('Hard Mode %')
ax16.legend()
ax16.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('word_structure_analysis.png', dpi=300, bbox_inches='tight')
print("\nMain visualization saved to 'word_structure_analysis.png'")

# Additional detailed analysis plots
fig2 = plt.figure(figsize=(20, 12))

# 1. Feature importance comparison
ax1 = plt.subplot(2, 3, 1)
corr_abs = correlations.abs().sort_values(ascending=True)
corr_abs = corr_abs[corr_abs.index != 'hard_mode_pct']
bars = ax1.barh(range(len(corr_abs)), corr_abs.values, alpha=0.7)
ax1.set_yticks(range(len(corr_abs)))
ax1.set_yticklabels(corr_abs.index)
ax1.set_title('Feature Importance (Absolute Correlation)', fontsize=14, fontweight='bold')
ax1.set_xlabel('|Correlation| with Hard Mode %')
ax1.grid(True, alpha=0.3, axis='x')
for i, (bar, val) in enumerate(zip(bars, corr_abs.values)):
    ax1.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
             f'{val:.3f}', va='center', fontsize=9)

# 2. Scatter matrix of top features
ax2 = plt.subplot(2, 3, 2)
top_features = corr_abs.tail(5).index.tolist()
if 'hard_mode_pct' not in top_features:
    top_features.append('hard_mode_pct')
sns.scatterplot(data=df, x=top_features[0], y='hard_mode_pct', hue=top_features[1] if len(top_features) > 1 else None, 
                alpha=0.6, ax=ax2)
ax2.set_title(f'{top_features[0]} vs Hard Mode %', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

# 3. Interaction: Unique letters and repeats
ax3 = plt.subplot(2, 3, 3)
df['unique_repeat_interaction'] = df['n_unique_letters'] * (df['n_repeated_letters'] + 1)
ax3.scatter(df['unique_repeat_interaction'], df['hard_mode_pct'], alpha=0.5)
z = np.polyfit(df['unique_repeat_interaction'], df['hard_mode_pct'], 1)
p = np.poly1d(z)
ax3.plot(df['unique_repeat_interaction'].sort_values(), p(df['unique_repeat_interaction'].sort_values()), "r--", alpha=0.8)
corr, p_val = pearsonr(df['unique_repeat_interaction'], df['hard_mode_pct'])
ax3.set_title(f'Unique × Repeats vs Hard Mode % (r={corr:.3f})', fontsize=14, fontweight='bold')
ax3.set_xlabel('Unique Letters × (Repeats + 1)')
ax3.set_ylabel('Hard Mode %')
ax3.grid(True, alpha=0.3)

# 4. Vowel/Consonant distribution
ax4 = plt.subplot(2, 3, 4)
vc_counts = df.groupby(['n_vowels', 'n_consonants']).size().reset_index(name='count')
vc_avg = df.groupby(['n_vowels', 'n_consonants'])['hard_mode_pct'].mean().reset_index()
vc_merged = vc_counts.merge(vc_avg, on=['n_vowels', 'n_consonants'])
scatter = ax4.scatter(vc_merged['n_vowels'], vc_merged['n_consonants'], 
                      s=vc_merged['count']*10, c=vc_merged['hard_mode_pct'], 
                      cmap='viridis', alpha=0.6, edgecolors='black')
ax4.set_title('Vowel/Consonant Distribution\n(Size=Count, Color=Hard Mode %)', fontsize=14, fontweight='bold')
ax4.set_xlabel('Number of Vowels')
ax4.set_ylabel('Number of Consonants')
plt.colorbar(scatter, ax=ax4, label='Hard Mode %')

# 5. Letter rarity distribution
ax5 = plt.subplot(2, 3, 5)
ax5.hist(df['letter_rarity'], bins=30, alpha=0.7, edgecolor='black')
ax5.axvline(df['letter_rarity'].mean(), color='r', linestyle='--', label=f'Mean: {df["letter_rarity"].mean():.3f}')
ax5.set_title('Letter Rarity Distribution', fontsize=14, fontweight='bold')
ax5.set_xlabel('Letter Rarity')
ax5.set_ylabel('Frequency')
ax5.legend()
ax5.grid(True, alpha=0.3, axis='y')

# 6. Summary statistics table
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')
summary_data = []
for feature in numeric_features:
    corr, p_val = pearsonr(df[feature], df['hard_mode_pct'])
    summary_data.append([
        feature.replace('_', ' ').title(),
        f"{corr:.4f}",
        f"{p_val:.4f}" + ('***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else '')
    ])
table = ax6.table(cellText=summary_data,
                  colLabels=['Feature', 'Correlation', 'P-value'],
                  cellLoc='center',
                  loc='center',
                  bbox=[0, 0, 1, 1])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)
ax6.set_title('Statistical Summary', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('word_structure_detailed_analysis.png', dpi=300, bbox_inches='tight')
print("Detailed analysis saved to 'word_structure_detailed_analysis.png'")

# Save results
output_df = df[['Word', 'Number of  reported results', 'Number in hard mode', 'hard_mode_pct',
                'n_unique_letters', 'n_repeated_letters', 'has_repeats', 'max_repeats',
                'n_vowels', 'n_consonants', 'vowel_ratio', 'vc_pattern', 'unique_vc_pattern',
                'letter_rarity', 'positional_rarity', 'word_rarity', 'n_anagrams', 'has_anagrams']].copy()
output_df.to_csv('word_structure_features.csv', index=False)
print("Features saved to 'word_structure_features.csv'")

# Summary
print("\n" + "="*80)
print("KEY FINDINGS")
print("="*80)

top_corr = correlations.abs().sort_values(ascending=False)
top_corr = top_corr[top_corr.index != 'hard_mode_pct']
print(f"\nTop 5 Features Correlated with Hard Mode %:")
for i, (feature, corr) in enumerate(top_corr.head(5).items(), 1):
    print(f"  {i}. {feature}: {corr:.4f}")

print(f"\nHard Mode % Statistics:")
print(f"  Mean: {df['hard_mode_pct'].mean():.2f}%")
print(f"  Std Dev: {df['hard_mode_pct'].std():.2f}%")
print(f"  Min: {df['hard_mode_pct'].min():.2f}%")
print(f"  Max: {df['hard_mode_pct'].max():.2f}%")

# ============================================================================
# ADDITIONAL DEEP DIVE ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("DEEP DIVE ANALYSIS")
print("="*80)

# Find outliers
print("\nOutliers (High Hard Mode %):")
outliers_high = df.nlargest(10, 'hard_mode_pct')[['Word', 'hard_mode_pct', 'n_unique_letters', 
                                                   'n_repeated_letters', 'n_vowels', 'letter_rarity', 
                                                   'word_rarity', 'n_anagrams']]
print(outliers_high.to_string(index=False))

print("\nOutliers (Low Hard Mode %):")
outliers_low = df.nsmallest(10, 'hard_mode_pct')[['Word', 'hard_mode_pct', 'n_unique_letters', 
                                                   'n_repeated_letters', 'n_vowels', 'letter_rarity', 
                                                   'word_rarity', 'n_anagrams']]
print(outliers_low.to_string(index=False))

# Analyze by repeat structure
print("\n" + "-"*80)
print("ANALYSIS BY REPEAT STRUCTURE")
print("-"*80)
repeat_analysis = df.groupby('has_repeats').agg({
    'hard_mode_pct': ['mean', 'std', 'count'],
    'n_unique_letters': 'mean',
    'letter_rarity': 'mean'
}).round(3)
print(repeat_analysis)

# Analyze by number of unique letters
print("\n" + "-"*80)
print("ANALYSIS BY NUMBER OF UNIQUE LETTERS")
print("-"*80)
unique_analysis = df.groupby('n_unique_letters').agg({
    'hard_mode_pct': ['mean', 'std', 'count'],
    'n_repeated_letters': 'mean',
    'letter_rarity': 'mean'
}).round(3)
print(unique_analysis)

# Analyze by number of vowels
print("\n" + "-"*80)
print("ANALYSIS BY NUMBER OF VOWELS")
print("-"*80)
vowel_analysis = df.groupby('n_vowels').agg({
    'hard_mode_pct': ['mean', 'std', 'count'],
    'vowel_ratio': 'mean',
    'letter_rarity': 'mean'
}).round(3)
print(vowel_analysis)

# Analyze words with anagrams
print("\n" + "-"*80)
print("ANALYSIS: WORDS WITH VS WITHOUT ANAGRAMS")
print("-"*80)
anagram_analysis = df.groupby('has_anagrams').agg({
    'hard_mode_pct': ['mean', 'std', 'count'],
    'n_unique_letters': 'mean',
    'letter_rarity': 'mean'
}).round(3)
print(anagram_analysis)

# Multiple regression analysis
print("\n" + "-"*80)
print("MULTIPLE REGRESSION ANALYSIS")
print("-"*80)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Prepare features
X_reg = df[numeric_features].values
y_reg = df['hard_mode_pct'].values

# Fit multiple regression
reg_model = LinearRegression()
reg_model.fit(X_reg, y_reg)
y_pred = reg_model.predict(X_reg)
r2_reg = r2_score(y_reg, y_pred)

print(f"Multiple Regression R²: {r2_reg:.4f}")
print("\nCoefficients:")
for i, feature in enumerate(numeric_features):
    print(f"  {feature:25s}: {reg_model.coef_[i]:8.4f}")

# Create additional visualization for outliers and patterns
fig3 = plt.figure(figsize=(20, 12))

# 1. Outlier analysis
ax1 = plt.subplot(2, 3, 1)
top_10 = df.nlargest(10, 'hard_mode_pct')
bottom_10 = df.nsmallest(10, 'hard_mode_pct')
ax1.scatter(top_10['n_unique_letters'], top_10['hard_mode_pct'], 
           s=100, alpha=0.7, label='Top 10', color='red', edgecolors='black')
ax1.scatter(bottom_10['n_unique_letters'], bottom_10['hard_mode_pct'], 
           s=100, alpha=0.7, label='Bottom 10', color='blue', edgecolors='black')
for _, row in top_10.iterrows():
    ax1.annotate(row['Word'], (row['n_unique_letters'], row['hard_mode_pct']), 
                fontsize=8, ha='center')
ax1.set_title('Outliers: Unique Letters vs Hard Mode %', fontsize=14, fontweight='bold')
ax1.set_xlabel('Number of Unique Letters')
ax1.set_ylabel('Hard Mode %')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Repeat structure comparison
ax2 = plt.subplot(2, 3, 2)
repeat_comparison = df.groupby('has_repeats')['hard_mode_pct'].agg(['mean', 'std', 'count'])
x = [0, 1]
bars = ax2.bar(x, repeat_comparison['mean'].values, yerr=repeat_comparison['std'].values, 
               alpha=0.7, capsize=5, edgecolor='black')
ax2.set_xticks(x)
ax2.set_xticklabels(['No Repeats', 'Has Repeats'])
ax2.set_title('Hard Mode %: Repeats vs No Repeats', fontsize=14, fontweight='bold')
ax2.set_ylabel('Hard Mode %')
ax2.grid(True, alpha=0.3, axis='y')
for i, (bar, val) in enumerate(zip(bars, repeat_comparison['mean'].values)):
    ax2.text(bar.get_x() + bar.get_width()/2, val + repeat_comparison['std'].iloc[i] + 0.5, 
             f'{val:.2f}%', ha='center', va='bottom', fontsize=10)

# 3. Unique letters detailed
ax3 = plt.subplot(2, 3, 3)
unique_stats = df.groupby('n_unique_letters')['hard_mode_pct'].agg(['mean', 'std', 'count'])
x = unique_stats.index
bars = ax3.bar(x, unique_stats['mean'].values, yerr=unique_stats['std'].values, 
               alpha=0.7, capsize=5, edgecolor='black')
ax3.set_title('Hard Mode % by Unique Letters', fontsize=14, fontweight='bold')
ax3.set_xlabel('Number of Unique Letters')
ax3.set_ylabel('Hard Mode %')
ax3.grid(True, alpha=0.3, axis='y')
for i, (bar, val) in enumerate(zip(bars, unique_stats['mean'].values)):
    ax3.text(bar.get_x() + bar.get_width()/2, val + unique_stats['std'].iloc[i] + 0.5, 
             f'{val:.2f}%', ha='center', va='bottom', fontsize=9)

# 4. Vowels detailed
ax4 = plt.subplot(2, 3, 4)
vowel_stats = df.groupby('n_vowels')['hard_mode_pct'].agg(['mean', 'std', 'count'])
x = vowel_stats.index
bars = ax4.bar(x, vowel_stats['mean'].values, yerr=vowel_stats['std'].values, 
               alpha=0.7, capsize=5, edgecolor='black')
ax4.set_title('Hard Mode % by Number of Vowels', fontsize=14, fontweight='bold')
ax4.set_xlabel('Number of Vowels')
ax4.set_ylabel('Hard Mode %')
ax4.grid(True, alpha=0.3, axis='y')
for i, (bar, val) in enumerate(zip(bars, vowel_stats['mean'].values)):
    ax4.text(bar.get_x() + bar.get_width()/2, val + vowel_stats['std'].iloc[i] + 0.5, 
             f'{val:.2f}%', ha='center', va='bottom', fontsize=9)

# 5. Rarity scatter with regression
ax5 = plt.subplot(2, 3, 5)
ax5.scatter(df['word_rarity'], df['hard_mode_pct'], alpha=0.5)
z = np.polyfit(df['word_rarity'], df['hard_mode_pct'], 1)
p = np.poly1d(z)
ax5.plot(df['word_rarity'].sort_values(), p(df['word_rarity'].sort_values()), "r--", linewidth=2, alpha=0.8)
corr, p_val = pearsonr(df['word_rarity'], df['hard_mode_pct'])
ax5.set_title(f'Word Rarity vs Hard Mode %\n(r={corr:.3f}, p={p_val:.4f})', fontsize=14, fontweight='bold')
ax5.set_xlabel('Word Rarity')
ax5.set_ylabel('Hard Mode %')
ax5.grid(True, alpha=0.3)

# 6. Feature importance from regression
ax6 = plt.subplot(2, 3, 6)
coef_abs = np.abs(reg_model.coef_)
coef_df = pd.DataFrame({'Feature': numeric_features, 'Coefficient': reg_model.coef_, 'Abs_Coeff': coef_abs})
coef_df = coef_df.sort_values('Abs_Coeff', ascending=True)
bars = ax6.barh(range(len(coef_df)), coef_df['Coefficient'].values, alpha=0.7)
colors = ['red' if x < 0 else 'green' for x in coef_df['Coefficient'].values]
for i, bar in enumerate(bars):
    bar.set_color(colors[i])
ax6.set_yticks(range(len(coef_df)))
ax6.set_yticklabels(coef_df['Feature'].values)
ax6.set_title(f'Multiple Regression Coefficients\n(R²={r2_reg:.4f})', fontsize=14, fontweight='bold')
ax6.set_xlabel('Coefficient Value')
ax6.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax6.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('word_structure_deep_dive.png', dpi=300, bbox_inches='tight')
print("\nDeep dive analysis saved to 'word_structure_deep_dive.png'")

# Save outlier analysis
outliers_high.to_csv('outliers_high_hard_mode.csv', index=False)
outliers_low.to_csv('outliers_low_hard_mode.csv', index=False)
print("Outlier data saved to CSV files")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)

