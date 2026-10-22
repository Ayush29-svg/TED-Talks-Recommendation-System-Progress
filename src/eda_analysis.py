"""
Module 2: Exploratory Data Analysis (EDA)
Generates distribution charts and correlation plots for TED talk metadata,
durations, community ratings, tags, and engagement metrics.
"""
import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
EDA_DIR = os.path.join(BASE_DIR, 'eda_visualizations')

sns.set_theme(style='whitegrid', font='sans-serif')
plt.rcParams['font.sans-serif'] = 'Arial'

def generate_eda_charts():
    os.makedirs(EDA_DIR, exist_ok=True)
    cleaned_path = os.path.join(PROCESSED_DIR, 'ted_talks_cleaned.csv')
    interactions_path = os.path.join(PROCESSED_DIR, 'user_item_interactions.csv')
    ratings_path = os.path.join(PROCESSED_DIR, 'talk_ratings_matrix.csv')
    
    logging.info(f"Loading data from {cleaned_path}...")
    df = pd.read_csv(cleaned_path)
    df_interactions = pd.read_csv(interactions_path)
    df_ratings = pd.read_csv(ratings_path)
    
    # 1. Talk Duration Distribution
    logging.info("Plotting duration distribution...")
    plt.figure(figsize=(10, 6))
    sns.histplot(df['duration_minutes'], bins=40, kde=True, color='#e62b1e', edgecolor='black', alpha=0.7)
    mean_dur = df['duration_minutes'].mean()
    med_dur = df['duration_minutes'].median()
    plt.axvline(mean_dur, color='#2c3e50', linestyle='--', linewidth=2, label=f'Mean: {mean_dur:.1f} mins')
    plt.axvline(med_dur, color='#27ae60', linestyle='-', linewidth=2, label=f'Median: {med_dur:.1f} mins')
    plt.title('TED Talks Duration Distribution', fontsize=14, pad=12, fontweight='bold')
    plt.xlabel('Duration (Minutes)', fontsize=11)
    plt.ylabel('Count', fontsize=11)
    plt.xlim(0, 45)
    plt.legend(frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, 'talk_duration_distribution.png'), dpi=300)
    plt.close()

    # 2. Top 25 Tags
    logging.info("Plotting top 25 tags...")
    tags_series = df['tags_string'].dropna().apply(lambda x: str(x).split())
    all_tags = [t for sublist in tags_series for t in sublist]
    tag_counts = pd.Series(all_tags).value_counts().head(25).reset_index()
    tag_counts.columns = ['tag', 'count']
    
    plt.figure(figsize=(12, 8))
    sns.barplot(data=tag_counts, x='count', y='tag', hue='tag', palette='flare', legend=False)
    plt.title('Top 25 Most Frequent Topic Tags', fontsize=14, pad=12, fontweight='bold')
    plt.xlabel('Occurrences', fontsize=11)
    plt.ylabel('Topic Tag', fontsize=11)
    for idx, row in tag_counts.iterrows():
        plt.text(row['count'] + 8, idx, str(row['count']), va='center', fontsize=9, color='#333333')
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, 'top_25_tags_distribution.png'), dpi=300)
    plt.close()

    # 3. Views and Comments Distribution
    logging.info("Plotting views and comments distributions...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    sns.histplot(df['views'], bins=40, kde=True, ax=axes[0], color='#2980b9', log_scale=True, edgecolor='black', alpha=0.7)
    axes[0].set_title('Distribution of Views (Log Scale)', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Views', fontsize=11)
    axes[0].set_ylabel('Count', fontsize=11)
    
    sns.histplot(df['comments'], bins=40, kde=True, ax=axes[1], color='#8e44ad', log_scale=True, edgecolor='black', alpha=0.7)
    axes[1].set_title('Distribution of Comments (Log Scale)', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Comments', fontsize=11)
    axes[1].set_ylabel('Count', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, 'views_and_comments_distribution.png'), dpi=300)
    plt.close()

    # 4. Community Ratings Breakdown
    logging.info("Plotting community rating categories...")
    rating_cols = [c for c in df_ratings.columns if c not in ['talk_id', 'title']]
    ratings_total = df_ratings[rating_cols].sum().sort_values(ascending=False).reset_index()
    ratings_total.columns = ['category', 'votes']
    
    plt.figure(figsize=(11, 6.5))
    sns.barplot(data=ratings_total, x=ratings_total['votes'] / 1e6, y='category', hue='category', palette='viridis', legend=False)
    plt.title('Total Community Ratings by Feedback Type (Millions)', fontsize=14, pad=12, fontweight='bold')
    plt.xlabel('Total Votes (Millions)', fontsize=11)
    plt.ylabel('Category', fontsize=11)
    for idx, row in ratings_total.iterrows():
        val = row['votes'] / 1e6
        plt.text(val + 0.04, idx, f"{val:.2f}M", va='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, 'rating_categories_distribution.png'), dpi=300)
    plt.close()

    # 5. Word Count and Speaking Rate
    logging.info("Plotting word counts and speaking rate...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    sns.histplot(df['transcript_word_count'], bins=35, kde=True, ax=axes[0], color='#16a085', edgecolor='black', alpha=0.7)
    axes[0].set_title('Transcript Word Count Distribution', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Words', fontsize=11)
    axes[0].set_ylabel('Count', fontsize=11)
    axes[0].set_xlim(0, 6000)
    
    valid_wpm = df[(df['words_per_minute'] > 50) & (df['words_per_minute'] < 250)]
    sns.histplot(valid_wpm['words_per_minute'], bins=35, kde=True, ax=axes[1], color='#d35400', edgecolor='black', alpha=0.7)
    axes[1].set_title('Speaking Rate (Words Per Minute)', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('WPM', fontsize=11)
    axes[1].set_ylabel('Count', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, 'transcript_word_count_distribution.png'), dpi=300)
    plt.close()

    # 6. Correlation Heatmap
    logging.info("Plotting correlation matrix...")
    numeric_features = [
        'duration_minutes', 'views', 'comments', 'languages', 
        'transcript_word_count', 'tag_count', 'total_rating_votes', 'views_per_comment'
    ]
    corr = df[numeric_features].corr()
    
    plt.figure(figsize=(9.5, 7.5))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', cbar=True, square=True, linewidths=0.5, annot_kws={'size': 9})
    plt.title('Correlation Heatmap of Metadata and Engagement Features', fontsize=13, pad=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=9.5)
    plt.yticks(rotation=0, fontsize=9.5)
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, 'feature_correlation_matrix.png'), dpi=300)
    plt.close()

    # 7. Top Speakers and Occupations
    logging.info("Plotting top speakers and occupations...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    
    top_speakers = df['main_speaker'].value_counts().head(12).reset_index()
    top_speakers.columns = ['speaker', 'talks']
    sns.barplot(data=top_speakers, x='talks', y='speaker', hue='speaker', ax=axes[0], palette='Blues_r', legend=False)
    axes[0].set_title('Top 12 Frequent TED Speakers', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Number of Talks', fontsize=11)
    axes[0].set_ylabel('')
    
    top_occ = df[df['speaker_occupation'] != 'Unknown']['speaker_occupation'].value_counts().head(12).reset_index()
    top_occ.columns = ['occupation', 'talks']
    sns.barplot(data=top_occ, x='talks', y='occupation', hue='occupation', ax=axes[1], palette='Purples_r', legend=False)
    axes[1].set_title('Top 12 Speaker Occupations', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Number of Talks', fontsize=11)
    axes[1].set_ylabel('')
    
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, 'top_speakers_and_occupations.png'), dpi=300)
    plt.close()

    # 8. User Interaction Rating Distribution
    logging.info("Plotting user-item interaction ratings...")
    plt.figure(figsize=(8.5, 5))
    rating_counts = df_interactions['rating'].value_counts().sort_index().reset_index()
    rating_counts.columns = ['stars', 'count']
    
    sns.barplot(data=rating_counts, x='stars', y='count', hue='stars', palette='viridis', edgecolor='black', alpha=0.85, legend=False)
    plt.title('User Rating Distribution (Collaborative Filtering Benchmark)', fontsize=13, pad=12, fontweight='bold')
    plt.xlabel('Rating (1 to 5 Stars)', fontsize=11)
    plt.ylabel('Interactions', fontsize=11)
    for idx, row in rating_counts.iterrows():
        plt.text(idx, row['count'] + 140, f"{row['count']:,}\n({row['count']/len(df_interactions):.1%})", ha='center', fontsize=9.5)
    plt.ylim(0, max(rating_counts['count']) * 1.15)
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, 'user_interaction_ratings_distribution.png'), dpi=300)
    plt.close()

    logging.info(f"Visualizations saved to {EDA_DIR}/")

if __name__ == '__main__':
    generate_eda_charts()
