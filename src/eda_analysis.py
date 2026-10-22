import os
import ast
import json
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
EDA_DIR = os.path.join(BASE_DIR, 'eda_visualizations')

# Set aesthetic styling
sns.set_theme(style='whitegrid', font='sans-serif')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

def generate_eda_charts():
    os.makedirs(EDA_DIR, exist_ok=True)
    cleaned_path = os.path.join(PROCESSED_DIR, 'ted_talks_cleaned.csv')
    interactions_path = os.path.join(PROCESSED_DIR, 'user_item_interactions.csv')
    ratings_path = os.path.join(PROCESSED_DIR, 'talk_ratings_matrix.csv')
    
    logging.info(f"Loading cleaned data from {cleaned_path}...")
    df = pd.read_csv(cleaned_path)
    df_interactions = pd.read_csv(interactions_path)
    df_ratings = pd.read_csv(ratings_path)
    
    # 1. Talk Duration Distribution
    logging.info("Generating Talk Duration Distribution plot...")
    plt.figure(figsize=(10, 6))
    sns.histplot(df['duration_minutes'], bins=40, kde=True, color='#e62b1e', edgecolor='black', alpha=0.7)
    mean_dur = df['duration_minutes'].mean()
    med_dur = df['duration_minutes'].median()
    plt.axvline(mean_dur, color='#2c3e50', linestyle='--', linewidth=2, label=f'Mean: {mean_dur:.1f} mins')
    plt.axvline(med_dur, color='#27ae60', linestyle='-', linewidth=2, label=f'Median: {med_dur:.1f} mins')
    plt.title('TED Talks Duration Distribution', fontsize=15, pad=15, fontweight='bold')
    plt.xlabel('Duration (Minutes)', fontsize=12)
    plt.ylabel('Number of Talks', fontsize=12)
    plt.xlim(0, 45)
    plt.legend(frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, 'talk_duration_distribution.png'), dpi=300)
    plt.close()

    # 2. Top 25 Tag Frequencies
    logging.info("Generating Top Tags Frequency plot...")
    tags_series = df['tags_string'].dropna().apply(lambda x: str(x).split())
    all_tags = [tag for sublist in tags_series for tag in sublist]
    tag_counts = pd.Series(all_tags).value_counts().head(25)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x=tag_counts.values, y=tag_counts.index, palette='flare', orient='h')
    plt.title('Top 25 Most Frequent Topic Tags in TED Talks', fontsize=15, pad=15, fontweight='bold')
    plt.xlabel('Frequency of Tag Occurrences', fontsize=12)
    plt.ylabel('Topic Tag', fontsize=12)
    for index, value in enumerate(tag_counts.values):
        plt.text(value + 10, index, str(value), va='center', fontsize=10, color='#333333')
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, 'top_25_tags_distribution.png'), dpi=300)
    plt.close()

    # 3. Views and Comments Distribution (Log Scaled)
    logging.info("Generating Views and Comments Distributions...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.histplot(df['views'], bins=40, kde=True, ax=axes[0], color='#2980b9', log_scale=True, edgecolor='black', alpha=0.7)
    axes[0].set_title('Distribution of Views (Log Scale)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Views', fontsize=12)
    axes[0].set_ylabel('Talk Count', fontsize=12)
    
    sns.histplot(df['comments'], bins=40, kde=True, ax=axes[1], color='#8e44ad', log_scale=True, edgecolor='black', alpha=0.7)
    axes[1].set_title('Distribution of Comments (Log Scale)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Comments', fontsize=12)
    axes[1].set_ylabel('Talk Count', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, 'views_and_comments_distribution.png'), dpi=300)
    plt.close()

    # 4. User Rating Sentiments / Quality Dimensions Breakdown
    logging.info("Generating Community Rating Categories Distribution...")
    rating_cols = [c for c in df_ratings.columns if c not in ['talk_id', 'title']]
    total_ratings_by_category = df_ratings[rating_cols].sum().sort_values(ascending=False)
    
    plt.figure(figsize=(12, 7))
    colors = ['#e74c3c' if x in ['Inspiring', 'Fascinating', 'Informative', 'Funny', 'Beautiful'] else '#34495e' for x in total_ratings_by_category.index]
    sns.barplot(x=total_ratings_by_category.values / 1e6, y=total_ratings_by_category.index, palette=colors, orient='h')
    plt.title('Aggregated Community Ratings by Category (Millions of Votes)', fontsize=15, pad=15, fontweight='bold')
    plt.xlabel('Total Votes (in Millions)', fontsize=12)
    plt.ylabel('Feedback Category', fontsize=12)
    for index, value in enumerate(total_ratings_by_category.values / 1e6):
        plt.text(value + 0.03, index, f"{value:.2f}M", va='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, 'rating_categories_distribution.png'), dpi=300)
    plt.close()

    # 5. Transcript Word Count & Speaking Rate Distribution
    logging.info("Generating Transcript Length & Speaking Rate plot...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.histplot(df['transcript_word_count'], bins=35, kde=True, ax=axes[0], color='#16a085', edgecolor='black', alpha=0.7)
    axes[0].set_title('Transcript Word Count Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Word Count', fontsize=12)
    axes[0].set_ylabel('Talk Count', fontsize=12)
    axes[0].set_xlim(0, 6000)
    
    valid_wpm = df[(df['words_per_minute'] > 50) & (df['words_per_minute'] < 250)]
    sns.histplot(valid_wpm['words_per_minute'], bins=35, kde=True, ax=axes[1], color='#d35400', edgecolor='black', alpha=0.7)
    axes[1].set_title('Speaking Rate (Words Per Minute)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Words Per Minute (WPM)', fontsize=12)
    axes[1].set_ylabel('Talk Count', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, 'transcript_word_count_distribution.png'), dpi=300)
    plt.close()

    # 6. Feature Correlation Heatmap
    logging.info("Generating Feature Correlation Matrix Heatmap...")
    numeric_features = [
        'duration_minutes', 'views', 'comments', 'languages', 
        'transcript_word_count', 'tag_count', 'total_rating_votes', 'views_per_comment'
    ]
    corr = df[numeric_features].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', cbar=True, square=True, linewidths=0.5, annot_kws={'size': 10})
    plt.title('Correlation Matrix of Talk Metadata & Engagement Features', fontsize=14, pad=15, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, 'feature_correlation_matrix.png'), dpi=300)
    plt.close()

    # 7. Top Speakers and Occupations
    logging.info("Generating Top Speakers and Occupations plot...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    top_speakers = df['main_speaker'].value_counts().head(12)
    sns.barplot(x=top_speakers.values, y=top_speakers.index, ax=axes[0], palette='Blues_r', orient='h')
    axes[0].set_title('Top 12 Most Frequent TED Speakers', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Number of Talks', fontsize=12)
    
    top_occ = df[df['speaker_occupation'] != 'Unknown']['speaker_occupation'].value_counts().head(12)
    sns.barplot(x=top_occ.values, y=top_occ.index, ax=axes[1], palette='Purples_r', orient='h')
    axes[1].set_title('Top 12 Speaker Occupations', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Number of Talks', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, 'top_speakers_and_occupations.png'), dpi=300)
    plt.close()

    # 8. User-Item Interaction Ratings Distribution for Collaborative Filtering
    logging.info("Generating User-Item Interaction Ratings Distribution...")
    plt.figure(figsize=(9, 5))
    rating_counts = df_interactions['rating'].value_counts().sort_index()
    sns.barplot(x=rating_counts.index, y=rating_counts.values, palette='viridis', edgecolor='black', alpha=0.85)
    plt.title('User-Item Ratings Distribution (Collaborative Filtering Benchmark)', fontsize=14, pad=15, fontweight='bold')
    plt.xlabel('Rating (1 to 5 Stars)', fontsize=12)
    plt.ylabel('Number of Interactions', fontsize=12)
    for idx, count in enumerate(rating_counts.values):
        plt.text(idx, count + 150, f"{count:,}\n({count/len(df_interactions):.1%})", ha='center', fontsize=10)
    plt.ylim(0, max(rating_counts.values) * 1.15)
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, 'user_interaction_ratings_distribution.png'), dpi=300)
    plt.close()

    logging.info(f"All 8 publication-grade visualization plots saved to {EDA_DIR}/")

if __name__ == '__main__':
    generate_eda_charts()
