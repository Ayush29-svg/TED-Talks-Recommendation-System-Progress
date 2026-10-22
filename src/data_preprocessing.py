"""
Module 2: Data Preprocessing
Cleans metadata and transcripts, handles missing values, engineers text
features, and generates interaction matrices for recommendation models.
"""
import os
import re
import ast
import json
import logging
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')

def clean_text(text: str, remove_stopwords: bool = True) -> str:
    """Clean text by removing laughter/applause cues, URLs, punctuation, and extra whitespace."""
    if not isinstance(text, str) or not text.strip():
        return ""
    
    # Strip transcript stage cues like (Laughter), (Applause), [Music]
    text = re.sub(r'\([A-Za-z\s]+\)', ' ', text)
    text = re.sub(r'\[[A-Za-z\s]+\]', ' ', text)
    
    # Remove web links
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    
    # Lowercase
    text = text.lower()
    
    # Strip punctuation and numbers/symbols, keep words
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    
    tokens = text.split()
    if remove_stopwords:
        tokens = [w for w in tokens if w not in ENGLISH_STOP_WORDS and len(w) > 2]
    
    return " ".join(tokens)

def parse_literal(val, default):
    """Safely parse string representations of lists or dicts."""
    try:
        if pd.isna(val) or val == "":
            return default
        return ast.literal_eval(val)
    except Exception:
        return default

def parse_ratings(ratings_list):
    """Extract total votes, primary sentiment, and voting breakdown across all 14 categories."""
    if not isinstance(ratings_list, list) or len(ratings_list) == 0:
        return 0, 'Unknown', {}
    
    total_votes = sum(item.get('count', 0) for item in ratings_list if isinstance(item, dict))
    sorted_items = sorted(ratings_list, key=lambda x: x.get('count', 0), reverse=True)
    top_sentiment = sorted_items[0].get('name', 'Unknown') if sorted_items else 'Unknown'
    
    breakdown = {item.get('name'): item.get('count', 0) for item in ratings_list if isinstance(item, dict)}
    return total_votes, top_sentiment, breakdown

def build_user_interaction_matrix(df_talks: pd.DataFrame, num_users: int = 500) -> pd.DataFrame:
    """
    Generate realistic user-item viewing and rating interactions across diverse topic interests.
    This creates the interaction matrix required for collaborative filtering in Milestone 2.
    """
    np.random.seed(42)
    logging.info(f"Building interaction matrix for {num_users} users...")
    
    # Define topic interests for user groups
    interest_groups = {
        'tech_enthusiast': ['technology', 'computers', 'ai', 'future', 'robots', 'internet', 'innovation'],
        'science_buff': ['science', 'biology', 'physics', 'space', 'environment', 'genetics', 'evolution'],
        'arts_and_humanities': ['culture', 'art', 'music', 'humanity', 'history', 'society', 'storytelling'],
        'global_issues': ['global issues', 'climate change', 'activism', 'peace', 'politics', 'inequality', 'women'],
        'business_and_economy': ['business', 'leadership', 'economics', 'entrepreneur', 'marketing', 'work'],
        'mind_and_health': ['health', 'psychology', 'brain', 'mental health', 'medicine', 'happiness', 'mind']
    }
    
    all_talks = df_talks[['talk_id', 'tags_list', 'views', 'duration_minutes']].copy()
    talk_ids = all_talks['talk_id'].values
    group_names = list(interest_groups.keys())
    
    records = []
    
    for i in range(1, num_users + 1):
        user_id = f"U{i:04d}"
        group = group_names[i % len(group_names)]
        fav_tags = set(interest_groups[group])
        
        # User views between 15 and 50 talks
        views_count = np.random.randint(15, 55)
        
        # Talks with matching tags have higher probability of being watched
        def get_weight(tags):
            overlap = len(set(tags).intersection(fav_tags))
            return 1.0 + 3.5 * overlap
            
        weights = all_talks['tags_list'].apply(get_weight).values
        probs = weights / weights.sum()
        
        watched_talks = np.random.choice(talk_ids, size=min(views_count, len(talk_ids)), replace=False, p=probs)
        
        for tid in watched_talks:
            row = all_talks[all_talks['talk_id'] == tid].iloc[0]
            talk_tags = set(row['tags_list'])
            overlap = len(talk_tags.intersection(fav_tags))
            
            # Score 1 to 5 stars
            base_score = 3.2 + (0.5 * min(overlap, 3)) + np.random.normal(0, 0.6)
            rating = int(np.clip(round(base_score), 1, 5))
            
            # Watch completion percentage (implicit feedback)
            watch_ratio = float(np.clip(np.random.normal(0.75 + (0.05 * rating), 0.15), 0.20, 1.0))
            
            records.append({
                'user_id': user_id,
                'talk_id': tid,
                'rating': rating,
                'watch_ratio': round(watch_ratio, 2),
                'interest_profile': group
            })
            
    interaction_df = pd.DataFrame(records)
    logging.info(f"Created {len(interaction_df):,} user interaction records across {interaction_df['talk_id'].nunique()} talks.")
    return interaction_df

def run_preprocessing_pipeline():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    raw_main_path = os.path.join(RAW_DIR, 'ted_main.csv')
    raw_trans_path = os.path.join(RAW_DIR, 'transcripts.csv')
    
    logging.info("Loading raw datasets...")
    df_main = pd.read_csv(raw_main_path, encoding='latin1')
    df_trans = pd.read_csv(raw_trans_path, encoding_errors='replace')
    
    logging.info(f"Loaded {len(df_main)} talks and {len(df_trans)} transcripts.")
    
    # Merge metadata and transcripts on unique URL
    df = pd.merge(df_main, df_trans, on='url', how='left')
    df.insert(0, 'talk_id', range(1001, 1001 + len(df)))
    
    # Impute missing transcripts using title and description so no talk is lost
    df['has_transcript'] = df['transcript'].notnull()
    df['transcript'] = df['transcript'].fillna(df['title'] + ". " + df['description'])
    
    # Impute missing speaker occupation
    df['speaker_occupation'] = df['speaker_occupation'].fillna('Unknown')
    
    # Parse tags and ratings
    logging.info("Parsing tags and community ratings...")
    df['tags_list'] = df['tags'].apply(lambda x: parse_literal(x, []))
    df['ratings_list'] = df['ratings'].apply(lambda x: parse_literal(x, []))
    
    rating_data = df['ratings_list'].apply(parse_ratings)
    df['total_rating_votes'] = [r[0] for r in rating_data]
    df['dominant_sentiment'] = [r[1] for r in rating_data]
    
    # Export talk-ratings matrix
    ratings_breakdowns = [r[2] for r in rating_data]
    ratings_matrix = pd.DataFrame(ratings_breakdowns).fillna(0)
    ratings_matrix.insert(0, 'talk_id', df['talk_id'])
    ratings_matrix.insert(1, 'title', df['title'])
    ratings_matrix.to_csv(os.path.join(PROCESSED_DIR, 'talk_ratings_matrix.csv'), index=False)
    
    # Date features
    df['film_datetime'] = pd.to_datetime(df['film_date'], unit='s', errors='coerce')
    df['published_datetime'] = pd.to_datetime(df['published_date'], unit='s', errors='coerce')
    df['film_year'] = df['film_datetime'].dt.year
    df['published_year'] = df['published_datetime'].dt.year
    df['published_month'] = df['published_datetime'].dt.month
    
    # Content metrics
    logging.info("Calculating content metrics...")
    df['duration_minutes'] = (df['duration'] / 60.0).round(2)
    df['transcript_word_count'] = df['transcript'].apply(lambda x: len(str(x).split()))
    df['transcript_char_count'] = df['transcript'].apply(lambda x: len(str(x)))
    df['description_word_count'] = df['description'].apply(lambda x: len(str(x).split()))
    df['tag_count'] = df['tags_list'].apply(lambda x: len(x) if isinstance(x, list) else 0)
    df['words_per_minute'] = (df['transcript_word_count'] / np.maximum(df['duration_minutes'], 0.1)).round(1)
    df['views_per_comment'] = (df['views'] / np.maximum(df['comments'], 1)).round(1)
    
    # Clean text columns
    logging.info("Cleaning text fields...")
    df['cleaned_title'] = df['title'].apply(lambda x: clean_text(x, remove_stopwords=False))
    df['cleaned_description'] = df['description'].apply(lambda x: clean_text(x, remove_stopwords=True))
    df['cleaned_transcript'] = df['transcript'].apply(lambda x: clean_text(x, remove_stopwords=True))
    df['tags_string'] = df['tags_list'].apply(lambda x: " ".join([re.sub(r'[^a-z0-9]', '', t.lower()) for t in x]) if isinstance(x, list) else "")
    df['clean_speaker'] = df['main_speaker'].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', str(x)).strip())
    
    # Combined content soup for recommendation modeling
    logging.info("Creating content soup feature...")
    df['content_soup'] = (
        df['cleaned_title'] + " " +
        df['clean_speaker'] + " " +
        df['speaker_occupation'].apply(lambda x: clean_text(str(x), remove_stopwords=True)) + " " +
        df['tags_string'] + " " +
        df['cleaned_description']
    )
    
    # Encode speaker IDs
    df['speaker_id'] = pd.factorize(df['main_speaker'])[0] + 1
    
    # Build user interaction dataset
    interactions_df = build_user_interaction_matrix(df, num_users=500)
    interactions_df.to_csv(os.path.join(PROCESSED_DIR, 'user_item_interactions.csv'), index=False)
    
    # Save summary stats
    summary_stats = {
        'total_talks': int(len(df)),
        'total_features': int(df.shape[1]),
        'talks_with_verbatim_transcripts': int(df['has_transcript'].sum()),
        'talks_with_imputed_transcripts': int((~df['has_transcript']).sum()),
        'unique_speakers': int(df['main_speaker'].nunique()),
        'unique_occupations': int(df['speaker_occupation'].nunique()),
        'total_tags_catalogued': int(len(set([t for sublist in df['tags_list'] for t in sublist]))),
        'duration_minutes_mean': float(df['duration_minutes'].mean()),
        'duration_minutes_median': float(df['duration_minutes'].median()),
        'duration_minutes_min': float(df['duration_minutes'].min()),
        'duration_minutes_max': float(df['duration_minutes'].max()),
        'views_mean': float(df['views'].mean()),
        'views_median': float(df['views'].median()),
        'views_min': int(df['views'].min()),
        'views_max': int(df['views'].max()),
        'transcript_words_mean': float(df['transcript_word_count'].mean()),
        'transcript_words_median': float(df['transcript_word_count'].median()),
        'tags_per_talk_mean': float(df['tag_count'].mean()),
        'collaborative_users_count': int(interactions_df['user_id'].nunique()),
        'collaborative_interactions_count': int(len(interactions_df)),
        'collaborative_interaction_sparsity_pct': float((1 - (len(interactions_df) / (interactions_df['user_id'].nunique() * len(df)))) * 100),
        'missing_values_after_cleaning': {k: int(v) for k, v in df.isnull().sum().items() if v > 0}
    }
    
    with open(os.path.join(PROCESSED_DIR, 'data_summary_stats.json'), 'w', encoding='utf-8') as f:
        json.dump(summary_stats, f, indent=4)
        
    # Save cleaned dataset
    output_csv = os.path.join(PROCESSED_DIR, 'ted_talks_cleaned.csv')
    df.to_csv(output_csv, index=False)
    logging.info(f"Preprocessing completed. Output saved to {output_csv}")
    
    return df, summary_stats

if __name__ == '__main__':
    run_preprocessing_pipeline()
