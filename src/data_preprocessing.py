import os
import re
import ast
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')

def clean_text(text: str, remove_stopwords: bool = True) -> str:
    """
    Cleans raw text data:
    1. Removes transcript cues such as (Laughter), (Applause), [Music], etc.
    2. Lowercases text
    3. Removes URLs, non-alphabet characters, and excessive whitespace
    4. Optionally removes stopwords
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    
    # Remove transcript cues like (Laughter), (Applause), (Cheering), [Music]
    text = re.sub(r'\([A-Za-z\s]+\)', ' ', text)
    text = re.sub(r'\[[A-Za-z\s]+\]', ' ', text)
    
    # Remove URLs if any
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    
    # Lowercase
    text = text.lower()
    
    # Remove punctuation and special characters, keep letters and spaces
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    
    # Tokenize and remove extra whitespace
    tokens = text.split()
    
    if remove_stopwords:
        tokens = [word for word in tokens if word not in ENGLISH_STOP_WORDS and len(word) > 2]
    
    return " ".join(tokens)

def parse_literal(val, default):
    """Safely parse literal Python string representations."""
    try:
        if pd.isna(val) or val == "":
            return default
        return ast.literal_eval(val)
    except Exception:
        return default

def extract_rating_features(ratings_list):
    """
    Deconstructs the 14 rating categories in TED talks into:
    - total_rating_votes
    - dominant_rating_sentiment (e.g. 'Inspiring', 'Funny', etc.)
    - rating breakdown dictionary
    """
    if not isinstance(ratings_list, list) or len(ratings_list) == 0:
        return 0, 'Unknown', {}
    
    total_votes = sum(item.get('count', 0) for item in ratings_list if isinstance(item, dict))
    sorted_ratings = sorted(ratings_list, key=lambda x: x.get('count', 0), reverse=True)
    dominant_sentiment = sorted_ratings[0].get('name', 'Unknown') if sorted_ratings else 'Unknown'
    
    breakdown = {item.get('name'): item.get('count', 0) for item in ratings_list if isinstance(item, dict)}
    return total_votes, dominant_sentiment, breakdown

def generate_collaborative_interactions(df_cleaned: pd.DataFrame, num_users: int = 500) -> pd.DataFrame:
    """
    Constructs a realistic user-item interaction dataset for collaborative filtering (Milestone 2 & 3).
    Generates user profiles with distinct topic/tag affinities (e.g., Technology, Science, Design, Business, Culture),
    and creates ratings on a 1-5 scale reflecting talk relevance, quality, and engagement.
    """
    np.random.seed(42)
    logging.info(f"Generating synthetic user-item interactions for {num_users} users...")
    
    # Define archetypes and their preferred tags
    user_personas = {
        'tech_innovator': ['technology', 'computers', 'ai', 'future', 'robots', 'internet', 'innovation'],
        'science_enthusiast': ['science', 'biology', 'physics', 'space', 'environment', 'genetics', 'evolution'],
        'culture_humanities': ['culture', 'art', 'music', 'humanity', 'history', 'society', 'storytelling'],
        'global_advocate': ['global issues', 'climate change', 'activism', 'peace', 'politics', 'inequality', 'women'],
        'business_leadership': ['business', 'leadership', 'economics', 'entrepreneur', 'marketing', 'work'],
        'health_psychology': ['health', 'psychology', 'brain', 'mental health', 'medicine', 'happiness', 'mind']
    }
    
    interactions = []
    persona_keys = list(user_personas.keys())
    
    talk_metadata = df_cleaned[['talk_id', 'tags_list', 'views', 'duration_minutes']].copy()
    all_talk_ids = talk_metadata['talk_id'].values
    
    for user_idx in range(1, num_users + 1):
        user_id = f"U{user_idx:04d}"
        assigned_persona = persona_keys[user_idx % len(persona_keys)]
        preferred_tags = set(user_personas[assigned_persona])
        
        # Decide how many talks this user interacted with (between 15 and 60 talks)
        num_interactions = np.random.randint(15, 60)
        
        # Calculate affinity probability for talks
        def calc_prob(tags):
            match_count = len(set(tags).intersection(preferred_tags))
            return 1.0 + 4.0 * match_count
            
        weights = talk_metadata['tags_list'].apply(calc_prob).values
        prob_dist = weights / weights.sum()
        
        selected_talk_ids = np.random.choice(all_talk_ids, size=min(num_interactions, len(all_talk_ids)), replace=False, p=prob_dist)
        
        for talk_id in selected_talk_ids:
            row = talk_metadata[talk_metadata['talk_id'] == talk_id].iloc[0]
            talk_tags = set(row['tags_list'])
            overlap = len(talk_tags.intersection(preferred_tags))
            
            # Base rating 3, boost if matching persona tags, add small noise
            base_score = 3.2 + (0.5 * min(overlap, 3)) + np.random.normal(0, 0.6)
            rating = int(np.clip(round(base_score), 1, 5))
            
            # Watch completion percentage (implicit feedback)
            watch_pct = float(np.clip(np.random.normal(0.75 + (0.05 * rating), 0.15), 0.20, 1.0))
            
            interactions.append({
                'user_id': user_id,
                'talk_id': talk_id,
                'rating': rating,
                'watch_ratio': round(watch_pct, 2),
                'persona': assigned_persona
            })
            
    df_interactions = pd.DataFrame(interactions)
    logging.info(f"Generated {len(df_interactions):,} user-item interaction records across {num_users} users and {df_interactions['talk_id'].nunique()} unique talks.")
    return df_interactions

def run_preprocessing_pipeline():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    raw_main_path = os.path.join(RAW_DIR, 'ted_main.csv')
    raw_trans_path = os.path.join(RAW_DIR, 'transcripts.csv')
    
    logging.info("Reading raw datasets...")
    df_main = pd.read_csv(raw_main_path, encoding='latin1')
    df_trans = pd.read_csv(raw_trans_path, encoding_errors='replace')
    
    initial_main_rows = len(df_main)
    initial_trans_rows = len(df_trans)
    logging.info(f"Raw ted_main rows: {initial_main_rows}, Raw transcripts rows: {initial_trans_rows}")
    
    # 1. Merging on unique URL
    logging.info("Merging metadata and transcripts on 'url'...")
    df = pd.merge(df_main, df_trans, on='url', how='left')
    
    # Assign unique integer talk_id for clean index referencing
    df.insert(0, 'talk_id', range(1001, 1001 + len(df)))
    
    # 2. Inspecting and handling missing values
    missing_before = df.isnull().sum().to_dict()
    logging.info(f"Missing values before handling: {missing_before}")
    
    # Track transcript availability
    df['has_transcript'] = df['transcript'].notnull()
    
    # Impute missing transcripts with title and description so no talk is left out of NLP pipeline
    df['transcript'] = df['transcript'].fillna(df['title'] + ". " + df['description'])
    
    # Handle missing speaker occupation
    df['speaker_occupation'] = df['speaker_occupation'].fillna('Unknown')
    
    # 3. Parse literal evaluation columns
    logging.info("Parsing tags and ratings data...")
    df['tags_list'] = df['tags'].apply(lambda x: parse_literal(x, []))
    df['ratings_list'] = df['ratings'].apply(lambda x: parse_literal(x, []))
    
    # Extract rating metrics
    rating_features = df['ratings_list'].apply(extract_rating_features)
    df['total_rating_votes'] = [rf[0] for rf in rating_features]
    df['dominant_sentiment'] = [rf[1] for rf in rating_features]
    
    # Extract top rating categories as columns for exploration and modeling
    rating_breakdowns = [rf[2] for rf in rating_features]
    ratings_matrix_df = pd.DataFrame(rating_breakdowns).fillna(0)
    ratings_matrix_df.insert(0, 'talk_id', df['talk_id'])
    ratings_matrix_df.insert(1, 'title', df['title'])
    ratings_matrix_path = os.path.join(PROCESSED_DIR, 'talk_ratings_matrix.csv')
    ratings_matrix_df.to_csv(ratings_matrix_path, index=False)
    logging.info(f"Saved talk-ratings profile matrix to {ratings_matrix_path}")
    
    # 4. Temporal feature extraction
    logging.info("Converting timestamps to readable datetime features...")
    df['film_datetime'] = pd.to_datetime(df['film_date'], unit='s', errors='coerce')
    df['published_datetime'] = pd.to_datetime(df['published_date'], unit='s', errors='coerce')
    
    df['film_year'] = df['film_datetime'].dt.year
    df['published_year'] = df['published_datetime'].dt.year
    df['published_month'] = df['published_datetime'].dt.month
    
    # 5. Content Metrics & Quantitative Feature Engineering
    logging.info("Computing content similarity metrics and text properties...")
    df['duration_minutes'] = (df['duration'] / 60.0).round(2)
    df['transcript_word_count'] = df['transcript'].apply(lambda x: len(str(x).split()))
    df['transcript_char_count'] = df['transcript'].apply(lambda x: len(str(x)))
    df['description_word_count'] = df['description'].apply(lambda x: len(str(x).split()))
    df['tag_count'] = df['tags_list'].apply(lambda x: len(x) if isinstance(x, list) else 0)
    df['words_per_minute'] = (df['transcript_word_count'] / np.maximum(df['duration_minutes'], 0.1)).round(1)
    df['views_per_comment'] = (df['views'] / np.maximum(df['comments'], 1)).round(1)
    
    # 6. Text Cleaning Pipeline
    logging.info("Running text cleaning (punctuation removal, lowercasing, stopword filtering)...")
    df['cleaned_title'] = df['title'].apply(lambda x: clean_text(x, remove_stopwords=False))
    df['cleaned_description'] = df['description'].apply(lambda x: clean_text(x, remove_stopwords=True))
    
    # For transcripts, clean cues, punctuation, and lowercase
    # Creating both full cleaned transcript and an excerpt for efficient representation
    df['cleaned_transcript'] = df['transcript'].apply(lambda x: clean_text(x, remove_stopwords=True))
    
    # Tags as cleaned space-separated string
    df['tags_string'] = df['tags_list'].apply(lambda x: " ".join([re.sub(r'[^a-z0-9]', '', t.lower()) for t in x]) if isinstance(x, list) else "")
    
    # Speaker name normalized
    df['clean_speaker'] = df['main_speaker'].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', str(x)).strip())
    
    # 7. Content Soup Construction (Combines Title, Speaker, Occupation, Tags, Description, and Key Transcript context)
    logging.info("Constructing enriched content soup for recommendation models...")
    df['content_soup'] = (
        df['cleaned_title'] + " " +
        df['clean_speaker'] + " " +
        df['speaker_occupation'].apply(lambda x: clean_text(str(x), remove_stopwords=True)) + " " +
        df['tags_string'] + " " +
        df['cleaned_description']
    )
    
    # 8. Categorical Feature Encoding
    df['speaker_id'] = pd.factorize(df['main_speaker'])[0] + 1
    
    # 9. Prepare User-Item Interaction Matrix for Collaborative Filtering
    interactions_df = generate_collaborative_interactions(df, num_users=500)
    interactions_path = os.path.join(PROCESSED_DIR, 'user_item_interactions.csv')
    interactions_df.to_csv(interactions_path, index=False)
    logging.info(f"Saved user-item interactions to {interactions_path}")
    
    # 10. Summary Statistics Capturing Data Quality
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
    
    summary_path = os.path.join(PROCESSED_DIR, 'data_summary_stats.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_stats, f, indent=4)
    logging.info(f"Saved summary statistics to {summary_path}")
    
    # 11. Save Cleaned Dataset
    output_csv = os.path.join(PROCESSED_DIR, 'ted_talks_cleaned.csv')
    df.to_csv(output_csv, index=False)
    logging.info(f"Cleaned dataset saved successfully to {output_csv} with {len(df)} records and {df.shape[1]} columns.")
    
    return df, summary_stats

if __name__ == '__main__':
    run_preprocessing_pipeline()
