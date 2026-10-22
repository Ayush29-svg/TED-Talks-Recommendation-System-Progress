# Personalized TED Talks Recommendation System Using Machine Learning

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/Milestone%201-Completed-success.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)]()

---

## 1. Project Statement
With an ever-growing repository of TED Talks covering diverse subjects—from artificial intelligence and biotechnology to design, psychology, and global issues—users struggle to efficiently discover talks aligned with their interests. 

This project develops an intelligent, machine learning–powered recommendation system combining **Content-Based Filtering** (leveraging talk transcripts, metadata, and tags) and **Collaborative Filtering** (leveraging user interaction patterns and preferences). The final deliverable will be deployed via an interactive **Streamlit** web application.

---

## 2. Milestone 1 (Week 1-2) Implementation Summary

### Module 1: Data Collection
- **Source**: Canonical TED Talks archive and transcripts repository.
- **Data Volume**:
  - `ted_main.csv`: **2,628** talks with 17 metadata fields.
  - `transcripts.csv`: **2,542** verbatim transcripts.
- **Richness**:
  - Over **2,228 unique speakers** and **1,513 distinct professions**.
  - **423 unique topic tags** spanning interdisciplinary domains.
  - **Over 4.5 million community ratings** across 14 emotional and cognitive feedback dimensions.

### Module 2: Data Exploration & Data Preprocessing
- **Data Inspection & Cleaning**:
  - Handled encoding nuances (`latin1` for metadata, UTF-8 with error replacement for international transcripts).
  - Merged talk metadata and transcripts on unique `url` identifiers.
  - Resolved missing transcripts (86 talks) by imputing title and description to preserve complete catalog coverage.
  - Resolved missing `speaker_occupation` (6 records) with `'Unknown'`.
  - Achieved **0 missing values** across the final 43-column dataset.
- **Text Normalization Pipeline**:
  - Stripped transcript sound cues (e.g., `(Laughter)`, `(Applause)`, `[Music]`).
  - Standardized to lowercase, removed special characters/punctuation.
  - Filtered English stopwords using scikit-learn's vetted lexicon.
  - Constructed enriched **`content_soup`** integrating title, speaker, occupation, tags, and cleaned description for downstream vectorization.
- **Feature Engineering & Similarity Metrics**:
  - `duration_minutes`: Talk length in minutes (mean: 13.73 mins, median: 14.06 mins).
  - `transcript_word_count`: Word count per talk (mean: 1,955 words).
  - `words_per_minute`: Speaking rate pacing metric.
  - `views_per_comment`: Audience interaction density ratio.
  - `tag_count`: Tag diversity metric (mean: 7.57 tags per talk).
  - `dominant_sentiment`: Primary emotional response from community voting.
- **Collaborative Filtering Setup**:
  - Extracted `talk_ratings_matrix.csv` with community voting distributions across all 14 TED rating attributes.
  - Generated `user_item_interactions.csv` representing **18,323 interactions** across **500 diverse user profiles** with 98.6% matrix sparsity, formatted for Matrix Factorization (SVD/ALS) in Milestone 2.
- **Visualizations**:
  - Generated 8 publication-grade distribution plots and correlation heatmaps saved to `eda_visualizations/`.

---

## 3. Repository Architecture

```text
C:\Drivers\
├── data\
│   ├── raw\
│   │   ├── ted_main.csv                     # Original metadata (2,628 talks)
│   │   └── transcripts.csv                  # Original transcripts (2,542 talks)
│   └── processed\
│       ├── ted_talks_cleaned.csv            # Cleaned master dataset (43 features)
│       ├── user_item_interactions.csv       # User-item rating interaction benchmark
│       ├── talk_ratings_matrix.csv          # Community ratings per talk (14 dimensions)
│       └── data_summary_stats.json          # Data quality and summary statistics
├── eda_visualizations\
│   ├── talk_duration_distribution.png       # Duration distribution (mean/median)
│   ├── top_25_tags_distribution.png         # Frequency of top 25 topic tags
│   ├── views_and_comments_distribution.png  # Log-scale views & comments reach
│   ├── rating_categories_distribution.png   # Millions of votes across 14 feedback types
│   ├── transcript_word_count_distribution.png # Word count & speaking rate (WPM)
│   ├── feature_correlation_matrix.png       # Correlation matrix of numerical features
│   ├── top_speakers_and_occupations.png     # Distribution of top speakers & jobs
│   └── user_interaction_ratings_distribution.png # Collaborative ratings (1-5 stars)
├── notebooks\
│   └── Milestone_1_EDA_and_Preprocessing.ipynb # Interactive demonstration notebook
├── src\
│   ├── data_collection.py                   # Automated raw data ingestion script
│   ├── data_preprocessing.py                # Preprocessing, text cleaning & matrix prep
│   ├── eda_analysis.py                      # EDA plotting and visualization generation
│   ├── create_notebook.py                   # Notebook generator
│   └── run_milestone1.py                    # Master end-to-end execution runner
└── README.md
```

---

## 4. Key EDA Insights

1. **Duration Norms**: TED talks center tightly around the classic 15–18 minute format (median: 14.06 mins; mean: 13.73 mins). Short talks (~3–5 minutes) and extended talks (>30 minutes) form distinct tails.
2. **Topic Diversity**: Top tags are led by `technology`, `science`, `global issues`, `culture`, `design`, and `business`, reflecting strong potential for thematic cluster discovery.
3. **Engagement Dynamics**: Both Views and Comments follow a log-normal distribution with long right tails. The most viewed talks exceed 40M+ views, while median reach is ~1.11M views.
4. **Dominant Sentiment Dimensions**: Community votes overwhelmingly classify top talks as *Inspiring* (2.1M+ votes), *Informative* (1.8M+ votes), and *Fascinating* (1.5M+ votes).
5. **Speaking Rate**: Most speakers deliver talks at an average rate of 130–165 words per minute.

---

## 5. How to Run the Pipeline

### Prerequisites
Ensure Python 3.10+ and required packages are installed:
```powershell
pip install pandas scikit-learn matplotlib seaborn nltk
```

### Run Entire Milestone 1 Pipeline
To run data collection, cleaning, feature engineering, and visualization generation:
```powershell
python src/run_milestone1.py
```

### Inspect Output Files
- Preprocessed dataset: `data/processed/ted_talks_cleaned.csv`
- Quality statistics: `data/processed/data_summary_stats.json`
- Visualization charts: `eda_visualizations/*.png`
- Notebook: `notebooks/Milestone_1_EDA_and_Preprocessing.ipynb`

---

## 6. Next Steps: Milestone 2 Preview (Week 3-4)
- **Module 3: Feature Engineering and Vectorization**:
  - TF-IDF vectorization on `content_soup` and transcripts.
  - Word2Vec / BERT dense embeddings for semantic search.
  - Sparse representation of user-item interaction matrix.
- **Module 4: Develop Recommendation Engine**:
  - Cosine similarity content-based recommender.
  - Collaborative filtering via Matrix Factorization (SVD / TruncatedSVD / ALS).
  - Hybrid scoring architecture combining content similarity and user affinities.
