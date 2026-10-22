# Personalized TED Talks Recommendation System

A machine learning project designed to recommend relevant TED Talks based on content similarity (transcripts, tags, talk descriptions) and collaborative user engagement signals.

Developed as part of the Springboard Internship Program.

---

## Project Overview

With thousands of TED Talks spanning diverse topics—such as technology, science, design, psychology, and global issues—it can be challenging for viewers to find talks matching their specific interests. 

This project aims to build a hybrid recommendation engine combining:
1. **Content-Based Filtering**: Recommends talks with similar subjects, transcripts, tags, and speakers using natural language processing.
2. **Collaborative Filtering**: Suggests talks based on user watch histories, ratings, and shared interest profiles.
3. **Streamlit Web Application**: An interactive interface allowing users to explore recommendations and filter by topic or speaker.

---

## Milestone 1: Data Collection & Data Preprocessing (Week 1–2)

### Module 1: Data Collection
- **Metadata (`ted_main.csv`)**: 2,628 talks containing 17 fields including title, speaker, speaker occupation, tags, views, comments, duration, publication date, and community ratings.
- **Transcripts (`transcripts.csv`)**: 2,542 full verbatim transcripts mapped to talks by URL.
- **Dataset Diversity**: Spans over 2,228 unique speakers, 1,513 occupations, and 423 unique topic tags.

### Module 2: Data Exploration & Preprocessing
- **Handling Missing Data**:
  - Imputed missing transcripts (86 talks) using talk titles and descriptions to ensure 100% catalog coverage for recommendation models.
  - Filled missing speaker occupations (6 talks) with `"Unknown"`.
  - Zero missing values in the final processed dataset (43 columns).
- **Text Normalization**:
  - Stripped transcript sound cues such as `(Laughter)`, `(Applause)`, and `[Music]`.
  - Converted text to lowercase, removed special characters, punctuation, and URLs.
  - Filtered common English stopwords using scikit-learn's stopword dictionary.
  - Built an aggregated `content_soup` feature combining title, speaker, occupation, tags, and description for vectorization in Milestone 2.
- **Feature Engineering**:
  - `duration_minutes`: Duration in minutes (mean: 13.73 mins, median: 14.06 mins).
  - `transcript_word_count`: Word count per transcript (mean: 1,955 words).
  - `words_per_minute`: Speaking rate pacing metric.
  - `views_per_comment`: Audience discussion ratio.
  - `tag_count`: Number of assigned topic tags (mean: 7.57 tags per talk).
  - `dominant_sentiment`: Most frequent community rating sentiment (e.g., Inspiring, Informative, Fascinating).
- **Collaborative Filtering Preparation**:
  - Extracted community rating distributions into `talk_ratings_matrix.csv`.
  - Generated a benchmark user-item interaction log (`user_item_interactions.csv`) with 18,323 ratings across 500 user profiles representing various interest groups (technology, science, arts, global issues, business, psychology).

---

## Key Exploratory Findings

1. **Talk Durations**: Most talks adhere closely to the classic 15–18 minute TED window, with a median duration of 14.06 minutes.
2. **Top Topic Tags**: The most frequent tags are `technology` (727), `science` (567), `global issues` (501), `culture` (486), `design` (418), and `business` (348).
3. **Audience Engagement**: Talk views and comments follow a right-skewed log-normal distribution. While median views are ~1.11 million, top talks reach over 40 million views.
4. **Community Feedback**: The most common audience reactions across talks are *Inspiring* (2.1M+ votes), *Informative* (1.8M+ votes), and *Fascinating* (1.5M+ votes).

Visualizations generated during EDA can be found in the [`eda_visualizations/`](eda_visualizations/) directory.

---

## Repository Structure

```text
├── data/
│   ├── raw/
│   │   ├── ted_main.csv                     # Original talk metadata
│   │   └── transcripts.csv                  # Original verbatim transcripts
│   └── processed/
│       ├── ted_talks_cleaned.csv            # Cleaned dataset (43 features)
│       ├── user_item_interactions.csv       # User interaction matrix for collaborative filtering
│       ├── talk_ratings_matrix.csv          # Community ratings per talk across 14 categories
│       └── data_summary_stats.json          # Data quality summary metrics
├── eda_visualizations/                      # EDA charts and distribution plots
├── notebooks/
│   └── Milestone_1_EDA_and_Preprocessing.ipynb # Step-by-step EDA notebook
├── src/
│   ├── data_collection.py                   # Dataset ingestion script
│   ├── data_preprocessing.py                # Preprocessing and text cleaning pipeline
│   ├── eda_analysis.py                      # Exploratory plotting script
│   └── run_milestone1.py                    # Pipeline runner
├── requirements.txt                         # Python dependencies
├── LICENSE                                  # MIT License
└── README.md                                # Project documentation
```

---

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/Ayush29-svg/TED-Talks-Recommendation-System-Progress.git
cd TED-Talks-Recommendation-System-Progress
```

### 2. Set up virtual environment & install requirements
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Run the Milestone 1 Pipeline
To re-run data collection, cleaning, and chart generation:
```bash
python src/run_milestone1.py
```

### 4. Open the Jupyter Notebook
```bash
jupyter notebook notebooks/Milestone_1_EDA_and_Preprocessing.ipynb
```

---

## Next Steps: Milestone 2 (Week 3–4)

- **Module 3: Feature Engineering and Vectorization**:
  - Implement TF-IDF vectorization and word/sentence embeddings on `content_soup` and transcripts.
  - Construct sparse user-item interaction matrices for collaborative filtering.
- **Module 4: Recommendation Engine Development**:
  - Develop content-based similarity models using cosine similarity.
  - Implement collaborative filtering using Matrix Factorization (SVD / ALS).
  - Design a hybrid recommendation model combining content and user preferences.

---

## Author
**Ayush Gupta**  
GitHub: [@Ayush29-svg](https://github.com/Ayush29-svg)
