"""
Master Pipeline Runner for Milestone 1:
Module 1: Data Collection
Module 2: Data Exploration (EDA) & Data Preprocessing
"""
import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_all():
    logging.info("=" * 60)
    logging.info("STARTING MILESTONE 1 PIPELINE")
    logging.info("=" * 60)
    
    # Step 1: Data Collection
    logging.info("\n--- STEP 1: MODULE 1 - DATA COLLECTION ---")
    from data_collection import collect_data
    collect_data()
    
    # Step 2: Data Preprocessing & Cleaning
    logging.info("\n--- STEP 2: MODULE 2 - DATA PREPROCESSING ---")
    from data_preprocessing import run_preprocessing_pipeline
    df_cleaned, summary_stats = run_preprocessing_pipeline()
    
    # Step 3: Exploratory Data Analysis & Visualizations
    logging.info("\n--- STEP 3: MODULE 2 - EXPLORATORY DATA ANALYSIS ---")
    from eda_analysis import generate_eda_charts
    generate_eda_charts()
    
    logging.info("\n" + "=" * 60)
    logging.info("MILESTONE 1 COMPLETED SUCCESSFULLY!")
    logging.info(f"Total Talks Processed: {summary_stats['total_talks']}")
    logging.info(f"Cleaned Dataset: data/processed/ted_talks_cleaned.csv")
    logging.info(f"User Interactions: data/processed/user_item_interactions.csv")
    logging.info(f"Talk Ratings Matrix: data/processed/talk_ratings_matrix.csv")
    logging.info(f"Data Summary Stats: data/processed/data_summary_stats.json")
    logging.info(f"EDA Plots Directory: eda_visualizations/")
    logging.info("=" * 60)

if __name__ == '__main__':
    # Add src to path if needed
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    run_all()
