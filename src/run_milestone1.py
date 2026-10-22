"""
Milestone 1 Pipeline Execution Script
Runs data collection, data preprocessing, and exploratory visualization generation.
"""
import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    logging.info("Starting Milestone 1 pipeline...")
    
    # 1. Collect raw datasets
    logging.info("--- Step 1: Data Collection ---")
    from data_collection import collect_data
    collect_data()
    
    # 2. Preprocess metadata and transcripts
    logging.info("--- Step 2: Data Preprocessing ---")
    from data_preprocessing import run_preprocessing_pipeline
    df_cleaned, summary = run_preprocessing_pipeline()
    
    # 3. Generate exploratory data plots
    logging.info("--- Step 3: Exploratory Data Analysis ---")
    from eda_analysis import generate_eda_charts
    generate_eda_charts()
    
    logging.info("Milestone 1 completed successfully.")
    logging.info(f"Total talks processed: {summary['total_talks']}")
    logging.info("Outputs generated in data/processed/ and eda_visualizations/")

if __name__ == '__main__':
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    main()
