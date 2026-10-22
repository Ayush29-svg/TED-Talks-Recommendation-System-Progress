"""
Module 1: Data Collection
Downloads the raw TED Talks metadata and transcript datasets.
"""
import os
import urllib.request
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Primary URLs for TED talks metadata and transcripts
DATA_SOURCES = {
    'ted_main.csv': 'https://raw.githubusercontent.com/owentemple/TED-talks/master/data/ted_main.csv',
    'transcripts.csv': 'https://raw.githubusercontent.com/owentemple/TED-talks/master/data/transcripts.csv'
}

RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'raw')

def download_dataset(url: str, output_path: str):
    """Download a dataset file if it does not already exist."""
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        logging.info(f"Already downloaded: {os.path.basename(output_path)} ({os.path.getsize(output_path):,} bytes)")
        return
    
    logging.info(f"Downloading {os.path.basename(output_path)} from {url}...")
    headers = {'User-Agent': 'Mozilla/5.0'}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as response, open(output_path, 'wb') as out_file:
        chunk_size = 64 * 1024
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
    logging.info(f"Saved {os.path.basename(output_path)} ({os.path.getsize(output_path):,} bytes)")

def collect_data():
    """Run data collection step."""
    os.makedirs(RAW_DIR, exist_ok=True)
    for filename, url in DATA_SOURCES.items():
        dest = os.path.join(RAW_DIR, filename)
        download_dataset(url, dest)
    logging.info("Data collection finished.")

if __name__ == '__main__':
    collect_data()
