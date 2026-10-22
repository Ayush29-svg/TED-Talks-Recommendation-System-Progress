import os
import urllib.request
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DATA_URLS = {
    'ted_main.csv': 'https://raw.githubusercontent.com/owentemple/TED-talks/master/data/ted_main.csv',
    'transcripts.csv': 'https://raw.githubusercontent.com/owentemple/TED-talks/master/data/transcripts.csv'
}

RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'raw')

def download_file(url: str, dest_path: str):
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000:
        logging.info(f"File already exists: {dest_path} ({os.path.getsize(dest_path):,} bytes)")
        return
    logging.info(f"Downloading from {url} to {dest_path}...")
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
    with urllib.request.urlopen(req) as resp, open(dest_path, 'wb') as f:
        total_bytes = 0
        while True:
            chunk = resp.read(1024 * 64)
            if not chunk:
                break
            f.write(chunk)
            total_bytes += len(chunk)
    logging.info(f"Downloaded {dest_path} successfully ({total_bytes:,} bytes).")

def collect_data():
    os.makedirs(RAW_DIR, exist_ok=True)
    logging.info(f"Data directory: {RAW_DIR}")
    for filename, url in DATA_URLS.items():
        dest = os.path.join(RAW_DIR, filename)
        download_file(url, dest)
    logging.info("Data collection complete.")

if __name__ == '__main__':
    collect_data()
