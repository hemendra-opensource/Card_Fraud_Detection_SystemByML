import os
import urllib.request
import zipfile
import sys

def download_and_extract():
    url = "https://storage.googleapis.com/spls/gsp774/archive.zip"
    raw_dir = os.path.abspath("data/raw")
    zip_path = os.path.join(raw_dir, "archive.zip")
    
    print(f"Creating raw data directory: {raw_dir}")
    os.makedirs(raw_dir, exist_ok=True)
    
    print(f"Downloading dataset from {url}...")
    try:
        urllib.request.urlretrieve(url, zip_path)
        print("Download complete!")
    except Exception as e:
        print(f"Error downloading dataset: {e}", file=sys.stderr)
        return False
        
    print(f"Extracting {zip_path} to {raw_dir}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(raw_dir)
        print("Extraction complete!")
    except Exception as e:
        print(f"Error extracting dataset: {e}", file=sys.stderr)
        return False
        
    print("Cleaning up zip file...")
    try:
        os.remove(zip_path)
        print("Cleanup complete!")
    except Exception as e:
        print(f"Error removing zip file: {e}", file=sys.stderr)
        
    print("Files in data/raw:")
    for f in os.listdir(raw_dir):
        print(f" - {f} ({os.path.getsize(os.path.join(raw_dir, f)) / (1024*1024):.2f} MB)")
        
    return True

if __name__ == "__main__":
    download_and_extract()
