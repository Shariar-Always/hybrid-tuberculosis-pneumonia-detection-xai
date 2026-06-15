import os
import pandas as pd
from pathlib import Path

def generate_metadata():
    # Define paths
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data"
    
    # Path to the downloaded dataset splits CSV (tries local Windows paths and Colab paths)
    paths_to_try = [
        Path(r"E:\Download\TB Data\data\dataset_splits_with_metadata.csv"),
        Path(r"/content/colab_data/data/dataset_splits_with_metadata.csv"),
        Path(r"/content/colab_data/dataset_splits_with_metadata.csv"),
        base_dir / "data" / "dataset_splits_with_metadata.csv"
    ]
    
    downloaded_csv = None
    for p in paths_to_try:
        if p.exists():
            downloaded_csv = p
            break
            
    print("=" * 60)
    print("      Tuberculosis & Pneumonia CXR Metadata Generator")
    print("=" * 60)
    
    if downloaded_csv is None:
        print(f"[-] Error: Could not find dataset splits CSV in any of the expected paths:")
        for p in paths_to_try:
            print(f"    - {p}")
        return False
        
    print(f"[+] Reading downloaded metadata from: {downloaded_csv}")
    df = pd.read_csv(downloaded_csv)
    
    # Filter out UNKNOWN labels
    df = df[df['label'] != 'UNKNOWN'].copy()
    
    # Construct image_path relative to dataset root
    df['image_path'] = df['split'] + '/' + df['label'] + '/' + df['name']
    
    # Map text labels to numeric labels
    label_map = {
        'NORMAL': 0,
        'TUBERCULOSIS': 1,
        'PNEUMONIA': 2
    }
    df['class_name'] = df['label']
    df['label'] = df['label'].map(label_map)
    
    # Reorder columns to match previous metadata structure
    df = df[['image_path', 'label', 'class_name', 'split', 'source']]
    
    # Save to data/metadata.csv inside workspace
    metadata_csv = data_dir / "metadata.csv"
    os.makedirs(data_dir, exist_ok=True)
    df.to_csv(metadata_csv, index=False)
    
    print("-" * 60)
    counts = df['class_name'].value_counts()
    for cls, cnt in counts.items():
        print(f"[+] Total {cls} scans found: {cnt}")
    print(f"[+] Metadata CSV successfully generated at: {metadata_csv}")
    print(f"[+] Total entries in dataset: {len(df)}")
    print("=" * 60)
    return True

if __name__ == "__main__":
    generate_metadata()
