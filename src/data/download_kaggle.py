"""Download UNSW-NB15 dataset using kagglehub and copy to data/raw/."""

import os
import shutil
from pathlib import Path
import kagglehub

def download_and_setup_dataset():
    print("[*] Downloading dataset 'dhoogla/unswnb15' via kagglehub...")
    download_path = kagglehub.dataset_download("dhoogla/unswnb15")
    print(f"[+] Download complete. Kagglehub cache path: {download_path}")

    source_dir = Path(download_path)
    target_dir = Path(__file__).resolve().parent.parent.parent / "data" / "raw"
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Copying files to project raw directory: {target_dir}")
    downloaded_files = list(source_dir.rglob("*"))
    copied_count = 0

    for file_path in downloaded_files:
        if file_path.is_file():
            # Relative path from source_dir
            rel_path = file_path.relative_to(source_dir)
            dest_file = target_dir / rel_path
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, dest_file)
            size_mb = dest_file.stat().st_size / (1024 * 1024)
            print(f"  -> Copied: {dest_file.name} ({size_mb:.2f} MB)")
            copied_count += 1

    print(f"[✓] Successfully copied {copied_count} file(s) into {target_dir}")

if __name__ == "__main__":
    download_and_setup_dataset()
