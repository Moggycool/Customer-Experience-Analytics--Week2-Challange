#!/usr/bin/env python3
"""
Master Script to Run Complete Analysis Pipeline
"""

import os
import sys
import subprocess
from datetime import datetime
# --------------------------
# Configure logging
# --------------------------
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "pipeline.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)



def run_command(cmd_description, cmd):
    """Run a shell command and handle errors."""
    print(f"\n{'='*60}")
    print(f"[INFO] {cmd_description}")
    print(f"{'='*60}")
    print(f"Command: {cmd}")

    try:
        result = subprocess.run(
            cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"[SUCCESS] {cmd_description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[FAILED] {cmd_description}")
        print(f"Error: {e.stderr}")
        return False


def main():
    """Run the complete analysis pipeline."""
    print("\n" + "="*60)
    print("[INFO] BANK REVIEWS ANALYSIS PIPELINE")
    print("="*60)

    # Define paths - CORRECTED: Use project root, not src directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)  # Go up one level from src/

    raw_data = os.path.join(project_root, "data", "raw",
                            "google_play_raw_reviews.csv")
    processed_dir = os.path.join(project_root, "data", "preprocessed")

    # Step 1: Preprocessing
    print(f"\n[INFO] Step 1: Preprocessing")
    print(f"[INFO] Looking for raw data at: {raw_data}")
    print(f"[INFO] Output will be saved to: {processed_dir}")

    if not os.path.exists(raw_data):
        print(f"[ERROR] Raw data file not found: {raw_data}")
        print("[INFO] Please check the file exists.")
        return

    if not run_command(
        "Preprocessing raw reviews",
        f"python -m src.preprocessing"
    ):
        print("\n[ERROR] Preprocessing failed. Exiting.")
        return

    # Step 2: Sentiment Analysis
    processed_data = os.path.join(
        processed_dir, "google_play_processed_reviews.csv")

    print(f"\n[INFO] Step 2: Sentiment Analysis")
    print(f"[INFO] Looking for processed data at: {processed_data}")

    if not os.path.exists(processed_data):
        print(f"[ERROR] Processed data not found: {processed_data}")
        print("[INFO] Please check if preprocessing completed successfully.")
        return

    if not run_command(
        "Running sentiment analysis",
        f"python -m src.sentiment_analysis {processed_data} {processed_dir}/sentiment_preprocessed.csv --method textblob"
    ):
        print(
            "\n[WARNING] Sentiment analysis failed, but continuing with thematic analysis...")

    # Step 3: Thematic Analysis
    print(f"\n[INFO] Step 3: Thematic Analysis")
    print(f"[INFO] Using processed data: {processed_data}")

    if not run_command(
        "Running thematic analysis",
        f"python -m src.thematic_analysis {processed_data}"
    ):
        print("\n[WARNING] Thematic analysis failed.")

    # Final Summary
    print("\n" + "="*60)
    print("[INFO] PIPELINE EXECUTION SUMMARY")
    print("="*60)

    files_to_check = [
        ("Preprocessed Reviews", "google_play_processed_reviews.csv"),
        ("Sentiment Results", "sentiment_preprocessed.csv"),
        ("Thematic Analysis", "thematic_analysis.csv"),
        ("Thematic Summary", "thematic_summary.csv"),
        ("Thematic Metrics", "thematic_metrics.csv"),
        ("Analysis Report", "thematic_analysis_report.json")
    ]

    created_count = 0
    missing_count = 0

    for desc, filename in files_to_check:
        filepath = os.path.join(processed_dir, filename)
        if os.path.exists(filepath):
            file_size = os.path.getsize(filepath) / 1024  # Size in KB
            print(f"✅ {desc}: {filename} ({file_size:.1f} KB)")
            created_count += 1
        else:
            print(f"❌ {desc}: {filename} (not found)")
            missing_count += 1

    print(
        f"\n[INFO] Summary: {created_count} files created, {missing_count} files missing")
    print(f"[INFO] All outputs saved to: {processed_dir}")
    print(
        f"[INFO] Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if missing_count == 0:
        print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    else:
        print(
            f"\n⚠️  PIPELINE PARTIALLY COMPLETED ({missing_count} files missing)")
>>>>>>> task-2


if __name__ == "__main__":
    main()
