#!/usr/bin/env python3
"""
Integrated pipeline script to download, preprocess, train, and evaluate
deepfake detection models (Xception and RGB+FFT Dual-Stream) end-to-end.
"""

import os
import sys
import argparse
import subprocess
import json
import glob
from pathlib import Path


def run_command(cmd, desc=None):
    """Run a shell command and print output."""
    if desc:
        print("\n" + "=" * 60)
        print(f" {desc}")
        print("=" * 60)
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"Error: Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    return result


def main():
    parser = argparse.ArgumentParser(description="End-to-end Deepfake Detection Pipeline")
    parser.add_argument("--model", type=str, default="all",
                        choices=["xception", "rgb_fft_dual_stream", "all"],
                        help="Model to run: xception, rgb_fft_dual_stream, or all")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"],
                        help="Dataset split to evaluate")
    parser.add_argument("--num_videos", type=int, default=50, help="Number of videos to download if missing")
    parser.add_argument("--server", type=str, default="EU2", help="FaceForensics mirror server (EU2/CA/EU)")
    parser.add_argument("--skip_download", action="store_true", help="Skip dataset download step")
    parser.add_argument("--skip_preprocess", action="store_true", help="Skip dataset preprocessing step")
    parser.add_argument("--skip_train", action="store_true", help="Skip training and just evaluate existing checkpoints")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save evaluation results")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    os.chdir(project_root)

    # 1. Dataset Downloading
    raw_dir = project_root / "data" / "faceforensics_raw"
    if not args.skip_download:
        # Check if videos exist
        videos_exist = len(list(raw_dir.rglob("*.mp4"))) > 0
        if not videos_exist:
            os.makedirs(raw_dir, exist_ok=True)
            print("\nDataset missing. Initiating download...")
            # Download Deepfakes
            run_command([
                sys.executable,
                str(project_root / "scripts" / "download" / "download_faceforensics.py"),
                str(raw_dir),
                "-d", "Deepfakes",
                "-c", "c23",
                "-t", "videos",
                "-n", str(args.num_videos),
                "--server", args.server,
                "-y"
            ], "Downloading FaceForensics++ Deepfakes Subset")

            # Download Original
            run_command([
                sys.executable,
                str(project_root / "scripts" / "download" / "download_faceforensics.py"),
                str(raw_dir),
                "-d", "original",
                "-c", "c23",
                "-t", "videos",
                "-n", str(args.num_videos),
                "--server", args.server,
                "-y"
            ], "Downloading FaceForensics++ Original Subset")
        else:
            print("\n✓ Raw videos already exist in data/faceforensics_raw. Skipping download.")

    # 2. Dataset Preprocessing
    metadata_train = project_root / "data" / "train_metadata.json"
    
    # Auto-force preprocessing if existing metadata contains only one class
    force_preprocess = False
    if metadata_train.exists():
        try:
            with open(metadata_train, "r") as f:
                meta = json.load(f)
                labels = {item['label'] for item in meta}
                if len(labels) < 2:
                    print("\n⚠️  Existing preprocessed metadata contains only 1 class. Forcing balanced preprocessing...")
                    force_preprocess = True
        except Exception:
            force_preprocess = True

    if not args.skip_preprocess:
        if force_preprocess or not metadata_train.exists() or not (project_root / "data" / "test_metadata.json").exists():
            run_command([
                sys.executable,
                str(project_root / "scripts" / "preprocess.py"),
                "--config", args.config,
                "--dataset-type", "faceforensics",
                "--videos-dir", str(raw_dir),
                "--max_videos", str(args.num_videos * 2)
            ], "Preprocessing Dataset (MTCNN alignment + FFT Extraction)")
        else:
            print("\n✓ Preprocessed dataset metadata already exists. Skipping preprocessing.")

    # Determine models to run
    models_to_run = ["xception", "rgb_fft_dual_stream"] if args.model == "all" else [args.model]

    # 3. Model Training & Evaluation
    for model_name in models_to_run:
        # Train model
        if not args.skip_train:
            run_command([
                sys.executable,
                str(project_root / "scripts" / "train.py"),
                "--model", model_name,
                "--config", args.config,
                "--epochs", str(args.epochs)
            ], f"Training Model: {model_name.upper()} ({args.epochs} Epochs)")

        # Determine checkpoint path (fallback to latest if best model is missing/not updated)
        checkpoint_path = project_root / "checkpoints" / f"best_model_{model_name}.pth"
        if not checkpoint_path.exists():
            fallback_path = project_root / "checkpoints" / f"latest_{model_name}.pth"
            if fallback_path.exists():
                print(f"⚠️  best_model_{model_name}.pth not found. Falling back to latest_{model_name}.pth")
                checkpoint_path = fallback_path

        # Evaluate model
        run_command([
            sys.executable,
            str(project_root / "scripts" / "evaluate.py"),
            "--model", model_name,
            "--config", args.config,
            "--checkpoint", str(checkpoint_path),
            "--split", args.split,
            "--optimal_threshold",
            "--output_dir", args.output_dir
        ], f"Evaluating Model: {model_name.upper()}")

    # 4. Generate & Display Summary Table
    print("\n" + "=" * 60)
    print(" PIPELINE RESULTS SUMMARY (FaceForensics++ Deepfakes, c23)")
    print("=" * 60)
    print(f"{'Method':<25} | {'Accuracy':<10} | {'F1-score':<10} | {'AUC':<8} | {'EER':<8}")
    print("-" * 65)

    summary_file = Path(args.output_dir) / "pipeline_summary.txt"
    with open(summary_file, "w") as sf:
        sf.write("=" * 60 + "\n")
        sf.write(" PIPELINE RESULTS SUMMARY (FaceForensics++ Deepfakes, c23)\n")
        sf.write("=" * 60 + "\n")
        sf.write(f"{'Method':<25} | {'Accuracy':<10} | {'F1-score':<10} | {'AUC':<8} | {'EER':<8}\n")
        sf.write("-" * 65 + "\n")

        # Read results files
        for model_name in models_to_run:
            results_path = Path(args.output_dir) / f"{model_name}_{args.split}_results.txt"
            if results_path.exists():
                metrics = {}
                with open(results_path, "r") as rf:
                    for line in rf:
                        if ":" in line:
                            parts = line.split(":")
                            key = parts[0].strip().lower()
                            val = parts[1].strip()
                            metrics[key] = val

                model_disp_name = "Xception (baseline)" if model_name == "xception" else "RGB+FFT dual-stream"
                accuracy = metrics.get("accuracy", "N/A")
                f1_score = metrics.get("f1-score", "N/A")
                auc = metrics.get("auc", "N/A")
                eer = metrics.get("eer", "N/A")

                row = f"{model_disp_name:<25} | {accuracy:<10} | {f1_score:<10} | {auc:<8} | {eer:<8}"
                print(row)
                sf.write(row + "\n")

    print("-" * 65)
    print(f"\nConsolidated results saved to: {summary_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
