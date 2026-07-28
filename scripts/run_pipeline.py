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
    parser.add_argument("--force_reprocess", action="store_true", help="Force regeneration of preprocessed metadata")
    parser.add_argument("--smoke_test", action="store_true", help="Run end-to-end smoke test on a tiny balanced subset")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2026], help="List of random seeds to run over")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    os.chdir(project_root)

    # Overrides for smoke_test mode
    if args.smoke_test:
        print("\n" + "=" * 60)
        print(" RUNNING INTEGRATED E2E SMOKE TEST")
        print("=" * 60)
        args.epochs = 1
        args.num_videos = 3  # Balance real/fake = 6 videos total
        args.force_reprocess = True
        args.split = "test"
        args.model = "all"

    raw_dir = project_root / "data" / "faceforensics_raw"
    if not args.skip_download:
        # Check if enough videos exist per category
        reals_existing = list((raw_dir / "original_sequences").rglob("*.mp4"))
        fakes_existing = list((raw_dir / "manipulated_sequences").rglob("*.mp4"))
        
        # If flat structure check
        if not reals_existing and not fakes_existing:
            all_videos = list(raw_dir.rglob("*.mp4"))
            reals_existing = [v for v in all_videos if "original" in str(v).lower() or "real" in str(v).lower()]
            fakes_existing = [v for v in all_videos if v not in reals_existing]
            
        needs_download = (len(reals_existing) < args.num_videos) or (len(fakes_existing) < args.num_videos)
        
        if needs_download:
            os.makedirs(raw_dir, exist_ok=True)
            print(f"\nDataset incomplete (Found: {len(reals_existing)} real, {len(fakes_existing)} fake; Needed: {args.num_videos} of each). Initiating download...")
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
            print(f"\n✓ Sufficient raw videos already exist in data/faceforensics_raw ({len(reals_existing)} real, {len(fakes_existing)} fake). Skipping download.")

    # 2. Dataset Preprocessing
    metadata_train = project_root / "data" / "train_metadata.json"
    settings_file = project_root / "data" / "preprocessing_settings.json"
    
    # Gather sorted raw video file names (reproducible sorting) to detect dataset changes
    raw_videos = []
    if raw_dir.exists():
        raw_videos = sorted([str(p.name) for p in raw_dir.rglob("*.mp4")] + [str(p.name) for p in raw_dir.rglob("*.avi")])
        
    # Track current preprocessing settings to auto-detect changes
    current_settings = {
        "dataset_type": "faceforensics",
        "videos_dir": str(raw_dir),
        "frame_sampling_rate": 3,
        "face_size": 224,
        "use_mtcnn": True,
        "raw_video_filenames": raw_videos
    }
    
    try:
        import yaml
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)
            current_settings["frame_sampling_rate"] = cfg.get('data', {}).get('frame_sampling_rate', 3)
            current_settings["face_size"] = cfg.get('data', {}).get('face_size', 224)
            current_settings["use_mtcnn"] = cfg.get('preprocessing', {}).get('use_mtcnn', True)
    except Exception:
        pass

    settings_changed = False
    if settings_file.exists():
        try:
            with open(settings_file, 'r') as f:
                old_settings = json.load(f)
                if old_settings != current_settings:
                    print("\n⚠️  Preprocessing configuration settings or raw video files changed. Forcing dataset regeneration...")
                    settings_changed = True
        except Exception:
            settings_changed = True
    else:
        settings_changed = True

    # Auto-force preprocessing if existing metadata contains only one class
    force_preprocess = args.force_reprocess or settings_changed
    if not force_preprocess and metadata_train.exists():
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
            # Update settings filenames before executing preprocess since new download might have run
            raw_videos = []
            if raw_dir.exists():
                raw_videos = sorted([str(p.name) for p in raw_dir.rglob("*.mp4")] + [str(p.name) for p in raw_dir.rglob("*.avi")])
            current_settings["raw_video_filenames"] = raw_videos

            run_command([
                sys.executable,
                str(project_root / "scripts" / "preprocess.py"),
                "--config", args.config,
                "--dataset-type", "faceforensics",
                "--videos-dir", str(raw_dir),
                "--max_videos", str(args.num_videos * 2)
            ], "Preprocessing Dataset (MTCNN alignment + FFT Extraction)")
            
            # Save settings upon successful preprocessing
            os.makedirs(project_root / "data", exist_ok=True)
            try:
                with open(settings_file, 'w') as f:
                    json.dump(current_settings, f, indent=2)
                print("✓ Preprocessing settings cached.")
            except Exception as e:
                print(f"Warning: Could not save preprocessing settings cache ({e}).")
        else:
            print("\n✓ Preprocessed dataset metadata already exists. Skipping preprocessing.")

    # Determine models to run
    models_to_run = ["xception", "rgb_fft_dual_stream"] if args.model == "all" else [args.model]

    # results collector: model -> metric_name -> list of values
    results_agg = {model: {m: [] for m in ['accuracy', 'f1', 'auc', 'eer']} for model in models_to_run}

    for seed in args.seeds:
        print(f"\n" + "=" * 60)
        print(f" PIPELINE RUN WITH SEED: {seed}")
        print("=" * 60)
        
        for model_name in models_to_run:
            # 3. Model Training
            if not args.skip_train:
                train_cmd = [
                    sys.executable,
                    str(project_root / "scripts" / "train.py"),
                    "--model", model_name,
                    "--config", args.config,
                    "--epochs", str(args.epochs),
                    "--seed", str(seed)
                ]
                if args.smoke_test:
                    train_cmd.append("--smoke_test")
                run_command(train_cmd, f"Training Model: {model_name.upper()} (Seed {seed})")

            # Determine checkpoint path (suffix checkpoint name with seed value)
            model_name_clean = model_name.lower().replace("-", "_")
            checkpoint_path = project_root / "checkpoints" / f"best_model_{model_name_clean}_seed{seed}.pth"
            if not checkpoint_path.exists():
                fallback_path = project_root / "checkpoints" / f"latest_{model_name_clean}_seed{seed}.pth"
                if fallback_path.exists():
                    print(f"⚠️  best checkpoint not found. Falling back to latest.")
                    checkpoint_path = fallback_path

            # 4. Evaluate Model
            eval_cmd = [
                sys.executable,
                str(project_root / "scripts" / "evaluate.py"),
                "--model", model_name,
                "--config", args.config,
                "--checkpoint", str(checkpoint_path),
                "--split", args.split,
                "--optimal_threshold",
                "--output_dir", args.output_dir,
                "--seed", str(seed)
            ]
            if args.smoke_test:
                eval_cmd.append("--allow_random_weights")
            run_command(eval_cmd, f"Evaluating Model: {model_name.upper()} (Seed {seed})")

            # Parse results
            results_path = Path(args.output_dir) / f"{model_name_clean}_{args.split}_results_seed{seed}.txt"
            if results_path.exists():
                metrics = {}
                with open(results_path, "r") as rf:
                    for line in rf:
                        if ":" in line:
                            parts = line.split(":")
                            key = parts[0].strip().lower()
                            if key == "f1-score":
                                key = "f1"
                            elif key.startswith("video_"):
                                key = key.replace("video_", "")
                            try:
                                val_str = parts[1].strip().replace('%', '')
                                val = float(val_str)
                                metrics[key] = val
                            except ValueError:
                                pass
                
                for m in ['accuracy', 'f1', 'auc', 'eer']:
                    if m in metrics:
                        val = metrics[m]
                        # Scale down percentage values to [0,1]
                        if m in ['accuracy', 'f1', 'eer'] and val > 1.0:
                            val /= 100.0
                        results_agg[model_name][m].append(val)

    # 4. Generate & Display Summary Table
    import math

    def mean_std(values):
        if not values:
            return "N/A", "N/A"
        n = len(values)
        mean = sum(values) / n
        if n > 1:
            variance = sum((x - mean) ** 2 for x in values) / (n - 1)
        else:
            variance = 0.0
        std = math.sqrt(variance)
        return mean, std

    print("\n" + "=" * 80)
    print(" PIPELINE RESULTS SUMMARY (FaceForensics++ Deepfakes, c23)")
    print(f" Aggregated across seeds: {args.seeds}")
    print("=" * 80)
    print(f"{'Method':<25} | {'Accuracy':<16} | {'F1-score':<16} | {'AUC':<16} | {'EER':<16}")
    print("-" * 101)

    summary_file = Path(args.output_dir) / "pipeline_summary.txt"
    with open(summary_file, "w") as sf:
        sf.write("=" * 80 + "\n")
        sf.write(" PIPELINE RESULTS SUMMARY (FaceForensics++ Deepfakes, c23)\n")
        sf.write(f" Aggregated across seeds: {args.seeds}\n")
        sf.write("=" * 80 + "\n")
        sf.write(f"{'Method':<25} | {'Accuracy':<16} | {'F1-score':<16} | {'AUC':<16} | {'EER':<16}\n")
        sf.write("-" * 101 + "\n")

        for model_name in models_to_run:
            model_disp_name = "Xception (baseline)" if model_name == "xception" else "RGB+FFT dual-stream"
            
            acc_m, acc_s = mean_std(results_agg[model_name]['accuracy'])
            f1_m, f1_s = mean_std(results_agg[model_name]['f1'])
            auc_m, auc_s = mean_std(results_agg[model_name]['auc'])
            eer_m, eer_s = mean_std(results_agg[model_name]['eer'])

            def format_metric(m, s, is_percent=True):
                if m == "N/A":
                    return "N/A"
                if is_percent:
                    return f"{m*100:.2f}% ± {s*100:.2f}%"
                else:
                    return f"{m:.4f} ± {s:.4f}"

            acc_str = format_metric(acc_m, acc_s, is_percent=True)
            f1_str = format_metric(f1_m, f1_s, is_percent=True)
            auc_str = format_metric(auc_m, auc_s, is_percent=False)
            eer_str = format_metric(eer_m, eer_s, is_percent=True)

            row = f"{model_disp_name:<25} | {acc_str:<16} | {f1_str:<16} | {auc_str:<16} | {eer_str:<16}"
            print(row)
            sf.write(row + "\n")

    print("-" * 101)
    print(f"\nConsolidated results saved to: {summary_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
