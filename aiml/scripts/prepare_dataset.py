"""
Retinal AI: Dataset Preparation and Manifest Generation CLI
Project: Smart India Hackathon 2026 - Retinal AI Diabetic Retinopathy Screening
Primary Specification: MATLAB Retinal AI SIH Presentation Guide.pdf & AI_SPEC.md

Usage:
  py -3.11 aiml/scripts/prepare_dataset.py [--data-dir PATH] [--output-dir outputs/manifests] [--seed 42]
"""

import sys
import os
import argparse
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from aiml.src.data.dataset_preparer import (
    RetinalDatasetPreparer,
    RetinalDatasetDiscovery,
    ICDR_LABEL_NAMES,
)


def run_dataset_preparation(
    data_dir: str = "data",
    output_dir: str = "outputs/manifests",
    manifest_prefix: str = "dataset_manifest_v1.0.0",
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
):
    print("=================================================================")
    print("RETINAL AI: BENCHMARK DATASET DISCOVERY & MANIFEST PREPARER")
    print("=================================================================")
    print(f"Target Search Root : {os.path.abspath(data_dir)}")
    print(f"Manifest Directory : {os.path.abspath(output_dir)}")
    print(f"Deterministic Seed : {seed}")
    print(f"Split Ratios       : Train={train_ratio*100:.0f}%, Val={val_ratio*100:.0f}%, Test={test_ratio*100:.0f}%\n")

    # Step 1: Discover candidate datasets
    print("Step 1: Running dataset discovery across standard benchmark formats...")
    discovered = RetinalDatasetDiscovery.discover(data_dir)
    print(f"  -> Discovered {len(discovered)} raw candidate image records.")

    if not discovered:
        print("\n[INFO] No external dataset images found in search root.")
        print("  -> Creating standard output directory structure.")
        os.makedirs(output_dir, exist_ok=True)
        print("  -> Ready to receive benchmark datasets (APTOS, IDRiD, Messidor-2, DDR) in data/ or datasets/.\n")
        return {
            "status": "no_data_found",
            "search_root": os.path.abspath(data_dir),
            "usable_images": 0,
            "rejected_count": 0,
            "duplicate_count": 0,
            "manifest_path": None,
        }

    # Step 2: Process dataset
    print("\nStep 2: Validating image integrity, labels, duplicates, and patient-level stratification...")
    preparer = RetinalDatasetPreparer(
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    result = preparer.process_dataset(
        raw_records=discovered,
        output_manifest_dir=output_dir,
        manifest_prefix=manifest_prefix,
    )

    # Step 3: Print Audit Report
    print("\n=================================================================")
    print("DATASET PREPARATION AUDIT REPORT")
    print("=================================================================")
    print(f"Total Discovered Images : {result['total_discovered']}")
    print(f"Usable Validated Images : {result['usable_images']}")
    print(f"Rejected Images         : {result['rejected_count']}")
    print(f"Duplicate Images Dropped: {result['duplicate_count']}")
    print(f"Unique Patient Groups   : {result['patient_count']}")
    print(f"Patient Leakage Detected: {result['has_patient_leakage']} (MUST BE False)")

    print("\n--- ICDR Class Distribution ---")
    for c in range(5):
        cnt = result["class_counts"][c]
        pct = cnt / max(1, result["usable_images"]) * 100
        print(f"  ICDR Grade {c} ({ICDR_LABEL_NAMES[c]:<23}): {cnt:>5} samples ({pct:>5.1f}%)")

    print(f"\nClass Imbalance Ratio   : {result['imbalance_ratio']}x (Max Class / Min Class)")
    print(f"Focal Loss Alpha Weights: {[round(x, 3) for x in result['focal_loss_alpha']]}")

    print("\n--- Split Partition Breakdown ---")
    for s_name in ["train", "val", "test"]:
        s_cnt = result["split_counts"][s_name]
        s_pct = s_cnt / max(1, result["usable_images"]) * 100
        print(f"  {s_name.capitalize():<10}: {s_cnt:>5} samples ({s_pct:>5.1f}%) | Classes: {result['split_class_counts'][s_name]}")

    print("\n--- Generated Manifest Artifacts ---")
    print(f"  JSON Manifest: {result['json_manifest_path']}")
    print(f"  CSV Manifest : {result['csv_manifest_path']}")
    print(f"  Metadata     : {result['metadata_path']}")
    print("=================================================================\n")

    return result


def main():
    parser = argparse.ArgumentParser(description="Retinal AI Dataset Preparation CLI")
    parser.add_argument("--data-dir", type=str, default="data", help="Root directory to search for datasets")
    parser.add_argument("--output-dir", type=str, default="outputs/manifests", help="Output directory for manifests")
    parser.add_argument("--manifest-prefix", type=str, default="dataset_manifest_v1.0.0", help="Prefix for manifest files")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Train partition ratio")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation partition ratio")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test partition ratio")

    args = parser.parse_args()

    run_dataset_preparation(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        manifest_prefix=args.manifest_prefix,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )


if __name__ == "__main__":
    main()
