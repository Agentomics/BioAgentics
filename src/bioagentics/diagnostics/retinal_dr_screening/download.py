"""Download and catalog DR screening datasets.

Supports:
  - EyePACS / Kaggle DR Detection (~88k images)
  - APTOS 2019 Blindness Detection (~5.5k images)
  - IDRiD (516 images with lesion annotations)
  - Messidor-2 (1,748 images)
  - ODIR-5K (5,000 patients, both eyes)

Each dataset produces a unified metadata CSV with columns:
  image_path, dr_grade (0-4), dataset_source, original_filename

Usage:
    uv run python -m bioagentics.diagnostics.retinal_dr_screening.download eyepacs [--force]
    uv run python -m bioagentics.diagnostics.retinal_dr_screening.download aptos2019 [--force]
    uv run python -m bioagentics.diagnostics.retinal_dr_screening.download all [--force]
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
from pathlib import Path

import pandas as pd

from bioagentics.diagnostics.retinal_dr_screening.config import DATA_DIR, DATASETS

logger = logging.getLogger(__name__)


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _kaggle_download(dataset_slug: str, dest: Path, *, force: bool = False) -> Path:
    """Download a Kaggle dataset/competition using the Kaggle CLI."""
    dest = _ensure_dir(dest)
    marker = dest / ".downloaded"
    if marker.exists() and not force:
        logger.info("Already downloaded: %s (use --force to re-download)", dest)
        return dest

    # Determine if it's a competition or dataset
    if "/" in dataset_slug:
        cmd = ["kaggle", "datasets", "download", "-d", dataset_slug, "-p", str(dest), "--unzip"]
    else:
        cmd = ["kaggle", "competitions", "download", "-c", dataset_slug, "-p", str(dest), "--force"]

    logger.info("Downloading from Kaggle: %s → %s", dataset_slug, dest)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if result.returncode != 0:
        raise RuntimeError(
            f"Kaggle download failed for {dataset_slug}: {result.stderr.strip()}"
        )

    marker.touch()
    return dest


# ── EyePACS (Kaggle DR Detection Competition) ──


def download_eyepacs(*, force: bool = False) -> Path:
    """Download EyePACS dataset from Kaggle competition."""
    dest = DATA_DIR / "eyepacs"
    return _kaggle_download("diabetic-retinopathy-detection", dest, force=force)


def catalog_eyepacs(data_dir: Path | None = None) -> pd.DataFrame:
    """Build unified metadata CSV for EyePACS.

    EyePACS has trainLabels.csv with columns: image, level (0-4).
    Images are in train/ and test/ directories as .jpeg files.
    """
    data_dir = data_dir or DATA_DIR / "eyepacs"
    records = []

    # Training labels
    labels_path = data_dir / "trainLabels.csv"
    if labels_path.exists():
        labels = pd.read_csv(labels_path)
        for _, row in labels.iterrows():
            img_name = row["image"]
            # Images may be in train/ subdirectory
            for subdir in ["train", "."]:
                img_path = data_dir / subdir / f"{img_name}.jpeg"
                if img_path.exists():
                    records.append({
                        "image_path": str(img_path),
                        "dr_grade": int(row["level"]),
                        "dataset_source": "eyepacs",
                        "original_filename": f"{img_name}.jpeg",
                    })
                    break

    # Test labels (retinopathy_solution.csv if available)
    test_labels_path = data_dir / "retinopathy_solution.csv"
    if test_labels_path.exists():
        test_labels = pd.read_csv(test_labels_path)
        # Filter out "public" usage column rows without valid levels
        if "Usage" in test_labels.columns:
            test_labels = test_labels[test_labels["Usage"].notna()]
        for _, row in test_labels.iterrows():
            img_name = row["image"]
            for subdir in ["test", "."]:
                img_path = data_dir / subdir / f"{img_name}.jpeg"
                if img_path.exists():
                    records.append({
                        "image_path": str(img_path),
                        "dr_grade": int(row["level"]),
                        "dataset_source": "eyepacs",
                        "original_filename": f"{img_name}.jpeg",
                    })
                    break

    df = pd.DataFrame(records)
    logger.info("EyePACS catalog: %d images", len(df))
    return df


# ── APTOS 2019 ──


def download_aptos2019(*, force: bool = False) -> Path:
    """Download APTOS 2019 dataset from Kaggle."""
    dest = DATA_DIR / "aptos2019"
    return _kaggle_download("aptos2019-blindness-detection", dest, force=force)


def catalog_aptos2019(data_dir: Path | None = None) -> pd.DataFrame:
    """Build unified metadata CSV for APTOS 2019.

    APTOS has train.csv with columns: id_code, diagnosis (0-4).
    Images are in train_images/ as .png files.
    """
    data_dir = data_dir or DATA_DIR / "aptos2019"
    records = []

    labels_path = data_dir / "train.csv"
    if labels_path.exists():
        labels = pd.read_csv(labels_path)
        for _, row in labels.iterrows():
            img_name = row["id_code"]
            for subdir in ["train_images", "."]:
                img_path = data_dir / subdir / f"{img_name}.png"
                if img_path.exists():
                    records.append({
                        "image_path": str(img_path),
                        "dr_grade": int(row["diagnosis"]),
                        "dataset_source": "aptos2019",
                        "original_filename": f"{img_name}.png",
                    })
                    break

    df = pd.DataFrame(records)
    logger.info("APTOS 2019 catalog: %d images", len(df))
    return df


# ── IDRiD ──


def download_idrid(*, force: bool = False) -> Path:
    """IDRiD requires manual download from IEEE DataPort.

    Download from: https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid
    Place files in data/diagnostics/smartphone-retinal-dr-screening/idrid/
    """
    dest = _ensure_dir(DATA_DIR / "idrid")
    marker = dest / ".downloaded"
    if marker.exists() and not force:
        logger.info("IDRiD already present at %s", dest)
        return dest

    logger.warning(
        "IDRiD requires manual download from IEEE DataPort.\n"
        "  1. Visit: https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid\n"
        "  2. Download and extract to: %s\n"
        "  3. Create marker: touch %s/.downloaded",
        dest,
        dest,
    )
    return dest


def catalog_idrid(data_dir: Path | None = None) -> pd.DataFrame:
    """Build unified metadata CSV for IDRiD.

    IDRiD has separate Disease Grading CSV with columns: Image name, Retinopathy grade.
    Images are in 1. Original Images/a. Training Set/ and b. Testing Set/.
    """
    data_dir = data_dir or DATA_DIR / "idrid"
    records = []

    # Check various possible label file locations
    possible_label_paths = [
        data_dir / "2. Groundtruths" / "a. IDRiD_Disease Grading_Training Labels.csv",
        data_dir / "IDRiD_Disease Grading_Training Labels.csv",
        data_dir / "labels" / "train.csv",
    ]

    for labels_path in possible_label_paths:
        if not labels_path.exists():
            continue

        labels = pd.read_csv(labels_path)
        # Column names vary; try common patterns
        img_col = next(
            (c for c in labels.columns if "image" in c.lower() or "name" in c.lower()),
            labels.columns[0],
        )
        grade_col = next(
            (c for c in labels.columns if "retinopathy" in c.lower() or "grade" in c.lower()),
            labels.columns[1],
        )

        for _, row in labels.iterrows():
            img_name = str(row[img_col]).strip()
            if not img_name.lower().endswith((".jpg", ".jpeg", ".png", ".tif")):
                img_name += ".jpg"

            # Search common image subdirectories
            for subdir in [
                "1. Original Images/a. Training Set",
                "1. Original Images/b. Testing Set",
                "images/train",
                "images/test",
                ".",
            ]:
                img_path = data_dir / subdir / img_name
                if img_path.exists():
                    records.append({
                        "image_path": str(img_path),
                        "dr_grade": int(row[grade_col]),
                        "dataset_source": "idrid",
                        "original_filename": img_name,
                    })
                    break
        break  # Found a valid label file

    df = pd.DataFrame(records)
    logger.info("IDRiD catalog: %d images", len(df))
    return df


# ── Messidor-2 ──


def download_messidor2(*, force: bool = False) -> Path:
    """Messidor-2 requires registration at ADCIS.

    Download from: https://www.adcis.net/en/third-party/messidor2/
    Place files in data/diagnostics/smartphone-retinal-dr-screening/messidor2/
    """
    dest = _ensure_dir(DATA_DIR / "messidor2")
    marker = dest / ".downloaded"
    if marker.exists() and not force:
        logger.info("Messidor-2 already present at %s", dest)
        return dest

    logger.warning(
        "Messidor-2 requires registration and manual download.\n"
        "  1. Visit: https://www.adcis.net/en/third-party/messidor2/\n"
        "  2. Download and extract to: %s\n"
        "  3. Create marker: touch %s/.downloaded",
        dest,
        dest,
    )
    return dest


def catalog_messidor2(data_dir: Path | None = None) -> pd.DataFrame:
    """Build unified metadata CSV for Messidor-2.

    Messidor-2 typically has an Excel file with grades.
    Images are TIFF or JPEG files in the root or subdirectories.
    """
    data_dir = data_dir or DATA_DIR / "messidor2"
    records = []

    # Look for label files
    possible_label_paths = [
        data_dir / "messidor_data.csv",
        data_dir / "Annotation_Base11.csv",
    ]

    # Also check for Excel files
    for xlsx in data_dir.glob("*.xlsx"):
        possible_label_paths.append(xlsx)
    for xls in data_dir.glob("*.xls"):
        possible_label_paths.append(xls)

    for labels_path in possible_label_paths:
        if not labels_path.exists():
            continue

        if labels_path.suffix in (".xlsx", ".xls"):
            labels = pd.read_excel(labels_path)
        else:
            labels = pd.read_csv(labels_path)

        img_col = next(
            (c for c in labels.columns if "image" in c.lower() or "name" in c.lower()),
            labels.columns[0],
        )
        grade_col = next(
            (c for c in labels.columns if "retinopathy" in c.lower() or "grade" in c.lower() or "adjudicated" in c.lower()),
            labels.columns[-1],
        )

        for _, row in labels.iterrows():
            img_name = str(row[img_col]).strip()
            grade = row[grade_col]
            if pd.isna(grade):
                continue
            grade = int(grade)
            if grade < 0 or grade > 4:
                continue

            for subdir in [".", "images", "IMAGES"]:
                img_path = data_dir / subdir / img_name
                if img_path.exists():
                    records.append({
                        "image_path": str(img_path),
                        "dr_grade": grade,
                        "dataset_source": "messidor2",
                        "original_filename": img_name,
                    })
                    break
        break

    df = pd.DataFrame(records)
    logger.info("Messidor-2 catalog: %d images", len(df))
    return df


# ── ODIR-5K ──


def download_odir5k(*, force: bool = False) -> Path:
    """ODIR-5K requires manual download.

    Download from: https://odir2019.grand-challenge.org/
    Place files in data/diagnostics/smartphone-retinal-dr-screening/odir5k/
    """
    dest = _ensure_dir(DATA_DIR / "odir5k")
    marker = dest / ".downloaded"
    if marker.exists() and not force:
        logger.info("ODIR-5K already present at %s", dest)
        return dest

    logger.warning(
        "ODIR-5K requires manual download.\n"
        "  1. Visit: https://odir2019.grand-challenge.org/\n"
        "  2. Download and extract to: %s\n"
        "  3. Create marker: touch %s/.downloaded",
        dest,
        dest,
    )
    return dest


def catalog_odir5k(data_dir: Path | None = None) -> pd.DataFrame:
    """Build unified metadata CSV for ODIR-5K.

    ODIR-5K has annotations with multiple disease labels per patient.
    We extract DR grade from the diagnostic keywords column.
    Columns typically: ID, Patient Age, Patient Sex, Left-Fundus, Right-Fundus,
    Left-Diagnostic Keywords, Right-Diagnostic Keywords, N, D, G, C, A, H, M, O
    where D = diabetic retinopathy.
    """
    data_dir = data_dir or DATA_DIR / "odir5k"
    records = []

    possible_label_paths = [
        data_dir / "ODIR-5K_Training_Annotations(Updated)_V2.xlsx",
        data_dir / "full_df.csv",
        data_dir / "annotations.csv",
    ]
    for xlsx in data_dir.glob("*.xlsx"):
        if xlsx not in possible_label_paths:
            possible_label_paths.append(xlsx)

    for labels_path in possible_label_paths:
        if not labels_path.exists():
            continue

        if labels_path.suffix in (".xlsx", ".xls"):
            labels = pd.read_excel(labels_path)
        else:
            labels = pd.read_csv(labels_path)

        # ODIR uses multi-label format — map DR keyword presence to binary (0 or 1)
        # For 5-class grading, ODIR only provides binary DR labels (D column)
        # We map: D=0 → grade 0, D=1 → grade 2 (referable DR, conservative estimate)
        d_col = "D" if "D" in labels.columns else None
        if d_col is None:
            continue

        for _, row in labels.iterrows():
            dr_flag = int(row[d_col]) if pd.notna(row[d_col]) else 0

            # Process both eyes
            for side, fundus_col in [("left", "Left-Fundus"), ("right", "Right-Fundus")]:
                if fundus_col not in labels.columns:
                    continue
                img_name = str(row[fundus_col]).strip()
                if not img_name or img_name == "nan":
                    continue

                for subdir in ["ODIR-5K_Training_Images", "images", "."]:
                    img_path = data_dir / subdir / img_name
                    if img_path.exists():
                        records.append({
                            "image_path": str(img_path),
                            "dr_grade": 2 if dr_flag == 1 else 0,
                            "dataset_source": "odir5k",
                            "original_filename": img_name,
                        })
                        break
        break

    df = pd.DataFrame(records)
    logger.info("ODIR-5K catalog: %d images", len(df))
    return df


# ── Unified catalog ──

CATALOG_FUNCTIONS: dict[str, tuple] = {
    "eyepacs": (download_eyepacs, catalog_eyepacs),
    "aptos2019": (download_aptos2019, catalog_aptos2019),
    "idrid": (download_idrid, catalog_idrid),
    "messidor2": (download_messidor2, catalog_messidor2),
    "odir5k": (download_odir5k, catalog_odir5k),
}


def build_unified_catalog(datasets: list[str] | None = None) -> pd.DataFrame:
    """Build a unified metadata CSV from all available datasets.

    Args:
        datasets: List of dataset keys to include. If None, includes all available.

    Returns:
        DataFrame with columns: image_path, dr_grade, dataset_source, original_filename
    """
    if datasets is None:
        datasets = list(CATALOG_FUNCTIONS.keys())

    dfs = []
    for name in datasets:
        if name not in CATALOG_FUNCTIONS:
            logger.warning("Unknown dataset: %s (skipping)", name)
            continue
        _, catalog_fn = CATALOG_FUNCTIONS[name]
        df = catalog_fn()
        if not df.empty:
            dfs.append(df)
        else:
            logger.warning("No images found for %s — is the data downloaded?", name)

    if not dfs:
        return pd.DataFrame(columns=["image_path", "dr_grade", "dataset_source", "original_filename"])

    combined = pd.concat(dfs, ignore_index=True)
    logger.info(
        "Unified catalog: %d images across %d datasets",
        len(combined),
        combined["dataset_source"].nunique(),
    )
    return combined


def save_catalog(df: pd.DataFrame, path: Path | None = None) -> Path:
    """Save catalog DataFrame to CSV."""
    if path is None:
        path = DATA_DIR / "catalog.csv"
    _ensure_dir(path.parent)
    df.to_csv(path, index=False)
    logger.info("Catalog saved to %s", path)
    return path


# ── CLI ──


def main():
    parser = argparse.ArgumentParser(description="Download and catalog DR datasets")
    parser.add_argument(
        "dataset",
        choices=list(CATALOG_FUNCTIONS.keys()) + ["all", "catalog"],
        help="Dataset to download, 'all' for everything, or 'catalog' to build catalog only",
    )
    parser.add_argument("--force", action="store_true", help="Re-download even if present")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.dataset == "catalog":
        df = build_unified_catalog()
        save_catalog(df)
        print(f"\nCatalog: {len(df)} images")
        if not df.empty:
            print(df["dataset_source"].value_counts().to_string())
        return

    if args.dataset == "all":
        datasets_to_download = list(CATALOG_FUNCTIONS.keys())
    else:
        datasets_to_download = [args.dataset]

    for name in datasets_to_download:
        download_fn, _ = CATALOG_FUNCTIONS[name]
        download_fn(force=args.force)

    # Build catalog after downloading
    df = build_unified_catalog(datasets_to_download)
    save_catalog(df)
    print(f"\nCatalog: {len(df)} images")
    if not df.empty:
        print(df["dataset_source"].value_counts().to_string())


if __name__ == "__main__":
    main()
