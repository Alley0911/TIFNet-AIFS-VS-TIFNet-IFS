"""Portable directory-level IFSMatch2AIFS generation."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

from .spectral_matching import spectral_match_isotropic_1d


DIRECT_COPY_VARIABLES = frozenset({"lsm", "slor", "z", "sdor"})


def _variable_name(filepath: Path) -> str:
    return filepath.name.split("_", 1)[0]


def _is_valid_array(filepath: Path) -> bool:
    try:
        array = np.load(filepath, mmap_mode="r")
        return array.ndim >= 2 and array.shape[-2] == array.shape[-1] and array.size > 0 and np.isfinite(array).all()
    except (OSError, ValueError, TypeError):
        return False


def match_one_file(aifs_filepath: Path, ifs_filepath: Path, output_filepath: Path, overwrite: bool = False) -> str:
    if not ifs_filepath.is_file():
        raise FileNotFoundError(f"IFS input file does not exist: {ifs_filepath}")
    if not _is_valid_array(aifs_filepath) or not _is_valid_array(ifs_filepath):
        raise ValueError(f"Invalid input NPY: AIFS={aifs_filepath}, IFS={ifs_filepath}")
    if output_filepath.exists() and not overwrite:
        return "skipped_existing"
    output_filepath.parent.mkdir(parents=True, exist_ok=True)
    variable_name = _variable_name(aifs_filepath)
    if variable_name in DIRECT_COPY_VARIABLES:
        shutil.copy2(ifs_filepath, output_filepath)
        return "copied_static"
    aifs_array = np.load(aifs_filepath)
    ifs_array = np.load(ifs_filepath)
    if aifs_array.shape != ifs_array.shape or aifs_array.ndim != 2:
        raise ValueError(f"Dynamic field shapes differ: AIFS={aifs_array.shape}, IFS={ifs_array.shape}")
    matched_array = spectral_match_isotropic_1d(
        aifs=aifs_array[None, ...], ifs=ifs_array[None, ...],
        dx_km=25.0, dy_km=25.0, n_bins=30, window="hann", fill_nan="zero",
        allow_amplify=False, max_gain=1.5, smooth_sigma=2.0,
    ).ifs_matched[0].astype(ifs_array.dtype, copy=False)
    if not np.isfinite(matched_array).all():
        raise ValueError(f"Matched field contains NaN or infinity: {output_filepath}")
    temporary_filepath = output_filepath.with_suffix(output_filepath.suffix + ".tmp")
    with temporary_filepath.open("wb") as output_file:
        np.save(output_file, matched_array)
    temporary_filepath.replace(output_filepath)
    return "spectral_matched"


def match_directory(aifs_directory: str | Path, ifs_directory: str | Path, output_directory: str | Path, overwrite: bool = False, on_error: str = "raise") -> dict[str, int]:
    aifs_directory, ifs_directory, output_directory = map(Path, (aifs_directory, ifs_directory, output_directory))
    if not aifs_directory.is_dir() or not ifs_directory.is_dir():
        raise FileNotFoundError("AIFS and IFS input directories must both exist")
    statistics = {"spectral_matched": 0, "copied_static": 0, "skipped_existing": 0, "failed": 0}
    for aifs_filepath in sorted(aifs_directory.glob("*/*/*.npy")):
        relative_filepath = aifs_filepath.relative_to(aifs_directory)
        ifs_filepath = ifs_directory / relative_filepath
        output_filepath = output_directory / relative_filepath
        try:
            result = match_one_file(aifs_filepath, ifs_filepath, output_filepath, overwrite=overwrite)
            statistics[result] += 1
        except Exception:
            statistics["failed"] += 1
            if on_error == "raise":
                raise
    return statistics


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate IFSMatch2AIFS NPY files from local AIFS and IFS caches.")
    parser.add_argument("--aifs-directory", required=True, help="AIFS NPY root containing <ty_id>/<init>/<files>.npy")
    parser.add_argument("--ifs-directory", required=True, help="IFS NPY root with the same relative file layout")
    parser.add_argument("--output-directory", required=True, help="IFSMatch2AIFS output NPY root")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--on-error", choices=("raise", "skip"), default="raise")
    arguments = parser.parse_args()
    statistics = match_directory(arguments.aifs_directory, arguments.ifs_directory, arguments.output_directory, arguments.overwrite, arguments.on_error)
    print("IFSMatch2AIFS generation complete:", statistics)


if __name__ == "__main__":
    main()
