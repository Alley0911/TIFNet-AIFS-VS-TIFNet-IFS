"""Download checked Release models and validate processed inference samples."""
from __future__ import annotations

import hashlib
from pathlib import Path
import urllib.error
import urllib.request

import numpy as np


def verify_model_file(model_filepath: Path, specification: dict) -> None:
    if not model_filepath.is_file():
        raise FileNotFoundError(f"Model file is missing: {model_filepath}")
    digest = hashlib.sha256()
    with model_filepath.open('rb') as model_stream:
        for block in iter(lambda: model_stream.read(1024 * 1024), b''):
            digest.update(block)
    if model_filepath.stat().st_size != specification['size_bytes'] or digest.hexdigest() != specification['sha256']:
        raise ValueError(
            f"Model size/SHA256 verification failed: {model_filepath}. "
            "Remove the corrupt file and rerun the download cell, or download the matching v1.0.0 asset manually."
        )


def download_verified_model(specification: dict, model_directory: Path) -> Path:
    model_directory = Path(model_directory)
    model_directory.mkdir(parents=True, exist_ok=True)
    model_filepath = model_directory / specification['filename']
    if model_filepath.exists():
        verify_model_file(model_filepath, specification)
        return model_filepath
    temporary_filepath = model_filepath.with_suffix('.part')
    try:
        with urllib.request.urlopen(specification['url'], timeout=120) as response, temporary_filepath.open('wb') as destination:
            while block := response.read(1024 * 1024):
                destination.write(block)
        verify_model_file(temporary_filepath, specification)
        temporary_filepath.replace(model_filepath)
    except (urllib.error.URLError, OSError) as error:
        raise RuntimeError(
            f"Model download failed. Check internet access or manually download {specification['url']} "
            f"to {model_filepath}, then rerun."
        ) from error
    finally:
        temporary_filepath.unlink(missing_ok=True)
    return model_filepath


def load_sample_inputs(sample_filepath: Path, metadata: dict) -> dict[str, np.ndarray]:
    if not sample_filepath.is_file():
        raise FileNotFoundError(f"Sample data missing: {sample_filepath}. Restore the repository's data directory.")
    if metadata.get('normalized') is not False:
        raise ValueError('Expected unnormalized inputs in physical units.')
    with np.load(sample_filepath, allow_pickle=False) as archive:
        if set(archive.files) != set(metadata['inputs']):
            raise ValueError('Sample must contain exactly the five inputs listed in metadata.json.')
        input_arrays = {name: archive[name] for name in metadata['inputs']}
    for input_name, input_array in input_arrays.items():
        expected_shape = tuple(metadata['inputs'][input_name])
        if input_array.shape != expected_shape or input_array.dtype != np.float32:
            raise ValueError(f'{input_name} must be float32 with shape {expected_shape}; got {input_array.dtype}, {input_array.shape}.')
        if not np.isfinite(input_array).all():
            raise ValueError(f'{input_name} contains NaN or infinity.')
    return input_arrays
