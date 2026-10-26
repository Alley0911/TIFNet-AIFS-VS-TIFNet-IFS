from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.ndimage import gaussian_filter1d


@dataclass
class Spectrum1DResult:
    k_cpkm: np.ndarray
    wavelength_km: np.ndarray
    E1D: np.ndarray
    variance_field: float
    variance_spec: float
    nx: int
    ny: int
    dx_km: float
    dy_km: float
    window_energy_U: float
    n_valid_bins: int


@dataclass
class MeanSpectrum1DResult:
    k_cpkm: np.ndarray
    wavelength_km: np.ndarray
    mean_E1D: np.ndarray
    std_E1D: np.ndarray
    sem_E1D: np.ndarray
    p25_E1D: np.ndarray
    p75_E1D: np.ndarray
    n_samples: int


@dataclass
class SpectralMatchResult:
    ifs_matched: np.ndarray
    k_cpkm: np.ndarray
    wavelength_km: np.ndarray
    mean_spec_aifs: MeanSpectrum1DResult
    mean_spec_ifs: MeanSpectrum1DResult
    filter_W: np.ndarray
    filter_meta: dict


def make_2d_hann_window(number_of_latitudes: int, number_of_longitudes: int) -> np.ndarray:
    longitude_window = np.hanning(number_of_longitudes)
    latitude_window = np.hanning(number_of_latitudes)
    return np.outer(latitude_window, longitude_window)


def preprocess_field(
    environmental_field: np.ndarray,
    fill_nan: Literal["zero", "error"] = "zero",
    detrend_mean: bool = True,
) -> np.ndarray:
    environmental_array = np.asarray(environmental_field, dtype=float).copy()
    if not np.isfinite(environmental_array).any():
        raise ValueError("field is all NaN or inf; cannot compute spectrum")

    if detrend_mean:
        environmental_array = environmental_array - np.nanmean(environmental_array)

    if fill_nan == "zero":
        return np.nan_to_num(environmental_array, nan=0.0, posinf=0.0, neginf=0.0)
    if fill_nan == "error":
        if not np.isfinite(environmental_array).all():
            raise ValueError("field contains NaN or inf while fill_nan='error'")
        return environmental_array
    raise ValueError("fill_nan must be 'zero' or 'error'")


def compute_2d_periodogram(
    environmental_field: np.ndarray,
    dx_km: float,
    dy_km: float | None = None,
    window: Literal["hann", "none"] = "hann",
    fill_nan: Literal["zero", "error"] = "zero",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    if dy_km is None:
        dy_km = dx_km

    environmental_array = preprocess_field(environmental_field, fill_nan=fill_nan, detrend_mean=True)
    number_of_latitudes, number_of_longitudes = environmental_array.shape
    variance_field = float(np.mean(environmental_array ** 2))

    if window == "hann":
        window_array = make_2d_hann_window(number_of_latitudes, number_of_longitudes)
    elif window == "none":
        window_array = np.ones((number_of_latitudes, number_of_longitudes), dtype=float)
    else:
        raise ValueError("window must be 'hann' or 'none'")

    window_energy = float(np.mean(window_array ** 2))
    windowed_array = environmental_array * window_array
    fourier_coefficients = np.fft.fft2(windowed_array)

    longitude_frequencies = np.fft.fftfreq(number_of_longitudes, d=dx_km)
    latitude_frequencies = np.fft.fftfreq(number_of_latitudes, d=dy_km)
    periodogram_2d = (
        (dx_km * dy_km)
        / (number_of_longitudes * number_of_latitudes * window_energy)
        * (np.abs(fourier_coefficients) ** 2)
    )

    longitude_frequency_spacing = 1.0 / (number_of_longitudes * dx_km)
    latitude_frequency_spacing = 1.0 / (number_of_latitudes * dy_km)
    variance_spec = float(np.sum(periodogram_2d) * longitude_frequency_spacing * latitude_frequency_spacing)
    return (
        longitude_frequencies,
        latitude_frequencies,
        periodogram_2d,
        variance_field,
        variance_spec,
        window_energy,
    )


def compute_isotropic_spectrum_1d(
    environmental_field: np.ndarray,
    dx_km: float,
    dy_km: float | None = None,
    window: Literal["hann", "none"] = "hann",
    fill_nan: Literal["zero", "error"] = "zero",
    n_bins: int | None = None,
    common_k_edges: np.ndarray | None = None,
) -> Spectrum1DResult:
    if dy_km is None:
        dy_km = dx_km

    (
        longitude_frequencies,
        latitude_frequencies,
        periodogram_2d,
        variance_field,
        _variance_spec,
        window_energy,
    ) = compute_2d_periodogram(
        environmental_field=environmental_field,
        dx_km=dx_km,
        dy_km=dy_km,
        window=window,
        fill_nan=fill_nan,
    )

    number_of_latitudes, number_of_longitudes = periodogram_2d.shape
    longitude_frequency_grid, latitude_frequency_grid = np.meshgrid(longitude_frequencies, latitude_frequencies)
    radial_frequency_grid = np.sqrt(longitude_frequency_grid ** 2 + latitude_frequency_grid ** 2)

    longitude_frequency_spacing = 1.0 / (number_of_longitudes * dx_km)
    latitude_frequency_spacing = 1.0 / (number_of_latitudes * dy_km)
    nyquist_frequency = min(1.0 / (2.0 * dx_km), 1.0 / (2.0 * dy_km))

    if common_k_edges is not None:
        frequency_bin_edges = np.asarray(common_k_edges, dtype=float)
    else:
        if n_bins is None:
            n_bins = min(number_of_longitudes, number_of_latitudes) // 2
        frequency_bin_edges = np.linspace(0.0, nyquist_frequency, n_bins + 1)

    isotropic_spectrum = np.full(len(frequency_bin_edges) - 1, np.nan, dtype=float)
    frequency_bin_centers = 0.5 * (frequency_bin_edges[:-1] + frequency_bin_edges[1:])

    for bin_index in range(len(frequency_bin_edges) - 1):
        lower_frequency = frequency_bin_edges[bin_index]
        upper_frequency = frequency_bin_edges[bin_index + 1]
        frequency_mask = (radial_frequency_grid >= lower_frequency) & (radial_frequency_grid < upper_frequency)
        if not np.any(frequency_mask):
            continue
        shell_variance = (
            np.sum(periodogram_2d[frequency_mask])
            * longitude_frequency_spacing
            * latitude_frequency_spacing
        )
        frequency_width = upper_frequency - lower_frequency
        if frequency_width > 0:
            isotropic_spectrum[bin_index] = shell_variance / frequency_width

    valid_bins = np.isfinite(isotropic_spectrum) & (frequency_bin_centers > 0.0)
    k_cpkm = frequency_bin_centers[valid_bins]
    isotropic_spectrum = isotropic_spectrum[valid_bins]
    wavelength_km = 1.0 / k_cpkm
    variance_spec_1d = float(np.trapz(isotropic_spectrum, k_cpkm)) if len(k_cpkm) >= 2 else np.nan

    return Spectrum1DResult(
        k_cpkm=k_cpkm,
        wavelength_km=wavelength_km,
        E1D=isotropic_spectrum,
        variance_field=variance_field,
        variance_spec=variance_spec_1d,
        nx=number_of_longitudes,
        ny=number_of_latitudes,
        dx_km=dx_km,
        dy_km=dy_km,
        window_energy_U=window_energy,
        n_valid_bins=len(k_cpkm),
    )


def compute_mean_isotropic_spectrum(
    environmental_fields: np.ndarray,
    dx_km: float,
    dy_km: float | None = None,
    window: Literal["hann", "none"] = "hann",
    fill_nan: Literal["zero", "error"] = "zero",
    n_bins: int | None = None,
) -> MeanSpectrum1DResult:
    environmental_array = np.asarray(environmental_fields, dtype=float)
    if environmental_array.ndim != 3:
        raise ValueError("fields must have shape=(n_samples, ny, nx)")

    number_of_samples, number_of_latitudes, number_of_longitudes = environmental_array.shape
    if dy_km is None:
        dy_km = dx_km
    if n_bins is None:
        n_bins = min(number_of_longitudes, number_of_latitudes) // 2

    nyquist_frequency = min(1.0 / (2.0 * dx_km), 1.0 / (2.0 * dy_km))
    common_k_edges = np.linspace(0.0, nyquist_frequency, n_bins + 1)

    spectra = [
        compute_isotropic_spectrum_1d(
            environmental_field=environmental_array[sample_index],
            dx_km=dx_km,
            dy_km=dy_km,
            window=window,
            fill_nan=fill_nan,
            common_k_edges=common_k_edges,
        ).E1D
        for sample_index in range(number_of_samples)
    ]
    spectrum_matrix = np.asarray(spectra, dtype=float)

    frequency_bin_centers = 0.5 * (common_k_edges[:-1] + common_k_edges[1:])
    valid_bins = frequency_bin_centers > 0
    k_cpkm = frequency_bin_centers[valid_bins]
    wavelength_km = 1.0 / k_cpkm

    mean_spectrum = np.nanmean(spectrum_matrix, axis=0)[valid_bins]
    standard_deviation_spectrum = (
        np.nanstd(spectrum_matrix, axis=0, ddof=1)
        if number_of_samples > 1
        else np.full(spectrum_matrix.shape[1], np.nan)
    )[valid_bins]
    standard_error_spectrum = (
        standard_deviation_spectrum / np.sqrt(number_of_samples)
        if number_of_samples > 1
        else np.full_like(standard_deviation_spectrum, np.nan)
    )
    percentile_25_spectrum = np.nanpercentile(spectrum_matrix, 25, axis=0)[valid_bins]
    percentile_75_spectrum = np.nanpercentile(spectrum_matrix, 75, axis=0)[valid_bins]

    return MeanSpectrum1DResult(
        k_cpkm=k_cpkm,
        wavelength_km=wavelength_km,
        mean_E1D=mean_spectrum,
        std_E1D=standard_deviation_spectrum,
        sem_E1D=standard_error_spectrum,
        p25_E1D=percentile_25_spectrum,
        p75_E1D=percentile_75_spectrum,
        n_samples=number_of_samples,
    )


def build_isotropic_matching_filter(
    k_target: np.ndarray,
    E_target: np.ndarray,
    k_source: np.ndarray,
    E_source: np.ndarray,
    allow_amplify: bool = False,
    max_gain: float = 1.0,
    smooth_sigma: float = 2.0,
    eps: float = 1e-12,
) -> np.ndarray:
    target_spectrum_interpolated = np.interp(k_source, k_target, E_target)
    matching_filter = np.sqrt(np.maximum(target_spectrum_interpolated, 0.0) / np.maximum(E_source, eps))

    if allow_amplify:
        matching_filter = np.minimum(matching_filter, max_gain)
    else:
        matching_filter = np.minimum(matching_filter, 1.0)

    if smooth_sigma is not None and smooth_sigma > 0:
        matching_filter = gaussian_filter1d(matching_filter, sigma=smooth_sigma, mode="nearest")

    if allow_amplify:
        return np.clip(matching_filter, 0.0, max_gain)
    return np.clip(matching_filter, 0.0, 1.0)


def apply_isotropic_filter_to_field(
    environmental_field: np.ndarray,
    dx_km: float,
    dy_km: float | None,
    k_filter_cpkm: np.ndarray,
    matching_filter: np.ndarray,
    remove_mean: bool = True,
    add_mean_back: bool = True,
) -> np.ndarray:
    environmental_array = np.asarray(environmental_field, dtype=float).copy()
    if not np.isfinite(environmental_array).any():
        raise ValueError("field is all NaN or inf; cannot filter")

    mean_value = float(np.nanmean(environmental_array)) if remove_mean else 0.0
    if remove_mean:
        environmental_array = environmental_array - mean_value
    environmental_array = np.nan_to_num(environmental_array, nan=0.0, posinf=0.0, neginf=0.0)

    number_of_latitudes, number_of_longitudes = environmental_array.shape
    if dy_km is None:
        dy_km = dx_km

    fourier_coefficients = np.fft.fft2(environmental_array)
    longitude_frequencies = np.fft.fftfreq(number_of_longitudes, d=dx_km)
    latitude_frequencies = np.fft.fftfreq(number_of_latitudes, d=dy_km)
    longitude_frequency_grid, latitude_frequency_grid = np.meshgrid(longitude_frequencies, latitude_frequencies)
    radial_frequency_grid = np.sqrt(longitude_frequency_grid ** 2 + latitude_frequency_grid ** 2)
    filter_2d = np.interp(
        radial_frequency_grid.ravel(),
        k_filter_cpkm,
        matching_filter,
        left=matching_filter[0],
        right=matching_filter[-1],
    ).reshape(radial_frequency_grid.shape)

    filtered_coefficients = fourier_coefficients * filter_2d
    filtered_array = np.fft.ifft2(filtered_coefficients).real
    if add_mean_back and remove_mean:
        filtered_array = filtered_array + mean_value
    return filtered_array


def spectral_match_isotropic_1d(
    aifs: np.ndarray,
    ifs: np.ndarray,
    dx_km: float,
    dy_km: float | None = None,
    n_bins: int | None = None,
    window: Literal["hann", "none"] = "hann",
    fill_nan: Literal["zero", "error"] = "zero",
    allow_amplify: bool = False,
    max_gain: float = 1.0,
    smooth_sigma: float = 2.0,
) -> SpectralMatchResult:
    aifs_array = np.asarray(aifs, dtype=float)
    ifs_array = np.asarray(ifs, dtype=float)
    if aifs_array.ndim != 3 or ifs_array.ndim != 3:
        raise ValueError("aifs and ifs must both have shape=(n_samples, ny, nx)")
    if aifs_array.shape != ifs_array.shape:
        raise ValueError("aifs and ifs must have the same shape")

    mean_spectrum_aifs = compute_mean_isotropic_spectrum(
        environmental_fields=aifs_array,
        dx_km=dx_km,
        dy_km=dy_km,
        window=window,
        fill_nan=fill_nan,
        n_bins=n_bins,
    )
    mean_spectrum_ifs = compute_mean_isotropic_spectrum(
        environmental_fields=ifs_array,
        dx_km=dx_km,
        dy_km=dy_km,
        window=window,
        fill_nan=fill_nan,
        n_bins=n_bins,
    )

    matching_filter = build_isotropic_matching_filter(
        k_target=mean_spectrum_aifs.k_cpkm,
        E_target=mean_spectrum_aifs.mean_E1D,
        k_source=mean_spectrum_ifs.k_cpkm,
        E_source=mean_spectrum_ifs.mean_E1D,
        allow_amplify=allow_amplify,
        max_gain=max_gain,
        smooth_sigma=smooth_sigma,
    )

    ifs_matched = np.empty_like(ifs_array)
    for sample_index in range(ifs_array.shape[0]):
        ifs_matched[sample_index] = apply_isotropic_filter_to_field(
            environmental_field=ifs_array[sample_index],
            dx_km=dx_km,
            dy_km=dy_km,
            k_filter_cpkm=mean_spectrum_ifs.k_cpkm,
            matching_filter=matching_filter,
            remove_mean=True,
            add_mean_back=True,
        )

    return SpectralMatchResult(
        ifs_matched=ifs_matched,
        k_cpkm=mean_spectrum_ifs.k_cpkm,
        wavelength_km=mean_spectrum_ifs.wavelength_km,
        mean_spec_aifs=mean_spectrum_aifs,
        mean_spec_ifs=mean_spectrum_ifs,
        filter_W=matching_filter,
        filter_meta={
            "allow_amplify": allow_amplify,
            "max_gain": max_gain,
            "smooth_sigma": smooth_sigma,
            "window_for_spectrum_estimation": window,
            "filter_domain": "isotropic_1d",
            "k_unit": "cycles/km",
        },
    )
