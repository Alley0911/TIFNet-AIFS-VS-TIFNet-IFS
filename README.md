# A comparison of AIFS- and IFS-based TIFNet


This companion provides the IFSMatch2AIFS spectral matching source code, two trained TIFNet models in ONNX format, and a CPU inference notebook. The AIFS-driven and IFS-driven models are used in the paper's case studies; source-specific fine-tuning updated their final prediction heads.

## Run the example

Use Python 3.10 or later in a fresh environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
jupyter notebook notebooks/aifs_inference.ipynb
```

Run all cells in order. The notebook downloads and verifies the AIFS model on first use, loads the processed inputs for tropical cyclone **202424 (Man-yi), initialized at 2024-11-14 12:00 UTC** from the paper's Man-yi case-study figure, runs ONNX Runtime on CPU, and displays the forecast alongside the original paper predictions and CMA observations for all 21 lead times. Outputs are a forecast CSV and a PDF intensity curve. No PyTorch, training checkpoints, or original project checkout is needed. Allow several GB of RAM for inference and additional disk space for the model download.

The notebook works when launched from the repository root or `notebooks/`. For IFS, change `input_source = "aifs"` to `"ifs"` and rerun all cells: both the model and sample change together.

## Models and data

- [Release v1.0.0](https://github.com/Alley0911/TIFNet-AIFS-VS-TIFNet-IFS/releases/tag/v1.0.0): `tifnet_aifs.onnx` and `tifnet_ifs.onnx` (ONNX opset 17, float32).
- [Model manifest](models/manifest.json): fixed download URLs, sizes, and SHA256 checksums.
- `data/202424_2024111412/`: compressed unnormalized input tensors, metadata, and the original paper predictions with CMA observations.

ONNX includes input normalization, network weights, target denormalization, and initial-wind addition. **Do not normalize inputs again.** Models are Release assets, not files in ordinary Git history.

## ONNX input and output

Both models take five finite `float32` arrays in physical units, with one sample per call. Keep the axis and channel order shown below; do not transpose or rescale the supplied arrays.

| Input | Shape | Axis order |
|---|---|---|
| `encoder_surface` | `(1, 9, 5, 72, 72)` | batch, variable, history time, latitude, longitude |
| `encoder_isobaric` | `(1, 5, 5, 7, 72, 72)` | batch, variable, history time, pressure level, latitude, longitude |
| `encoder_artificial` | `(1, 5, 1)` | batch, history time, historical maximum wind speed |
| `decoder_surface` | `(1, 9, 21, 72, 72)` | batch, variable, forecast time, latitude, longitude |
| `decoder_isobaric` | `(1, 5, 21, 7, 72, 72)` | batch, variable, forecast time, pressure level, latitude, longitude |

History covers −24 to 0 h; forecasts cover 0 to 120 h, both at 6 h intervals. The pressure levels are 100, 200, 300, 500, 700, 850 and 925 hPa. Surface channels are `u10`, `v10` (m/s), `t2m` (K), `slp` (Pa), `skt` (K), `z` (m²/s²), `lsm` (dimensionless), `sdor` (m) and `slor` (dimensionless). Pressure-level channels are `u`, `v` (m/s), `tmp` (K), `sh` (kg/kg) and `gp` (m²/s², geopotential rather than geopotential height). Historical intensity and the output `predicted_maximum_wind_speed` `(1, 21)` are in m/s. The supplied IFS arrays already include conversion from geopotential height to geopotential.

The example inputs have been centered on diagnosed cyclone centers and interpolated to a 72 × 72 grid at 0.25° spacing. Latitude runs north to south and longitude west to east. Each time uses its corresponding diagnosed center. The model applies its saved normalization and adds the final historical maximum wind speed after predicting intensity change; do not perform either step again. Future CMA observations in the reference CSV are for comparison only. Match each source to its own ONNX model and input NPZ.

## Model pseudocode

[TIFNet model pseudocode](models/tifnet_pseudocode.py) follows the original TIFNet forward pass used for the released AIFS and IFS models. It shows patch embedding, temporal and spatial attention, historical-intensity fusion, cross-attention, and the prediction head. The final function summarizes how the ONNX graph converts raw inputs into maximum wind speed. The pseudocode is explanatory; run the notebook above for inference with the trained weights.

## Generate IFSMatch2AIFS fields

```bash
ifsmatch2aifs   --aifs-directory /path/to/AIFS/npy   --ifs-directory /path/to/IFS/npy   --output-directory /path/to/IFSMatch2AIFS/npy
```

Dynamic variables use paired one-dimensional isotropic spectral attenuation with 25 km spacing, 30 radial bins, a two-dimensional Hann window, Gaussian smoothing with `sigma=2`, and gain capped at 1. Static variables (`lsm`, `slor`, `z`, `sdor`) are copied from IFS. The matching implementation and its parameters are unchanged.
