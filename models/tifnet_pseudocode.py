"""Pseudocode for the TIFNet models distributed with this paper.

This file summarizes the TIFNet encoder-decoder architecture used by the
AIFS-driven and IFS-driven models accompanying the paper. The two models
share the architecture and have separate trained weights. Source-specific
fine-tuning updated only the final linear prediction head.

The functions below describe tensor operations; they are not executable
inference code. Use ../notebooks/aifs_inference.ipynb with the ONNX models
for inference. Input variable order, units, and preprocessing are specified
in ../README.md.

Shapes use (batch, time, tokens, features) unless a comment says otherwise.
The released ONNX interface fixes batch=1, history=5, forecast=21, and the
spatial grid at 72 x 72. All 21 forecast times are processed together.
"""


def encode_environmental_sequence(surface_fields, pressure_level_fields, branch):
    """Encode one historical or forecast sequence with its own TIFNet branch.

    Surface fields:       (batch, 9, time, 72, 72)
    Pressure-level fields: (batch, 5, time, 7, 72, 72)
    Result:               (batch, time, 768)
    """
    # Each time is embedded independently. The surface convolution uses
    # 4 x 4 patches and produces 256 features per patch. The pressure-level
    # convolution spans all seven levels and a 4 x 4 area, producing 512.
    surface_patches = branch.surface_patch_embedding(surface_fields)
    pressure_level_patches = branch.pressure_level_patch_embedding(
        pressure_level_fields
    )
    patch_tokens = concatenate_features(
        flatten_spatial_patches(surface_patches),
        flatten_spatial_patches(pressure_level_patches),
    )  # (batch, time, 324, 768): 18 x 18 patches

    # The learned summary token is repeated at every time. Its position is
    # included alongside the 324 patch positions in the learned embedding.
    tokens = prepend_summary_token(patch_tokens, branch.summary_token)
    tokens = tokens + branch.spatial_position_embedding  # 325 tokens/time
    number_of_time_steps = surface_fields.shape[2]
    tokens = tokens + sinusoidal_encoding_for_sequence_positions(number_of_time_steps)

    for attention_block in branch.space_time_blocks:  # eight distinct blocks
        tokens = apply_divided_space_time_block(tokens, attention_block)

    tokens = branch.final_layer_normalization(tokens)
    return select_summary_token_at_each_time(tokens)


def apply_divided_space_time_block(tokens, attention_block):
    """Update the summary and patch tokens using temporal, then spatial attention."""
    tokens = tokens + attention_block.temporal_attention(
        attention_block.temporal_layer_normalization(tokens),
        sequence_axis="time",
        group_by="token position",
    )
    tokens = tokens + attention_block.spatial_attention(
        attention_block.spatial_layer_normalization(tokens),
        sequence_axis="all 325 tokens within each time",
    )
    tokens = tokens + attention_block.feed_forward_network(
        attention_block.feed_forward_layer_normalization(tokens)
    )
    return tokens


def predict_normalized_intensity_change(
    historical_surface_fields,
    historical_pressure_level_fields,
    historical_maximum_wind_speed,
    forecast_surface_fields,
    forecast_pressure_level_fields,
    tifnet_model,
):
    """Original TIFNet forward pass before target denormalization.

    Historical maximum wind speed has shape (batch, 5, 1). No future
    observed intensity is supplied to the forecast branch.
    """
    historical_features = encode_environmental_sequence(
        historical_surface_fields,
        historical_pressure_level_fields,
        tifnet_model.encoder,
    )  # (batch, 5, 768)
    forecast_features = encode_environmental_sequence(
        forecast_surface_fields,
        forecast_pressure_level_fields,
        tifnet_model.decoder,
    )  # (batch, 21, 768); independent branch parameters

    # TIFNet fuses intensity AFTER the historical environmental blocks.
    historical_features = tifnet_model.encoder.merge_intensity(
        concatenate_features(historical_features, historical_maximum_wind_speed)
    )  # linear 769 -> 1536, GELU, linear 1536 -> 768

    for cross_attention_block in tifnet_model.cross_attention_blocks:  # two
        # At every forecast time, the query attends to all five history times.
        forecast_features = forecast_features + cross_attention_block.attention(
            query=cross_attention_block.query_normalization(forecast_features),
            key=historical_features,
            value=historical_features,
        )
        forecast_features = forecast_features + cross_attention_block.feed_forward_network(
            cross_attention_block.feed_forward_normalization(forecast_features)
        )

    normalized_intensity_change = tifnet_model.prediction_head(forecast_features)
    # The same biased 768 -> 1 head is used at every lead time.
    return normalized_intensity_change.squeeze_last_axis()  # (batch, 21)


def predict_maximum_wind_speed(
    encoder_surface,
    encoder_isobaric,
    encoder_artificial,
    decoder_surface,
    decoder_isobaric,
    tifnet_model,
    saved_statistics,
):
    """ONNX inference wrapper: raw physical inputs to maximum wind in m/s."""
    initial_maximum_wind_speed = encoder_artificial[:, -1, 0:1]
    normalized_encoder_surface = normalize(encoder_surface, saved_statistics.surface)
    normalized_encoder_isobaric = normalize(encoder_isobaric, saved_statistics.isobaric)
    normalized_encoder_artificial = normalize(
        encoder_artificial, saved_statistics.intensity
    )
    normalized_decoder_surface = normalize(decoder_surface, saved_statistics.surface)
    normalized_decoder_isobaric = normalize(decoder_isobaric, saved_statistics.isobaric)
    normalized_intensity_change = predict_normalized_intensity_change(
        normalized_encoder_surface,
        normalized_encoder_isobaric,
        normalized_encoder_artificial,
        normalized_decoder_surface,
        normalized_decoder_isobaric,
        tifnet_model,
    )
    intensity_change = denormalize_by_forecast_lead_time(
        normalized_intensity_change, saved_statistics.intensity_change
    )
    return initial_maximum_wind_speed + intensity_change  # (1, 21), m/s


def normalize(physical_values, statistics):
    """Apply the saved per-channel offset and scale used by the ONNX model."""
    return (physical_values - statistics.offset) / statistics.scale


# Self-attention and cross-attention use 16 heads and 768 features.
# Their feed-forward layers expand 768 features fourfold to 3072.
# The two source-specific models retain the original TIFNet backbone and
# each has its own fine-tuned final linear head.
