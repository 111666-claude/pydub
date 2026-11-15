from __future__ import division

import array

from .utils import register_pydub_effect

try:
    import numpy as np
except ImportError:
    np = None


# Default STFT parameters. A Hann window with 75% overlap gives a constant
# window-sum envelope in the signal interior and smooth frame transitions.
DEFAULT_FRAME_SIZE = 2048
DEFAULT_FRAMES_PER_CHUNK = 256


def _require_numpy():
    if np is None:
        raise ImportError(
            "reduce_noise() requires numpy. "
            "Install it with `pip install numpy`."
        )


def _integer_dtype(sample_width):
    # pydub stores 8-bit audio signed and converts 24-bit audio to 32-bit,
    # so only 1, 2 and 4 byte sample widths ever reach this point.
    return {1: np.int8, 2: np.int16, 4: np.int32}[sample_width]


def _integer_range(sample_width):
    bits = sample_width * 8
    return -(2 ** (bits - 1)), 2 ** (bits - 1) - 1


def _segment_to_channels(seg):
    """
    Return the raw samples of an AudioSegment as a float64 array of shape
    (channels, sample_count) plus the numpy integer dtype matching the
    segment's sample width.
    """
    dtype = _integer_dtype(seg.sample_width)
    samples = np.frombuffer(seg.get_array_of_samples(), dtype=dtype)
    samples = samples.astype(np.float64)
    return samples.reshape(-1, seg.channels).T, dtype


def _convert_sample_width(seg, target_width):
    """
    Resample raw PCM to another sample width using numpy, avoiding the
    stdlib audioop dependency (unavailable on Python >= 3.13). Values are
    scaled between the integer ranges rather than reinterpreted.
    """
    if seg.sample_width == target_width:
        return seg

    source_min, source_max = _integer_range(seg.sample_width)
    target_min, target_max = _integer_range(target_width)
    source_dtype = _integer_dtype(seg.sample_width)
    target_dtype = _integer_dtype(target_width)

    samples = np.frombuffer(seg.get_array_of_samples(),
                            dtype=source_dtype).astype(np.float64)
    scaled = samples * (target_max - target_min) / (source_max - source_min)
    scaled = np.clip(np.rint(scaled), target_min, target_max)

    raw = scaled.astype(target_dtype).tobytes()
    out_array = array.array(
        {1: "b", 2: "h", 4: "i"}[target_width], raw
    )
    return seg._spawn(
        data=out_array,
        overrides={
            "sample_width": target_width,
            "frame_width": target_width * seg.channels,
        }
    )


def _channels_to_array(channels, dtype, array_type, min_value, max_value):
    """Inverse of _segment_to_channels: re-interleave and quantize."""
    interleaved = np.clip(np.rint(channels.T.reshape(-1)), min_value, max_value)
    return array.array(array_type, interleaved.astype(dtype).tobytes())


def _noise_power_profile(noise_channels, frame_size, window):
    """
    Average power spectrum of the reference noise clip, estimated
    independently for every channel.
    """
    channels, sample_count = noise_channels.shape
    profile = np.zeros((channels, frame_size // 2 + 1), dtype=np.float64)
    frame_count = 0
    for start in range(0, max(sample_count - frame_size, 0) + 1, frame_size):
        frames = noise_channels[:, start:start + frame_size]
        if frames.shape[1] < frame_size:
            padded = np.zeros((channels, frame_size), dtype=np.float64)
            padded[:, :frames.shape[1]] = frames
            frames = padded
        spectrum = np.fft.rfft(frames * window, axis=1)
        profile += (spectrum.real ** 2 + spectrum.imag ** 2)
        frame_count += 1

    # A reference shorter than one analysis frame is zero-padded once.
    if frame_count == 0:
        spectrum = np.fft.rfft(noise_channels * window, axis=1)
        profile = spectrum.real ** 2 + spectrum.imag ** 2
    else:
        profile /= frame_count
    return profile


def _spectral_gains(power, noise_power, strength, floor_gain):
    """
    Over-subtracting spectral subtraction gain (Berouti-style):

        |Y|^2 = max(|X|^2 - alpha * |N|^2, floor_gain^2 * |X|^2)

    Bins dominated by the reference noise spectrum are attenuated towards
    ``floor_gain`` while bins whose energy clearly exceeds the noise (e.g.
    voice fundamentals and harmonics) keep a gain close to one.
    """
    over_subtraction = 1.0 + 2.0 * strength
    target = np.maximum(
        power - over_subtraction * noise_power,
        (floor_gain ** 2) * power
    )
    gains = np.ones_like(power)
    nonzero = power > 0.0
    gains[nonzero] = np.sqrt(target[nonzero] / power[nonzero])
    return gains


def _denoise_channel(samples, noise_power, frame_size, hop_size, strength):
    """
    Stream one channel through the STFT -> gain -> inverse STFT pipeline in
    bounded-size chunks. Temporal gain state and overlap-add tails are carried
    across chunk boundaries so processing length does not change the result.
    """
    sample_count = samples.shape[0]
    window = np.hanning(frame_size)
    pad = frame_size // 2
    floor_gain = max(0.02, 1.0 - 0.98 * strength)
    # Temporal smoothing of the gain curve: suppresses "musical noise" and
    # guarantees frame-to-frame continuity at every analysis boundary.
    smoothing = 0.85

    output = np.zeros(sample_count, dtype=np.float64)
    carried_signal = np.zeros(0, dtype=np.float64)
    carried_window = np.zeros(0, dtype=np.float64)
    previous_gains = np.ones(frame_size // 2 + 1, dtype=np.float64)

    base = 0
    frame_index = 0
    # The buffer must hold every frame this chunk could touch. With a
    # custom hop size the span differs from the default 75%-overlap layout.
    chunk_span = DEFAULT_FRAMES_PER_CHUNK * hop_size + frame_size

    while base < sample_count:
        accum = np.zeros(chunk_span, dtype=np.float64)
        envelope = np.zeros(chunk_span, dtype=np.float64)
        accum[:carried_signal.shape[0]] = carried_signal
        envelope[:carried_window.shape[0]] = carried_window

        frames_done = 0
        while frames_done < DEFAULT_FRAMES_PER_CHUNK:
            start = frame_index * hop_size - pad
            if start >= sample_count:
                break

            frame = np.zeros(frame_size, dtype=np.float64)
            src_start = max(start, 0)
            src_end = min(start + frame_size, sample_count)
            dst_start = src_start - start
            frame[dst_start:dst_start + (src_end - src_start)] = \
                samples[src_start:src_end]

            spectrum = np.fft.rfft(frame * window)
            power = spectrum.real ** 2 + spectrum.imag ** 2
            gains = _spectral_gains(power, noise_power, strength, floor_gain)
            gains = smoothing * gains + (1.0 - smoothing) * previous_gains
            previous_gains = gains

            reconstructed = np.fft.irfft(spectrum * gains, frame_size) * window
            window_energy = window ** 2

            offset = src_start - base
            length = src_end - src_start
            accum[offset:offset + length] += reconstructed[
                dst_start:dst_start + length
            ]
            envelope[offset:offset + length] += window_energy[
                dst_start:dst_start + length
            ]

            frame_index += 1
            frames_done += 1

        next_start = frame_index * hop_size - pad
        flush_end = min(max(next_start - base, 0), sample_count - base)
        target = envelope[:flush_end] > 1e-12
        output[base:base + flush_end][target] = (
            accum[:flush_end][target] / envelope[:flush_end][target]
        )

        carried_signal = accum[flush_end:].copy()
        carried_window = envelope[flush_end:].copy()
        base += flush_end

    return output


@register_pydub_effect
def reduce_noise(seg, noise_clip, strength=0.7, frame_size=DEFAULT_FRAME_SIZE,
                 hop_size=None):
    """
    Frequency-domain noise reduction driven by a reference noise clip.

    The reference clip should contain only the noise to remove (e.g. a short
    recording of the steady air-conditioning hum between voice-overs). Its
    average power spectrum is estimated per channel and subtracted from every
    analysis frame of ``seg`` with over-subtracting spectral subtraction.

    Keyword Arguments:

        noise_clip - an AudioSegment containing a sample of the noise.
            Must use the same frame rate and channel count as ``seg``.

        strength - float in the range [0.0, 1.0], default 0.7.
            How aggressively spectra matching the reference noise are
            suppressed. 0.0 leaves the signal untouched; 1.0 applies the
            strongest subtraction and deepest attenuation floor.

        frame_size - STFT window length in samples, default 2048.
            Must be an integer of at least 4.

        hop_size - STFT hop length in samples, default ``frame_size // 4``
            (75% overlap). Must be between 1 and ``frame_size``.

    Returns a new AudioSegment with the same frame rate, channel count,
    sample width and duration as ``seg``. The result can be concatenated,
    overlaid and exported like any other AudioSegment.

    An empty ``noise_clip`` (or ``strength`` of 0.0) returns ``seg``
    unchanged. Incompatible frame rates or channel counts and out-of-range
    parameters raise ValueError.
    """
    _require_numpy()

    if not isinstance(strength, (int, float)) or isinstance(strength, bool):
        raise ValueError("strength must be a number between 0.0 and 1.0")
    if not 0.0 <= float(strength) <= 1.0:
        raise ValueError("strength must be between 0.0 and 1.0")
    if not isinstance(frame_size, int) or frame_size < 4:
        raise ValueError("frame_size must be an integer of at least 4")
    if hop_size is None:
        hop_size = frame_size // 4
    if not isinstance(hop_size, int) or not 1 <= hop_size <= frame_size:
        raise ValueError(
            "hop_size must be an integer between 1 and frame_size"
        )

    # An empty reference carries no noise information.
    if len(noise_clip._data) == 0 or float(strength) == 0.0:
        return seg

    if noise_clip.frame_rate != seg.frame_rate:
        raise ValueError(
            "noise_clip frame rate ({0}) does not match the segment "
            "frame rate ({1})".format(noise_clip.frame_rate, seg.frame_rate)
        )
    if noise_clip.channels != seg.channels:
        raise ValueError(
            "noise_clip channel count ({0}) does not match the segment "
            "channel count ({1})".format(noise_clip.channels, seg.channels)
        )
    if noise_clip.sample_width != seg.sample_width:
        # Resample the reference width only; the segment width is preserved.
        noise_clip = _convert_sample_width(
            noise_clip, seg.sample_width
        )

    if len(seg._data) == 0:
        return seg._spawn(data=b'')

    signal_channels, dtype = _segment_to_channels(seg)
    noise_channels, _ = _segment_to_channels(noise_clip)

    window = np.hanning(frame_size)
    noise_profile = _noise_power_profile(noise_channels, frame_size, window)
    # The reference arrays are no longer needed beyond its power profile.
    del noise_channels

    # Single output buffer, filled one channel at a time. Apart from this
    # buffer and the input copy the working set per channel is bounded by
    # the chunk size regardless of the total segment length.
    sample_count = signal_channels.shape[1]
    result_channels = np.empty(
        (seg.channels, sample_count), dtype=np.float64
    )
    for channel in range(seg.channels):
        result_channels[channel] = _denoise_channel(
            signal_channels[channel], noise_profile[channel],
            frame_size, hop_size, float(strength)
        )
    del signal_channels

    min_value, max_value = _integer_range(seg.sample_width)
    out_array = _channels_to_array(
        result_channels, dtype, seg.array_type, min_value, max_value
    )
    return seg._spawn(data=out_array)
