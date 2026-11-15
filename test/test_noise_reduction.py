import os
import sys
import unittest

# Allow running both under pytest (which loads the root conftest.py) and via
# plain `python -m unittest`. The bare `import pyaudioop` fallback in
# pydub.utils requires the package directory itself to be on sys.path.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in (ROOT, os.path.join(ROOT, "pydub")):
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np

from pydub import AudioSegment
from pydub.noise_reduction import (
    DEFAULT_FRAME_SIZE,
    DEFAULT_FRAMES_PER_CHUNK,
)

FRAME_RATE = 16000
SEED = 1701

INT_DTYPES = {1: np.int8, 2: np.int16, 4: np.int32}


def _to_segment(samples, sample_width=2, channels=1,
                frame_rate=FRAME_RATE):
    """Build an AudioSegment from float64 samples in roughly [-1, 1]."""
    peak = 2 ** (8 * sample_width - 1)
    quantized = np.clip(np.rint(samples * peak), -peak, peak - 1)
    data = quantized.astype(INT_DTYPES[sample_width]).tobytes()
    return AudioSegment(data, frame_rate=frame_rate,
                        sample_width=sample_width, channels=channels)


def _samples(segment, sample_width=2):
    return np.frombuffer(
        segment.get_array_of_samples(),
        dtype=INT_DTYPES[sample_width]
    ).astype(np.float64)


def _rms(samples):
    return float(np.sqrt(np.mean(samples ** 2)))


def _snr(clean, observed):
    noise = observed - clean
    return 20.0 * np.log10(_rms(clean) / _rms(noise))


def _band_energy(samples, frame_rate, center, bandwidth=12.0):
    spectrum = np.abs(np.fft.rfft(samples))
    frequencies = np.fft.rfftfreq(samples.size, 1.0 / frame_rate)
    mask = np.abs(frequencies - center) <= bandwidth
    return float(spectrum[mask].sum())


class NoiseReductionTests(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(SEED)
        duration = FRAME_RATE  # one second
        self.time = np.arange(duration) / float(FRAME_RATE)
        self.tone = 0.5 * np.sin(2 * np.pi * 440.0 * self.time)
        self.noise = 0.2 * rng.standard_normal(duration)
        self.mixed = self.tone + self.noise

        self.segment = _to_segment(self.mixed)
        self.clean_segment = _to_segment(self.tone)
        self.noise_segment = _to_segment(self.noise)

    def test_method_is_registered(self):
        self.assertTrue(hasattr(AudioSegment, "reduce_noise"))
        self.assertTrue(callable(AudioSegment.reduce_noise))

    def test_improves_snr_while_preserving_tone_frequency(self):
        denoised = self.segment.reduce_noise(
            self.noise_segment, strength=0.8
        )
        before = _snr(_samples(self.clean_segment),
                      _samples(self.segment))
        after = _snr(_samples(self.clean_segment),
                     _samples(denoised))

        # Spectral subtraction against the exact reference noise should give
        # a substantial, stable SNR gain.
        self.assertGreater(after - before, 6.0)

        # The voice/tone fundamental must survive almost untouched.
        tone_before = _band_energy(_samples(self.segment),
                                   FRAME_RATE, 440.0)
        tone_after = _band_energy(_samples(denoised),
                                  FRAME_RATE, 440.0)
        self.assertGreater(tone_after / tone_before, 0.95)

        # Energy well away from the tone should be clearly suppressed.
        spectrum_before = np.abs(
            np.fft.rfft(_samples(self.segment))
        )
        spectrum_after = np.abs(
            np.fft.rfft(_samples(denoised))
        )
        frequencies = np.fft.rfftfreq(self.time.size,
                                      1.0 / FRAME_RATE)
        high_band = frequencies > 1500.0
        self.assertLess(
            spectrum_after[high_band].mean()
            / spectrum_before[high_band].mean(),
            0.5
        )

    def test_preserves_metadata_and_duration(self):
        denoised = self.segment.reduce_noise(
            self.noise_segment, strength=0.7
        )
        self.assertEqual(len(denoised), len(self.segment))
        self.assertEqual(denoised.frame_rate, self.segment.frame_rate)
        self.assertEqual(denoised.channels, self.segment.channels)
        self.assertEqual(denoised.sample_width,
                         self.segment.sample_width)

    def test_strength_zero_and_empty_reference_are_passthrough(self):
        untouched = self.segment.reduce_noise(
            self.noise_segment, strength=0.0
        )
        self.assertEqual(untouched._data, self.segment._data)

        empty_reference = AudioSegment.silent(
            0, frame_rate=FRAME_RATE
        )
        self.assertEqual(
            self.segment.reduce_noise(empty_reference)._data,
            self.segment._data
        )

        empty_segment = AudioSegment(b"", metadata={
            "channels": 1, "sample_width": 2,
            "frame_rate": FRAME_RATE, "frame_width": 2,
        })
        result = empty_segment.reduce_noise(self.noise_segment)
        self.assertEqual(len(result), 0)

    def test_invalid_strength_raises_value_error(self):
        for bad in (-0.1, 1.1, 2.0, "0.7", None, True, False):
            with self.assertRaises(ValueError):
                self.segment.reduce_noise(self.noise_segment,
                                          strength=bad)

    def test_incompatible_reference_raises_value_error(self):
        wrong_rate = AudioSegment(
            self.noise_segment._data,
            metadata={
                "channels": 1, "sample_width": 2,
                "frame_rate": 8000, "frame_width": 2,
            }
        )
        with self.assertRaises(ValueError):
            self.segment.reduce_noise(wrong_rate)

        wrong_channels = self.noise_segment._spawn(
            self.noise_segment.get_array_of_samples(),
            overrides={"channels": 2, "frame_width": 4}
        )
        with self.assertRaises(ValueError):
            self.segment.reduce_noise(wrong_channels)

    def test_invalid_frame_parameters_raise_value_error(self):
        with self.assertRaises(ValueError):
            self.segment.reduce_noise(self.noise_segment,
                                      frame_size=3)
        with self.assertRaises(ValueError):
            self.segment.reduce_noise(self.noise_segment,
                                      frame_size=2048, hop_size=0)
        with self.assertRaises(ValueError):
            self.segment.reduce_noise(self.noise_segment,
                                      frame_size=2048, hop_size=4096)

    def test_supported_sample_widths(self):
        for sample_width in (1, 2, 4):
            segment = _to_segment(self.mixed, sample_width=sample_width)
            reference = _to_segment(self.noise,
                                    sample_width=sample_width)
            denoised = segment.reduce_noise(reference, strength=0.8)

            self.assertEqual(denoised.sample_width, sample_width)
            self.assertEqual(len(denoised), len(segment))
            self.assertLess(
                _rms(_samples(denoised, sample_width)),
                _rms(_samples(segment, sample_width))
            )

    def test_reference_sample_width_is_resampled(self):
        # A reference with the same data and length but a different width
        # must be accepted without altering the output segment's width.
        reference = _to_segment(self.noise, sample_width=4)
        denoised = self.segment.reduce_noise(reference, strength=0.8)
        self.assertEqual(denoised.sample_width,
                         self.segment.sample_width)
        self.assertEqual(len(denoised), len(self.segment))

    def test_stereo_channels_are_independent(self):
        rng = np.random.default_rng(SEED + 1)
        left = self.tone + self.noise
        right = 0.2 * rng.standard_normal(self.time.size)

        interleaved = np.empty(2 * self.time.size)
        interleaved[0::2] = left
        interleaved[1::2] = right
        stereo = _to_segment(interleaved, channels=2)

        reference_noise = np.repeat(self.noise, 2)
        reference = _to_segment(reference_noise, channels=2)

        denoised = stereo.reduce_noise(reference, strength=0.9)
        output = _samples(denoised).reshape(-1, 2)
        source = _samples(stereo).reshape(-1, 2)

        # The tone on the left channel is preserved...
        left_tone_before = _band_energy(source[:, 0],
                                        FRAME_RATE, 440.0)
        left_tone_after = _band_energy(output[:, 0],
                                       FRAME_RATE, 440.0)
        self.assertGreater(left_tone_after / left_tone_before, 0.95)

        # ...while the noise-only right channel is strongly attenuated.
        self.assertLess(
            _rms(output[:, 1]) / _rms(source[:, 1]), 0.4
        )

        # No cross-talk: the tone must not leak into the right channel.
        right_tone = _band_energy(output[:, 1],
                                  FRAME_RATE, 440.0)
        self.assertLess(right_tone, 0.05 * left_tone_after)

    def test_short_clip_shorter_than_frame(self):
        short = _to_segment(self.noise[:800])
        denoised = short.reduce_noise(self.noise_segment,
                                      strength=0.8)
        self.assertEqual(len(denoised), len(short))
        self.assertEqual(denoised.sample_width, short.sample_width)

    def test_no_jump_at_chunk_boundaries(self):
        # Twelve seconds crosses several internal processing chunks.
        long_mix = np.tile(self.mixed, 12)
        long_noise = np.tile(self.noise, 12)
        segment = _to_segment(long_mix)
        reference = _to_segment(long_noise)

        denoised = segment.reduce_noise(reference, strength=0.9)
        deltas = np.abs(np.diff(_samples(denoised)))
        percentile = np.percentile(deltas, 99)

        # Each chunk processes DEFAULT_FRAMES_PER_CHUNK frames; the first
        # flush ends at frames * hop - frame_size / 2 samples.
        hop_size = DEFAULT_FRAME_SIZE // 4
        first_boundary = (
            DEFAULT_FRAMES_PER_CHUNK * hop_size - DEFAULT_FRAME_SIZE // 2
        )
        checked = 0
        boundary = first_boundary
        while boundary + 300 < deltas.size:
            neighborhood = deltas[boundary - 300:boundary + 300]
            self.assertLess(neighborhood.max(), 2.0 * percentile)
            checked += 1
            boundary += DEFAULT_FRAMES_PER_CHUNK * hop_size
        self.assertGreaterEqual(checked, 1)

    def test_silence_is_preserved(self):
        silence = AudioSegment.silent(500, frame_rate=FRAME_RATE)
        denoised = silence.reduce_noise(self.noise_segment)
        self.assertEqual(len(denoised), 500)
        self.assertEqual(_rms(_samples(denoised)), 0.0)

    def test_result_remains_composable_and_exportable(self):
        denoised = self.segment.reduce_noise(
            self.noise_segment, strength=0.7
        )
        concatenated = denoised + self.segment
        self.assertEqual(
            len(concatenated), len(denoised) + len(self.segment)
        )

        exported = denoised.export(format="wav")
        reloaded = AudioSegment.from_wav(exported)
        self.assertEqual(len(reloaded), len(denoised))
        self.assertEqual(reloaded.frame_rate, denoised.frame_rate)
        self.assertEqual(reloaded.channels, denoised.channels)

    def test_strength_monotonicity(self):
        weak = self.segment.reduce_noise(self.noise_segment,
                                         strength=0.2)
        strong = self.segment.reduce_noise(self.noise_segment,
                                           strength=1.0)
        self.assertLess(
            _rms(_samples(strong) - _samples(self.clean_segment)),
            _rms(_samples(weak) - _samples(self.clean_segment))
        )


if __name__ == "__main__":
    unittest.main()
