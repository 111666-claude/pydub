from functools import partial
import math
import os
import sys
import unittest
from array import array
from tempfile import (
    NamedTemporaryFile,
    mkdtemp,
    gettempdir
)
import tempfile
import struct

from pydub import AudioSegment
from pydub.audio_segment import (
    CROSSFADE_LINEAR,
    CROSSFADE_EQUAL_POWER,
    CROSSFADE_CURVES,
    extract_wav_headers,
)
from pydub.utils import (
    db_to_float,
    ratio_to_db,
    make_chunks,
    mediainfo,
    get_encoder_name,
    get_supported_decoders,
    get_supported_encoders,
    get_array_type,
    audioop,
)
from pydub.exceptions import (
    InvalidTag,
    InvalidID3TagVersion,
    InvalidDuration,
    CouldntDecodeError,
    MissingAudioParameter,
)
from pydub.silence import (
    detect_silence,
    split_on_silence,
)
from pydub.generators import (
    Sine,
    Square,
    Pulse,
    Triangle,
    Sawtooth,
    WhiteNoise,
)

data_dir = os.path.join(os.path.dirname(__file__), 'data')


class UtilityTests(unittest.TestCase):

    def test_db_float_conversions(self):
        self.assertEqual(db_to_float(20), 10)
        self.assertEqual(db_to_float(10, using_amplitude=False), 10)
        self.assertEqual(db_to_float(0), 1)
        self.assertEqual(ratio_to_db(1), 0)
        self.assertEqual(ratio_to_db(10), 20)
        self.assertEqual(ratio_to_db(10, using_amplitude=False), 10)
        self.assertEqual(3, db_to_float(ratio_to_db(3)))
        self.assertEqual(12, ratio_to_db(db_to_float(12)))
        self.assertEqual(3, db_to_float(ratio_to_db(3, using_amplitude=False), using_amplitude=False))
        self.assertEqual(12, ratio_to_db(db_to_float(12, using_amplitude=False), using_amplitude=False))


if sys.version_info >= (3, 6):
    class PathLikeObjectTests(unittest.TestCase):

        class MyPathLike:
            def __init__(self, path):
                self.path = path

            def __fspath__(self):
                return self.path

        def setUp(self):
            self.mp3_path_str = os.path.join(data_dir, 'test1.mp3')

            from pathlib import Path
            self.mp3_pathlib_path = Path(self.mp3_path_str)

            self.mp3_path_like_str = self.MyPathLike(self.mp3_path_str)
            self.mp3_path_like_bytes = self.MyPathLike(bytes(self.mp3_path_str, sys.getdefaultencoding()))

        def test_audio_segment_from_pathlib_path(self):
            seg1 = AudioSegment.from_file(self.mp3_path_str)
            seg2 = AudioSegment.from_file(self.mp3_pathlib_path)

            self.assertEqual(len(seg1), len(seg2))
            self.assertEqual(seg1._data, seg2._data)
            self.assertTrue(len(seg1) > 0)

        def test_audio_segment_from_path_like_str(self):
            seg1 = AudioSegment.from_file(self.mp3_path_str)
            seg2 = AudioSegment.from_file(self.mp3_path_like_str)

            self.assertEqual(len(seg1), len(seg2))
            self.assertEqual(seg1._data, seg2._data)
            self.assertTrue(len(seg1) > 0)

        def test_audio_segment_from_path_like_bytes(self):
            seg1 = AudioSegment.from_file(self.mp3_path_str)
            seg2 = AudioSegment.from_file(self.mp3_path_like_bytes)

            self.assertEqual(len(seg1), len(seg2))
            self.assertEqual(seg1._data, seg2._data)
            self.assertTrue(len(seg1) > 0)

        def test_non_existant_pathlib_path(self):
            from pathlib import Path
            path = Path('this/path/should/not/exist/do/not/make/this/exist')
            with self.assertRaises(FileNotFoundError):
                _ = AudioSegment.from_file(path)

            path = Path('')
            # On Unicies this will raise a IsADirectoryError, on Windows this
            # will result in a PermissionError. Both of these are subclasses of
            # OSError. We aren't so much worried about the specific exception
            # here, just that reading a file from an empty path is an error.
            with self.assertRaises(OSError):
                _ = AudioSegment.from_file(path)

        def test_non_existant_path_like_str(self):
            path = self.MyPathLike('this/path/should/not/exist/do/not/make/this/exist')
            with self.assertRaises(FileNotFoundError):
                _ = AudioSegment.from_file(path)

            path = self.MyPathLike('')
            with self.assertRaises(FileNotFoundError):
                _ = AudioSegment.from_file(path)

        def test_non_existant_path_like_bytes(self):
            path = self.MyPathLike(bytes('this/path/should/not/exist/do/not/make/this/exist', sys.getdefaultencoding()))
            with self.assertRaises(FileNotFoundError):
                _ = AudioSegment.from_file(path)

            path = self.MyPathLike(bytes('', sys.getdefaultencoding()))
            with self.assertRaises(FileNotFoundError):
                _ = AudioSegment.from_file(path)

        def assertWithinRange(self, val, lower_bound, upper_bound):
            self.assertTrue(lower_bound <= val <= upper_bound,
                            "%s is not in the acceptable range: %s - %s" %
                            (val, lower_bound, upper_bound))

        def assertWithinTolerance(self, val, expected, tolerance=None,
                                  percentage=None):
            if percentage is not None:
                tolerance = expected * percentage
            lower_bound = expected - tolerance
            upper_bound = expected + tolerance
            self.assertWithinRange(val, lower_bound, upper_bound)

        def test_export_pathlib_path(self):
            seg1 = AudioSegment.from_file(self.mp3_path_str)
            from pathlib import Path
            path = Path(tempfile.gettempdir()) / 'pydub-test-export-8ajds.mp3'
            try:
                seg1.export(path, format='mp3')
                seg2 = AudioSegment.from_file(path, format='mp3')

                self.assertTrue(len(seg1) > 0)
                self.assertWithinTolerance(len(seg1),
                                           len(seg2),
                                           percentage=0.01)
            finally:
                os.unlink(path)


class FileAccessTests(unittest.TestCase):

    def setUp(self):
        self.mp3_path = os.path.join(data_dir, 'test1.mp3')

    def test_audio_segment_from_mp3(self):
        seg1 = AudioSegment.from_mp3(os.path.join(data_dir, 'test1.mp3'))

        mp3_file = open(os.path.join(data_dir, 'test1.mp3'), 'rb')
        seg2 = AudioSegment.from_mp3(mp3_file)

        self.assertEqual(len(seg1), len(seg2))
        self.assertTrue(seg1._data == seg2._data)
        self.assertTrue(len(seg1) > 0)


test1wav = test4wav = test1 = test2 = test3 = testparty = testdcoffset = None


class AudioSegmentTests(unittest.TestCase):

    def setUp(self):
        global test1, test2, test3, testparty, testdcoffset
        if not test1:
            a = os.path.join(data_dir, 'test1.mp3')
            test1 = AudioSegment.from_mp3(os.path.join(data_dir, 'test1.mp3'))
            test2 = AudioSegment.from_mp3(os.path.join(data_dir, 'test2.mp3'))
            test3 = AudioSegment.from_mp3(os.path.join(data_dir, 'test3.mp3'))
            testdcoffset = AudioSegment.from_mp3(
                os.path.join(data_dir, 'test-dc_offset.wav'))
            testparty = AudioSegment.from_mp3(
                os.path.join(data_dir, 'party.mp3'))

        self.seg1 = test1
        self.seg2 = test2
        self.seg3 = test3
        self.mp3_seg_party = testparty
        self.seg_dc_offset = testdcoffset

        self.ogg_file_path = os.path.join(data_dir, 'bach.ogg')
        self.mp4_file_path = os.path.join(data_dir, 'creative_common.mp4')
        self.mp3_file_path = os.path.join(data_dir, 'party.mp3')
        self.webm_file_path = os.path.join(data_dir, 'test5.webm')

        self.jpg_cover_path = os.path.join(data_dir, 'cover.jpg')
        self.png_cover_path = os.path.join(data_dir, 'cover.png')

    def assertWithinRange(self, val, lower_bound, upper_bound):
        self.assertTrue(lower_bound <= val <= upper_bound,
                        "%s is not in the acceptable range: %s - %s" %
                        (val, lower_bound, upper_bound))

    def assertWithinTolerance(self, val, expected, tolerance=None,
                              percentage=None):
        if percentage is not None:
            tolerance = expected * percentage
        lower_bound = expected - tolerance
        upper_bound = expected + tolerance
        self.assertWithinRange(val, lower_bound, upper_bound)

    def test_direct_instantiation_with_bytes(self):
        seg = AudioSegment(
            b'RIFF\x28\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x02\x00\x00}\x00\x00\x00\xf4\x01\x00\x04\x00\x10\x00data\x04\x00\x00\x00\x00\x00\x00\x00')
        self.assertEqual(seg.frame_count(), 1)
        self.assertEqual(seg.channels, 2)
        self.assertEqual(seg.sample_width, 2)
        self.assertEqual(seg.frame_rate, 32000)

    def test_24_bit_audio(self):
        path24 = os.path.join(data_dir, 'test1-24bit.wav')
        seg24 = AudioSegment._from_safe_wav(path24)
        # The data length lies at bytes 40-44
        with open(path24, 'rb') as f:
            raw24 = f.read()
        len24 = struct.unpack("<L", raw24[40:44])[0]

        # should have been converted to 32 bit
        self.assertEqual(seg24.sample_width, 4)
        # the data length should have grown by exactly 4:3 (24 bits turn into 32 bits)
        self.assertEqual(len(seg24.raw_data) * 3, len24 * 4)

    def test_8_bit_audio(self):
        original_path = os.path.join(data_dir,'test1.wav')
        original_segment = AudioSegment.from_file(original_path)
        target_rms = original_segment.rms//2**8

        path_with_8bits = os.path.join(data_dir,'test1-8bit.wav')

        def check_8bit_segment(segment):
            self.assertWithinTolerance(segment.rms,target_rms,tolerance=0)

        # check reading directly
        check_8bit_segment(AudioSegment.from_file(path_with_8bits))

        # check using ffmpeg on it
        with open(path_with_8bits,'rb') as file_8bit:
            check_8bit_segment(AudioSegment.from_file(file_8bit))

        # check conversion from higher-width sample
        check_8bit_segment(AudioSegment.from_file(original_path).set_sample_width(1))

        # check audio export
        with NamedTemporaryFile('w+b', suffix='.wav') as tmp_file:
            original_segment.set_sample_width(1).export(tmp_file,format='wav')
            tmp_file.seek(0)
            check_8bit_segment(AudioSegment.from_file(tmp_file))

    def test_192khz_audio(self):
        test_files = [('test-192khz-16bit.wav', 16),
                      ('test-192khz-24bit.wav', 32),
                      ('test-192khz-32bit.flac', 32),
                      ('test-192khz-32bit.wav', 32),
                      ('test-192khz-64bit.wav', 64)]
        base_file, bit_depth = test_files[0]
        path = os.path.join(data_dir, base_file)
        base = AudioSegment.from_file(path)

        headers = extract_wav_headers(open(path, 'rb').read())
        data16_size = headers[-1].size
        self.assertEqual(len(base.raw_data), data16_size)
        self.assertEqual(base.frame_rate, 192000)
        self.assertEqual(base.sample_width, bit_depth / 8)

        for test_file, bit_depth in test_files[1:]:
            path = os.path.join(data_dir, test_file)
            seg = AudioSegment.from_file(path)
            self.assertEqual(seg.sample_width, bit_depth / 8)
            self.assertEqual(seg.frame_rate, 192000)
            self.assertEqual(len(seg.raw_data), len(base.raw_data) *
                             seg.sample_width / base.sample_width)
            self.assertEqual(seg.frame_rate, 192000)

    def test_concat(self):
        catted_audio = self.seg1 + self.seg2

        expected = len(self.seg1) + len(self.seg2)
        self.assertWithinTolerance(len(catted_audio), expected, 1)

    def test_append(self):
        merged1 = self.seg3.append(self.seg1, crossfade=100)
        merged2 = self.seg3.append(self.seg2, crossfade=100)

        self.assertEqual(len(merged1), len(self.seg1) + len(self.seg3) - 100)
        self.assertEqual(len(merged2), len(self.seg2) + len(self.seg3) - 100)

    def test_too_long_crossfade(self):
        s1 = self.seg1[:1000]
        s2 = self.seg2[:500]
        self.assertRaises(ValueError, lambda: s1.append(s2, crossfade=len(s1) + 10))

    def test_sum(self):
        def gen():
            yield self.seg1
            yield self.seg2
            yield self.seg3

        try:
            summed = sum(gen())
        except TypeError as e:
            if "unsupported operand" in str(e):
                self.fail("Could not sum() audio segments.")
            else:
                raise

    def test_volume_with_add_sub(self):
        quieter = self.seg1 - 6
        self.assertAlmostEqual(ratio_to_db(quieter.rms, self.seg1.rms),
                               -6,
                               places=2)

        louder = quieter + 2.5
        self.assertAlmostEqual(ratio_to_db(louder.rms, quieter.rms),
                               2.5,
                               places=2)

    def test_repeat_with_multiply(self):
        seg = self.seg1 * 3
        expected = len(self.seg1) * 3
        expected = (expected - 2, expected + 2)
        self.assertTrue(expected[0] < len(seg) < expected[1])

    def test_overlay(self):
        seg_mult = self.seg1[:5000] * self.seg2[:3000]
        seg_over = self.seg1[:5000].overlay(self.seg2[:3000], loop=True)

        self.assertEqual(len(seg_mult), len(seg_over))
        self.assertTrue(seg_mult._data == seg_over._data)

        self.assertEqual(len(seg_mult), 5000)
        self.assertEqual(len(seg_over), 5000)

    def test_overlay_times(self):
        # infinite
        seg_mult = self.seg1[:5000] * self.seg2[:3000]
        seg_over = self.seg1[:5000].overlay(self.seg2[:3000], times=99999999)
        self.assertEqual(len(seg_mult), len(seg_over))
        self.assertEqual(len(seg_over), 5000)
        self.assertTrue(seg_mult._data == seg_over._data)

        # no times, no-op
        piece = self.seg2[:1000]
        seg_manual = self.seg1[:4000]
        seg_over = self.seg1[:4000].overlay(piece, times=0)
        self.assertEqual(len(seg_manual), len(seg_over))
        self.assertEqual(len(seg_over), 4000)
        self.assertFalse(seg_mult._data == seg_over._data)

        # 1 loop
        seg_manual = self.seg1[:4000].overlay(piece, position=500)
        seg_over = self.seg1[:4000].overlay(piece, times=1)
        self.assertEqual(len(seg_manual), len(seg_over))
        self.assertEqual(len(seg_over), 4000)
        self.assertFalse(seg_mult._data == seg_over._data)

        # 2 loops
        seg_manual = self.seg1[:4000].overlay(piece, position=500) \
            .overlay(piece, position=1500)
        seg_over = self.seg1[:4000].overlay(piece, times=2)
        self.assertEqual(len(seg_manual), len(seg_over))
        self.assertEqual(len(seg_over), 4000)
        self.assertFalse(seg_mult._data == seg_over._data)

        # 3 loops
        seg_manual = self.seg1[:4000].overlay(piece, position=500) \
            .overlay(piece, position=1500).overlay(piece, position=2500)
        seg_over = self.seg1[:4000].overlay(piece, times=3)
        self.assertEqual(len(seg_manual), len(seg_over))
        self.assertEqual(len(seg_over), 4000)
        self.assertFalse(seg_mult._data == seg_over._data)

        # 4 loops (last will pass end)
        seg_manual = self.seg1[:4000].overlay(piece, position=500) \
            .overlay(piece, position=1500).overlay(piece, position=2500) \
            .overlay(piece, position=3500)
        seg_over = self.seg1[:4000].overlay(piece, times=4)
        self.assertEqual(len(seg_manual), len(seg_over))
        self.assertEqual(len(seg_over), 4000)
        self.assertFalse(seg_mult._data == seg_over._data)

        # 5 loops (last won't happen b/c past end)
        seg_manual = self.seg1[:4000].overlay(piece, position=500) \
            .overlay(piece, position=1500).overlay(piece, position=2500) \
            .overlay(piece, position=3500).overlay(piece, position=3500)
        seg_over = self.seg1[:4000].overlay(piece, times=5)
        self.assertEqual(len(seg_manual), len(seg_over))
        self.assertEqual(len(seg_over), 4000)
        self.assertFalse(seg_mult._data == seg_over._data)

        # ~infinite, same (as 4 and 5 really)
        seg_over = self.seg1[:4000].overlay(piece, times=999999999)
        self.assertEqual(len(seg_manual), len(seg_over))
        self.assertEqual(len(seg_over), 4000)
        self.assertFalse(seg_mult._data == seg_over._data)

    def test_overlay_with_gain_change(self):
        # Use overlay silence with volume change
        seg_one = self.seg1[:5000]
        seg_silent = AudioSegment.silent(duration=2000)
        seg_over = seg_one.overlay(seg_silent, gain_during_overlay=-7)

        # Manually lower first segment
        seg_one_lower = seg_one - 7
        seg_manual = seg_one_lower[:2000] + seg_one[2000:]

        self.assertEqual(len(seg_over), len(seg_manual))
        self.assertAlmostEqual(seg_over.dBFS, seg_manual.dBFS)
        self.assertEqual(len(seg_manual), 5000)
        self.assertEqual(len(seg_over), 5000)

    def test_slicing(self):
        empty = self.seg1[:0]
        second_long_slice = self.seg1[:1000]
        remainder = self.seg1[1000:]

        self.assertEqual(len(empty), 0)
        self.assertEqual(len(second_long_slice), 1000)
        self.assertEqual(len(remainder), len(self.seg1) - 1000)

        last_5_seconds = self.seg1[-5000:]
        before = self.seg1[:-5000]

        self.assertEqual(len(last_5_seconds), 5000)
        self.assertEqual(len(before), len(self.seg1) - 5000)

        past_end = second_long_slice[:1500]
        self.assertTrue(second_long_slice._data == past_end._data)

    def test_slicing_by_step(self):
        audio = self.seg1[:10000]
        chunks = audio[:0]

        for chunk in audio[::1000]:
            self.assertTrue(isinstance(chunk, AudioSegment))
            self.assertEqual(len(chunk), 1000)
            chunks += chunk

        self.assertEqual(len(audio), len(chunks))

    def test_indexing(self):
        short = self.seg1[:100]

        rebuilt1 = self.seg1[:0]
        for part in short:
            rebuilt1 += part

        rebuilt2 = sum([part for part in short])

        self.assertTrue(short._data == rebuilt1._data)
        self.assertTrue(short._data == rebuilt2._data)

    def test_set_channels(self):
        mono = self.seg1.set_channels(1)
        stereo = mono.set_channels(2)

        self.assertEqual(len(self.seg1), len(mono))
        self.assertEqual(len(self.seg1), len(stereo))

        mono = self.seg2.set_channels(1)
        mono = mono.set_frame_rate(22050)

        self.assertEqual(len(mono), len(self.seg2))

        with NamedTemporaryFile('w+b', suffix='.mp3') as tmp_file:
            if sys.platform == 'win32':
                tmp_file.close()

            mono.export(tmp_file.name, 'mp3')
            monomp3 = AudioSegment.from_mp3(tmp_file.name)

            self.assertWithinTolerance(
                len(monomp3),
                len(self.seg2),
                tolerance=105
            )

            if sys.platform == 'win32':
                os.remove(tmp_file.name)

        merged = mono.append(stereo, crossfade=100)
        self.assertWithinTolerance(len(merged),
                                   len(self.seg1) + len(self.seg2) - 100,
                                   tolerance=1)

    def test_split_to_mono(self):
        seg = self.seg1
        mono_segments = seg.split_to_mono()
        seg_lchannel = mono_segments[0]
        seg_rchannel = mono_segments[1]

        self.assertEqual(len(seg_lchannel), len(seg))
        self.assertEqual(len(seg_rchannel), len(seg))

        self.assertEqual(seg_lchannel.frame_rate, seg.frame_rate)
        self.assertEqual(seg_rchannel.frame_rate, seg.frame_rate)

        self.assertEqual(seg_lchannel.frame_count(), seg.frame_count())
        self.assertEqual(seg_rchannel.frame_count(), seg.frame_count())

    def test_apply_gain_stereo(self):
        seg = self.seg1

        orig_l, orig_r = seg.split_to_mono()
        orig_dbfs_l = orig_l.dBFS
        orig_dbfs_r = orig_r.dBFS

        # for readability: infinity
        inf = float("inf")

        def assertAlmostEqual(v1, v2, **kwargs):
            if v1 in (inf, -inf):
                self.assertEqual(v1, v2)
            else:
                self.assertAlmostEqual(v1, v2, **kwargs)

        def check_stereo_gain(left_dbfs_change, right_dbfs_change):
            panned = seg.apply_gain_stereo(left_dbfs_change, right_dbfs_change)
            self.assertEqual(panned.channels, 2)

            l, r = panned.split_to_mono()
            assertAlmostEqual(l.dBFS, orig_dbfs_l + left_dbfs_change, places=2)
            assertAlmostEqual(r.dBFS, orig_dbfs_r + right_dbfs_change, places=2)

        # hard left
        check_stereo_gain(0.0, -inf)
        check_stereo_gain(0.0, -6.0)
        check_stereo_gain(0.0, 0.0)
        check_stereo_gain(-6.0, 0.0)
        check_stereo_gain(-inf, 0.0)

    def test_pan(self):
        seg = self.seg1

        orig_l, orig_r = seg.split_to_mono()
        orig_dbfs_l = orig_l.dBFS
        orig_dbfs_r = orig_r.dBFS

        # for readability: infinity
        inf = float("inf")

        def assertAlmostEqual(v1, v2, **kwargs):
            if v1 in (inf, -inf):
                self.assertEqual(v1, v2)
            else:
                self.assertAlmostEqual(v1, v2, **kwargs)

        def check_pan(pan, left_dbfs_change, right_dbfs_change):
            panned = seg.pan(pan)

            l, r = panned.split_to_mono()
            assertAlmostEqual(l.dBFS, orig_dbfs_l + left_dbfs_change, places=1)
            assertAlmostEqual(r.dBFS, orig_dbfs_r + right_dbfs_change, places=1)

        check_pan(-1.0, 3.0, -inf)
        check_pan(-0.5, 1.5, -4.65)
        check_pan(0.0, 0.0, 0.0)
        check_pan(0.5, -4.65, 1.5)
        check_pan(1.0, -inf, 3.0)

    def test_export_as_mp3(self):
        seg = self.seg1
        exported_mp3 = seg.export()
        seg_exported_mp3 = AudioSegment.from_mp3(exported_mp3)

        self.assertWithinTolerance(len(seg_exported_mp3),
                                   len(seg),
                                   percentage=0.01)

    def test_export_as_wav(self):
        seg = self.seg1
        exported_wav = seg.export(format='wav')
        seg_exported_wav = AudioSegment.from_wav(exported_wav)

        self.assertWithinTolerance(len(seg_exported_wav),
                                   len(seg),
                                   percentage=0.01)

    def test_export_as_wav_with_codec(self):
        seg = self.seg1
        exported_wav = seg.export(format='wav', codec='pcm_s32le')
        seg_exported_wav = AudioSegment.from_wav(exported_wav)

        self.assertWithinTolerance(len(seg_exported_wav),
                                   len(seg),
                                   percentage=0.01)
        self.assertEqual(seg_exported_wav.sample_width, 4)

    def test_export_as_wav_with_parameters(self):
        seg = self.seg1
        exported_wav = seg.export(format='wav', parameters=['-ar', '16000', '-ac', '1'])
        seg_exported_wav = AudioSegment.from_wav(exported_wav)

        self.assertWithinTolerance(len(seg_exported_wav),
                                   len(seg),
                                   percentage=0.01)
        self.assertEqual(seg_exported_wav.frame_rate, 16000)
        self.assertEqual(seg_exported_wav.channels, 1)

    def test_export_as_raw(self):
        seg = self.seg1
        exported_raw = seg.export(format='raw')
        seg_exported_raw = AudioSegment.from_raw(exported_raw, sample_width=seg.sample_width, frame_rate=seg.frame_rate,
                                                 channels=seg.channels)

        self.assertWithinTolerance(len(seg_exported_raw),
                                   len(seg),
                                   percentage=0.01)

    def test_export_as_raw_with_codec(self):
        seg = self.seg1
        with self.assertRaises(AttributeError):
            seg.export(format='raw', codec='pcm_s32le')

    def test_export_as_raw_with_parameters(self):
        seg = self.seg1
        with self.assertRaises(AttributeError):
            seg.export(format='raw', parameters=['-ar', '16000', '-ac', '1'])

    def test_export_as_ogg(self):
        seg = self.seg1
        exported_ogg = seg.export(format='ogg')
        seg_exported_ogg = AudioSegment.from_ogg(exported_ogg)

        self.assertWithinTolerance(len(seg_exported_ogg),
                                   len(seg),
                                   percentage=0.01)

    def test_export_forced_codec(self):
        seg = self.seg1 + self.seg2

        with NamedTemporaryFile('w+b', suffix='.ogg') as tmp_file:
            if sys.platform == 'win32':
                tmp_file.close()

            seg.export(tmp_file.name, 'ogg', codec='libvorbis')
            exported = AudioSegment.from_ogg(tmp_file.name)
            self.assertWithinTolerance(len(exported),
                                       len(seg),
                                       percentage=0.01)
            if sys.platform == 'win32':
                os.remove(tmp_file.name)

    def test_fades(self):
        seg = self.seg1[:10000]

        # 1 ms difference in the position of the end of the fade out
        inf_end = seg.fade(start=0, end=float('inf'), to_gain=-120)
        negative_end = seg.fade(start=0, end=-1, to_gain=-120)

        self.assertWithinTolerance(inf_end.rms, negative_end.rms,
                                   percentage=0.001)
        self.assertTrue(negative_end.rms <= inf_end.rms)
        self.assertTrue(inf_end.rms < seg.rms)

        self.assertEqual(len(inf_end), len(seg))

        self.assertTrue(-6 < ratio_to_db(inf_end.rms, seg.rms) < -5)

        # use a slice out of the middle to make sure there is audio
        seg = self.seg2[2000:8000]
        fade_out = seg.fade_out(1000)
        fade_in = seg.fade_in(1000)

        self.assertTrue(0 < fade_out.rms < seg.rms)
        self.assertTrue(0 < fade_in.rms < seg.rms)

        self.assertEqual(len(fade_out), len(seg))
        self.assertEqual(len(fade_in), len(seg))

        db_at_beginning = ratio_to_db(fade_in[:1000].rms, seg[:1000].rms)
        db_at_end = ratio_to_db(fade_in[-1000:].rms, seg[-1000:].rms)
        self.assertTrue(db_at_beginning < db_at_end)

        db_at_beginning = ratio_to_db(fade_out[:1000].rms, seg[:1000].rms)
        db_at_end = ratio_to_db(fade_out[-1000:].rms, seg[-1000:].rms)
        self.assertTrue(db_at_end < db_at_beginning)

    def test_reverse(self):
        seg = self.seg1
        rseg = seg.reverse()

        # the reversed audio should be exactly equal in playback duration
        self.assertEqual(len(seg), len(rseg))

        r2seg = rseg.reverse()

        # if you reverse it twice you should get an identical AudioSegment
        self.assertEqual(seg, r2seg)

    def test_normalize(self):
        seg = self.seg1
        normalized = seg.normalize(0.0)

        self.assertEqual(len(normalized), len(seg))
        self.assertTrue(normalized.rms > seg.rms)
        self.assertWithinTolerance(
            normalized.max,
            normalized.max_possible_amplitude,
            percentage=0.0001
        )

    def test_for_accidental_shortening(self):
        seg = self.mp3_seg_party
        with NamedTemporaryFile('w+b', suffix='.mp3') as tmp_mp3_file:
            if sys.platform == 'win32':
                tmp_mp3_file.close()

            fd = seg.export(tmp_mp3_file.name)
            fd.close()

            for i in range(3):
                fd = AudioSegment.from_mp3(tmp_mp3_file.name).export(tmp_mp3_file.name, "mp3")
                fd.close()

            tmp_seg = AudioSegment.from_mp3(tmp_mp3_file.name)
            self.assertAlmostEqual(len(tmp_seg), len(seg), places=1)

            if sys.platform == 'win32':
                os.remove(tmp_mp3_file.name)

    @unittest.skipUnless('aac' in get_supported_decoders(),
                         "Unsupported codecs")
    def test_formats(self):
        seg_m4a = AudioSegment.from_file(
            os.path.join(data_dir, 'format_test.m4a'), "m4a")
        self.assertTrue(len(seg_m4a))

    def test_equal_and_not_equal(self):
        wav_file = self.seg1.export(format='wav')
        wav = AudioSegment.from_wav(wav_file)
        self.assertTrue(self.seg1 == wav)
        self.assertFalse(self.seg1 != wav)

    def test_duration(self):
        self.assertEqual(int(self.seg1.duration_seconds), 10)

        wav_file = self.seg1.export(format='wav')
        wav = AudioSegment.from_wav(wav_file)
        self.assertEqual(wav.duration_seconds, self.seg1.duration_seconds)

    @unittest.skipUnless('aac' in get_supported_decoders(),
                         "Unsupported codecs")
    def test_autodetect_format(self):
        aac_path = os.path.join(data_dir, 'wrong_extension.aac')
        fn = partial(AudioSegment.from_file, aac_path, "aac")
        self.assertRaises(CouldntDecodeError, fn)

        # Trying to auto detect input file format
        aac_file = AudioSegment.from_file(
            os.path.join(data_dir, 'wrong_extension.aac'))
        self.assertEqual(int(aac_file.duration_seconds), 9)

    def test_export_ogg_as_mp3(self):
        with NamedTemporaryFile('w+b', suffix='.mp3') as tmp_mp3_file:
            AudioSegment.from_file(self.ogg_file_path).export(tmp_mp3_file,
                                                              format="mp3")

    def test_export_mp3_as_ogg(self):
        with NamedTemporaryFile('w+b', suffix='.ogg') as tmp_ogg_file:
            AudioSegment.from_file(self.mp3_file_path).export(tmp_ogg_file,
                                                              format="ogg")

    def test_export_webm_as_mp3(self):
        with NamedTemporaryFile('w+b', suffix='.mp3') as tmp_mp3_file:
            AudioSegment.from_file(
                self.webm_file_path,
                codec="opus"
            ).export(tmp_mp3_file, format="mp3")

    def test_export_mp3_as_webm(self):
        with NamedTemporaryFile('w+b', suffix='.webm') as tmp_webm_file:
            AudioSegment.from_file(self.mp3_file_path).export(tmp_webm_file,
                                                              format="webm")

    @unittest.skipUnless('aac' in get_supported_decoders(),
                         "Unsupported codecs")
    def test_export_mp4_as_ogg(self):
        with NamedTemporaryFile('w+b', suffix='.ogg') as tmp_ogg_file:
            AudioSegment.from_file(self.mp4_file_path).export(tmp_ogg_file,
                                                              format="ogg")

    @unittest.skipUnless('aac' in get_supported_decoders(),
                         "Unsupported codecs")
    def test_export_mp4_as_mp3(self):
        with NamedTemporaryFile('w+b', suffix='.mp3') as tmp_mp3_file:
            AudioSegment.from_file(self.mp4_file_path).export(tmp_mp3_file,
                                                              format="mp3")

    @unittest.skipUnless('aac' in get_supported_decoders(),
                         "Unsupported codecs")
    def test_export_mp4_as_wav(self):
        with NamedTemporaryFile('w+b', suffix='.wav') as tmp_wav_file:
            AudioSegment.from_file(self.mp4_file_path).export(tmp_wav_file,
                                                              format="mp3")

    @unittest.skipUnless('aac' in get_supported_decoders(),
                         "Unsupported codecs")
    def test_export_mp4_as_mp3_with_tags(self):
        with NamedTemporaryFile('w+b', suffix='.mp3') as tmp_mp3_file:
            tags_dict = {
                'title': "The Title You Want",
                'artist': "Artist's name",
                'album': "Name of the Album"
            }
            AudioSegment.from_file(self.mp4_file_path).export(tmp_mp3_file,
                                                              format="mp3",
                                                              tags=tags_dict)

    @unittest.skipUnless('aac' in get_supported_decoders(),
                         "Unsupported codecs")
    def test_export_mp4_as_mp3_with_tags_raises_exception_when_tags_are_not_a_dictionary(self):
        with NamedTemporaryFile('w+b', suffix='.mp3') as tmp_mp3_file:
            json = '{"title": "The Title You Want", "album": "Name of the Album", "artist": "Artist\'s name"}'
            func = partial(
                AudioSegment.from_file(self.mp4_file_path).export, tmp_mp3_file,
                format="mp3", tags=json)
            self.assertRaises(InvalidTag, func)

    @unittest.skipUnless('aac' in get_supported_decoders(),
                         "Unsupported codecs")
    def test_export_mp4_as_mp3_with_tags_raises_exception_when_id3version_is_wrong(self):
        tags = {'artist': 'Artist', 'title': 'Title'}
        with NamedTemporaryFile('w+b', suffix='.mp3') as tmp_mp3_file:
            func = partial(
                AudioSegment.from_file(self.mp4_file_path).export,
                tmp_mp3_file,
                format="mp3",
                tags=tags,
                id3v2_version='BAD VERSION'
            )
            self.assertRaises(InvalidID3TagVersion, func)

    @unittest.skipUnless('aac' in get_supported_decoders(),
                         "Unsupported codecs")
    def test_export_mp3_with_tags(self):
        tags = {'artist': 'Mozart', 'title': 'The Magic Flute'}

        delete = sys.platform != 'win32'

        with NamedTemporaryFile('w+b', suffix='.mp3', delete=delete) as tmp_mp3_file:
            AudioSegment.from_file(self.mp4_file_path).export(tmp_mp3_file, format="mp3", tags=tags)

            if sys.platform == 'win32':
                tmp_mp3_file.close()

            info = mediainfo(filepath=tmp_mp3_file.name)
            info_tags = info["TAG"]

            self.assertEqual(info_tags["artist"], "Mozart")
            self.assertEqual(info_tags["title"], "The Magic Flute")

            if sys.platform == 'win32':
                os.remove(tmp_mp3_file.name)

    def test_mp3_with_jpg_cover_img(self):
        with NamedTemporaryFile('w+b', suffix='.mp3') as tmp_mp3_file:
            outf = self.seg1.export(tmp_mp3_file, format="mp3", cover=self.jpg_cover_path)
            testseg = AudioSegment.from_file(outf, format="mp3")

            # should be within a 150ms and 1.5dB (not perfectly equal due to codecs)
            self.assertWithinTolerance(len(self.seg1), len(testseg), 150)
            self.assertWithinTolerance(self.seg1.dBFS, testseg.dBFS, 1.5)

    def test_mp3_with_png_cover_img(self):
        with NamedTemporaryFile('w+b', suffix='.mp3') as tmp_mp3_file:
            outf = self.seg1.export(tmp_mp3_file, format="mp3", cover=self.png_cover_path)
            testseg = AudioSegment.from_file(outf, format="mp3")

            # should be within a 150ms and 1.5dB (not perfectly equal due to codecs)
            self.assertWithinTolerance(len(self.seg1), len(testseg), 150)
            self.assertWithinTolerance(self.seg1.dBFS, testseg.dBFS, 1.5)

    def test_fade_raises_exception_when_duration_start_end_are_none(self):
        seg = self.seg1
        func = partial(seg.fade, start=1, end=1, duration=1)
        self.assertRaises(TypeError, func)

    def test_silent(self):
        seg = AudioSegment.silent(len(self.seg1))
        self.assertEqual(len(self.seg1), len(seg))
        self.assertEqual(seg.rms, 0)
        self.assertEqual(seg.frame_width, 2)

        seg_8bit = seg.set_sample_width(1)
        self.assertEqual(seg_8bit.sample_width, 1)
        self.assertEqual(seg_8bit.frame_width, 1)
        self.assertEqual(seg_8bit.rms, 0)

        seg *= self.seg1
        self.assertEqual(seg.rms, self.seg1.rms)
        self.assertEqual(len(seg), len(self.seg1))
        self.assertEqual(seg.frame_width, self.seg1.frame_width)
        self.assertEqual(seg.frame_rate, self.seg1.frame_rate)

    def test_from_mono_audiosegments(self):
        monoseg1 = self.seg1.set_channels(1)
        monoseg2 = monoseg1.reverse()
        stereo_sound = AudioSegment.from_mono_audiosegments(monoseg1, monoseg2)

        self.assertEqual(stereo_sound.channels, 2)
        self.assertEqual(stereo_sound.dBFS, monoseg1.dBFS)
        self.assertEqual(len(stereo_sound), len(monoseg1))

    def test_fade_raises_exception_when_duration_is_negative(self):
        seg = self.seg1
        func = partial(seg.fade,
                       to_gain=1,
                       from_gain=1,
                       start=None,
                       end=None,
                       duration=-1)
        self.assertRaises(InvalidDuration, func)

    def test_make_chunks(self):
        seg = self.seg1
        chunks = make_chunks(seg, 100)
        seg2 = chunks[0]
        for chunk in chunks[1:]:
            seg2 += chunk
        self.assertEqual(len(seg), len(seg2))

    def test_empty(self):
        self.assertEqual(len(self.seg1), len(self.seg1 + AudioSegment.empty()))
        self.assertEqual(len(self.seg2), len(self.seg2 + AudioSegment.empty()))
        self.assertEqual(len(self.seg3), len(self.seg3 + AudioSegment.empty()))

    def test_speedup(self):
        speedup_seg = self.seg1.speedup(2.0)

        self.assertWithinTolerance(
            len(self.seg1) / 2, len(speedup_seg), percentage=0.02)

    def test_dBFS(self):
        seg_8bit = self.seg1.set_sample_width(1)
        self.assertWithinTolerance(seg_8bit.dBFS, -18.06, tolerance=1.5)
        self.assertWithinTolerance(self.seg1.dBFS, -17.76, tolerance=1.5)
        self.assertWithinTolerance(self.seg2.dBFS, -20.78, tolerance=1.5)
        self.assertWithinTolerance(self.seg3.dBFS, -12.94, tolerance=1.5)

    def test_compress(self):
        compressed = self.seg1.compress_dynamic_range()
        self.assertWithinTolerance(self.seg1.dBFS - compressed.dBFS,
                                   10.0,
                                   tolerance=10.0)

        # Highest peak should be lower
        self.assertTrue(compressed.max < self.seg1.max)

        # average volume should be reduced
        self.assertTrue(compressed.rms < self.seg1.rms)

    @unittest.skipUnless('aac' in get_supported_decoders(),
                         "Unsupported codecs")
    def test_exporting_to_ogg_uses_default_codec_when_codec_param_is_none(self):
        delete = sys.platform != 'win32'

        with NamedTemporaryFile('w+b', suffix='.ogg', delete=delete) as tmp_ogg_file:
            AudioSegment.from_file(self.mp4_file_path).export(tmp_ogg_file, format="ogg")

            if sys.platform == 'win32':
                tmp_ogg_file.close()

            info = mediainfo(filepath=tmp_ogg_file.name)

            if sys.platform == 'win32':
                os.remove(tmp_ogg_file.name)

        self.assertEqual(info["codec_name"], "vorbis")
        self.assertEqual(info["format_name"], "ogg")

    def test_zero_length_segment(self):
        self.assertEqual(0, len(self.seg1[0:0]))

    def test_invert(self):
        s_mono = Sine(100).to_audio_segment()
        s = s_mono.set_channels(2)

        try:
            s_mono.invert_phase(channels=(1, 0))
        except Exception:
            pass
        else:
            raise Exception("AudioSegment.invert_phase() didn't catch a bad input (mono)")

        s_inv = s.invert_phase()
        self.assertFalse(s == s_inv)
        self.assertTrue(s.rms == s_inv.rms)
        self.assertTrue(s == s_inv.invert_phase())

        s_inv_right = s.invert_phase(channels=(0, 1))
        left, right = s_inv_right.split_to_mono()

        self.assertFalse(s_mono == s_inv_right)
        self.assertFalse(s_inv == s_inv_right)
        self.assertTrue(left == s_mono)
        self.assertFalse(right == s_mono)

        s_inv_left = s.invert_phase(channels=(1, 0))
        left, right = s_inv_left.split_to_mono()

        self.assertFalse(s_mono == s_inv_left)
        self.assertFalse(s_inv == s_inv_left)
        self.assertFalse(left == s_mono)
        self.assertTrue(right == s_mono)

    def test_max_dBFS(self):
        sine_0_dbfs = Sine(1000).to_audio_segment()
        sine_minus_3_dbfs = Sine(1000).to_audio_segment(volume=-3.0)
        self.assertAlmostEqual(-0.0, sine_0_dbfs.max_dBFS, 2)
        self.assertAlmostEqual(-3.0, sine_minus_3_dbfs.max_dBFS, 2)

    def test_array_type(self):
        self.assertEqual(self.seg1.array_type, "h")
        self.assertEqual(self.seg2.array_type, "h")
        self.assertEqual(self.seg3.array_type, "h")
        self.assertEqual(self.mp3_seg_party.array_type, "h")

        silence = AudioSegment.silent(50)
        self.assertEqual(silence.array_type, "h")
        self.assertEqual(silence.set_sample_width(1).array_type, "b")
        self.assertEqual(silence.set_sample_width(4).array_type, "i")

    def test_sample_array(self):
        samples = Sine(450).to_audio_segment().get_array_of_samples()
        self.assertEqual(
            list(samples[:8]),
            [0, 2099, 4190, 6263, 8311, 10325, 12296, 14217]
        )

    def test_get_dc_offset(self):
        seg = self.seg_dc_offset
        self.assertWithinTolerance(seg.get_dc_offset(), -0.16, tolerance=0.01)
        self.assertWithinTolerance(seg.get_dc_offset(1), -0.16, tolerance=0.01)
        self.assertWithinTolerance(seg.get_dc_offset(2), 0.1, tolerance=0.01)

    def test_remove_dc_offset(self):
        seg = self.seg_dc_offset

        seg1 = seg.remove_dc_offset()
        self.assertWithinTolerance(seg1.get_dc_offset(1), 0.0, tolerance=0.0001)
        self.assertWithinTolerance(seg1.get_dc_offset(2), 0.0, tolerance=0.0001)

        seg1 = seg.remove_dc_offset(1)
        self.assertWithinTolerance(seg1.get_dc_offset(1), 0.0, tolerance=0.0001)
        self.assertWithinTolerance(seg1.get_dc_offset(2), 0.1, tolerance=0.01)

        seg1 = seg.remove_dc_offset(2)
        self.assertWithinTolerance(seg1.get_dc_offset(1), -0.16, tolerance=0.01)
        self.assertWithinTolerance(seg1.get_dc_offset(2), 0.0, tolerance=0.0001)

        seg1 = seg.remove_dc_offset(channel=1, offset=(-0.06))
        self.assertWithinTolerance(seg1.get_dc_offset(1), -0.1, tolerance=0.01)

    def test_from_file_clean_fail(self):
        delete = sys.platform != 'win32'
        orig_tmpdir = gettempdir()
        new_tmpdir = mkdtemp()
        tempfile.tempdir = new_tmpdir

        with NamedTemporaryFile('w+b', suffix='.wav', delete=delete) as tmp_wav_file:
            tmp_wav_file.write("not really a wav".encode('utf-8'))
            tmp_wav_file.flush()
            self.assertRaises(CouldntDecodeError, AudioSegment.from_file, tmp_wav_file.name)
            files = os.listdir(tempfile.tempdir)
            self.assertEquals(files, [os.path.basename(tmp_wav_file.name)])

        if sys.platform == 'win32':
            os.remove(tmp_wav_file.name)

        tempfile.tempdir = orig_tmpdir
        os.rmdir(new_tmpdir)


class SilenceTests(unittest.TestCase):

    def setUp(self):
        global test1wav, test4wav
        if not test1wav:
            test1wav = AudioSegment.from_wav(os.path.join(data_dir, 'test1.wav'))
        if not test4wav:
            test4wav = AudioSegment.from_wav(os.path.join(data_dir, 'test4.wav'))

        self.seg1 = test1wav
        self.seg4 = test4wav

    def test_split_on_silence_complete_silence(self):
        seg = AudioSegment.silent(5000)
        self.assertEquals( split_on_silence(seg), [] )

    def test_split_on_silence_test1(self):
        self.assertEqual(
            len(split_on_silence(self.seg1, min_silence_len=500, silence_thresh=-20)),
            3
        )
    def test_split_on_silence_no_silence(self):
        splits = split_on_silence(self.seg1, min_silence_len=5000, silence_thresh=-200, keep_silence=True)
        lens = [len(split) for split in splits]
        self.assertEqual( lens, [len(self.seg1)] )

    def test_detect_completely_silent_segment(self):
        seg = AudioSegment.silent(5000)
        silent_ranges = detect_silence(seg, min_silence_len=1000, silence_thresh=-20)
        self.assertEqual(silent_ranges, [[0, 5000]])

    def test_detect_tight_silent_segment(self):
        seg = AudioSegment.silent(1000)
        silent_ranges = detect_silence(seg, min_silence_len=1000, silence_thresh=-20)
        self.assertEqual(silent_ranges, [[0, 1000]])

    def test_detect_too_long_silence(self):
        seg = AudioSegment.silent(3000)
        silent_ranges = detect_silence(seg, min_silence_len=5000, silence_thresh=-20)
        self.assertEqual(silent_ranges, [])

    def test_detect_silence_seg1(self):
        silent_ranges = detect_silence(self.seg1, min_silence_len=500, silence_thresh=-20)
        self.assertEqual(silent_ranges, [[0, 775], [3141, 4033], [5516, 6051]])

    def test_detect_silence_seg1_with_seek_split(self):
        silent_ranges = detect_silence(self.seg1, min_silence_len=500, silence_thresh=-20,
                                       seek_step=10)
        self.assertEqual(silent_ranges, [[0, 770], [3150, 4030], [5520, 6050]])

    def test_realistic_audio(self):
        silent_ranges = detect_silence(self.seg4, min_silence_len=1000, silence_thresh=self.seg4.dBFS)

        prev_end = -1
        for start, end in silent_ranges:
            self.assertTrue(start > prev_end)
            prev_end = end


class GeneratorTests(unittest.TestCase):

    def test_with_smoke(self):
        Sine(440).to_audio_segment()
        Square(440).to_audio_segment()
        Triangle(440).to_audio_segment()

        Pulse(440, duty_cycle=0.75).to_audio_segment()
        Sawtooth(440, duty_cycle=0.75).to_audio_segment()

        WhiteNoise().to_audio_segment()

    def test_loudness(self):
        sine_dbfs = Sine(440).to_audio_segment().dBFS
        square_dbfs = Square(440).to_audio_segment().dBFS
        white_noise_dbfs = WhiteNoise().to_audio_segment().dBFS

        self.assertAlmostEqual(sine_dbfs, -3.0, places=1)
        self.assertAlmostEqual(square_dbfs, 0.0, places=1)
        self.assertAlmostEqual(white_noise_dbfs, -5, places=0)

    def test_duration(self):
        one_sec = Sine(440).to_audio_segment(duration=1000)
        five_sec = Sine(440).to_audio_segment(duration=5000)
        half_sec = Sine(440).to_audio_segment(duration=500)

        self.assertAlmostEqual(len(one_sec), 1000)
        self.assertAlmostEqual(len(five_sec), 5000)
        self.assertAlmostEqual(len(half_sec), 500)


class NoConverterTests(unittest.TestCase):

    def setUp(self):
        self.wave_file = os.path.join(data_dir, 'test1.wav')
        self.wave24_file = os.path.join(data_dir, 'test1-24bit.wav')
        self.wave_empty = os.path.join(data_dir, 'test1_empty.wav')
        self.mp3_file = os.path.join(data_dir, 'test1.mp3')
        self.raw_file = os.path.join(data_dir, 'test1.raw')
        AudioSegment.converter = "definitely-not-a-path-to-anything-asdjklqwop"

    def tearDown(self):
        AudioSegment.converter = get_encoder_name()

    def test_opening_wav_file(self):
        seg = AudioSegment.from_wav(self.wave_file)
        self.assertTrue(len(seg) > 1000)

        seg = AudioSegment.from_file(self.wave_file)
        self.assertTrue(len(seg) > 1000)

        seg = AudioSegment.from_file(self.wave_file, "wav")
        self.assertTrue(len(seg) > 1000)

        seg = AudioSegment.from_file(self.wave_file, format="wav")
        self.assertTrue(len(seg) > 1000)

    def test_opening_wav24_file(self):
        seg = AudioSegment.from_wav(self.wave24_file)
        self.assertTrue(len(seg) > 1000)

        seg = AudioSegment.from_file(self.wave24_file)
        self.assertTrue(len(seg) > 1000)

        seg = AudioSegment.from_file(self.wave24_file, "wav")
        self.assertTrue(len(seg) > 1000)

        seg = AudioSegment.from_file(self.wave24_file, format="wav")
        self.assertTrue(len(seg) > 1000)

    def test_opening_raw_file(self):
        seg = AudioSegment.from_raw(self.raw_file, sample_width=2, frame_rate=32000, channels=2)
        self.assertTrue(len(seg) > 1000)

        seg = AudioSegment.from_file(self.raw_file, "raw", sample_width=2, frame_rate=32000, channels=2)
        self.assertTrue(len(seg) > 1000)

        seg = AudioSegment.from_file(self.raw_file, format="raw", sample_width=2, frame_rate=32000, channels=2)
        self.assertTrue(len(seg) > 1000)

    def test_opening_raw_file_with_missing_args_fails(self):
        func = partial(AudioSegment.from_raw, self.raw_file)
        self.assertRaises(KeyError, func)

    def test_opening_mp3_file_fails(self):
        func = partial(AudioSegment.from_mp3, self.mp3_file)
        self.assertRaises(OSError, func)

        func = partial(AudioSegment.from_file, self.mp3_file)
        self.assertRaises(OSError, func)

        func = partial(AudioSegment.from_file, self.mp3_file, "mp3")
        self.assertRaises(OSError, func)

        func = partial(AudioSegment.from_file, self.mp3_file, format="mp3")
        self.assertRaises(OSError, func)

    def test_init_AudioSegment_data_buffer(self):
        seg = AudioSegment(data="\0" * 34, sample_width=2, frame_rate=4, channels=1)

        self.assertEqual(seg.duration_seconds, 4.25)

        self.assertEqual(seg.sample_width, 2)

        self.assertEqual(seg.frame_rate, 4)

    def test_init_AudioSegment_data_buffer_with_missing_args_fails(self):
        func = partial(AudioSegment, data="\0" * 16, sample_width=2, frame_rate=2)
        self.assertRaises(MissingAudioParameter, func)

        func = partial(AudioSegment, data="\0" * 16, sample_width=2, channels=1)
        self.assertRaises(MissingAudioParameter, func)

        func = partial(AudioSegment, data="\0" * 16, frame_rate=2, channels=1)
        self.assertRaises(MissingAudioParameter, func)

    def test_init_AudioSegment_data_buffer_with_bad_values_fails(self):
        func = partial(AudioSegment, data="\0" * 14, sample_width=4, frame_rate=2, channels=1)
        self.assertRaises(ValueError, func)

    def test_exporting(self):
        seg = AudioSegment.from_wav(self.wave_file)
        exported = AudioSegment.from_wav(seg.export(format="wav"))

        self.assertEqual(len(exported), len(seg))

    def test_opening_empty_wav_file(self):
        seg = AudioSegment.from_wav(self.wave_empty)
        self.assertTrue(len(seg) == 0)

        seg = AudioSegment.from_file(self.wave_empty)
        self.assertTrue(len(seg) == 0)

        seg = AudioSegment.from_file(self.wave_empty, "wav")
        self.assertTrue(len(seg) == 0)

        seg = AudioSegment.from_file(self.wave_empty, format="wav")
        self.assertTrue(len(seg) == 0)


class FilterTests(unittest.TestCase):

    def setUp(self):
        global test1wav
        if not test1wav:
            test1wav = AudioSegment.from_wav(os.path.join(data_dir, 'test1.wav'))

        self.seg1 = test1wav

    def test_highpass_works_on_multichannel_segments(self):
        self.assertEqual(self.seg1.channels, 2)
        less_bass = self.seg1.high_pass_filter(800)
        self.assertTrue(less_bass.dBFS < self.seg1.dBFS)

    def test_highpass_filter_reduces_loudness(self):
        s = Square(200).to_audio_segment()
        less_bass = s.high_pass_filter(400)
        self.assertTrue(less_bass.dBFS < s.dBFS)

    def test_highpass_filter_cutoff_frequency(self):
        # A Sine wave should not be affected by a HPF 3 octaves lower
        s = Sine(800).to_audio_segment()
        less_bass = s.high_pass_filter(100)
        self.assertAlmostEqual(less_bass.dBFS, s.dBFS, places=0)

    def test_lowpass_filter_reduces_loudness(self):
        s = Square(200).to_audio_segment()
        less_treble = s.low_pass_filter(400)
        self.assertTrue(less_treble.dBFS < s.dBFS)

    def test_lowpass_filter_cutoff_frequency(self):
        # A Sine wave should not be affected by a LPF 3 octaves Higher
        s = Sine(100).to_audio_segment()
        less_treble = s.low_pass_filter(800)
        self.assertAlmostEqual(less_treble.dBFS, s.dBFS, places=0)


class PartialAudioSegmentLoadTests(unittest.TestCase):

    def setUp(self):
        self.wave_path_str = os.path.join(data_dir, 'test1.wav')
        self.mp3_path_str = os.path.join(data_dir, 'test1.mp3')
        self.raw_path_str = os.path.join(data_dir, 'test1.raw')

    def tearDown(self):
        AudioSegment.converter = get_encoder_name()

    def test_partial_load_duration_equals_cropped_mp3_audio_segment(self):
        partial_seg1 = AudioSegment.from_file(self.mp3_path_str)[:1000]
        partial_seg2 = AudioSegment.from_file(self.mp3_path_str, duration=1.)
        self.assertEqual(len(partial_seg1), len(partial_seg2))
        self.assertEqual(partial_seg1._data, partial_seg2._data)

    def test_partial_load_start_second_equals_cropped_mp3_audio_segment(self):
        partial_seg1 = AudioSegment.from_file(self.mp3_path_str)[1000:]
        partial_seg2 = AudioSegment.from_file(self.mp3_path_str, start_second=1.)[0:]
        self.assertEqual(len(partial_seg1), len(partial_seg2))
        self.assertEqual(partial_seg1._data, partial_seg2._data)

    def test_partial_load_start_second_and_duration_equals_cropped_mp3_audio_segment(self):
        partial_seg1 = AudioSegment.from_file(self.mp3_path_str)[1000:2000]
        partial_seg2 = AudioSegment.from_file(self.mp3_path_str, start_second=1., duration=1.)
        self.assertEqual(len(partial_seg1), len(partial_seg2))
        self.assertEqual(partial_seg1._data, partial_seg2._data)

    def test_partial_load_duration_equals_cropped_wav_audio_segment(self):
        partial_seg1 = AudioSegment.from_file(self.wave_path_str)[:1000]
        partial_seg2 = AudioSegment.from_file(self.wave_path_str, duration=1.)
        self.assertEqual(len(partial_seg1), len(partial_seg2))
        self.assertEqual(partial_seg1._data, partial_seg2._data)

    def test_partial_load_start_second_equals_cropped_wav_audio_segment(self):
        partial_seg1 = AudioSegment.from_file(self.wave_path_str)[1000:]
        partial_seg2 = AudioSegment.from_file(self.wave_path_str, start_second=1.)[0:]
        self.assertEqual(len(partial_seg1), len(partial_seg2))
        self.assertEqual(partial_seg1._data, partial_seg2._data)

    def test_partial_load_start_second_and_duration_equals_cropped_wav_audio_segment(self):
        partial_seg1 = AudioSegment.from_file(self.wave_path_str)[1000:2000]
        partial_seg2 = AudioSegment.from_file(self.wave_path_str, start_second=1., duration=1.)
        self.assertEqual(len(partial_seg1), len(partial_seg2))
        self.assertEqual(partial_seg1._data, partial_seg2._data)

    def test_partial_load_duration_equals_cropped_raw_audio_segment(self):
        partial_seg1 = AudioSegment.from_file(self.raw_path_str, format="raw", sample_width=2, frame_rate=32000, channels=2)[:1000]
        partial_seg2 = AudioSegment.from_file(self.raw_path_str, format="raw", sample_width=2, frame_rate=32000, channels=2, duration=1.)
        self.assertEqual(len(partial_seg1), len(partial_seg2))
        self.assertEqual(partial_seg1._data, partial_seg2._data)

    def test_partial_load_start_second_equals_cropped_raw_audio_segment(self):
        partial_seg1 = AudioSegment.from_file(self.raw_path_str, format="raw", sample_width=2, frame_rate=32000, channels=2)[1000:]
        partial_seg2 = AudioSegment.from_file(self.raw_path_str, format="raw", sample_width=2, frame_rate=32000, channels=2, start_second=1.)[0:]
        self.assertEqual(len(partial_seg1), len(partial_seg2))
        self.assertEqual(partial_seg1._data, partial_seg2._data)

    def test_partial_load_start_second_and_duration_equals_cropped_raw_audio_segment(self):
        partial_seg1 = AudioSegment.from_file(self.raw_path_str, format="raw", sample_width=2, frame_rate=32000, channels=2)[1000:2000]
        partial_seg2 = AudioSegment.from_file(self.raw_path_str, format="raw", sample_width=2, frame_rate=32000, channels=2, start_second=1., duration=1.)
        self.assertEqual(len(partial_seg1), len(partial_seg2))
        self.assertEqual(partial_seg1._data, partial_seg2._data)


class CrossfadeCurveTests(unittest.TestCase):
    """
    Tests for the crossfade curves of AudioSegment.append(). Synthetic
    frame-aligned PCM segments are built directly so no decoder is needed and
    every sample in the crossfade region can be checked exactly.
    """

    frame_rate = 1000  # 1 frame per millisecond, so ms == frame indices
    segment_frames = 30

    def _segment_from_frames(self, frames, channels=1, sample_width=2):
        """
        Build an AudioSegment from a list of per-frame sample lists, e.g.
        [[1, 2], [3, 4]] for a two-frame stereo segment.
        """
        samples = [sample for frame in frames for sample in frame]
        data = array(get_array_type(sample_width * 8), samples).tobytes()
        return AudioSegment(
            data,
            metadata={
                "channels": channels,
                "sample_width": sample_width,
                "frame_rate": self.frame_rate,
                "frame_width": channels * sample_width,
            }
        )

    def _constant_segment(self, value, frames=None, channels=1, sample_width=2):
        frames = frames or self.segment_frames
        return self._segment_from_frames(
            [[value] * channels for _ in range(frames)],
            channels=channels,
            sample_width=sample_width,
        )

    def _sample_at(self, seg, frame_index, channel=0):
        frame = seg.get_frame(frame_index)
        return audioop.getsample(frame, seg.sample_width, channel)

    def assertWithinTolerance(self, val, expected, tolerance=None,
                              percentage=None):
        if percentage is not None:
            tolerance = expected * percentage
        self.assertTrue(
            abs(val - expected) <= tolerance,
            "%s is not within %s of %s" % (val, tolerance, expected)
        )

    def test_default_curve_is_legacy_linear(self):
        # Omitting the curve must produce byte-identical output to the
        # original linear implementation (regression guard).
        seg1 = self._constant_segment(4000)
        seg2 = self._constant_segment(-4000)
        crossfade = 8

        merged_default = seg1.append(seg2, crossfade=crossfade)
        merged_linear = seg1.append(seg2, crossfade=crossfade,
                                    crossfade_curve=CROSSFADE_LINEAR)

        legacy_xf = seg1[-crossfade:].fade(to_gain=-120, start=0, end=float('inf'))
        legacy_xf *= seg2[:crossfade].fade(from_gain=-120, start=0, end=float('inf'))
        legacy = seg1._spawn(
            seg1[:-crossfade]._data + legacy_xf._data + seg2[crossfade:]._data
        )

        self.assertEqual(merged_default._data, legacy._data)
        self.assertEqual(merged_linear._data, legacy._data)
        self.assertEqual(len(merged_default),
                         len(seg1) + len(seg2) - crossfade)

    def test_equal_power_preserves_inputs_and_metadata(self):
        seg1 = self._constant_segment(4000)
        seg2 = self._constant_segment(-4000)
        seg1_before, seg2_before = seg1._data, seg2._data

        merged = seg1.append(seg2, crossfade=8,
                             crossfade_curve=CROSSFADE_EQUAL_POWER)

        # the original AudioSegments are immutable
        self.assertEqual(seg1._data, seg1_before)
        self.assertEqual(seg2._data, seg2_before)

        # channels and sample rate metadata are carried over
        self.assertEqual(merged.channels, seg1.channels)
        self.assertEqual(merged.frame_rate, seg1.frame_rate)
        self.assertEqual(merged.sample_width, seg1.sample_width)
        self.assertEqual(len(merged), len(seg1) + len(seg2) - 8)

    def test_equal_power_endpoint_frames_are_exact_sources(self):
        seg1 = self._constant_segment(5000)
        seg2 = self._constant_segment(-5000)
        crossfade = 9
        n = crossfade  # frames, frame_rate == 1000

        merged = seg1.append(seg2, crossfade=crossfade,
                             crossfade_curve=CROSSFADE_EQUAL_POWER)

        # untouched material outside the crossfade region passes through
        self.assertEqual(merged[:len(seg1) - n]._data, seg1[:-n]._data)
        self.assertEqual(merged[len(seg1):]._data, seg2[n:]._data)

        xf_start = len(seg1) - n
        # first crossfade frame: cos(0) == 1, sin(0) == 0 -> pure seg1 frame
        self.assertEqual(self._sample_at(merged, xf_start), 5000)
        # last crossfade frame: cos(pi/2) == 0, sin(pi/2) == 1
        # -> pure seg2 frame (exact, not an approximation)
        self.assertEqual(self._sample_at(merged, xf_start + n - 1), -5000)

    def test_constant_amplitude_midpoint_gains(self):
        # Two identical constant-amplitude (fully coherent) segments.
        # Linear ramps sum to 1 at every frame, so the amplitude is kept
        # exactly; the equal-power ramps reach sqrt(2) at the midpoint
        # (the known 3dB coherence boost of an equal-power crossfade).
        amplitude = 2000
        seg1 = self._constant_segment(amplitude)
        seg2 = self._constant_segment(amplitude)
        crossfade = 9  # odd number of frames -> exact midpoint frame

        merged_eq = seg1.append(seg2, crossfade=crossfade,
                                crossfade_curve=CROSSFADE_EQUAL_POWER)
        merged_lin = seg1.append(seg2, crossfade=crossfade,
                                 crossfade_curve=CROSSFADE_LINEAR)

        xf_start = len(seg1) - crossfade
        mid = xf_start + crossfade // 2

        # equal power: cos(pi/4) + sin(pi/4) == sqrt(2)
        self.assertAlmostEqual(
            self._sample_at(merged_eq, mid),
            amplitude * math.sqrt(2),
            delta=1
        )
        # linear: (1 - i/n) + (i/n) == 1, frame for frame
        self.assertAlmostEqual(
            self._sample_at(merged_lin, mid),
            amplitude,
            delta=1
        )
        for frame_index in range(xf_start, xf_start + crossfade):
            self.assertAlmostEqual(self._sample_at(merged_lin, frame_index),
                                   amplitude, delta=1)
        # both curves keep the boundary frames at the original amplitude
        self.assertEqual(self._sample_at(merged_eq, xf_start), amplitude)
        self.assertEqual(self._sample_at(merged_eq, xf_start + crossfade - 1),
                         amplitude)

    def test_quadrature_signal_midpoint_and_region_energy(self):
        # A sine/cosine pair over whole periods: the two signals are
        # sample-wise orthogonal in aggregate (mean(x1 * x2) == 0), like two
        # independent environmental sounds. Equal-power ramps keep their
        # combined power flat; linear ramps let it sag ~3dB at the middle.
        n = self.segment_frames
        amp = 10000
        periods = 3
        frames1, frames2 = [], []
        for i in range(n):
            phase = 2 * math.pi * periods * i / n
            frames1.append([int(round(amp * math.cos(phase)))])
            frames2.append([int(round(amp * math.sin(phase)))])
        seg1 = self._segment_from_frames(frames1)
        seg2 = self._segment_from_frames(frames2)
        crossfade = n

        merged_eq = seg1.append(seg2, crossfade=crossfade,
                                crossfade_curve=CROSSFADE_EQUAL_POWER)
        merged_lin = seg1.append(seg2, crossfade=crossfade,
                                 crossfade_curve=CROSSFADE_LINEAR)

        def rms(seg):
            samples = array(seg.array_type, seg._data)
            return math.sqrt(sum(s * s for s in samples) / len(samples))

        # floating-point reference implementation of both ramps, so the
        # expected energy includes weighted cross terms exactly
        lin_ref = [
            frames1[i][0] * (1 - i / n) + frames2[i][0] * (i / n)
            for i in range(n)
        ]
        eq_ref = [
            frames1[i][0] * math.cos((math.pi / 2) * i / (n - 1))
            + frames2[i][0] * math.sin((math.pi / 2) * i / (n - 1))
            for i in range(n)
        ]
        rms_lin_ref = math.sqrt(sum(s * s for s in lin_ref) / n)
        rms_eq_ref = math.sqrt(sum(s * s for s in eq_ref) / n)

        input_rms = rms(seg1)  # amp / sqrt(2) over whole periods
        # equal power: region energy is preserved (cos^2 + sin^2 == 1);
        # quantization is the only source of error against the float reference
        self.assertWithinTolerance(rms(merged_eq), rms_eq_ref,
                                   tolerance=rms_eq_ref * 0.01)
        self.assertWithinTolerance(rms(merged_eq), input_rms,
                                   tolerance=input_rms * 0.03)
        # linear: uncorrelated power sags in the middle (~1.5-3dB dip),
        # strictly quieter than the equal-power crossfade
        self.assertWithinTolerance(rms(merged_lin), rms_lin_ref,
                                   tolerance=rms_lin_ref * 0.01)
        self.assertLess(rms(merged_lin), rms(merged_eq))
        self.assertLess(rms(merged_lin), input_rms)

        # exact midpoint frame for both curves, computed from the actual
        # source samples and their per-frame ramp gains
        mid = crossfade // 2
        x1 = frames1[mid][0]
        x2 = frames2[mid][0]
        mid_angle = (math.pi / 2) * mid / (crossfade - 1)
        expected_eq = int(round(
            x1 * math.cos(mid_angle) + x2 * math.sin(mid_angle)
        ))
        self.assertAlmostEqual(self._sample_at(merged_eq, mid),
                               expected_eq, delta=2)
        expected_lin = int(round(x1 * (1 - mid / n) + x2 * (mid / n)))
        self.assertAlmostEqual(self._sample_at(merged_lin, mid),
                               expected_lin, delta=2)

    def test_equal_power_stereo_and_all_pcm_widths(self):
        for sample_width in (1, 2, 4):
            # stay inside the signed range of every supported PCM width
            # (8-bit pydub samples span -128..127)
            amplitude = 60 if sample_width == 1 else 6000
            seg1 = self._constant_segment(amplitude, channels=2,
                                          sample_width=sample_width)
            seg2 = self._constant_segment(-amplitude, channels=2,
                                          sample_width=sample_width)
            crossfade = 9

            merged = seg1.append(seg2, crossfade=crossfade,
                                 crossfade_curve=CROSSFADE_EQUAL_POWER)

            self.assertEqual(merged.channels, 2)
            self.assertEqual(merged.sample_width, sample_width)
            self.assertEqual(merged.frame_rate, self.frame_rate)
            self.assertEqual(len(merged),
                             len(seg1) + len(seg2) - crossfade)

            xf_start = len(seg1) - crossfade
            first_frame = merged.get_frame(xf_start)
            last_frame = merged.get_frame(xf_start + crossfade - 1)
            for channel in range(2):
                self.assertEqual(audioop.getsample(first_frame,
                                                   sample_width, channel),
                                 amplitude)
                self.assertEqual(audioop.getsample(last_frame,
                                                   sample_width, channel),
                                 -amplitude)

    def test_equal_power_hard_clips_peaks_to_pcm_range(self):
        # Two full-scale segments overlap: no headroom exists in integer PCM,
        # so the mix-down saturates at the sample-width limit instead of
        # wrapping around (same semantics as overlay()/the linear curve).
        max16 = 32767
        seg1 = self._constant_segment(max16)
        seg2 = self._constant_segment(max16)

        merged = seg1.append(seg2, crossfade=9,
                             crossfade_curve=CROSSFADE_EQUAL_POWER)

        # at the midpoint both ramps sit above 0.7, so the sum would exceed
        # the 16-bit ceiling; it must be clipped, never wrap to a negative int
        mid = len(seg1) - 9 + 4
        self.assertEqual(self._sample_at(merged, mid), max16)
        self.assertTrue(all(
            -max16 - 1 <= self._sample_at(merged, i) <= max16
            for i in range(len(merged))
        ))

    def test_zero_crossfade_concatenates_for_every_curve(self):
        seg1 = self._constant_segment(1000, frames=10)
        seg2 = self._constant_segment(-1000, frames=10)
        for curve in CROSSFADE_CURVES:
            merged = seg1.append(seg2, crossfade=0, crossfade_curve=curve)
            self.assertEqual(merged._data, seg1._data + seg2._data)
            self.assertEqual(len(merged), len(seg1) + len(seg2))

    def test_single_frame_crossfade_does_not_crash(self):
        # a 1ms crossfade covers exactly one frame; the ramps must degenerate
        # without raising (and without producing or dropping half a frame)
        seg1 = self._constant_segment(3000, frames=5)
        seg2 = self._constant_segment(-3000, frames=5)

        for curve in CROSSFADE_CURVES:
            merged = seg1.append(seg2, crossfade=1, crossfade_curve=curve)
            self.assertEqual(len(merged), len(seg1) + len(seg2) - 1)
            self.assertEqual(
                len(merged._data),
                (len(seg1) + len(seg2) - 1) * merged.frame_width
            )

    def test_negative_crossfade_raises(self):
        seg1 = self._constant_segment(1000, frames=10)
        seg2 = self._constant_segment(1000, frames=10)
        for curve in CROSSFADE_CURVES:
            self.assertRaises(
                ValueError,
                lambda: seg1.append(seg2, crossfade=-5, crossfade_curve=curve)
            )

    def test_crossfade_longer_than_either_segment_raises(self):
        seg1 = self._constant_segment(1000, frames=10)
        seg2 = self._constant_segment(1000, frames=6)
        for curve in CROSSFADE_CURVES:
            self.assertRaises(
                ValueError,
                lambda: seg1.append(seg2, crossfade=11, crossfade_curve=curve)
            )
            self.assertRaises(
                ValueError,
                lambda: seg2.append(seg1, crossfade=7, crossfade_curve=curve)
            )

    def test_unknown_curve_raises(self):
        seg1 = self._constant_segment(1000, frames=10)
        seg2 = self._constant_segment(1000, frames=10)
        self.assertRaises(
            ValueError,
            lambda: seg1.append(seg2, crossfade=4, crossfade_curve="logarithmic")
        )

    def test_unalignable_crossfade_raises_without_half_frames(self):
        # at 333Hz, 1ms is worth 0.333 frames: the crossfade cannot be split
        # into an integer number of frames, so it must fail loudly. Build the
        # segments with that frame rate directly (set_frame_rate would
        # resample and change the frame count).
        seg1 = self._segment_from_frames(
            [[1000] for _ in range(10)]
        )
        seg1 = AudioSegment(
            seg1._data,
            metadata={
                "channels": 1,
                "sample_width": 2,
                "frame_rate": 333,
                "frame_width": 2,
            }
        )
        seg2 = AudioSegment(
            seg1._data,
            metadata={
                "channels": 1,
                "sample_width": 2,
                "frame_rate": 333,
                "frame_width": 2,
            }
        )
        seg1_before, seg2_before = seg1._data, seg2._data

        for curve in CROSSFADE_CURVES:
            self.assertRaises(
                ValueError,
                lambda: seg1.append(seg2, crossfade=1, crossfade_curve=curve)
            )

        # failed alignment leaves no partial output and inputs untouched
        self.assertEqual(seg1._data, seg1_before)
        self.assertEqual(seg2._data, seg2_before)


if __name__ == "__main__":
    import sys

    if sys.version_info >= (3, 1):
        unittest.main(warnings="ignore")
    else:
        unittest.main()
