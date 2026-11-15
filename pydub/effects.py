import sys
import math
import array
import numbers
from .utils import (
    db_to_float,
    ratio_to_db,
    register_pydub_effect,
    make_chunks,
    audioop,
    get_min_max_value
)
from .silence import split_on_silence
from .exceptions import TooManyMissingFrames, InvalidDuration

if sys.version_info >= (3, 0):
    xrange = range
    basestring = str

try:
    _isfinite = math.isfinite
except AttributeError:  # python 2.7 has no math.isfinite
    def _isfinite(value):
        return (not math.isinf(value)) and (not math.isnan(value))


@register_pydub_effect
def apply_mono_filter_to_each_channel(seg, filter_fn):
    n_channels = seg.channels

    channel_segs = seg.split_to_mono()
    channel_segs = [filter_fn(channel_seg) for channel_seg in channel_segs]

    out_data = seg.get_array_of_samples()
    for channel_i, channel_seg in enumerate(channel_segs):
        for sample_i, sample in enumerate(channel_seg.get_array_of_samples()):
            index = (sample_i * n_channels) + channel_i
            out_data[index] = sample

    return seg._spawn(out_data)


@register_pydub_effect
def normalize(seg, headroom=0.1):
    """
    headroom is how close to the maximum volume to boost the signal up to (specified in dB)
    """
    peak_sample_val = seg.max
    
    # if the max is 0, this audio segment is silent, and can't be normalized
    if peak_sample_val == 0:
        return seg
    
    target_peak = seg.max_possible_amplitude * db_to_float(-headroom)

    needed_boost = ratio_to_db(target_peak / peak_sample_val)
    return seg.apply_gain(needed_boost)


@register_pydub_effect
def speedup(seg, playback_speed=1.5, chunk_size=150, crossfade=25):
    # we will keep audio in 150ms chunks since one waveform at 20Hz is 50ms long
    # (20 Hz is the lowest frequency audible to humans)

    # portion of AUDIO TO KEEP. if playback speed is 1.25 we keep 80% (0.8) and
    # discard 20% (0.2)
    atk = 1.0 / playback_speed

    if playback_speed < 2.0:
        # throwing out more than half the audio - keep 50ms chunks
        ms_to_remove_per_chunk = int(chunk_size * (1 - atk) / atk)
    else:
        # throwing out less than half the audio - throw out 50ms chunks
        ms_to_remove_per_chunk = int(chunk_size)
        chunk_size = int(atk * chunk_size / (1 - atk))

    # the crossfade cannot be longer than the amount of audio we're removing
    crossfade = min(crossfade, ms_to_remove_per_chunk - 1)

    # DEBUG
    #print("chunk: {0}, rm: {1}".format(chunk_size, ms_to_remove_per_chunk))

    chunks = make_chunks(seg, chunk_size + ms_to_remove_per_chunk)
    if len(chunks) < 2:
        raise Exception("Could not speed up AudioSegment, it was too short {2:0.2f}s for the current settings:\n{0}ms chunks at {1:0.1f}x speedup".format(
            chunk_size, playback_speed, seg.duration_seconds))

    # we'll actually truncate a bit less than we calculated to make up for the
    # crossfade between chunks
    ms_to_remove_per_chunk -= crossfade

    # we don't want to truncate the last chunk since it is not guaranteed to be
    # the full chunk length
    last_chunk = chunks[-1]
    chunks = [chunk[:-ms_to_remove_per_chunk] for chunk in chunks[:-1]]

    out = chunks[0]
    for chunk in chunks[1:]:
        out = out.append(chunk, crossfade=crossfade)

    out += last_chunk
    return out
    

@register_pydub_effect
def strip_silence(seg, silence_len=1000, silence_thresh=-16, padding=100):
    if padding > silence_len:
        raise InvalidDuration("padding cannot be longer than silence_len")

    chunks = split_on_silence(seg, silence_len, silence_thresh, padding)
    crossfade = padding / 2

    if not len(chunks):
        return seg[0:0]

    seg = chunks[0]
    for chunk in chunks[1:]:
        seg = seg.append(chunk, crossfade=crossfade)

    return seg


@register_pydub_effect
def compress_dynamic_range(seg, threshold=-20.0, ratio=4.0, attack=5.0, release=50.0):
    """
    Keyword Arguments:
        
        threshold - default: -20.0
            Threshold in dBFS. default of -20.0 means -20dB relative to the
            maximum possible volume. 0dBFS is the maximum possible value so
            all values for this argument sould be negative.

        ratio - default: 4.0
            Compression ratio. Audio louder than the threshold will be 
            reduced to 1/ratio the volume. A ratio of 4.0 is equivalent to
            a setting of 4:1 in a pro-audio compressor like the Waves C1.
        
        attack - default: 5.0
            Attack in milliseconds. How long it should take for the compressor
            to kick in once the audio has exceeded the threshold.

        release - default: 50.0
            Release in milliseconds. How long it should take for the compressor
            to stop compressing after the audio has falled below the threshold.

    
    For an overview of Dynamic Range Compression, and more detailed explanation
    of the related terminology, see: 

        http://en.wikipedia.org/wiki/Dynamic_range_compression
    """

    thresh_rms = seg.max_possible_amplitude * db_to_float(threshold)
    
    look_frames = int(seg.frame_count(ms=attack))
    def rms_at(frame_i):
        return seg.get_sample_slice(frame_i - look_frames, frame_i).rms
    def db_over_threshold(rms):
        if rms == 0: return 0.0
        db = ratio_to_db(rms / thresh_rms)
        return max(db, 0)

    output = []

    # amount to reduce the volume of the audio by (in dB)
    attenuation = 0.0
    
    attack_frames = seg.frame_count(ms=attack)
    release_frames = seg.frame_count(ms=release)
    for i in xrange(int(seg.frame_count())):
        rms_now = rms_at(i)
        
        # with a ratio of 4.0 this means the volume will exceed the threshold by
        # 1/4 the amount (of dB) that it would otherwise
        max_attenuation = (1 - (1.0 / ratio)) * db_over_threshold(rms_now)
        
        attenuation_inc = max_attenuation / attack_frames
        attenuation_dec = max_attenuation / release_frames
        
        if rms_now > thresh_rms and attenuation <= max_attenuation:
            attenuation += attenuation_inc
            attenuation = min(attenuation, max_attenuation)
        else:
            attenuation -= attenuation_dec
            attenuation = max(attenuation, 0)
        
        frame = seg.get_frame(i)
        if attenuation != 0.0:
            frame = audioop.mul(frame,
                                seg.sample_width,
                                db_to_float(-attenuation))
        
        output.append(frame)
    
    return seg._spawn(data=b''.join(output))


# Invert the phase of the signal.

@register_pydub_effect

def invert_phase(seg, channels=(1, 1)):
    """
    channels- specifies which channel (left or right) to reverse the phase of.
    Note that mono AudioSegments will become stereo.
    """
    if channels == (1, 1):
        inverted = audioop.mul(seg._data, seg.sample_width, -1.0)  
        return seg._spawn(data=inverted)
    
    else:
        if seg.channels == 2:
            left, right = seg.split_to_mono()
        else:
            raise Exception("Can't implicitly convert an AudioSegment with " + str(seg.channels) + " channels to stereo.")
            
        if channels == (1, 0):    
            left = left.invert_phase()
        else:
            right = right.invert_phase()
        
        return seg.from_mono_audiosegments(left, right)
        


# ---------------------------------------------------------------------------
# Multi-band parametric EQ
# ---------------------------------------------------------------------------

# Accepted spellings for the keys of a band passed as a dict. Keys are
# stripped and lower-cased before lookup, so "Q" and "q" are equivalent.
_EQ_BAND_KEYS = {
    "frequency": ("freq", "frequency", "f"),
    "q": ("q", "quality"),
    "gain_db": ("gain_db", "gain", "gaindb", "gain_db (db)"),
}


def _eq_unpack_band(band, index):
    """
    Turn one band specification into a ``(frequency, q, gain_db)`` tuple
    of floats, validating its shape. Runtime value validation happens in
    :func:`_eq_parse_bands`.
    """
    if isinstance(band, dict):
        normalized = {}
        for key, value in band.items():
            if not isinstance(key, basestring):
                raise ValueError(
                    "parametric_eq() band %d: keys must be strings, got %r"
                    % (index, key))
            normalized[key.strip().lower()] = value

        names = {}
        for field, aliases in _EQ_BAND_KEYS.items():
            for alias in aliases:
                if alias in normalized:
                    names[field] = normalized[alias]
                    break
            else:
                raise ValueError(
                    "parametric_eq() band %d: missing %r (accepted keys: "
                    "freq/frequency, q/quality, gain_db/gain)"
                    % (index, field))

        unknown = set(normalized) - set(
            alias for aliases in _EQ_BAND_KEYS.values() for alias in aliases)
        if unknown:
            raise ValueError(
                "parametric_eq() band %d: unknown key(s) %s (accepted keys: "
                "freq/frequency, q/quality, gain_db/gain)"
                % (index, ", ".join(repr(key) for key in sorted(unknown))))

        return names["frequency"], names["q"], names["gain_db"]

    if isinstance(band, (tuple, list)):
        if len(band) != 3:
            raise ValueError(
                "parametric_eq() band %d: expected a (frequency, q, gain_db) "
                "tuple, got %d value(s)" % (index, len(band)))
        return band[0], band[1], band[2]

    raise ValueError(
        "parametric_eq() band %d: expected a (frequency, q, gain_db) tuple "
        "or a dict, got %r" % (index, band))


def _eq_as_real(value, field, index):
    """Coerce a band parameter to float, rejecting bools and non-numbers."""
    # bool is a subclass of int, but True/False are not meaningful here
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(
            "parametric_eq() band %d: %s must be a real number, got %r"
            % (index, field, value))
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(
            "parametric_eq() band %d: %s must be a real number, got %r"
            % (index, field, value))
    if not _isfinite(value):
        raise ValueError(
            "parametric_eq() band %d: %s must be finite, got %r"
            % (index, field, value))
    return value


def _eq_parse_bands(seg, bands):
    """
    Validate a sequence of band specifications and return a list of
    ``(frequency, q, gain_db)`` float tuples. Raises ValueError for any
    malformed or out-of-range parameter.

    The Nyquist limit is checked for non-empty segments only: an empty
    segment carries no samples (its placeholder metadata may use an
    arbitrary frame rate), so there is nothing to filter.
    """
    if bands is None:
        raise ValueError("parametric_eq() bands must be a sequence, got None")
    try:
        iterator = iter(bands)
    except TypeError:
        raise ValueError(
            "parametric_eq() bands must be a sequence of "
            "(frequency, q, gain_db) tuples, got %r" % (bands,))

    nyquist = seg.frame_rate / 2.0
    check_nyquist = seg.frame_count() > 0

    specs = []
    for index, band in enumerate(iterator):
        frequency, q, gain_db = _eq_unpack_band(band, index)
        frequency = _eq_as_real(frequency, "frequency", index)
        q = _eq_as_real(q, "q", index)
        gain_db = _eq_as_real(gain_db, "gain_db", index)

        if frequency <= 0.0:
            raise ValueError(
                "parametric_eq() band %d: frequency must be > 0 Hz, got %r"
                % (index, frequency))
        if q <= 0.0:
            raise ValueError(
                "parametric_eq() band %d: q must be > 0, got %r"
                % (index, q))
        if check_nyquist and frequency >= nyquist:
            raise ValueError(
                "parametric_eq() band %d: frequency must be below the "
                "Nyquist frequency (%.1f Hz for a %d Hz sample rate), "
                "got %r" % (index, nyquist, seg.frame_rate, frequency))

        specs.append((frequency, q, gain_db))
    return specs


def _eq_peaking_coefficients(frame_rate, frequency, q, gain_db):
    """
    RBJ audio-EQ-cookbook peaking biquad, returning normalized
    ``(b0, b1, b2, a1, a2)``.

    ``gain_db`` is the gain at the center frequency: the amplitude
    response at ``frequency`` is exactly ``10 ** (gain_db / 20)``.
    """
    try:
        amplitude = 10.0 ** (gain_db / 40.0)
    except OverflowError:
        raise ValueError(
            "parametric_eq(): gain_db %r is too large to implement"
            % (gain_db,))

    omega = 2.0 * math.pi * frequency / frame_rate
    alpha = math.sin(omega) / (2.0 * q)
    cos_omega = math.cos(omega)

    b0 = 1.0 + alpha * amplitude
    b1 = -2.0 * cos_omega
    b2 = 1.0 - alpha * amplitude
    a0 = 1.0 + alpha / amplitude
    a1 = -2.0 * cos_omega
    a2 = 1.0 - alpha / amplitude

    coefficients = (b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)
    for coefficient in coefficients:
        if not _isfinite(coefficient):
            raise ValueError(
                "parametric_eq(): band (frequency=%r, q=%r, gain_db=%r) "
                "produces non-finite filter coefficients"
                % (frequency, q, gain_db))
    return coefficients


def _eq_apply_biquad(channel, coefficients):
    """
    Apply one biquad in transposed direct form II to a float array,
    in place. Each call starts from zero state, so a segment shorter
    than the filter's history is simply filtered from silence.
    """
    b0, b1, b2, a1, a2 = coefficients
    z1 = 0.0
    z2 = 0.0
    for i in range(len(channel)):
        x = channel[i]
        y = b0 * x + z1
        channel[i] = y
        z1 = b1 * x - a1 * y + z2
        z2 = b2 * x - a2 * y


def _eq_quantize(value, minval, maxval):
    """
    Convert one float sample to an integer with deterministic saturation:
    values beyond the sample width's range clip to its minimum/maximum
    (never wrap around), and NaN (which can only come from pathological
    gains) becomes silence.
    """
    if value != value:  # NaN
        return 0
    if value >= maxval:
        return maxval
    if value <= minval:
        return minval
    return int(round(value))


@register_pydub_effect
def parametric_eq(seg, bands):
    """
    Apply a sequence of peaking (bell) EQ bands to the segment.

    ``bands`` is a sequence of ``(frequency, q, gain_db)`` tuples, or of
    dicts with equivalent keys::

        # the two forms below are the same
        seg.parametric_eq([(250, 3.0, -9.0), (4000, 3.0, 6.0)])
        seg.parametric_eq([
            {"freq": 250, "q": 3.0, "gain_db": -9.0},
            {"frequency": 4000, "quality": 3.0, "gain": 6.0},
        ])

    frequency
        Center frequency in Hz. Must be finite, greater than 0 and below
        the Nyquist frequency (``frame_rate / 2``) for non-empty segments;
        anything else raises ValueError.
    q
        Quality factor (bandwidth). Must be finite and greater than 0;
        anything else raises ValueError.
    gain_db
        Gain at the center frequency in dB. Positive values boost, negative
        values cut. Must be finite. A gain of exactly 0 leaves that band
        out entirely (so a zero-gain band list is an exact identity).

    Bands are applied in the order given, one after another, as a cascade
    of second-order peaking filters (RBJ audio EQ cookbook). They are
    applied independently to every channel: filtering the left channel can
    never affect the right one. The frame rate, sample width, channel
    count and frame count are preserved.

    Filtering starts from zero state (no padding is added), so segments
    shorter than the filter's two-sample history are handled normally and
    keep their length. Intermediate results are kept in double precision
    and quantized once at the end; values outside the sample width's range
    are hard-clipped to its minimum/maximum, so overflow can never wrap
    around. Empty segments return an empty segment (the Nyquist check is
    skipped because there are no samples to process).

    The existing ``low_pass_filter()`` and ``high_pass_filter()`` effects
    are untouched; this is a separate, multi-band entry point.
    """
    specs = _eq_parse_bands(seg, bands)

    if not specs or seg.frame_count() == 0:
        return seg._spawn(seg._data)

    if seg.sample_width not in (1, 2, 4):
        raise ValueError(
            "parametric_eq() supports 8, 16 and 32 bit samples, got a "
            "sample width of %r bytes" % (seg.sample_width,))

    # Zero-gain bands are skipped rather than run as identity biquads, so
    # a zero-gain request returns the input samples bit for bit.
    coefficients = [
        _eq_peaking_coefficients(seg.frame_rate, frequency, q, gain_db)
        for frequency, q, gain_db in specs
        if gain_db != 0.0
    ]
    if not coefficients:
        return seg._spawn(seg._data)

    channels = seg.channels
    samples = seg.get_array_of_samples()
    minval, maxval = get_min_max_value(seg.sample_width * 8)
    out_samples = array.array(seg.array_type, [0]) * len(samples)

    for channel_index in range(channels):
        # Work on one channel at a time so every channel gets its own
        # filter state (no bleed) and the intermediate buffer stays tight.
        channel = array.array("d", samples[channel_index::channels])
        for band_coefficients in coefficients:
            _eq_apply_biquad(channel, band_coefficients)
        out_samples[channel_index::channels] = array.array(
            seg.array_type,
            [_eq_quantize(value, minval, maxval) for value in channel])

    return seg._spawn(data=out_samples)


# High and low pass filters based on implementation found on Stack Overflow:
#   http://stackoverflow.com/questions/13882038/implementing-simple-high-and-low-pass-filters-in-c

@register_pydub_effect
def low_pass_filter(seg, cutoff):
    """
        cutoff - Frequency (in Hz) where higher frequency signal will begin to
            be reduced by 6dB per octave (doubling in frequency) above this point
    """
    RC = 1.0 / (cutoff * 2 * math.pi)
    dt = 1.0 / seg.frame_rate

    alpha = dt / (RC + dt)
    
    original = seg.get_array_of_samples()
    filteredArray = array.array(seg.array_type, original)
    
    frame_count = int(seg.frame_count())

    last_val = [0] * seg.channels
    for i in range(seg.channels):
        last_val[i] = filteredArray[i] = original[i]

    for i in range(1, frame_count):
        for j in range(seg.channels):
            offset = (i * seg.channels) + j
            last_val[j] = last_val[j] + (alpha * (original[offset] - last_val[j]))
            filteredArray[offset] = int(last_val[j])

    return seg._spawn(data=filteredArray)


@register_pydub_effect
def high_pass_filter(seg, cutoff):
    """
        cutoff - Frequency (in Hz) where lower frequency signal will begin to
            be reduced by 6dB per octave (doubling in frequency) below this point
    """
    RC = 1.0 / (cutoff * 2 * math.pi)
    dt = 1.0 / seg.frame_rate

    alpha = RC / (RC + dt)

    minval, maxval = get_min_max_value(seg.sample_width * 8)
    
    original = seg.get_array_of_samples()
    filteredArray = array.array(seg.array_type, original)
    
    frame_count = int(seg.frame_count())

    last_val = [0] * seg.channels
    for i in range(seg.channels):
        last_val[i] = filteredArray[i] = original[i]

    for i in range(1, frame_count):
        for j in range(seg.channels):
            offset = (i * seg.channels) + j
            offset_minus_1 = ((i-1) * seg.channels) + j

            last_val[j] = alpha * (last_val[j] + original[offset] - original[offset_minus_1])
            filteredArray[offset] = int(min(max(last_val[j], minval), maxval))

    return seg._spawn(data=filteredArray)
    
    
@register_pydub_effect
def pan(seg, pan_amount):
    """
    pan_amount should be between -1.0 (100% left) and +1.0 (100% right)
    
    When pan_amount == 0.0 the left/right balance is not changed.
    
    Panning does not alter the *perceived* loundness, but since loudness
    is decreasing on one side, the other side needs to get louder to
    compensate. When panned hard left, the left channel will be 3dB louder.
    """
    if not -1.0 <= pan_amount <= 1.0:
        raise ValueError("pan_amount should be between -1.0 (100% left) and +1.0 (100% right)")
    
    max_boost_db = ratio_to_db(2.0)
    boost_db = abs(pan_amount) * max_boost_db
    
    boost_factor = db_to_float(boost_db)
    reduce_factor = db_to_float(max_boost_db) - boost_factor
    
    reduce_db = ratio_to_db(reduce_factor)
    
    # Cut boost in half (max boost== 3dB) - in reality 2 speakers
    #   do not sum to a full 6 dB.
    boost_db = boost_db / 2.0
    
    if pan_amount < 0:
        return seg.apply_gain_stereo(boost_db, reduce_db)
    else:
        return seg.apply_gain_stereo(reduce_db, boost_db)
        
    
@register_pydub_effect
def apply_gain_stereo(seg, left_gain=0.0, right_gain=0.0):
    """
    left_gain - amount of gain to apply to the left channel (in dB)
    right_gain - amount of gain to apply to the right channel (in dB)
    
    note: mono audio segments will be converted to stereo
    """
    if seg.channels == 1:
        left = right = seg
    elif seg.channels == 2:
        left, right = seg.split_to_mono()
    
    l_mult_factor = db_to_float(left_gain)
    r_mult_factor = db_to_float(right_gain)
    
    left_data = audioop.mul(left._data, left.sample_width, l_mult_factor)
    left_data = audioop.tostereo(left_data, left.sample_width, 1, 0)
    
    right_data = audioop.mul(right._data, right.sample_width, r_mult_factor)
    right_data = audioop.tostereo(right_data, right.sample_width, 0, 1)
    
    output = audioop.add(left_data, right_data, seg.sample_width)
    
    return seg._spawn(data=output,
                overrides={'channels': 2,
                           'frame_width': 2 * seg.sample_width})
