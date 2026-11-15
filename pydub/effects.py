import sys
import math
import array
import wave
import subprocess
from io import BytesIO
from .utils import (
    db_to_float,
    ratio_to_db,
    register_pydub_effect,
    make_chunks,
    audioop,
    which,
    get_available_filters,
    get_min_max_value
)
from .logging_utils import log_conversion, log_subprocess_output
from .audio_segment import AudioSegment
from .silence import split_on_silence
from .exceptions import (
    TooManyMissingFrames,
    InvalidDuration,
    InvalidSpeed,
    MissingConverterError,
    CouldntProcessError,
)

if sys.version_info >= (3, 0):
    xrange = range

# pitch-preserving time stretching is supported within this speed range
MIN_PLAYBACK_SPEED = 0.25
MAX_PLAYBACK_SPEED = 4.0

# the rubberband/atempo filters buffer whole analysis windows and can drop
# the tail of inputs (observed losses of ~15ms with rubberband and ~80ms
# with a chained atempo filter, measured on the output timeline). A silence
# is appended before processing and trimmed back off afterwards; this is
# the amount of spare output (in milliseconds on the output timeline) the
# padding guarantees.
_OUTPUT_TAIL_HEADROOM_MS = 500

# PCM codecs matching the sample widths pydub uses (8-bit audio is unsigned
# in WAV containers)
_PCM_CODECS_BY_SAMPLE_WIDTH = {
    1: "pcm_u8",
    2: "pcm_s16le",
    4: "pcm_s32le",
}


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


def _atempo_filter_chain(playback_speed):
    """
    Build an ffmpeg ``atempo`` filter chain for playback_speed.

    A single atempo filter only accepts values in [0.5, 2.0], so speeds
    outside that interval are decomposed into factors that each fall inside
    it (the supported speed range 0.25 - 4.0 needs at most two filters).
    The factors are chosen so their product is exactly playback_speed.
    """
    factors = []
    remaining = float(playback_speed)
    while remaining < 0.5:
        factors.append(0.5)
        remaining /= 0.5
    while remaining > 2.0:
        factors.append(2.0)
        remaining /= 2.0
    factors.append(remaining)
    return ",".join("atempo={0:.10g}".format(factor) for factor in factors)


@register_pydub_effect
def time_stretch(seg, playback_speed=1.0):
    """
    Pitch-preserving time stretch: change the playback speed without
    changing the pitch of the audio (unlike simply changing the frame
    rate, which shifts pitch, or ``speedup()``, which approximates
    acceleration by discarding chunks).

    playback_speed - positive float in the range 0.25 - 4.0:
        values above 1.0 speed the audio up, values below 1.0 slow it
        down. 1.0 returns a copy of the segment unchanged.

    The output duration is ``input_duration / playback_speed`` while the
    frame rate, channel count and sample width are preserved.

    Requires ffmpeg (the ``rubberband`` filter is used when available,
    otherwise an ``atempo`` filter chain is used). Audio is processed
    entirely through pipes, so no temporary files are created.
    """
    # bool is a subclass of int but does not make sense as a speed
    if isinstance(playback_speed, bool) or not isinstance(
            playback_speed, (int, float)):
        raise TypeError(
            "playback_speed must be a number, got {0}".format(
                type(playback_speed).__name__))

    playback_speed = float(playback_speed)
    if not math.isfinite(playback_speed):
        raise InvalidSpeed(
            "playback_speed must be a finite number, got {0}".format(
                playback_speed))
    if not (MIN_PLAYBACK_SPEED <= playback_speed <= MAX_PLAYBACK_SPEED):
        raise InvalidSpeed(
            "playback_speed must be between {0} and {1}, got {2}".format(
                MIN_PLAYBACK_SPEED, MAX_PLAYBACK_SPEED, playback_speed))

    # 1.0 means no time stretching at all
    if playback_speed == 1.0:
        return seg._spawn(seg._data)

    if len(seg) == 0:
        raise CouldntProcessError(
            "Cannot time stretch an empty (zero-length) AudioSegment")

    converter = seg.converter
    if not which(converter):
        raise MissingConverterError(
            "Could not find {0!r}; time_stretch() requires ffmpeg or "
            "avconv to be installed and available on PATH".format(converter))

    pcm_codec = _PCM_CODECS_BY_SAMPLE_WIDTH.get(seg.sample_width)
    if pcm_codec is None:
        raise CouldntProcessError(
            "time_stretch() cannot process audio with a sample width of "
            "{0} bytes; use set_sample_width() to convert to 1, 2 or 4 "
            "bytes first".format(seg.sample_width))

    # The time-stretch filters process the signal in fixed-size windows and
    # may truncate the tail (observed up to ~80ms on the output timeline
    # with a chained atempo filter). A silence is appended before processing
    # so the whole signal fits inside the output, then the output is trimmed
    # to the exact target duration. The headroom is requested on the output
    # timeline, hence it is multiplied by the speed to get the input-side
    # padding duration (slowing down compresses the padding on output).
    padding_ms = int(math.ceil(
        _OUTPUT_TAIL_HEADROOM_MS * playback_speed))
    # append() syncs channel count, sample width and frame rate by itself
    padded = seg + AudioSegment.silent(
        padding_ms, frame_rate=seg.frame_rate)

    # pipe the segment's raw PCM data into the converter as an uncompressed
    # WAV stream; everything stays in memory (no temporary files)
    wav_buffer = BytesIO()
    wav_data = padded._data
    if seg.sample_width == 1:
        # WAV stores 8-bit PCM as unsigned integers
        wav_data = audioop.bias(wav_data, 1, 128)
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(seg.channels)
        wav_file.setsampwidth(seg.sample_width)
        wav_file.setframerate(seg.frame_rate)
        wav_file.writeframes(wav_data)

    available_filters = get_available_filters()
    if "rubberband" in available_filters:
        # rubberband is the highest quality option and natively covers the
        # whole supported speed range; pitch defaults to 1.0 (unchanged)
        audio_filter = "rubberband=tempo={0:.10g}:pitch=1.0".format(
            playback_speed)
    elif "atempo" in available_filters:
        audio_filter = _atempo_filter_chain(playback_speed)
    else:
        raise CouldntProcessError(
            "The installed {0} build provides neither the 'rubberband' nor "
            "the 'atempo' audio filter, which time_stretch() requires".format(
                converter))

    conversion_command = [
        converter,
        "-y",
        "-hide_banner",
        "-f", "wav", "-i", "-",
        "-vn",
        "-acodec", pcm_codec,
        "-filter:a", audio_filter,
        "-f", "wav", "-",
    ]

    log_conversion(conversion_command)

    try:
        process = subprocess.Popen(
            conversion_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
    except OSError as exc:
        raise MissingConverterError(
            "Could not invoke {0!r}: {1}".format(converter, exc))

    p_out, p_err = process.communicate(input=wav_buffer.getvalue())
    log_subprocess_output(p_err)

    if process.returncode != 0 or not p_out:
        raise CouldntProcessError(
            "Time stretch failed. {0} returned error code {1}\n\n"
            "Output from {0}:\n\n{2}".format(
                converter, process.returncode,
                p_err.decode(errors='ignore')))

    # parse the output WAV and re-apply the segment's own metadata so the
    # frame rate, channel count and sample width are preserved exactly
    try:
        out_seg = seg.__class__(p_out)
    except Exception as exc:
        raise CouldntProcessError(
            "Time stretch produced unreadable output: {0}".format(exc))

    # trim the tail padding back off so the output duration is
    # input_duration / playback_speed, using the same millisecond rounding
    # (floor) that len() uses
    frame_width = seg.frame_width
    target_ms = int(len(seg) / playback_speed)
    target_frames = int(target_ms * seg.frame_rate / 1000.0)
    target_bytes = target_frames * frame_width
    if len(out_seg._data) < target_bytes:
        raise CouldntProcessError(
            "Time stretch produced shorter output than expected "
            "({0} frames instead of {1})".format(
                len(out_seg._data) // frame_width, target_frames))
    out_data = out_seg._data[:target_bytes]

    return seg._spawn(
        out_data,
        overrides={
            'channels': seg.channels,
            'sample_width': seg.sample_width,
            'frame_rate': seg.frame_rate,
            'frame_width': frame_width,
        })


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
