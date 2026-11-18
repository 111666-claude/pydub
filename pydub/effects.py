import sys
import math
import array
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


@register_pydub_effect
def feedback_delay(seg, delay_ms=250, feedback=0.4, wet=0.5, dry=1.0,
                   max_tail_ms=5000):
    """
    Add a feedback delay (echo) effect to an AudioSegment.

    The effect is a classic feedback delay line: each echo is a copy of the
    previous echo, scaled by ``feedback``. For a sample at frame n the output
    is::

        out[n] = dry * in[n] + wet * delay_line[n]
        delay_line[n] = in[n] + feedback * delay_line[n - delay_frames]

    so a sample at frame n reappears as the k-th echo at frame
    n + k * delay_frames with an amplitude of wet * feedback**(k - 1)
    (as long as no clipping occurs).

    Keyword Arguments:

        delay_ms - default: 250
            Delay between the original signal and the first echo, in
            milliseconds. It is converted to a whole number of *samples
            (frames)* with half-up rounding (``int(ms * frame_rate / 1000
            + 0.5)``) rather than by slicing whole milliseconds, so echo
            timing is sample-accurate. Must be >= 0. With a zero delay no
            delay line is possible, so the segment is returned with only the
            dry gain applied and no tail.

        feedback - default: 0.4
            Amount of the delayed signal fed back into the delay line, a
            linear amplitude factor in the range [0.0, 1.0). Values < 0 or
            >= 1.0 raise a ValueError: feedback >= 1 would make the echoes
            grow or never decay, so a finite tail could not represent the
            result. 0.0 produces a single echo with no repeats.

        wet - default: 0.5
            Linear amplitude factor applied to the echoed (wet) signal. Must
            be >= 0.0.

        dry - default: 1.0
            Linear amplitude factor applied to the original (dry) signal.
            Must be >= 0.0. Set to 0.0 to keep only the echoes.

        max_tail_ms - default: 5000
            Maximum amount of silence/echoes appended after the end of the
            original segment, in milliseconds. Converted to frames with the
            same half-up rounding as ``delay_ms``. Echoes that would fall
            beyond the original segment plus this tail are dropped, not
            wrapped or truncated mid-echo processing; the output is simply
            cut to that length. Must be >= 0.

    Returns a new AudioSegment. The number of channels, frame rate and
    sample width (and therefore the sample format) are preserved; the source
    segment is never modified.

    Clipping rule: all arithmetic is done in the segment's integer sample
    domain. Every sample written to the delay line or to the output is hard
    clipped (saturated) to the range of the sample width, the same rule used
    by ``audioop.add``/``audioop.mul`` (e.g. -32768..32767 for 16-bit audio).

    An empty (zero-frame) segment returns an empty segment of the same
    format, regardless of the tail setting.
    """
    if delay_ms < 0:
        raise ValueError("delay_ms must be greater than or equal to 0")
    if not 0.0 <= feedback < 1.0:
        raise ValueError("feedback must be in the range [0.0, 1.0)")
    if wet < 0:
        raise ValueError("wet must be greater than or equal to 0.0")
    if dry < 0:
        raise ValueError("dry must be greater than or equal to 0.0")
    if max_tail_ms < 0:
        raise ValueError("max_tail_ms must be greater than or equal to 0")

    in_frames = int(seg.frame_count())

    # an empty segment has nothing to echo; keep its (arbitrary) format
    if in_frames == 0:
        return seg._spawn(seg._data)

    # half-up rounding gives sample-accurate, non-negative frame counts
    delay_frames = int(delay_ms * seg.frame_rate / 1000.0 + 0.5)
    tail_frames = int(max_tail_ms * seg.frame_rate / 1000.0 + 0.5)

    # zero delay cannot recurse; apply the dry gain only, with no tail
    if delay_frames == 0:
        if dry == 1.0:
            return seg._spawn(seg._data)
        min_sample, max_sample = get_min_max_value(seg.sample_width * 8)
        scaled = seg.get_array_of_samples()
        for i, sample in enumerate(scaled):
            val = int(sample * dry)
            scaled[i] = min(max(val, min_sample), max_sample)
        return seg._spawn(scaled)

    min_sample, max_sample = get_min_max_value(seg.sample_width * 8)

    def clip(sample):
        if sample > max_sample:
            return max_sample
        if sample < min_sample:
            return min_sample
        return sample

    n_channels = seg.channels
    input_samples = seg.get_array_of_samples()
    total_frames = in_frames + tail_frames
    total_samples = total_frames * n_channels
    delay_samples = delay_frames * n_channels

    # the delay line holds the (clipped) recirculated signal; the output
    # holds the dry/wet mix. Both start silent.
    silent_template = array.array(seg.array_type, [0])
    delay_line = silent_template * total_samples
    output = array.array(seg.array_type, [0]) * total_samples

    input_sample_count = in_frames * n_channels
    for i in xrange(total_samples):
        in_sample = input_samples[i] if i < input_sample_count else 0

        if i >= delay_samples:
            delayed = delay_line[i - delay_samples]
        else:
            delayed = 0

        # clip first so the values fed back are exactly the stored ones
        delay_line[i] = clip(in_sample + int(feedback * delayed))
        output[i] = clip(int(dry * in_sample) + int(wet * delayed))

    return seg._spawn(output)
