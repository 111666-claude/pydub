"""
Seamless loop construction for :class:`AudioSegment <pydub.AudioSegment>`.

Environmental recordings rarely begin and end on matching samples, so looping
them produces an audible click at the wrap boundary.  This module searches the
head and tail of a segment for the pair of cut points whose waveforms and
energies line up best, then joins them with an equal-power crossfade so that
the returned segment repeats without a discontinuity.
"""
from __future__ import division

import array
import math
from collections import namedtuple

from .utils import audioop, get_array_type


SeamlessLoop = namedtuple('SeamlessLoop', [
    'segment',
    'loop_start_sample',
    'loop_end_sample',
    'crossfade_samples',
    'loop_start_ms',
    'loop_end_ms',
])
SeamlessLoop.__doc__ = """
Result of a seamless-loop search.

``segment`` is the loopable :class:`~pydub.AudioSegment`, laid out as a
faded-in copy of the chosen head zone, the unmodified interior, and a
faded-out copy of the chosen tail zone, so concatenating the segment with
itself is smooth at the join.  ``loop_start_sample`` / ``loop_end_sample``
are the frame indices at which the loop was cut out of the source segment;
``crossfade_samples`` is the crossfade length in frames.  The ``*_ms``
fields mirror the sample positions in milliseconds so the cut points are
directly interpretable.
"""


def _ms_to_frames(ms, frame_rate):
    return int(ms * (frame_rate / 1000.0))


def _to_bytes(samples):
    try:
        return samples.tobytes()
    except AttributeError:  # pragma: no cover - Python 2
        return samples.tostring()


def _channel_windows(interleaved_bytes, channels, sample_width):
    """
    Split an interleaved PCM byte string into one byte string per channel.

    Each returned byte string is ordinary packed mono PCM: ``sample_width``
    bytes per sample, ``num_frames * sample_width`` bytes total, so it can be
    sliced and passed to :mod:`audioop` directly.
    """
    if channels == 1:
        return [interleaved_bytes]
    samples = array.array(
        get_array_type(sample_width * 8), interleaved_bytes)
    return [
        _to_bytes(samples[ch::channels])
        for ch in range(channels)
    ]


def _mono_slice(mono_bytes, start_sample, num_samples, sample_width):
    """
    Slice ``num_samples`` mono samples starting at ``start_sample`` out of a
    packed-mono byte string produced by :func:`_channel_windows`.
    """
    start_byte = start_sample * sample_width
    end_byte = (start_sample + num_samples) * sample_width
    return mono_bytes[start_byte:end_byte]


def _apply_gain_ramp(data, frame_width, sample_width, gain_for_frame):
    """
    Apply a per-frame gain curve to interleaved PCM bytes.
    ``gain_for_frame(i)`` returns the gain of frame ``i`` (zero-based).
    Only the search/crossfade windows are ever ramped, so the amount of
    per-sample work stays bounded by the window size.
    """
    frame_count = len(data) // frame_width
    if frame_count == 0:
        return data
    out = bytearray()
    for i in range(frame_count):
        frame = data[i * frame_width:(i + 1) * frame_width]
        out += audioop.mul(frame, sample_width, gain_for_frame(i, frame_count))
    return bytes(out)


def _equal_power_fade(data, frame_width, sample_width, fade_in):
    """
    Equal-power fade ramps sampled at integer frame indices, so the endpoints
    land on exact gains rather than half-step gains:

    * fade-in:  ``gain(i) = sin(i/(C-1) * pi/2)`` -> exactly 0 at the first
      frame, exactly 1 at the last;
    * fade-out: ``gain(i) = cos(i/(C-1) * pi/2)`` -> exactly 1 at the first
      frame, exactly 0 at the last.

    At matching indices the two ramps satisfy ``g_in^2 + g_out^2 == 1``.
    Because the loop wrap joins the fade-out's last frame (gain 0) to the
    fade-in's first frame (gain 0), both boundary samples are silent and the
    repeat join is guaranteed not to click.  A one-frame fade degenerates to
    a single silent frame, which is the only possible behaviour.
    """
    def fade_in_gain(i, frame_count):
        if frame_count == 1 or i == 0:
            # the outer endpoint must be an exact zero: cos(pi/2) computed via
            # floats would otherwise round a loud sample down to +/-1
            return 0.0
        if i == frame_count - 1:
            return 1.0
        return math.sin(i / (frame_count - 1) * math.pi / 2.0)

    def fade_out_gain(i, frame_count):
        if frame_count == 1 or i == frame_count - 1:
            return 0.0
        if i == 0:
            return 1.0
        return math.cos(i / (frame_count - 1) * math.pi / 2.0)

    gain_for_frame = fade_in_gain if fade_in else fade_out_gain
    return _apply_gain_ramp(
        data, frame_width, sample_width, gain_for_frame)


def find_seamless_loop(segment, search_window_ms=100, crossfade_ms=10,
                       energy_weight=1.0):
    """
    Cut a seamlessly loopable segment out of ``segment``.

    The head and tail of the segment are each examined over a window of
    ``search_window_ms``.  A 1-D sweep over the relative offset of the two cut
    positions (one cut pinned to its window edge per offset) scores every
    candidate on three continuity measures computed independently per channel:
    the sample-by-sample waveform mismatch across the crossfade zone, the exact
    boundary pair crossed on every repeat, and the short-term energy gap.  The
    chosen head is faded in and the tail faded out with equal-power ramps whose
    outer endpoints sit on an exact gain of zero, so the wrap boundary of the
    returned period is silent on both sides and cannot click.

    The returned period is laid out as the faded-in head zone, the untouched
    interior, and the faded-out tail zone.  Because the only per-candidate work
    is proportional to the crossfade and just the two bounded windows are read,
    runtime and memory do not grow with the source length apart from writing
    the single output buffer.

    Arguments:
        segment (AudioSegment): source audio.
        search_window_ms (int): length of the head and tail search windows,
            in milliseconds.
        crossfade_ms (int): crossfade duration at the loop seam, in
            milliseconds.  Must be shorter than the search window and leave
            at least one frame of non-crossfaded audio.
        energy_weight (float): relative weight of the energy-continuity term
            versus the waveform-mismatch terms (which have weight 1).

    Returns:
        SeamlessLoop: the loopable segment and the cut positions used.

    Raises:
        ValueError: for empty audio, non-finite/non-positive/non-numeric
            parameters, a crossfade longer than the search window, or source
            material with too few samples for the requested windows.
    """
    def _require_positive_ms(value, name):
        if isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not math.isfinite(value):
            raise ValueError(
                "%s must be a finite positive number of milliseconds" % name)
        if value <= 0:
            raise ValueError(
                "%s must be a positive number of milliseconds" % name)
        return value

    search_window_ms = _require_positive_ms(
        search_window_ms, "search_window_ms")
    crossfade_ms = _require_positive_ms(crossfade_ms, "crossfade_ms")
    if isinstance(energy_weight, bool) or not isinstance(
            energy_weight, (int, float)) or not math.isfinite(energy_weight):
        raise ValueError("energy_weight must be a finite non-negative number")
    if energy_weight < 0:
        raise ValueError("energy_weight must be non-negative")

    total_frames = int(segment.frame_count())
    if total_frames == 0:
        raise ValueError(
            "cannot build a seamless loop from an empty AudioSegment")

    window_frames = _ms_to_frames(search_window_ms, segment.frame_rate)
    crossfade_frames = _ms_to_frames(crossfade_ms, segment.frame_rate)

    if window_frames <= 0:
        raise ValueError(
            "search_window_ms is shorter than one frame at %d Hz"
            % segment.frame_rate)
    if crossfade_frames <= 0:
        raise ValueError(
            "crossfade_ms is shorter than one frame at %d Hz"
            % segment.frame_rate)
    if crossfade_frames >= window_frames:
        raise ValueError(
            "crossfade_ms (%dms) must be shorter than search_window_ms "
            "(%dms)" % (crossfade_ms, search_window_ms))
    # The head and tail windows are taken from opposite ends, and at least one
    # frame of unmodified interior must remain between the two fades.
    if total_frames <= 2 * window_frames:
        raise ValueError(
            "not enough samples: the segment must be longer than twice the "
            "search window to leave an uncrossfaded interior (%d frames "
            "available, %d needed)"
            % (total_frames, 2 * window_frames + 1))
    # The head and tail crossfade zones must not overlap.
    min_frames = 2 * crossfade_frames + 1
    if total_frames < min_frames:
        raise ValueError(
            "not enough samples: a %dms crossfade needs at least %d frames "
            "(%d available)"
            % (crossfade_ms, min_frames, total_frames))

    frame_width = segment.frame_width
    sample_width = segment.sample_width
    channels = segment.channels
    raw = segment._data
    full_scale = float(segment.max_possible_amplitude)

    head_window = raw[:window_frames * frame_width]
    tail_window = raw[-window_frames * frame_width:]

    head_channels = _channel_windows(head_window, channels, sample_width)
    tail_channels = _channel_windows(tail_window, channels, sample_width)

    # Search the relative offset between the head and tail cut positions.
    # This is a deliberately restricted 1-D sweep rather than an exhaustive
    # 2-D search: for each offset delta == tail_start - head_start, one cut is
    # pinned to its window edge, so each offset is scored once.  Offsets run
    # from -(window-crossfade) to +(window-crossfade), and every candidate
    # only does work proportional to the crossfade, never to the segment.
    best = None
    # Each side must leave room for a full crossfade zone inside its window.
    reach = window_frames - crossfade_frames
    last_zone_sample = crossfade_frames - 1

    for delta in range(-reach, reach + 1):
        if delta < 0:
            head_start = -delta
            tail_start = 0
        else:
            head_start = 0
            tail_start = delta

        waveform_cost = 0.0
        boundary_cost = 0.0
        energy_cost = 0.0
        for head_chan, tail_chan in zip(head_channels, tail_channels):
            head_zone = _mono_slice(head_chan, head_start,
                                    crossfade_frames, sample_width)
            tail_zone = _mono_slice(tail_chan, tail_start,
                                    crossfade_frames, sample_width)

            head_rms = float(audioop.rms(head_zone, sample_width))
            tail_rms = float(audioop.rms(tail_zone, sample_width))

            # Waveform continuity over the whole crossfade zone: RMS of the
            # sample-by-sample difference.  audioop has no subtraction, so
            # negate the head then add it.
            inverted_head = audioop.mul(head_zone, sample_width, -1.0)
            difference = audioop.add(tail_zone, inverted_head, sample_width)
            waveform_cost += audioop.rms(difference, sample_width) / full_scale

            # The physical wrap joins the tail zone's final frame (the last
            # faded-out sample) to the head zone's first frame (the first
            # faded-in sample).  Score that exact boundary pair so the search
            # optimises the seam the listener actually crosses.
            seam_gap = abs(
                audioop.getsample(tail_zone, sample_width, last_zone_sample)
                - audioop.getsample(head_zone, sample_width, 0))
            boundary_cost += seam_gap / full_scale

            # Energy continuity: gap between the two zones' RMS levels.
            energy_cost += abs(tail_rms - head_rms) / full_scale

        waveform_cost /= channels
        boundary_cost /= channels
        energy_cost /= channels
        cost = (waveform_cost + boundary_cost
                + energy_weight * energy_cost)

        if best is None or cost < best[0]:
            best = (cost, head_start, tail_start)

    _, head_cut, tail_cut = best

    # Frame indices of the chosen cut inside the source segment.
    loop_start_sample = head_cut
    tail_window_start = total_frames - window_frames
    loop_end_sample = tail_window_start + tail_cut

    # Build one loop period from three contiguous pieces.  The head and tail
    # zones are ramped in opposite directions but kept apart by the untouched
    # interior rather than summed on top of each other, so there is no
    # overlap-add amplitude doubling:
    #
    #   [ head zone fading in ] + [ unmodified interior ] + [ tail fading out ]
    #    \______ C frames ______/   \_ (end-start)-C _/     \___ C frames ___/
    #
    # The wrap boundary joins the tail's last frame (exact gain 0) to the next
    # period's first head frame (exact gain 0), so both boundary samples are
    # silent and the repeat join cannot click.
    tail_zone = raw[
        loop_end_sample * frame_width:
        (loop_end_sample + crossfade_frames) * frame_width
    ]
    head_zone = raw[
        loop_start_sample * frame_width:
        (loop_start_sample + crossfade_frames) * frame_width
    ]

    faded_in = _equal_power_fade(head_zone, frame_width, sample_width, True)
    faded_out = _equal_power_fade(tail_zone, frame_width, sample_width, False)

    interior_start = loop_start_sample + crossfade_frames
    interior_end = loop_end_sample
    if interior_end <= interior_start:
        # Defensive: the up-front total_frames > 2*window check normally
        # guarantees a non-empty interior before we reach here.
        raise ValueError(
            "not enough samples: no uncrossfaded audio remains between the "
            "chosen cut points (try a shorter search window)")
    interior_frames = interior_end - interior_start

    # Assemble into one preallocated buffer instead of chaining bytes '+'
    # (which would allocate a full-size temporary copy at every step).  The
    # buffer is retained in place by the spawned segment, so the only
    # whole-output allocation is the output itself.
    output_frames = 2 * crossfade_frames + interior_frames
    loop_data = bytearray(output_frames * frame_width)
    head_bytes = crossfade_frames * frame_width
    interior_bytes = interior_frames * frame_width
    loop_data[:head_bytes] = faded_in
    # memoryview copies straight into the preallocated buffer without first
    # materialising an interior-sized temporary copy of the source.
    loop_data[head_bytes:head_bytes + interior_bytes] = memoryview(raw)[
        interior_start * frame_width:interior_end * frame_width
    ]
    loop_data[head_bytes + interior_bytes:] = faded_out
    # _spawn retains the buffer without copying it; wave/raw export both accept
    # bytearray, so keeping it here avoids another full-length allocation.
    loop_segment = segment._spawn(loop_data)

    return SeamlessLoop(
        segment=loop_segment,
        loop_start_sample=loop_start_sample,
        loop_end_sample=loop_end_sample,
        crossfade_samples=crossfade_frames,
        loop_start_ms=round(1000 * loop_start_sample / segment.frame_rate),
        loop_end_ms=round(1000 * loop_end_sample / segment.frame_rate),
    )
