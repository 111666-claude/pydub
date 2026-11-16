"""
Windowed phase correlation analysis for stereo ``AudioSegment`` objects.

Folding a stereo signal down to mono sums the left and right channels. When
the two channels are out of phase the sum partially cancels, so the folded
signal is quieter than either channel - on a phone's single speaker this can
sound like the ambience suddenly dropping out. This module measures the
normalized correlation between the channels window by window so those
regions can be located in a long recording.

The correlation used here is the zero-lag Pearson correlation between the
two channels::

    r = sum((L[i] - mean(L)) * (R[i] - mean(R))) /
        sqrt(sum((L[i] - mean(L)) ** 2) * sum((R[i] - mean(R)) ** 2))

Removing the means first means a shared DC offset (an inaudible constant
voltage, unrelated to fold-down cancellation) cannot inflate the coefficient
and hide an anti-phase signal. Apart from that, this is exactly the quantity
that governs mono fold-down: ``r = 1`` means the channels are identical up
to gain and phase, ``r = 0`` means they are uncorrelated (folding costs about
3 dB of power), and ``r = -1`` means folding cancels the signal completely.
``r`` is always clamped to the ``[-1, 1]`` range.

The analysis never modifies the ``AudioSegment`` and never materializes the
whole sample array. PCM data is read through a memory view and decoded a
block at a time inside each window, so memory use does not grow with the
length of the audio.
"""

from __future__ import division

import array
import math
import numbers
import operator
import sys
from collections import namedtuple

from .utils import register_pydub_effect, get_array_type

__all__ = [
    "PhaseCorrelationWindow",
    "phase_correlation",
    "phase_correlation_windows",
    "windowed_phase_correlation",
    "analyze_phase_correlation",
    "detect_phase_correlation",
]

try:
    _isfinite = math.isfinite
except AttributeError:  # python 2.7 has no math.isfinite
    def _isfinite(value):
        return (not math.isinf(value)) and (not math.isnan(value))

if sys.version_info >= (3, 0):
    basestring = str
    # python 2's memoryview cannot reliably slice the str objects pydub
    # stores its PCM in, so only use zero-copy views on python 3
    _MEMORYVIEW_SUPPORTED = True
else:  # pragma: no cover - python 2 support only
    _MEMORYVIEW_SUPPORTED = False


# Phase correlation is only meaningful between exactly two channels.
_REQUIRED_CHANNELS = 2

# Correlation sums are accumulated in fixed-size blocks so a long window is
# never decoded into one big array. The value is a byte count and is rounded
# down to a whole number of frames.
_PHASE_BLOCK_BYTES = 64 * 1024

# Correlations this close to the risk threshold are treated as numerical
# noise. Integer PCM quantisation and float rounding can turn a genuinely
# uncorrelated window into a correlation of a few parts in 10**8; comparing
# with an epsilon keeps such windows from being flagged as a fold risk.
_RISK_EPSILON = 1e-6

# Sentinel used to tell "argument not passed" apart from an explicit None,
# which matters for the window/window_ms and step/step_ms aliases.
_UNSET = object()

# Default analysis window, in milliseconds.
_DEFAULT_WINDOW_MS = 100

_PHASE_WINDOW_FIELDS = (
    'start_ms',
    'end_ms',
    'correlation',
    'mono_fold_risk',
    'silent',
    'partial',
)


class PhaseCorrelationWindow(namedtuple('PhaseCorrelationWindow',
                                        _PHASE_WINDOW_FIELDS)):
    """
    The result of analyzing one window of a stereo ``AudioSegment``.

    Fields:

        start_ms - start of the window in milliseconds from the beginning of
            the analyzed segment (a float)
        end_ms - end of the window in milliseconds from the beginning of the
            analyzed segment (a float)
        correlation - normalized correlation between the left and right
            channels in the window, clamped to ``[-1, 1]``. Silent windows
            have no defined correlation and report ``0.0``.
        mono_fold_risk - ``True`` when ``correlation`` is below the requested
            risk threshold (``0.0`` by default), i.e. summing the channels to
            mono reduces their level. Silent windows are never a risk.
        silent - ``True`` when both channels contain only zero samples. A
            window with one silent channel is not marked silent; its
            correlation is ``0.0`` because it is undefined.
        partial - ``True`` for a shorter window at the end of the audio when
            ``include_partial=True`` was requested.

    The fields can be read as attributes (``window.correlation``), as mapping
    items (``window['correlation']``) or by index. ``start``, ``end``,
    ``risk``, ``fold_risk`` and ``coefficient`` are aliases for ``start_ms``,
    ``end_ms``, ``mono_fold_risk`` and ``correlation`` respectively.
    """

    __slots__ = ()

    _ALIASES = {
        'start': 'start_ms',
        'end': 'end_ms',
        'risk': 'mono_fold_risk',
        'fold_risk': 'mono_fold_risk',
        'is_silent': 'silent',
        'coefficient': 'correlation',
    }

    def __getitem__(self, key):
        if isinstance(key, basestring):
            key = self._ALIASES.get(key, key)
            if key not in _PHASE_WINDOW_FIELDS:
                raise KeyError(key)
            key = _PHASE_WINDOW_FIELDS.index(key)
        return tuple.__getitem__(self, key)

    @property
    def start(self):
        """Alias for ``start_ms``."""
        return self.start_ms

    @property
    def end(self):
        """Alias for ``end_ms``."""
        return self.end_ms

    @property
    def risk(self):
        """Alias for ``mono_fold_risk``."""
        return self.mono_fold_risk

    @property
    def fold_risk(self):
        """Alias for ``mono_fold_risk``."""
        return self.mono_fold_risk

    @property
    def coefficient(self):
        """Alias for ``correlation``."""
        return self.correlation

    @property
    def duration_ms(self):
        """Length of the window in milliseconds."""
        return self.end_ms - self.start_ms

    def as_dict(self):
        """Return the window as a plain ``dict``."""
        return dict(zip(_PHASE_WINDOW_FIELDS, self))

    def __repr__(self):
        return (
            "PhaseCorrelationWindow(start_ms={0!r}, end_ms={1!r}, "
            "correlation={2!r}, mono_fold_risk={3!r}, silent={4!r}, "
            "partial={5!r})".format(*self)
        )


def _as_real(value, name):
    """Coerce ``value`` to a finite float, rejecting bools and non-numbers."""
    # bool is a subclass of int, but True/False are not durations
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(
            "{0} must be a real number of milliseconds, got {1!r}".format(
                name, value))
    value = float(value)
    if not _isfinite(value):
        raise ValueError(
            "{0} must be finite, got {1!r}".format(name, value))
    return value


def _ms_to_frames(milliseconds, frame_rate):
    """Convert milliseconds to the nearest whole sample frame."""
    # floor(x + 0.5) rounds half up, independent of the interpreter's
    # round-half-to-even behavior, so window boundaries are predictable.
    return int(math.floor(milliseconds * frame_rate / 1000.0 + 0.5))


def _data_view(data):
    """
    Return a zero-copy view of raw PCM data when the interpreter supports it.

    Pythons that cannot create a ``memoryview`` from the data (python 2)
    fall back to ordinary slices. Either way only one window (and inside it
    one block) is ever copied, never the whole segment.
    """
    if _MEMORYVIEW_SUPPORTED:
        try:
            return memoryview(data)
        except TypeError:
            pass
    return data


def _window_sums(view, start_byte, end_byte, frame_width, array_type):
    """
    Accumulate the integer sums needed to correlate the two channels of the
    frames in ``view[start_byte:end_byte]``.

    Returns a tuple ``(sums, silent)`` where ``sums`` is ``(sum_l, sum_r,
    sum_lr, sum_ll, sum_rr, frame_count)`` and ``silent`` is true when every
    sample of both channels is zero. The region is decoded one block of
    frames at a time. Sums are accumulated as Python integers, which cannot
    overflow regardless of sample width or window length.
    """
    block_frames = max(1, _PHASE_BLOCK_BYTES // frame_width)
    block_bytes = block_frames * frame_width

    sum_l = 0
    sum_r = 0
    sum_lr = 0
    sum_ll = 0
    sum_rr = 0
    frame_count = 0
    silent = True

    position = start_byte
    while position < end_byte:
        stop = min(position + block_bytes, end_byte)
        # Window boundaries are frame aligned, but keep the decoder honest
        # if a caller ever hands us a partial frame.
        stop -= (stop - position) % frame_width
        if stop <= position:
            break

        chunk = array.array(array_type)
        try:
            chunk.frombytes(view[position:stop])
        except AttributeError:  # python 2
            chunk.fromstring(view[position:stop])

        # De-interleave inside the block only; this never touches the whole
        # segment's samples.
        left = chunk[0::2]
        right = chunk[1::2]
        sum_l += sum(left)
        sum_r += sum(right)
        sum_lr += sum(map(operator.mul, left, right))
        sum_ll += sum(map(operator.mul, left, left))
        sum_rr += sum(map(operator.mul, right, right))
        frame_count += len(left)
        if silent and (any(left) or any(right)):
            silent = False
        position = stop

    return (sum_l, sum_r, sum_lr, sum_ll, sum_rr, frame_count), silent


def _correlation(sums):
    """
    Turn the integer sums into a correlation in ``[-1, 1]``.

    The means are removed before correlating so a shared DC offset does not
    masquerade as correlation. The centred cross-products are computed as
    the integer identities::

        n * sum(l*r) - sum(l) * sum(r)
        n * sum(l*l) - sum(l) ** 2
        n * sum(r*r) - sum(r) ** 2

    Python integers are arbitrary precision, so this is exact: no
    catastrophic cancellation even with a full-scale DC offset on top of
    very quiet audio. The final division is the only floating point step.

    A window in which either channel has no variance has no defined
    correlation (the divisor would be zero), so it reports ``0.0`` - the
    neutral value that does not look like cancellation.
    """
    sum_l, sum_r, sum_lr, sum_ll, sum_rr, n = sums

    # A window whose frame count came back empty (only possible for data
    # that is not a whole number of frames) has no defined correlation.
    if n == 0:
        return 0.0

    covariance_num = n * sum_lr - sum_l * sum_r
    variance_l_num = n * sum_ll - sum_l * sum_l
    variance_r_num = n * sum_rr - sum_r * sum_r

    if variance_l_num <= 0 or variance_r_num <= 0:
        return 0.0

    # sqrt(variance) * sqrt(variance) keeps the intermediate values inside
    # the float range even for very long windows with 32 bit samples.
    denominator = math.sqrt(variance_l_num) * math.sqrt(variance_r_num)
    if denominator == 0:
        return 0.0

    correlation = covariance_num / denominator
    if not _isfinite(correlation):
        return 0.0

    # Floating point rounding can produce values a hair outside the unit
    # interval for perfectly correlated signals.
    return max(-1.0, min(1.0, correlation))


def _window_ranges(start_frame, end_frame, window_frames, step_frames,
                   include_partial):
    """
    Yield ``(start, end, partial)`` frame ranges for the analysis.

    Full windows start at ``start_frame`` and advance by ``step_frames``
    while they fit entirely inside ``[start_frame, end_frame)``. When
    ``include_partial`` is true, any remaining tail is returned as one
    shorter window; if the window is longer than the selected audio the
    whole selection is returned as a single partial window.
    """
    if window_frames > end_frame - start_frame:
        if include_partial:
            yield start_frame, end_frame, True
        return

    position = start_frame
    last_start = start_frame
    while position + window_frames <= end_frame:
        yield position, position + window_frames, False
        last_start = position
        position += step_frames

    if include_partial:
        tail_start = last_start + step_frames
        if tail_start < end_frame:
            yield tail_start, end_frame, True


@register_pydub_effect("phase_correlation")
def phase_correlation(audio_segment, window=_UNSET, step=_UNSET, start=None,
                      end=None, risk_threshold=0.0, include_partial=False,
                      window_ms=_UNSET, step_ms=_UNSET):
    """
    Analyze a stereo ``AudioSegment`` window by window and return the phase
    correlation between the left and right channels for each window.

    A negative correlation means the channels partially cancel when summed
    to mono, which is what makes a stereo ambience lose level on a phone's
    single speaker. The returned windows identify the time ranges where
    that happens.

    Keyword Arguments:

        window - default: 100
            Window length in milliseconds. ``window_ms`` is accepted as an
            alias. Must be greater than zero and represent at least one
            sample frame.

        step - default: None
            Distance in milliseconds between the starts of consecutive
            windows. ``step_ms`` is accepted as an alias. ``None`` (the
            default) uses the window length, i.e. non-overlapping windows.
            Smaller values produce overlapping windows. Must be greater
            than zero and represent at least one sample frame.

        start - default: None
            Offset in milliseconds at which to begin the analysis. ``None``
            starts at the beginning of the segment.

        end - default: None
            Offset in milliseconds at which to stop. ``None`` analyzes to
            the end of the segment. Values outside the audio raise
            ``ValueError``.

        risk_threshold - default: 0.0
            A window is marked with ``mono_fold_risk=True`` when its
            correlation is below this value. The default marks only
            negatively correlated windows, for which folding to mono
            actually cancels signal. Must be between -1 and 1.

        include_partial - default: False
            When true, a final shorter window is emitted for the remaining
            tail of the audio (and a selection shorter than one window is
            analyzed as a single partial window). Partial windows have
            ``partial=True``. By default only complete windows are
            analyzed.

        window_ms, step_ms - aliases for ``window`` and ``step``; pass at
            most one spelling of each.

    Returns a list of :class:`PhaseCorrelationWindow` named tuples in
    chronological order, each carrying the window start and end time, the
    correlation (always in ``[-1, 1]``) and the mono fold risk flag.

    Edge cases all have a defined result:

        * empty audio returns an empty list (this check comes first, so the
          empty mono segment returned by ``AudioSegment.empty()`` is not an
          error);
        * audio (or a selected ``start``/``end`` range) shorter than one
          window returns an empty list, or a single partial window when
          ``include_partial=True`` - a window is never silently clipped;
        * a window in which both channels are silent reports
          ``correlation == 0.0``, ``silent == True`` and
          ``mono_fold_risk == False`` (no division by zero);
        * a window with only one silent channel reports ``correlation ==
          0.0`` (undefined) but is not marked silent, because the other
          channel still carries signal;
        * non-empty mono or multichannel segments raise ``ValueError``
          because phase correlation is only defined between two channels -
          convert with ``set_channels(2)`` first;
        * a zero/negative/non-finite window or step raises ``ValueError``
          (``TypeError`` for non-numbers), and ``start``/``end`` outside
          the audio also raise ``ValueError``.

    Supported sample widths are the ones pydub itself stores: 8 bit (one
    byte), 16 bit (two bytes) and 32 bit (four bytes, which also holds 24 bit
    source material after pydub widens it). The input segment is never
    modified and samples are decoded in blocks rather than being copied in
    their entirety.
    """
    # ---- argument validation -------------------------------------------
    # window and window_ms are two spellings of the same argument; likewise
    # step and step_ms. The sentinel tells "not passed" apart from an
    # explicit value so that passing both spellings with different values is
    # reported instead of silently picking one.
    if window is _UNSET and window_ms is _UNSET:
        window = _DEFAULT_WINDOW_MS
    elif window is _UNSET:
        window = window_ms
    elif window_ms is not _UNSET and window != window_ms:
        raise TypeError(
            "pass either window or window_ms, not both (got {0!r} and "
            "{1!r})".format(window, window_ms))

    if step is _UNSET and step_ms is _UNSET:
        step = None
    elif step is _UNSET:
        step = step_ms
    elif step_ms is not _UNSET and step != step_ms:
        raise TypeError(
            "pass either step or step_ms, not both (got {0!r} and "
            "{1!r})".format(step, step_ms))

    # A step of None means "non-overlapping windows".
    if step is None:
        step = window

    window_ms = _as_real(window, "window")
    if window_ms <= 0:
        raise ValueError(
            "window must be greater than zero milliseconds, got "
            "{0!r}".format(window_ms))

    step_ms = _as_real(step, "step")
    if step_ms <= 0:
        raise ValueError(
            "step must be greater than zero milliseconds, got "
            "{0!r}".format(step_ms))

    risk_threshold = _as_real(risk_threshold, "risk_threshold")
    if not -1.0 <= risk_threshold <= 1.0:
        raise ValueError(
            "risk_threshold must be between -1.0 and 1.0, got "
            "{0!r}".format(risk_threshold))

    start_ms = 0.0 if start is None else _as_real(start, "start")
    if start_ms < 0:
        raise ValueError(
            "start must be greater than or equal to zero, got {0!r}".format(
                start))

    frame_rate = audio_segment.frame_rate
    if not frame_rate or frame_rate <= 0:
        raise ValueError(
            "AudioSegment has an invalid frame rate {0!r}".format(frame_rate))

    total_frames = int(audio_segment.frame_count())

    # ---- empty audio ---------------------------------------------------
    # An empty segment has nothing to analyze. This check comes before the
    # channel check because AudioSegment.empty() is mono: returning an empty
    # list is more useful than raising for audio that contains no frames at
    # all, and it keeps the "no windows" result distinct from the errors for
    # bad arguments.
    if total_frames == 0:
        return []

    # ---- channel count -------------------------------------------------
    if audio_segment.channels != _REQUIRED_CHANNELS:
        raise ValueError(
            "phase correlation requires a stereo AudioSegment with exactly "
            "{0} channels, got {1}; use set_channels(2) first".format(
                _REQUIRED_CHANNELS, audio_segment.channels))

    sample_width = audio_segment.sample_width
    try:
        array_type = get_array_type(sample_width * 8)
    except KeyError:
        raise ValueError(
            "unsupported sample width {0} bytes; phase correlation supports "
            "8, 16 and 32 bit PCM (24 bit audio is stored by pydub as 32 "
            "bit)".format(sample_width))

    frame_width = audio_segment.frame_width
    if frame_width != sample_width * _REQUIRED_CHANNELS:
        raise ValueError(
            "AudioSegment frame width {0} does not match {1} channels of "
            "{2} bytes".format(frame_width, _REQUIRED_CHANNELS, sample_width))

    # ---- time range ----------------------------------------------------
    duration_ms = total_frames * 1000.0 / frame_rate
    # ``len(segment)`` is rounded to a whole millisecond, so allow half a
    # sample frame of slack past the exact duration and clamp to the audio.
    max_ms = duration_ms + 500.0 / frame_rate

    end_ms = duration_ms if end is None else _as_real(end, "end")
    if end_ms < 0:
        raise ValueError(
            "end must be greater than or equal to zero, got {0!r}".format(
                end))

    if start_ms > max_ms:
        raise ValueError(
            "start of {0!r}ms is outside the audio (duration {1!r}ms)".format(
                start_ms, duration_ms))
    if end_ms > max_ms:
        raise ValueError(
            "end of {0!r}ms is outside the audio (duration {1!r}ms)".format(
                end_ms, duration_ms))

    start_frame = min(_ms_to_frames(start_ms, frame_rate), total_frames)
    end_frame = min(_ms_to_frames(end_ms, frame_rate), total_frames)

    if end_frame <= start_frame:
        raise ValueError(
            "end ({0!r}ms) must be greater than start ({1!r}ms)".format(
                end_ms, start_ms))

    window_frames = _ms_to_frames(window_ms, frame_rate)
    if window_frames < 1:
        raise ValueError(
            "window of {0!r}ms is shorter than one sample frame at "
            "{1}Hz".format(window_ms, frame_rate))

    step_frames = _ms_to_frames(step_ms, frame_rate)
    if step_frames < 1:
        raise ValueError(
            "step of {0!r}ms is shorter than one sample frame at "
            "{1}Hz".format(step_ms, frame_rate))

    # ---- analysis ------------------------------------------------------
    view = _data_view(audio_segment._data)
    results = []

    for window_start, window_end, partial in _window_ranges(
            start_frame, end_frame, window_frames, step_frames,
            include_partial):
        sums, silent = _window_sums(
            view,
            window_start * frame_width,
            window_end * frame_width,
            frame_width,
            array_type,
        )

        correlation = _correlation(sums)
        # Silent windows cannot cancel anything and are deliberately not
        # reported as a fold risk. The epsilon keeps float noise from
        # flagging a genuinely zero correlation with the default threshold.
        risk = (not silent) and correlation < risk_threshold - _RISK_EPSILON

        results.append(PhaseCorrelationWindow(
            start_ms=window_start * 1000.0 / frame_rate,
            end_ms=window_end * 1000.0 / frame_rate,
            correlation=correlation,
            mono_fold_risk=risk,
            silent=silent,
            partial=partial,
        ))

    return results


# The same analysis is available under more descriptive names, both at
# module level and as AudioSegment methods.
windowed_phase_correlation = register_pydub_effect(
    phase_correlation, "windowed_phase_correlation")

analyze_phase_correlation = register_pydub_effect(
    phase_correlation, "analyze_phase_correlation")

detect_phase_correlation = register_pydub_effect(
    phase_correlation, "detect_phase_correlation")

phase_correlation_windows = register_pydub_effect(
    phase_correlation, "phase_correlation_windows")
