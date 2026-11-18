"""
A drop-in replacement for the stdlib ``audioop`` module (removed in
Python 3.13), covering only the functions pydub uses.

Unlike the original ctypes/struct based implementation, every function here
works on ``array.array`` objects so that the per-sample work is done in C.
The clipping and rounding behaviour matches the stdlib module: arithmetic
results are truncated with ``int()`` and saturated to the sample range.
"""
import math
import array
try:
    from fractions import gcd
except ImportError:  # Python 3.9+
    from math import gcd

# keep references to the builtins: this module defines its own max() and
# minmax() which would otherwise shadow them inside the helper functions
_builtin_max = max
_builtin_min = min


class error(Exception):
    pass


_ARRAY_TYPES = {1: 'b', 2: 'h', 4: 'i'}
_MIN_VALUES = {1: -0x80, 2: -0x8000, 4: -0x80000000}
_MAX_VALUES = {1: 0x7f, 2: 0x7fff, 4: 0x7fffffff}


def _check_size(size):
    if size not in (1, 2, 4):
        raise error("Size should be 1, 2 or 4")


def _check_params(length, size):
    _check_size(size)
    if length % size != 0:
        raise error("not a whole number of frames")


def _as_array(cp, size):
    _check_params(len(cp), size)
    return array.array(_ARRAY_TYPES[size], bytes(cp))


def _to_bytes(arr):
    return arr.tobytes()


def _clip(values, size):
    """Saturate an iterable of ints to the sample range of ``size``."""
    minval = _MIN_VALUES[size]
    maxval = _MAX_VALUES[size]
    return [_builtin_min(_builtin_max(v, minval), maxval) for v in values]


def getsample(cp, size, i):
    _check_params(len(cp), size)
    samples = _as_array(cp, size)
    if not (0 <= i < len(samples)):
        raise error("Index out of range")
    return samples[i]


def max(cp, size):
    samples = _as_array(cp, size)
    if not len(samples):
        return 0
    return _builtin_max(abs(sample) for sample in samples)


def minmax(cp, size):
    samples = _as_array(cp, size)
    max_sample, min_sample = 0, 0
    for sample in samples:
        max_sample = _builtin_max(sample, max_sample)
        min_sample = _builtin_min(sample, min_sample)
    return min_sample, max_sample


def avg(cp, size):
    samples = _as_array(cp, size)
    if not len(samples):
        return 0
    return sum(samples) // len(samples)


def rms(cp, size):
    samples = _as_array(cp, size)
    if not len(samples):
        return 0
    return int(math.sqrt(sum(sample * sample for sample in samples)
                         // len(samples)))


def _sum2(cp1, cp2, length):
    size = 2
    a1 = _as_array(cp1, size)
    a2 = _as_array(cp2, size)
    return sum(a1[i] * a2[i] for i in range(length))


def findfit(cp1, cp2):
    size = 2

    if len(cp1) % 2 != 0 or len(cp2) % 2 != 0:
        raise error("Strings should be even-sized")

    if len(cp1) < len(cp2):
        raise error("First sample should be longer")

    a1 = _as_array(cp1, size)
    a2 = _as_array(cp2, size)
    len1 = len(a1)
    len2 = len(a2)

    sum_ri_2 = _sum2(cp2, cp2, len2)
    sum_aij_2 = _sum2(cp1, cp1, len2)
    sum_aij_ri = _sum2(cp1, cp2, len2)

    result = (sum_ri_2 * sum_aij_2 - sum_aij_ri * sum_aij_ri) // sum_aij_2

    best_result = result
    best_i = 0

    for i in range(1, len1 - len2 + 1):
        aj_m1 = a1[i - 1]
        aj_lm1 = a1[i + len2 - 1]

        sum_aij_2 += aj_lm1 ** 2 - aj_m1 ** 2
        sum_aij_ri = _sum2(a1[i * size:].tobytes(), cp2, len2)

        result = (sum_ri_2 * sum_aij_2 - sum_aij_ri * sum_aij_ri) // sum_aij_2

        if result < best_result:
            best_result = result
            best_i = i

    factor = _sum2(a1[best_i * size:].tobytes(), cp2, len2) // sum_ri_2

    return best_i, factor


def findfactor(cp1, cp2):
    size = 2

    if len(cp1) % 2 != 0:
        raise error("Strings should be even-sized")

    if len(cp1) != len(cp2):
        raise error("Samples should be same size")

    a1 = _as_array(cp1, size)
    a2 = _as_array(cp2, size)
    sample_count = len(a1)

    sum_ri_2 = _sum2(cp2, cp2, sample_count)
    sum_aij_ri = _sum2(cp1, cp2, sample_count)

    return sum_aij_ri // sum_ri_2


def findmax(cp, len2):
    size = 2
    a = _as_array(cp, size)
    sample_count = len(a)

    if len(cp) % 2 != 0:
        raise error("Strings should be even-sized")

    if len2 < 0 or sample_count < len2:
        raise error("Input sample should be longer")

    if sample_count == 0:
        return 0

    result = _sum2(cp, cp, len2)
    best_result = result
    best_i = 0

    for i in range(1, sample_count - len2 + 1):
        sample_leaving_window = a[i - 1]
        sample_entering_window = a[i + len2 - 1]

        result -= sample_leaving_window ** 2
        result += sample_entering_window ** 2

        if result > best_result:
            best_result = result
            best_i = i

    return best_i


def avgpp(cp, size):
    samples = _as_array(cp, size)
    sample_count = len(samples)

    prevextremevalid = False
    prevextreme = None
    avg = 0
    nextreme = 0

    prevval = samples[0]
    val = samples[1]

    prevdiff = val - prevval

    for i in range(1, sample_count):
        val = samples[i]
        diff = val - prevval

        if diff * prevdiff < 0:
            if prevextremevalid:
                avg += abs(prevval - prevextreme)
                nextreme += 1

            prevextremevalid = True
            prevextreme = prevval

        prevval = val
        if diff != 0:
            prevdiff = diff

    if nextreme == 0:
        return 0

    return avg // nextreme


def maxpp(cp, size):
    samples = _as_array(cp, size)
    sample_count = len(samples)

    prevextremevalid = False
    prevextreme = None
    max_diff = 0

    prevval = samples[0]
    val = samples[1]

    prevdiff = val - prevval

    for i in range(1, sample_count):
        val = samples[i]
        diff = val - prevval

        if diff * prevdiff < 0:
            if prevextremevalid:
                extremediff = abs(prevval - prevextreme)
                if extremediff > max_diff:
                    max_diff = extremediff
            prevextremevalid = True
            prevextreme = prevval

        prevval = val
        if diff != 0:
            prevdiff = diff

    return max_diff


def cross(cp, size):
    samples = _as_array(cp, size)

    crossings = 0
    last_sample = 0
    for sample in samples:
        if sample <= 0 < last_sample or sample >= 0 > last_sample:
            crossings += 1
        last_sample = sample

    return crossings


def mul(cp, size, factor):
    samples = _as_array(cp, size)
    result = array.array(samples.typecode,
                         _clip((int(sample * factor) for sample in samples),
                               size))
    return _to_bytes(result)


def tomono(cp, size, fac1, fac2):
    samples = _as_array(cp, size)
    mixed = (int(samples[i] * fac1 + samples[i + 1] * fac2)
             for i in range(0, len(samples), 2))
    result = array.array(samples.typecode, _clip(mixed, size))
    return _to_bytes(result)


def tostereo(cp, size, fac1, fac2):
    samples = _as_array(cp, size)
    result = array.array(samples.typecode)
    # extend with pre-clipped left/right pairs, two output frames per input
    left = _clip((int(sample * fac1) for sample in samples), size)
    right = _clip((int(sample * fac2) for sample in samples), size)
    for l_sample, r_sample in zip(left, right):
        result.append(l_sample)
        result.append(r_sample)
    return _to_bytes(result)


def add(cp1, cp2, size):
    _check_params(len(cp1), size)

    if len(cp1) != len(cp2):
        raise error("Lengths should be the same")

    a1 = _as_array(cp1, size)
    a2 = _as_array(cp2, size)
    result = array.array(a1.typecode,
                         _clip((s1 + s2 for s1, s2 in zip(a1, a2)), size))
    return _to_bytes(result)


def bias(cp, size, bias):
    samples = _as_array(cp, size)
    minval = _MIN_VALUES[size]
    maxval = _MAX_VALUES[size]
    bits = size * 8
    offset = 2 ** (bits - 1)

    def overflow(val):
        if minval <= val <= maxval:
            return val
        return ((val + offset) % (2 ** bits)) - offset

    result = array.array(samples.typecode,
                         (overflow(sample + bias) for sample in samples))
    return _to_bytes(result)


def reverse(cp, size):
    samples = _as_array(cp, size)
    samples.reverse()
    return _to_bytes(samples)


def lin2lin(cp, size, size2):
    _check_params(len(cp), size)
    _check_size(size2)

    if size == size2:
        return bytes(cp)

    samples = _as_array(cp, size)
    minval = _MIN_VALUES[size2]
    maxval = _MAX_VALUES[size2]
    bits = size2 * 8
    offset = 2 ** (bits - 1)

    def overflow(val):
        if minval <= val <= maxval:
            return val
        return ((val + offset) % (2 ** bits)) - offset

    if size < size2:
        shift = 4 * size2 // size
        converted = (overflow(sample << shift) for sample in samples)
    else:
        shift = 4 * size // size2
        converted = (overflow(sample >> shift) for sample in samples)

    result = array.array(_ARRAY_TYPES[size2], converted)
    return _to_bytes(result)


def ratecv(cp, size, nchannels, inrate, outrate, state, weightA=1, weightB=0):
    _check_params(len(cp), size)
    if nchannels < 1:
        raise error("# of channels should be >= 1")

    bytes_per_frame = size * nchannels
    samples = _as_array(cp, size)
    frame_count = len(samples) // nchannels

    if bytes_per_frame // nchannels != size:
        raise OverflowError("width * nchannels too big for a C int")

    if weightA < 1 or weightB < 0:
        raise error("weightA should be >= 1, weightB should be >= 0")

    if len(cp) % bytes_per_frame != 0:
        raise error("not a whole number of frames")

    if inrate <= 0 or outrate <= 0:
        raise error("sampling rate not > 0")

    d = gcd(inrate, outrate)
    inrate //= d
    outrate //= d

    prev_i = [0] * nchannels
    cur_i = [0] * nchannels

    if state is None:
        d = -outrate
    else:
        d, samps = state

        if len(samps) != nchannels:
            raise error("illegal state argument")

        prev_i, cur_i = zip(*samps)
        prev_i, cur_i = list(prev_i), list(cur_i)

    q = frame_count // inrate
    ceiling = (q + 1) * outrate
    result = array.array(samples.typecode, [0]) * (ceiling * nchannels)

    sample_iter = iter(samples)
    in_frame = 0
    out_i = 0
    minval = _MIN_VALUES[size]
    maxval = _MAX_VALUES[size]
    bits = size * 8
    offset = 2 ** (bits - 1)

    def overflow(val):
        val = int(val)
        if minval <= val <= maxval:
            return val
        return ((val + offset) % (2 ** bits)) - offset

    while True:
        while d < 0:
            if frame_count == 0:
                samps = tuple(zip(prev_i, cur_i))
                retval = result[:out_i * nchannels].tobytes()
                return retval, (d, samps)

            for chan in range(nchannels):
                prev_i[chan] = cur_i[chan]
                cur_i[chan] = next(sample_iter)

                cur_i[chan] = (
                    (weightA * cur_i[chan] + weightB * prev_i[chan])
                    // (weightA + weightB)
                )

            frame_count -= 1
            in_frame += 1
            d += outrate

        while d >= 0:
            for chan in range(nchannels):
                cur_o = (
                    (prev_i[chan] * d + cur_i[chan] * (outrate - d))
                    // outrate
                )
                result[out_i * nchannels + chan] = overflow(cur_o)
            out_i += 1
            d -= inrate


def lin2ulaw(cp, size):
    raise NotImplementedError()


def ulaw2lin(cp, size):
    raise NotImplementedError()


def lin2alaw(cp, size):
    raise NotImplementedError()


def alaw2lin(cp, size):
    raise NotImplementedError()


def lin2adpcm(cp, size, state):
    raise NotImplementedError()


def adpcm2lin(cp, size, state):
    raise NotImplementedError()
