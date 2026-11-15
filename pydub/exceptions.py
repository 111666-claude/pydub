class PydubException(Exception):
    """
    Base class for any Pydub exception
    """


class TooManyMissingFrames(PydubException):
    pass


class InvalidDuration(PydubException):
    pass


class InvalidTag(PydubException):
    pass


class InvalidID3TagVersion(PydubException):
    pass


class CouldntDecodeError(PydubException):
    pass


class CouldntEncodeError(PydubException):
    pass


class MissingAudioParameter(PydubException):
    pass


class MissingConverterError(PydubException):
    """Raised when an operation needs ffmpeg/avconv but none can be found."""
    pass


class InvalidSpeed(PydubException, ValueError):
    """Raised when a playback speed is not a valid, supported number."""
    pass


class CouldntProcessError(PydubException):
    """Raised when ffmpeg/avconv fails to process the audio segment."""
    pass
