"""Errors the Dream Walk service raises; each maps to an HTTP status and a stable code."""


class GameError(Exception):
    status = 500
    code = "error"


class BadInput(GameError):
    status, code = 422, "bad_input"


class NotStarted(GameError):
    status, code = 409, "not_started"


class Busy(GameError):
    status, code = 409, "busy"


class GpuUnavailable(GameError):
    status, code = 503, "gpu_unavailable"


class UnknownLabel(GameError):
    status, code = 400, "unknown_label"


class UnknownSession(GameError, KeyError):
    status, code = 404, "unknown_session"

    def __str__(self):
        return "That session has expired, reload the page."
