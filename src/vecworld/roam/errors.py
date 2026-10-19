"""Errors the Dream Roam service raises; each maps to an HTTP status and a stable code."""


class GameError(Exception):
    status = 500
    code = "error"


class BadInput(GameError):
    status, code = 422, "bad_input"


class NotStarted(GameError):
    status, code = 409, "not_started"


class Busy(GameError):
    status, code = 409, "busy"


class UnknownSpot(GameError):
    status, code = 409, "unknown_spot"


class GpuUnavailable(GameError):
    status, code = 503, "gpu_unavailable"


class UnknownSession(GameError, KeyError):
    status, code = 404, "unknown_session"

    def __str__(self):
        return "That session has expired, reload the page."
