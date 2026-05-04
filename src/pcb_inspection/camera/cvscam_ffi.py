"""ctypes binding for libcvsCamCtrl.so (CREVIS C SDK).

The CREVIS Linux SDK ships only a C library (``libcvsCamCtrl.so`` exposing
``ST_*`` functions). This module wraps the subset that :mod:`crevis` consumes
into the Python OO surface (``CvsSystem`` / ``CvsDevice`` / ``ConvertToNumpy``)
that ``crevis.py`` was originally written against.

If the shared library cannot be loaded (e.g. dev/CI environments without the
SDK), :data:`AVAILABLE` is False and instantiating the classes raises
:class:`RuntimeError` — callers should branch on :func:`is_available` first.
"""

from __future__ import annotations

import ctypes
import logging
from ctypes import (
    POINTER,
    Structure,
    byref,
    c_bool,
    c_char_p,
    c_double,
    c_int32,
    c_int64,
    c_uint32,
    c_uint64,
    c_void_p,
    create_string_buffer,
)
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants (mirrored from cvsCamCtrl.h)
# ---------------------------------------------------------------------------

MCAM_ERR_OK = 0
MCAM_ERR_TIMEOUT = -1011
MCAM_ERR_BUFFER_TOO_SMALL = -1013

MCAM_DEVICEINFO_USER_ID = 10000
MCAM_DEVICEINFO_MODEL_NAME = 10001
MCAM_DEVICEINFO_SERIAL_NUMBER = 10002
MCAM_DEVICEINFO_DEVICE_VERSION = 10003
MCAM_DEVICEINFO_MAC_ADDRESS = 10004
MCAM_DEVICEINFO_IP_ADDRESS = 10005
MCAM_DEVICEINFO_CURRENT_SPEED = 10006

_STR_BUF_SIZE = 512


# ---------------------------------------------------------------------------
# Structures
# ---------------------------------------------------------------------------


class CVS_IMAGE(Structure):
    _fields_ = [
        ("width", c_int32),
        ("height", c_int32),
        ("step", c_int32),
        ("channels", c_int32),
        ("pImage", c_void_p),
    ]


class CVS_BUFFER(Structure):
    _fields_ = [
        ("isAllocated", c_bool),
        ("blockID", c_uint64),
        ("timestamp", c_uint64),
        ("size", c_uint32),
        ("image", CVS_IMAGE),
    ]


# ---------------------------------------------------------------------------
# Library load
# ---------------------------------------------------------------------------


def _try_load() -> ctypes.CDLL | None:
    for name in ("libcvsCamCtrl.so", "libcvsCamCtrl.so.1"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None


_lib = _try_load()
AVAILABLE = _lib is not None


def is_available() -> bool:
    """Returns True if libcvsCamCtrl.so was loaded successfully."""
    return AVAILABLE


def _bind() -> None:
    """Set argtypes/restype on every C function we call."""
    assert _lib is not None
    sigs: dict[str, tuple[Any, list[Any]]] = {
        # System
        "ST_InitSystem": (c_int32, []),
        "ST_FreeSystem": (c_int32, []),
        "ST_UpdateDevice": (c_int32, []),
        "ST_UpdateDeviceWithTimeout": (c_int32, [c_uint32]),
        "ST_GetAvailableCameraNum": (c_int32, [POINTER(c_uint32)]),
        "ST_GetEnumDeviceID": (c_int32, [c_uint32, c_char_p, POINTER(c_uint32)]),
        "ST_GetEnumDeviceInfo": (
            c_int32,
            [c_uint32, c_int32, c_char_p, POINTER(c_uint32)],
        ),
        # Device lifecycle
        "ST_OpenDevice": (c_int32, [c_uint32, POINTER(c_int32), c_bool]),
        "ST_CloseDevice": (c_int32, [c_int32]),
        # Acquisition
        "ST_AcqStart": (c_int32, [c_int32]),
        "ST_AcqStop": (c_int32, [c_int32]),
        "ST_SetGrabTimeout": (c_int32, [c_int32, c_uint32]),
        "ST_SetBufferCount": (c_int32, [c_int32, c_uint32]),
        "ST_GrabImage": (c_int32, [c_int32, POINTER(CVS_BUFFER)]),
        "ST_SingleGrabImage": (c_int32, [c_int32, POINTER(CVS_BUFFER)]),
        "ST_InitBuffer": (c_int32, [c_int32, POINTER(CVS_BUFFER), c_int32]),
        "ST_FreeBuffer": (c_int32, [POINTER(CVS_BUFFER)]),
        # GenICam node access
        "ST_SetIntReg": (c_int32, [c_int32, c_char_p, c_int64]),
        "ST_GetIntReg": (c_int32, [c_int32, c_char_p, POINTER(c_int64)]),
        "ST_SetFloatReg": (c_int32, [c_int32, c_char_p, c_double]),
        "ST_GetFloatReg": (c_int32, [c_int32, c_char_p, POINTER(c_double)]),
        "ST_SetBoolReg": (c_int32, [c_int32, c_char_p, c_bool]),
        "ST_GetBoolReg": (c_int32, [c_int32, c_char_p, POINTER(c_bool)]),
        "ST_SetEnumReg": (c_int32, [c_int32, c_char_p, c_char_p]),
        "ST_GetEnumReg": (c_int32, [c_int32, c_char_p, c_char_p, POINTER(c_uint32)]),
        "ST_SetStrReg": (c_int32, [c_int32, c_char_p, c_char_p]),
        "ST_GetStrReg": (c_int32, [c_int32, c_char_p, c_char_p, POINTER(c_uint32)]),
        "ST_SetCmdReg": (c_int32, [c_int32, c_char_p]),
        "ST_GetIntRegRange": (
            c_int32,
            [c_int32, c_char_p, POINTER(c_int64), POINTER(c_int64), POINTER(c_int64)],
        ),
        "ST_GetFloatRegRange": (
            c_int32,
            [c_int32, c_char_p, POINTER(c_double), POINTER(c_double)],
        ),
        "ST_GetEnumEntrySize": (c_int32, [c_int32, c_char_p, POINTER(c_int32)]),
        "ST_GetEnumEntryValue": (
            c_int32,
            [c_int32, c_char_p, c_int32, c_char_p, POINTER(c_uint32)],
        ),
        "ST_GetLastErrorDescription": (c_char_p, [c_int32]),
    }
    for fname, (restype, argtypes) in sigs.items():
        fn = getattr(_lib, fname)
        fn.restype = restype
        fn.argtypes = argtypes


if AVAILABLE:
    _bind()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class CvsCamError(RuntimeError):
    """Raised when an ST_* call returns a non-zero error code."""

    def __init__(self, fn: str, code: int, detail: str = "") -> None:
        msg = f"{fn} failed: code={code}"
        if detail:
            msg += f" ({detail})"
        super().__init__(msg)
        self.code = code


def _check(code: int, fn: str, h_device: int | None = None) -> None:
    if code == MCAM_ERR_OK:
        return
    detail = ""
    if h_device is not None and AVAILABLE:
        try:
            ptr = _lib.ST_GetLastErrorDescription(h_device)
            if ptr:
                detail = ptr.decode("utf-8", "replace")
        except Exception:
            pass
    raise CvsCamError(fn, code, detail)


def _require_lib() -> None:
    if not AVAILABLE:
        raise RuntimeError(
            "libcvsCamCtrl.so not loadable. Install the CREVIS cvsCam SDK first."
        )


def _read_string(
    fetch: Any,
    *prefix_args: Any,
) -> str:
    """Call a (..., char* buf, uint32* size)-shaped ST_ getter and decode."""
    size = c_uint32(_STR_BUF_SIZE)
    buf = create_string_buffer(_STR_BUF_SIZE)
    code = fetch(*prefix_args, buf, byref(size))
    if code == MCAM_ERR_BUFFER_TOO_SMALL:
        size = c_uint32(size.value)
        buf = create_string_buffer(size.value)
        code = fetch(*prefix_args, buf, byref(size))
    _check(code, fetch.__name__)
    return buf.value.decode("utf-8", "replace")


# ---------------------------------------------------------------------------
# Public API (matches the OO surface crevis.py was written against)
# ---------------------------------------------------------------------------


class CvsSystem:
    """Discovery scope: ST_InitSystem → enumerate → ST_FreeSystem."""

    def __init__(self) -> None:
        _require_lib()
        _check(_lib.ST_InitSystem(), "ST_InitSystem")
        self._initialized = True

    def UpdateDevice(self, timeout_ms: int | None = None) -> None:
        if timeout_ms is None:
            _check(_lib.ST_UpdateDevice(), "ST_UpdateDevice")
        else:
            _check(
                _lib.ST_UpdateDeviceWithTimeout(c_uint32(int(timeout_ms))),
                "ST_UpdateDeviceWithTimeout",
            )

    def GetAvailableCameraNum(self) -> int:
        n = c_uint32(0)
        _check(_lib.ST_GetAvailableCameraNum(byref(n)), "ST_GetAvailableCameraNum")
        return int(n.value)

    def GetEnumDeviceID(self, index: int) -> str:
        return _read_string(_lib.ST_GetEnumDeviceID, c_uint32(index))

    def GetEnumDeviceInfo(self, index: int, cmd: int) -> str:
        return _read_string(
            _lib.ST_GetEnumDeviceInfo, c_uint32(index), c_int32(cmd)
        )

    def Free(self) -> None:
        if self._initialized:
            _check(_lib.ST_FreeSystem(), "ST_FreeSystem")
            self._initialized = False

    def __del__(self) -> None:
        try:
            self.Free()
        except Exception:
            pass


class CvsDevice:
    """Single camera handle: open/close + GenICam node access + grab."""

    def __init__(self, index: int) -> None:
        _require_lib()
        self._index = int(index)
        self._h = c_int32(-1)
        self._opened = False
        self._buffer = CVS_BUFFER()
        self._buffer_inited = False

    # -- lifecycle --------------------------------------------------------

    def Open(self) -> None:
        _check(
            _lib.ST_OpenDevice(c_uint32(self._index), byref(self._h), c_bool(False)),
            "ST_OpenDevice",
        )
        self._opened = True
        # Pre-allocate buffer; SDK auto-resizes on grab.
        rc = _lib.ST_InitBuffer(self._h, byref(self._buffer), c_int32(0))
        _check(rc, "ST_InitBuffer", self._h.value)
        self._buffer_inited = True

    def Close(self) -> None:
        if self._buffer_inited:
            try:
                _lib.ST_FreeBuffer(byref(self._buffer))
            except Exception:
                pass
            self._buffer_inited = False
        if self._opened:
            _check(_lib.ST_CloseDevice(self._h), "ST_CloseDevice")
            self._opened = False

    # -- acquisition ------------------------------------------------------

    def AcqStart(self) -> None:
        _check(_lib.ST_AcqStart(self._h), "ST_AcqStart", self._h.value)

    def AcqStop(self) -> None:
        _check(_lib.ST_AcqStop(self._h), "ST_AcqStop", self._h.value)

    def SetGrabTimeout(self, ms: int) -> None:
        _check(
            _lib.ST_SetGrabTimeout(self._h, c_uint32(int(ms))),
            "ST_SetGrabTimeout",
            self._h.value,
        )

    def SetBufferCount(self, count: int) -> None:
        _check(
            _lib.ST_SetBufferCount(self._h, c_uint32(int(count))),
            "ST_SetBufferCount",
            self._h.value,
        )

    def GrabImage(self) -> CVS_BUFFER:
        _check(
            _lib.ST_GrabImage(self._h, byref(self._buffer)),
            "ST_GrabImage",
            self._h.value,
        )
        return self._buffer

    def SingleGrabImage(self) -> CVS_BUFFER:
        _check(
            _lib.ST_SingleGrabImage(self._h, byref(self._buffer)),
            "ST_SingleGrabImage",
            self._h.value,
        )
        return self._buffer

    # -- GenICam scalars --------------------------------------------------

    def GetIntReg(self, name: str) -> int:
        v = c_int64(0)
        _check(
            _lib.ST_GetIntReg(self._h, name.encode(), byref(v)),
            "ST_GetIntReg",
            self._h.value,
        )
        return int(v.value)

    def SetIntReg(self, name: str, value: int) -> None:
        _check(
            _lib.ST_SetIntReg(self._h, name.encode(), c_int64(int(value))),
            "ST_SetIntReg",
            self._h.value,
        )

    def GetFloatReg(self, name: str) -> float:
        v = c_double(0.0)
        _check(
            _lib.ST_GetFloatReg(self._h, name.encode(), byref(v)),
            "ST_GetFloatReg",
            self._h.value,
        )
        return float(v.value)

    def SetFloatReg(self, name: str, value: float) -> None:
        _check(
            _lib.ST_SetFloatReg(self._h, name.encode(), c_double(float(value))),
            "ST_SetFloatReg",
            self._h.value,
        )

    def GetBoolReg(self, name: str) -> bool:
        v = c_bool(False)
        _check(
            _lib.ST_GetBoolReg(self._h, name.encode(), byref(v)),
            "ST_GetBoolReg",
            self._h.value,
        )
        return bool(v.value)

    def SetBoolReg(self, name: str, value: bool) -> None:
        _check(
            _lib.ST_SetBoolReg(self._h, name.encode(), c_bool(bool(value))),
            "ST_SetBoolReg",
            self._h.value,
        )

    def GetEnumReg(self, name: str) -> str:
        return _read_string(_lib.ST_GetEnumReg, self._h, name.encode())

    def SetEnumReg(self, name: str, value: str) -> None:
        _check(
            _lib.ST_SetEnumReg(self._h, name.encode(), value.encode()),
            "ST_SetEnumReg",
            self._h.value,
        )

    def GetStrReg(self, name: str) -> str:
        return _read_string(_lib.ST_GetStrReg, self._h, name.encode())

    def SetStrReg(self, name: str, value: str) -> None:
        _check(
            _lib.ST_SetStrReg(self._h, name.encode(), value.encode()),
            "ST_SetStrReg",
            self._h.value,
        )

    def SetCmdReg(self, name: str) -> None:
        _check(
            _lib.ST_SetCmdReg(self._h, name.encode()),
            "ST_SetCmdReg",
            self._h.value,
        )

    # -- ranges / enum introspection -------------------------------------

    def GetIntRegRange(self, name: str) -> tuple[int, int, int]:
        mn = c_int64(0)
        mx = c_int64(0)
        inc = c_int64(0)
        _check(
            _lib.ST_GetIntRegRange(
                self._h, name.encode(), byref(mn), byref(mx), byref(inc)
            ),
            "ST_GetIntRegRange",
            self._h.value,
        )
        return int(mn.value), int(mx.value), int(inc.value)

    def GetFloatRegRange(self, name: str) -> tuple[float, float]:
        mn = c_double(0.0)
        mx = c_double(0.0)
        _check(
            _lib.ST_GetFloatRegRange(self._h, name.encode(), byref(mn), byref(mx)),
            "ST_GetFloatRegRange",
            self._h.value,
        )
        return float(mn.value), float(mx.value)

    def GetEnumEntrySize(self, name: str) -> int:
        n = c_int32(0)
        _check(
            _lib.ST_GetEnumEntrySize(self._h, name.encode(), byref(n)),
            "ST_GetEnumEntrySize",
            self._h.value,
        )
        return int(n.value)

    def GetEnumEntryValue(self, name: str, idx: int) -> str:
        return _read_string(
            _lib.ST_GetEnumEntryValue, self._h, name.encode(), c_int32(idx)
        )

    def __del__(self) -> None:
        try:
            self.Close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Buffer → numpy conversion
# ---------------------------------------------------------------------------


def ConvertToNumpy(buf: CVS_BUFFER) -> np.ndarray:
    """Copy a freshly-grabbed CVS_BUFFER into a numpy array.

    Layout is inferred from ``image.step / image.width`` (bytes-per-pixel) and
    ``image.channels``:

    * 1 channel, 1 BPP  → ``(H, W)``  ``uint8``     (Mono8 / Bayer*8 raw)
    * 1 channel, 2 BPP  → ``(H, W)``  ``uint16``    (Mono16 / Bayer*16 raw)
    * 3 channels        → ``(H, W, 3)`` ``uint8``   (BGR8 / RGB8)
    * 3 channels, 2 BPP → ``(H, W, 3)`` ``uint16``  (BGR16 / RGB16)

    The returned array owns its memory — the SDK is free to reuse the buffer.
    """
    img = buf.image
    if img.pImage == 0 or img.width <= 0 or img.height <= 0:
        raise RuntimeError(
            f"Empty image buffer (w={img.width}, h={img.height}, ptr={img.pImage})"
        )

    h, w, step, ch = img.height, img.width, img.step, max(img.channels, 1)
    bpp_total = step // h if h > step else step  # step is bytes-per-row
    # step is always bytes-per-row; bytes-per-pixel = step / width
    bpp = max(step // max(w, 1), 1)
    per_channel_bytes = max(bpp // ch, 1)
    dtype = np.uint16 if per_channel_bytes >= 2 else np.uint8

    nbytes = step * h
    raw = (ctypes.c_uint8 * nbytes).from_address(img.pImage)
    flat = np.frombuffer(raw, dtype=np.uint8, count=nbytes).copy()

    if ch == 1:
        if per_channel_bytes == 2:
            arr = flat.view(np.uint16).reshape(h, w)
        else:
            # row stride may exceed w (alignment); slice to logical width
            arr = flat.reshape(h, step)[:, :w].astype(dtype, copy=False)
    else:
        row_pixels = step // bpp
        arr = flat.view(dtype).reshape(h, row_pixels, ch)[:, :w, :]
        arr = np.ascontiguousarray(arr)

    return arr
