"""Compatibility shim — canonical home is backend.app.camera.capture."""
from backend.app.camera.capture import FrameHub, OpenCVFrameSource

__all__ = ["FrameHub", "OpenCVFrameSource"]
