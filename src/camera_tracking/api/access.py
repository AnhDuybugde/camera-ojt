"""Compatibility shim — canonical home is backend.app.api.access."""
from backend.app.api.access import authorized, signed_stream_url

__all__ = ["authorized", "signed_stream_url"]
