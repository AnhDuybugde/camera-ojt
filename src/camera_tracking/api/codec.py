"""JSON codec for existing application method arguments (never pickle)."""
import base64
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, time


def encode(value):
    if isinstance(value, bytes):
        return {"$bytes": base64.b64encode(value).decode("ascii")}
    if isinstance(value, (datetime, date, time)):
        return {"$" + type(value).__name__: value.isoformat()}
    if is_dataclass(value):
        return encode(asdict(value))
    if isinstance(value, dict):
        return {k: encode(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [encode(v) for v in value]
    return value


def decode(value):
    if isinstance(value, list):
        return [decode(v) for v in value]
    if isinstance(value, dict):
        if set(value) == {"$bytes"}:
            return base64.b64decode(value["$bytes"], validate=True)
        for cls in (datetime, date, time):
            if set(value) == {"$" + cls.__name__}:
                return cls.fromisoformat(value["$" + cls.__name__])
        return {k: decode(v) for k, v in value.items()}
    return value
