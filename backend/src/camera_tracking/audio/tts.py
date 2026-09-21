"""Vietnamese text-to-speech and AAC encoding helpers."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path


DEFAULT_VOICE = "vi-VN-NamMinhNeural"


async def _save_edge_audio(text: str, voice: str, destination: Path) -> None:
    import edge_tts

    communicator = edge_tts.Communicate(text, voice=voice)
    await communicator.save(str(destination))


def synthesize_aac(
    text: str,
    *,
    voice: str = DEFAULT_VOICE,
    sample_rate: int = 16_000,
    bit_rate: int = 32_000,
) -> list[bytes]:
    """Synthesize Vietnamese speech and return raw AAC access units."""
    try:
        import av
        from av.audio.resampler import AudioResampler
    except ImportError as error:
        raise RuntimeError(
            "Camera TTS requires optional dependencies: pip install edge-tts av"
        ) from error

    with tempfile.TemporaryDirectory(prefix="camera-tts-") as directory:
        source = Path(directory) / "speech.mp3"
        asyncio.run(_save_edge_audio(text, voice, source))
        decoder = av.open(str(source))
        encoder = av.CodecContext.create("aac", "w")
        encoder.sample_rate = sample_rate
        encoder.layout = "mono"
        encoder.format = "fltp"
        encoder.bit_rate = bit_rate
        encoder.open()
        resampler = AudioResampler(
            format="fltp",
            layout="mono",
            rate=sample_rate,
        )
        packets: list[bytes] = []
        try:
            for frame in decoder.decode(audio=0):
                converted = resampler.resample(frame)
                for resampled in converted if isinstance(converted, list) else [converted]:
                    if resampled is None:
                        continue
                    packets.extend(bytes(packet) for packet in encoder.encode(resampled))
            packets.extend(bytes(packet) for packet in encoder.encode(None))
        finally:
            decoder.close()
        if not packets:
            raise RuntimeError("TTS encoder produced no AAC audio")
        return packets
