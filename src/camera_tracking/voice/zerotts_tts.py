"""Text-to-speech tieng Viet bang ZeroTTS (zeroweight-ai/ZeroTTS, MIT).

Dung de tao san WAV chao (offline sau lan dau tai weights ~900MB):
- unknown  -> 1 file: cau unknown_phrase (mac dinh "Xin chào quý khách").
- employee -> 1 file/cau: "Xin chào <ten>" (1 cau = 1 file duy nhat nen khi
  phat qua P2P trong 1 VisualTalk session khong the ngat quang giua chung).

API goc (pip install zerotts):
    from zerotts import ZeroTTS
    tts = ZeroTTS.from_pretrained("zeroweight-ai/ZeroTTS")
    audio = tts.synthesize("Xin chào các bạn.", voice="maichi")  # (1, n) float32 48kHz
    tts.save_audio(audio, "out.wav")
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DEFAULT_MODEL = "zeroweight-ai/ZeroTTS"
DEFAULT_VOICE = "maichi"


@dataclass
class ZeroTTSBackend:
    """Lazy wrapper quanh ``zerotts.ZeroTTS`` (CPU, 48 kHz)."""

    model: str = DEFAULT_MODEL
    voice: str = DEFAULT_VOICE
    _tts: object = field(default=None, init=False, repr=False)

    def _load(self):  # type: ignore[no-untyped-def]
        if self._tts is None:
            try:
                from zerotts import ZeroTTS
            except ImportError as error:
                raise RuntimeError(
                    "zerotts chua cai. Chay: python -m pip install zerotts "
                    "(lan dau can mang de tai weights ~900MB)"
                ) from error
            self._tts = ZeroTTS.from_pretrained(self.model)
        return self._tts

    def list_voices(self) -> list[str]:
        return list(self._load().list_voices())

    @property
    def sample_rate(self) -> int:
        return int(self._load().sample_rate)

    def synthesize(self, text: str) -> np.ndarray:
        """Tra mono float32 shape (n,) o sample_rate goc (48 kHz)."""
        tts = self._load()
        audio = tts.synthesize(text, voice=self.voice)
        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        return samples

    def save_wav(self, text: str, path: str | Path) -> Path:
        """Synthesize 1 cau hoan chinh -> 1 file WAV (khong ghep noi)."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        samples = self.synthesize(text)
        try:
            import soundfile as sf
        except ImportError as error:
            raise RuntimeError(
                "soundfile chua cai (di kem zerotts). Chay: pip install zerotts"
            ) from error
        sf.write(str(out), samples, self.sample_rate)
        return out


def concat_wavs_gapless(parts: list[str | Path], out: str | Path) -> Path:
    """Noi nhieu WAV thanh 1 file duy nhat, sample-accurate, khong chen silence.

    Chi dung khi bat buoc ghep 2 doan roi (vi du "Xin chào" + ten). Khuyen
    nghi: synthesize ca cau 1 lan (ZeroTTSBackend.save_wav) de tu nhien nhat.
    Tat ca parts phai cung sample rate; khac rate -> bao loi ro rang.
    """
    destination = Path(out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not parts:
        raise ValueError("concat_wavs_gapless needs at least one part")
    try:
        import soundfile as sf
    except ImportError as error:
        raise RuntimeError(
            "soundfile chua cai (di kem zerotts). Chay: pip install zerotts"
        ) from error
    chunks: list[np.ndarray] = []
    sample_rate: int | None = None
    for part in parts:
        samples, rate = sf.read(str(part), dtype="float32", always_2d=False)
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        if sample_rate is None:
            sample_rate = int(rate)
        elif int(rate) != sample_rate:
            raise ValueError(
                f"Sample rate mismatch: {part} is {rate} Hz, "
                f"expected {sample_rate} Hz (resample truoc khi noi)"
            )
        chunks.append(samples)
    sf.write(str(destination), np.concatenate(chunks), sample_rate)
    return destination


def load_phrase_files(greeting_dir: str | Path) -> dict[str, Path]:
    """Doc manifest.json -> {cau chao nguyen van: file wav ton tai}.

    Manifest do scripts/build_greeting_wavs.py sinh. File thieu/rong thi bo
    qua (greeter fallback TTS runtime). Khong bao gio raise.
    """
    import json

    phrase_files: dict[str, Path] = {}
    try:
        root = Path(greeting_dir)
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        entries = manifest.get("phrases") or {}
        for phrase, filename in entries.items():
            wav = root / str(filename)
            if wav.is_file() and wav.stat().st_size > 0:
                phrase_files[str(phrase)] = wav
    except (OSError, ValueError, AttributeError):
        return {}
    return phrase_files


def add_display_name_aliases(
    phrase_files: dict[str, Path], display_names: list[str]
) -> dict[str, Path]:
    """Map ten rut gon cua gallery vao WAV ho-ten day du trong manifest.

    Vi du gallery hien ``Anh Duy`` trong khi WAV da tao cho
    ``Xin chào Lê Hồ Anh Duy``. So khop bo dau/case va theo hau to ten,
    giup dung WAV offline thay vi roi sang Edge TTS can mang.
    """
    output = dict(phrase_files)

    def normalized(value: str) -> str:
        ascii_text = unicodedata.normalize("NFKD", value)
        ascii_text = "".join(ch for ch in ascii_text if not unicodedata.combining(ch))
        ascii_text = ascii_text.replace("Đ", "D").replace("đ", "d")
        return " ".join(re.findall(r"[a-z0-9]+", ascii_text.casefold()))

    normalized_sources = [
        (normalized(phrase), path, len(normalized(phrase)))
        for phrase, path in phrase_files.items()
    ]
    for display_name in display_names:
        desired = f"Xin chào {display_name.strip()}"
        if not display_name.strip() or desired in output:
            continue
        name_key = normalized(display_name)
        matches = [
            (length, path) for phrase_key, path, length in normalized_sources
            if phrase_key.startswith("xin chao ") and phrase_key.endswith(name_key)
        ]
        if matches:
            output[desired] = min(matches, key=lambda item: item[0])[1]
    return output


__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_VOICE",
    "ZeroTTSBackend",
    "concat_wavs_gapless",
    "add_display_name_aliases",
    "load_phrase_files",
]
