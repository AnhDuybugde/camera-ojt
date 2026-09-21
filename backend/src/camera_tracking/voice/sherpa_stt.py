"""STT tieng Viet bang Zipformer transducer tren CPU (sherpa-onnx).

Model mac dinh: sherpa-onnx-zipformer-vi-30M-int8-2026-02-09 (~30M params,
int8, train ~6000h tieng Viet, nhat VLSP 2025). 12s audio -> ~0.3s CPU,
RAM/VRAM khong dang ke — thay the faster-whisper medium (~9s CPU) cho
vong nghe lenh Ha Linh.

Khac faster-whisper:
- Output chu thuong, khong dau cau (transducer). Cac cho so khop
  (is_wake/fast_answer) deu normalize truoc nen khong anh huong.
- Khong co no_speech_prob/avg_logprob: cong RMS/SNR truoc STT van giu,
  output rong coi nhu im lang.
- Input float32 [-1, 1] @16kHz (tuong duong PCM s16 ma pipeline dang dung).

Wake giu faster-whisper small (can initial_prompt bias + cua chat luong).
"""

from __future__ import annotations

from pathlib import Path

MODEL_DIR_NAME = "sherpa-onnx-zipformer-vi-30M-int8-2026-02-09"
SAMPLE_RATE = 16000


def default_model_dir() -> Path:
    """Thu muc model: models/stt/<MODEL_DIR_NAME> (tu repo root)."""
    here = Path(__file__).resolve()
    for parent in (here.parents[2], here.parents[3]):
        candidate = parent / "models" / "stt" / MODEL_DIR_NAME
        if candidate.is_dir():
            return candidate
    return here.parents[2] / "models" / "stt" / MODEL_DIR_NAME


class SherpaZipformerSTT:
    """Wrapper OfflineRecognizer transducer, load 1 lan, CPU-only."""

    def __init__(
        self,
        model_dir: str | Path | None = None,
        num_threads: int = 6,
    ) -> None:
        self.model_dir = Path(model_dir) if model_dir else default_model_dir()
        self.num_threads = max(1, int(num_threads))
        self._recognizer = None

    def _load(self):  # type: ignore[no-untyped-def]
        if self._recognizer is None:
            try:
                import sherpa_onnx
            except ImportError as error:
                raise RuntimeError(
                    "chua cai sherpa-onnx: chay "
                    "/home/jloy/venvs/cv/bin/pip install sherpa-onnx"
                ) from error
            root = self.model_dir
            for name in ("tokens.txt", "encoder.int8.onnx",
                         "decoder.onnx", "joiner.int8.onnx"):
                if not (root / name).is_file():
                    raise RuntimeError(f"thieu model file: {root / name}")
            self._recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
                tokens=str(root / "tokens.txt"),
                encoder=str(root / "encoder.int8.onnx"),
                decoder=str(root / "decoder.onnx"),
                joiner=str(root / "joiner.int8.onnx"),
                num_threads=self.num_threads,
                sample_rate=SAMPLE_RATE,
                decoding_method="greedy_search",
                provider="cpu",
            )
        return self._recognizer

    def warmup(self) -> float:
        """Load model + transcribe 1s im lang de am session. Tra giay load."""
        import time

        t0 = time.monotonic()
        recognizer = self._load()
        import numpy as np

        stream = recognizer.create_stream()
        stream.accept_waveform(SAMPLE_RATE,
                               np.zeros(SAMPLE_RATE, dtype=np.float32))
        recognizer.decode_stream(stream)
        return time.monotonic() - t0

    def transcribe_pcm(
        self,
        pcm: bytes,
        *,
        agc_peak: float = 0.6,
        agc_max_gain: float = 50.0,
    ) -> str:
        """PCM s16le mono 16kHz (giong faster-whisper path) -> text."""
        import array

        import numpy as np

        samples = array.array("h")
        try:
            samples.frombytes(bytes(pcm))
        except (ValueError, OverflowError):
            return ""
        if not samples:
            return ""
        peak = max(abs(s) for s in samples)
        if peak <= 0:
            return ""
        gain = min((agc_peak * 32768.0) / peak, agc_max_gain)
        audio = np.asarray(samples, dtype=np.float32) / 32768.0
        if gain > 1.0:
            audio = audio * gain
        recognizer = self._load()
        stream = recognizer.create_stream()
        stream.accept_waveform(SAMPLE_RATE, audio)
        recognizer.decode_stream(stream)
        return str(stream.result.text or "").strip()

    def transcribe_wav(self, path: str | Path) -> str:
        """Doc file wav (16k mono; tu resample neu khac) -> text. Tien bench."""
        import soundfile as sf

        samples, rate = sf.read(str(path), dtype="float32", always_2d=False)
        import numpy as np

        audio = np.asarray(samples, dtype=np.float32).reshape(-1)
        if int(rate) != SAMPLE_RATE:
            ratio = SAMPLE_RATE / float(rate)
            idx = (np.arange(int(len(audio) * ratio)) / ratio).astype(int)
            audio = audio[np.clip(idx, 0, len(audio) - 1)]
        recognizer = self._load()
        stream = recognizer.create_stream()
        stream.accept_waveform(SAMPLE_RATE, audio)
        recognizer.decode_stream(stream)
        return str(stream.result.text or "").strip()


__all__ = ["MODEL_DIR_NAME", "SAMPLE_RATE", "SherpaZipformerSTT",
           "default_model_dir"]
