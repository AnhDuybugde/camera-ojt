"""Mic camera -> keyword "hello/xin chào" (RTSP + faster-whisper, thread nền).

Pipeline (chỉ nghe, không chạm chiều phát loa):
  mic camera -> RTSP :554 /cam/realmonitor -> ffmpeg audio-only
  -> PCM 16k mono -> energy VAD -> speech segment -> faster-whisper (vi)
  -> VoiceTrigger.note_heard()

Thiết kế cho run_workstate.py: daemon thread, lỗi dep (thiếu
faster-whisper/ffmpeg, sai RTSP) thì tự disable + log, không chết pipeline.
Chống hú: bỏ segment khi đang inhibit (loa TTS vừa phát).
"""
from __future__ import annotations

import collections
import math
import shutil
import struct
import subprocess
import threading
import time
import wave
from pathlib import Path

from camera_tracking.voice.voice_trigger import VoiceTrigger

TARGET_RATE = 16000
FRAME_MS = 30
# 16000 * 30 / 1000 = 480 samples * 2 bytes = 960 bytes @16k mono s16le
FRAME_BYTES = 960

DEFAULT_AUDIO_FILTER = "highpass=f=80,afftdn=nr=12:nf=-25"


def find_ffmpeg() -> str | None:
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg  # type: ignore

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and Path(exe).exists():
            return exe
    except Exception:
        pass
    return None


def rms_level(pcm: bytes) -> float:
    if len(pcm) < 2:
        return 0.0
    n = len(pcm) // 2
    total = 0
    for i in range(0, len(pcm) - 1, 2):
        s = struct.unpack_from("<h", pcm, i)[0]
        total += s * s
    return math.sqrt(total / max(1, n)) / 32768.0


def transcribe_segment_fw(
    fw_model, pcm: bytes, lang: str = "vi", *,
    max_no_speech_prob: float = 0.55,
    min_avg_logprob: float = -0.85,
) -> str:
    """Chạy 1 speech segment qua faster-whisper (qua file wav tạm)."""
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_f:
        tmp = tmp_f.name
        with wave.open(tmp, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(TARGET_RATE)
            w.writeframes(pcm)
    try:
        # Keyword ngan khong can beam search rong; beam=1 giam ro do tre CPU.
        segments, _info = fw_model.transcribe(
            tmp,
            language=lang,
            beam_size=1,
            vad_filter=False,
            condition_on_previous_text=False,
        )
        accepted: list[str] = []
        for segment in segments:
            no_speech = float(getattr(segment, "no_speech_prob", 0.0))
            avg_logprob = float(getattr(segment, "avg_logprob", 0.0))
            if no_speech > max_no_speech_prob or avg_logprob < min_avg_logprob:
                continue
            accepted.append(segment.text.strip())
        return " ".join(accepted).strip()
    except Exception:
        return ""
    finally:
        try:
            Path(tmp).unlink(missing_ok=True)
        except OSError:
            pass


class RtspVoiceListener(threading.Thread):
    """Thread nền nghe mic camera, đẩy keyword vào VoiceTrigger."""

    def __init__(
        self,
        trigger: VoiceTrigger,
        rtsp_url: str,
        *,
        fw_model_name: str = "medium",
        fw_lang: str = "vi",
        vad_threshold: float = 0.03,
        min_seg_rms: float = 0.025,
        silence_ms: int = 450,
        start_speech_ms: int = 90,
        min_speech_ms: int = 300,
        max_segment_s: float = 3.0,
        audio_filter: str = DEFAULT_AUDIO_FILTER,
        stop_event: threading.Event | None = None,
        verbose: bool = False,
        max_no_speech_prob: float = 0.55,
        min_avg_logprob: float = -0.85,
    ) -> None:
        super().__init__(name="rtsp-voice-listen", daemon=True)
        self.trigger = trigger
        self.rtsp_url = rtsp_url
        self.fw_model_name = fw_model_name
        self.fw_lang = fw_lang
        self.vad_threshold = vad_threshold
        self.min_seg_rms = min_seg_rms
        self.silence_ms = silence_ms
        self.start_speech_ms = max(FRAME_MS, int(start_speech_ms))
        self.min_speech_ms = min_speech_ms
        self.max_segment_s = max_segment_s
        self.audio_filter = audio_filter
        self.stop_event = stop_event or threading.Event()
        self.verbose = verbose
        self.max_no_speech_prob = float(max_no_speech_prob)
        self.min_avg_logprob = float(min_avg_logprob)
        self.disabled_reason: str | None = None

    def stop(self) -> None:
        self.stop_event.set()

    def run(self) -> None:  # noqa: C901 - vòng lặp nghe tuyến tính
        try:
            from faster_whisper import WhisperModel  # type: ignore
        except ImportError:
            self.disabled_reason = "thiếu faster-whisper (pip install faster-whisper)"
            print(f"[VoiceTrigger] disabled ({self.disabled_reason})")
            return
        ffmpeg = find_ffmpeg()
        if not ffmpeg:
            self.disabled_reason = "không tìm thấy ffmpeg"
            print(f"[VoiceTrigger] disabled ({self.disabled_reason})")
            return
        try:
            print(f"[VoiceTrigger] loading faster-whisper {self.fw_model_name} (int8 CPU) ...",
                  flush=True)
            fw_model = WhisperModel(self.fw_model_name, device="cpu", compute_type="int8")
        except Exception as error:
            self.disabled_reason = f"không load được STT: {error}"
            print(f"[VoiceTrigger] disabled ({self.disabled_reason})")
            return

        cmd = [
            ffmpeg, "-hide_banner", "-loglevel", "warning",
            "-rtsp_transport", "tcp", "-i", self.rtsp_url,
            "-map", "0:a:0", "-vn", "-ac", "1", "-ar", str(TARGET_RATE),
        ]
        if self.audio_filter:
            cmd += ["-af", self.audio_filter]
        cmd += ["-f", "s16le", "pipe:1"]
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                    bufsize=0)
        except Exception as error:
            self.disabled_reason = f"không mở được ffmpeg: {error}"
            print(f"[VoiceTrigger] disabled ({self.disabled_reason})")
            return
        assert proc.stdout is not None
        print(
            f"[VoiceTrigger] listening mic camera (hello/xin chào), "
            f"VAD={self.vad_threshold:.3f}/{self.min_seg_rms:.3f} ...",
            flush=True,
        )

        pre_roll: collections.deque[bytes] = collections.deque(
            maxlen=max(1, int(300 / FRAME_MS)))
        speaking = False
        segment = bytearray()
        silence_ms = 0
        voiced_ms = 0
        leftover = bytearray()
        min_bytes = int(self.min_speech_ms / 1000 * TARGET_RATE * 2)
        max_bytes = int(self.max_segment_s * TARGET_RATE * 2)

        while not self.stop_event.is_set():
            chunk = proc.stdout.read(4096)
            if not chunk:
                self.disabled_reason = "RTSP audio stream ended"
                print(f"[VoiceTrigger] disabled ({self.disabled_reason})", flush=True)
                break
            leftover += chunk
            while len(leftover) >= FRAME_BYTES:
                frame = bytes(leftover[:FRAME_BYTES])
                del leftover[:FRAME_BYTES]
                now = time.monotonic()
                level = rms_level(frame)
                is_speech = level >= self.vad_threshold and not self.trigger.inhibited(now)
                if not speaking:
                    pre_roll.append(frame)
                    if is_speech:
                        voiced_ms += FRAME_MS
                        if voiced_ms >= self.start_speech_ms:
                            speaking = True
                            segment = bytearray(b"".join(pre_roll))
                            silence_ms = 0
                            voiced_ms = 0
                    else:
                        # Giong noi co khoang nghi ngan giua am tiet; decay
                        # thay vi reset giup khoi dong VAD ma khong doi 300 ms
                        # lien tuc vuot nguong.
                        voiced_ms = max(0, voiced_ms - FRAME_MS)
                else:
                    segment += frame
                    if is_speech:
                        silence_ms = 0
                    else:
                        silence_ms += FRAME_MS
                    if self.trigger.inhibited(time.monotonic()):
                        speaking = False
                        segment.clear()
                        silence_ms = 0
                        continue
                    if silence_ms >= self.silence_ms or len(segment) >= max_bytes:
                        seg = bytes(segment)
                        speaking = False
                        segment = bytearray()
                        silence_ms = 0
                        pre_roll.clear()
                        if len(seg) < min_bytes or rms_level(seg) < self.min_seg_rms:
                            continue
                        seg_rms = rms_level(seg)
                        print(
                            f"[VoiceTrigger] speech {len(seg) * 1000 // (TARGET_RATE * 2)}ms "
                            f"rms={seg_rms:.3f} -> STT",
                            flush=True,
                        )
                        text = transcribe_segment_fw(
                            fw_model, seg, self.fw_lang,
                            max_no_speech_prob=self.max_no_speech_prob,
                            min_avg_logprob=self.min_avg_logprob,
                        )
                        if not text:
                            print("[VoiceTrigger] STT không có văn bản", flush=True)
                            continue
                        # Log mọi text mic thu được (vd hello, xin chào, 1 2 3);
                        # khớp keyword thì log thêm dòng keyword.
                        print(f"[Voice][mic] nghe: {text}", flush=True)
                        if self.trigger.note_heard(text):
                            print(f"[VoiceTrigger] keyword: {text}", flush=True)
        try:
            proc.terminate()
        except Exception:
            pass


def build_listener_from_env(
    trigger: VoiceTrigger,
    *,
    channel: int = 1,
    subtype: int = 1,
    fw_model_name: str = "medium",
    fw_lang: str = "vi",
    vad_threshold: float = 0.03,
    min_seg_rms: float = 0.025,
    max_no_speech_prob: float = 0.55,
    min_avg_logprob: float = -0.85,
    stop_event: threading.Event | None = None,
) -> RtspVoiceListener | None:
    """Dựng listener từ IMOU_IP/USER/PASSWORD; None khi thiếu env."""
    from camera_tracking.camera.camera_imou import imou_url

    url = imou_url(int(channel), int(subtype))
    if not url:
        print("[VoiceTrigger] disabled (thiếu IMOU_IP/USER/PASSWORD cho mic RTSP)")
        return None
    return RtspVoiceListener(
        trigger, url,
        fw_model_name=fw_model_name, fw_lang=fw_lang,
        vad_threshold=vad_threshold, min_seg_rms=min_seg_rms,
        max_no_speech_prob=max_no_speech_prob,
        min_avg_logprob=min_avg_logprob,
        stop_event=stop_event,
    )


__all__ = [
    "DEFAULT_AUDIO_FILTER",
    "RtspVoiceListener",
    "build_listener_from_env",
    "find_ffmpeg",
    "rms_level",
    "transcribe_segment_fw",
]
