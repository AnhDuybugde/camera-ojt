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
# Bias decoder ve dung chinh ta cum goi (STT ep language='vi' hay be
# "hello imou" thanh "xin gao emu"). Re hon fuzzy-match ma khong cham.
DEFAULT_STT_PROMPT = "Hello imou. Xin chào imou."


def db_to_amplitude_ratio(db: float) -> float:
    """Đổi SNR dB sang tỉ số biên độ (P3: 10 dB ~= 3.16 lần)."""
    import math as _math

    return 10.0 ** (max(0.0, float(db)) / 20.0)


class NoiseFloorTracker:
    """P1 — ngưỡng VAD động bám nền ồn (trị ồn đều, không cần đo phòng).

    Chỉ học từ các frame yên (dưới ngưỡng hiện tại) bằng trung bình trượt;
    ngưỡng hiệu lực = min(max(nền × factor, sàn tuyệt đối), trần).
    factor <= 0 -> hành vi cũ (ngưỡng tĩnh).
    """

    def __init__(
        self,
        static_threshold: float = 0.03,
        factor: float = 3.0,
        abs_min: float = 0.02,
        ceiling: float = 0.15,
        adapt_rate: float = 0.03,
    ) -> None:
        self.static_threshold = max(0.0, float(static_threshold))
        self.factor = float(factor)
        self.abs_min = max(1e-4, float(abs_min))
        self.ceiling = max(self.abs_min, float(ceiling))
        self.adapt_rate = min(1.0, max(0.0, float(adapt_rate)))
        self.floor = self.abs_min

    @property
    def threshold(self) -> float:
        """Ngưỡng VAD hiệu lực cho frame kế tiếp."""
        if self.factor <= 0.0:
            return self.static_threshold
        dynamic = max(self.floor * self.factor, self.abs_min)
        return min(dynamic, self.ceiling)

    def update(self, rms: float) -> float:
        """Cập nhật nền từ 1 frame; trả ngưỡng hiệu lực mới."""
        level = max(0.0, float(rms))
        if level < self.threshold:
            self.floor += self.adapt_rate * (level - self.floor)
        else:
            # Ồn to kéo dài (điều hòa/TV mở to) vẫn phải nâng nền theo,
            # nhưng rất chậm để tiếng nói ngắn không kịp kéo nền lên.
            self.floor += (self.adapt_rate / 20.0) * (level - self.floor)
        self.floor = max(1e-4, self.floor)
        return self.threshold


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
    max_no_speech_prob: float = 0.40,
    min_avg_logprob: float = -0.70,
    agc_peak: float = 0.6,
    agc_max_gain: float = 50.0,
    initial_prompt: str = DEFAULT_STT_PROMPT,
) -> str:
    """Chạy 1 speech segment qua faster-whisper (qua file wav tạm).

    Mic camera thường thu rất nhỏ (rms ~0.005): chuẩn hóa đỉnh lên
    `agc_peak` trước khi STT để Whisper không loại vì "im lặng".
    """
    import array
    import tempfile

    samples = array.array("h")
    try:
        samples.frombytes(pcm)
    except (ValueError, OverflowError):
        return ""
    if not samples:
        return ""
    peak = max(abs(s) for s in samples)
    if peak <= 0:
        return ""
    gain = min((agc_peak * 32768.0) / peak, agc_max_gain)
    if gain > 1.0:
        for i, s in enumerate(samples):
            v = int(s * gain)
            samples[i] = 32767 if v > 32767 else (-32768 if v < -32768 else v)
        pcm = samples.tobytes()

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
            initial_prompt=initial_prompt or None,
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
        max_no_speech_prob: float = 0.40,
        min_avg_logprob: float = -0.70,
        # P1: ngưỡng động bám nền (factor<=0 -> ngưỡng tĩnh như cũ).
        vad_floor_factor: float = 3.0,
        vad_floor_min: float = 0.02,
        vad_ceiling: float = 0.15,
        # P3: đoạn cắt ra chỉ đi STT khi to hơn nền >= snr_min_db.
        snr_min_db: float = 10.0,
        # Dump mọi đoạn đi STT ra WAV để nghe lại/debug (None/"" = tắt).
        dump_dir: str | Path | None = None,
        dump_keep: int = 500,
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
        self.noise_floor = NoiseFloorTracker(
            static_threshold=vad_threshold,
            factor=vad_floor_factor,
            abs_min=vad_floor_min,
            ceiling=vad_ceiling,
        )
        self.snr_ratio = db_to_amplitude_ratio(snr_min_db)
        self.stats = {"segments": 0, "stt": 0, "dropped_snr": 0, "dropped_rms": 0}
        self.disabled_reason: str | None = None
        # Mức frame mới nhất (cho đồng hồ đo live); -1 = chưa có frame nào.
        self.last_level: float = -1.0
        # Đỉnh mức kể từ lần đọc meter trước (meter 1Hz dễ chộp trúng
        # khoảng ngắt giữa âm tiết nên phải giữ đỉnh, không đọc tức thời).
        self.level_peak: float = 0.0
        self.dump_dir = Path(dump_dir) if dump_dir else None
        self.dump_keep = max(0, int(dump_keep))
        if self.dump_dir is not None:
            try:
                self.dump_dir.mkdir(parents=True, exist_ok=True)
            except OSError:
                self.dump_dir = None

    def _dump_segment(self, seg: bytes, seg_rms: float) -> Path | None:
        """Lưu 1 đoạn đi STT ra WAV; tỉa file cũ khi quá dump_keep."""
        if self.dump_dir is None:
            return None
        try:
            stamp = time.strftime("%Y%m%d-%H%M%S")
            path = self.dump_dir / f"seg_{stamp}_{self.stats['stt']:04d}_rms{seg_rms:.3f}.wav"
            with wave.open(str(path), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(TARGET_RATE)
                w.writeframes(seg)
            if self.dump_keep > 0:
                files = sorted(self.dump_dir.glob("seg_*.wav"))
                for old in files[:-self.dump_keep]:
                    try:
                        old.unlink()
                    except OSError:
                        pass
            return path
        except OSError:
            return None

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
            f"[VoiceTrigger] listening mic camera (hello imou/xin chào imou), "
            f"VAD dyn~{self.noise_floor.threshold:.3f} "
            f"(static={self.vad_threshold:.3f}, x{self.noise_floor.factor:g}, "
            f"snr>={self.snr_ratio:.1f}x) ...",
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
                self.last_level = level
                if level > self.level_peak:
                    self.level_peak = level
                # P1: ngưỡng động bám nền ồn; frame yên nuôi lại nền.
                vad_now = self.noise_floor.update(level)
                is_speech = level >= vad_now and not self.trigger.inhibited(now)
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
                        self.stats["segments"] += 1
                        seg_rms = rms_level(seg)
                        if len(seg) < min_bytes or seg_rms < self.min_seg_rms:
                            self.stats["dropped_rms"] += 1
                            continue
                        # P3: đoạn phải to hơn nền đang theo dõi (loại ồn to
                        # đều + tiếng nói xa nhỏ, khỏi tốn lượt STT).
                        if seg_rms < self.noise_floor.floor * self.snr_ratio:
                            self.stats["dropped_snr"] += 1
                            if self.verbose:
                                print(
                                    f"[VoiceTrigger] drop SNR "
                                    f"rms={seg_rms:.3f} floor={self.noise_floor.floor:.3f}",
                                    flush=True,
                                )
                            continue
                        self.stats["stt"] += 1
                        dumped = self._dump_segment(seg, seg_rms)
                        if self.stats["segments"] % 100 == 0:
                            print(
                                f"[VoiceTrigger] stats seg={self.stats['segments']} "
                                f"stt={self.stats['stt']} "
                                f"drop_rms={self.stats['dropped_rms']} "
                                f"drop_snr={self.stats['dropped_snr']} "
                                f"floor={self.noise_floor.floor:.3f}",
                                flush=True,
                            )
                        print(
                            f"[VoiceTrigger] speech {len(seg) * 1000 // (TARGET_RATE * 2)}ms "
                            f"rms={seg_rms:.3f} -> STT"
                            + (f" [{dumped.name}]" if dumped is not None else ""),
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
    max_no_speech_prob: float = 0.40,
    min_avg_logprob: float = -0.70,
    vad_floor_factor: float = 3.0,
    vad_floor_min: float = 0.02,
    vad_ceiling: float = 0.15,
    snr_min_db: float = 10.0,
    stop_event: threading.Event | None = None,
    dump_dir: str | Path | None = None,
    dump_keep: int = 500,
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
        vad_floor_factor=vad_floor_factor,
        vad_floor_min=vad_floor_min,
        vad_ceiling=vad_ceiling,
        snr_min_db=snr_min_db,
        stop_event=stop_event,
        dump_dir=dump_dir,
        dump_keep=dump_keep,
    )


__all__ = [
    "DEFAULT_AUDIO_FILTER",
    "DEFAULT_STT_PROMPT",
    "NoiseFloorTracker",
    "RtspVoiceListener",
    "build_listener_from_env",
    "db_to_amplitude_ratio",
    "find_ffmpeg",
    "rms_level",
    "transcribe_segment_fw",
]
