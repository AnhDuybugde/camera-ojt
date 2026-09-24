"""Realtime Bé Xinh voice assistant using Gemini Live + IMOU camera audio.

Flow
----
1. A tiny/local STT loop listens only for the wake phrase "Bé Xinh ơi".
2. One Gemini Live session is opened for a conversation window.
3. Subsequent user turns are sent as 16 kHz PCM directly to Gemini Live.
4. By default, Gemini returns text and ZeroTTS reads it with the Hạ My voice;
   native Gemini 24 kHz audio remains available as a low-latency fallback.
5. Local camera/runtime tools are executed through Live function calling.

The existing ``be_xinh_assistant.py`` remains the classic fallback path.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from camera_tracking.audio.announcer import CameraCheckInAnnouncer
from camera_tracking.camera.camera_imou import imou_url
from camera_tracking.config import load_config
from camera_tracking.voice.qa_tools import TOOL_DECLARATIONS, TOOL_FUNCS
from camera_tracking.voice.turn_guard import clear_turn, mark_turn
from camera_tracking.voice.voice_trigger import normalize_trigger_text

try:
    from halinh_assistant import (
        BE_XINH_PHRASES,
        WAKE_STT_PROMPT,
        active_voice_volume,
        acquire_chat_lock,
        capture_utterance,
        is_wake,
        release_chat_lock,
        strip_wake_command,
    )
except ModuleNotFoundError:  # imported as scripts.be_xinh_live_assistant in tests
    from scripts.halinh_assistant import (
        BE_XINH_PHRASES,
        WAKE_STT_PROMPT,
        active_voice_volume,
        acquire_chat_lock,
        capture_utterance,
        is_wake,
        release_chat_lock,
        strip_wake_command,
    )
from camera_tracking.voice.rtsp_voice_listener import (
    TARGET_RATE,
    find_ffmpeg,
    transcribe_segment_fw,
)

LIVE_SYSTEM = """Bạn là Bé Xinh, người bạn đồng hành bằng giọng nói của hệ thống Camera-OJT.

Phong cách giọng nói:
- Luôn nói tiếng Việt tự nhiên, trẻ trung, trong sáng và thân thiện, như đang mỉm cười nhẹ.
  Kể cả khi bản chép lời nhận nhầm thành ngôn ngữ khác, vẫn trả lời bằng tiếng Việt;
  chỉ đổi ngôn ngữ khi người dùng yêu cầu thật rõ ràng.
- Phát âm rõ, nhịp vừa phải, có cảm xúc nhưng không lên giọng quá mức, không nói kiểu em bé.
- Mặc định chỉ trả lời một câu rõ ràng, khoảng 8-20 từ; chỉ dài hơn khi người dùng
  yêu cầu giải thích. Có thể dùng nhẹ các từ "nè", "nhé", "ạ" khi hợp ngữ cảnh,
  nhưng không chèn vào mọi câu và không dùng tiếng cảm thán gây ồn.
- Thay đổi cách mở đầu và kết thúc để không lặp máy móc. Không liên tục nói
  "Bé Xinh có thể giúp gì" hoặc tự giới thiệu lại trong cùng một phiên.
- Nếu biết chắc tên người dùng từ dữ liệu hệ thống thì gọi tên một cách tự nhiên;
  nếu chưa biết thì gọi là "bạn", tuyệt đối không đoán tên.
- Khi người dùng nghiêm túc, khó chịu hoặc đang cần trợ giúp, hạ giọng bình tĩnh và đi thẳng vào việc.

Hãy nhớ mạch hội thoại trong phiên hiện tại để hiểu các câu nối tiếp như
"còn người đó?", "vậy ngày mai?" hoặc "tại sao?". Khi cần dữ liệu camera,
chấm công, thời gian, thời tiết hoặc trạng thái hệ thống, hãy dùng function tool
thay vì đoán. Nếu dữ liệu không đủ, nói rõ điều chưa chắc; không tự bịa danh tính
hoặc trạng thái chấm công. Không dùng Markdown trong lời nói. Tự xưng là Bé Xinh.
"""


class LaptopSpeaker:
    """Small synchronous speaker adapter backed by ffplay."""

    def __init__(self) -> None:
        ffmpeg = find_ffmpeg()
        sibling = Path(ffmpeg).with_name(
            "ffplay.exe" if os.name == "nt" else "ffplay"
        ) if ffmpeg else None
        self.ffplay = (
            str(sibling) if sibling and sibling.is_file()
            else shutil.which("ffplay")
        )
        if not self.ffplay:
            raise RuntimeError("Không tìm thấy ffplay để phát loa laptop")
        self.gain = 0.85
        self._process: subprocess.Popen | None = None

    def play_file_sync(self, audio_path: str | Path) -> None:
        volume = max(10, min(100, round(float(self.gain) * 100)))
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        self._process = subprocess.Popen(
            [
                self.ffplay, "-nodisp", "-autoexit", "-loglevel", "quiet",
                "-volume", str(volume), str(audio_path),
            ],
            creationflags=flags,
        )
        try:
            code = self._process.wait()
            if code != 0:
                raise RuntimeError(f"ffplay thoát với mã {code}")
        finally:
            self._process = None

    def cancel_playback(self) -> None:
        process = self._process
        if process is not None and process.poll() is None:
            process.kill()

    def open_pcm_stream(self, *, input_sample_rate: int = 24_000):
        raise RuntimeError(
            "Loa laptop chưa hỗ trợ native PCM stream; dùng edge/zerotts"
        )

    def close(self) -> None:
        self.cancel_playback()


def _laptop_mic_input() -> tuple[list[str], str]:
    """Resolve a Windows microphone for ffmpeg DirectShow capture."""
    configured = os.getenv("BE_XINH_LAPTOP_MIC", "").strip()
    if configured:
        return [
            "-f", "dshow", "-audio_buffer_size", "50",
            "-i", f"audio={configured}",
        ], configured
    if os.name != "nt":
        return ["-f", "pulse", "-i", "default"], "default"
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("Không tìm thấy ffmpeg để mở microphone laptop")
    probe = subprocess.run(
        [ffmpeg, "-hide_banner", "-list_devices", "true", "-f", "dshow",
         "-i", "dummy"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=subprocess.CREATE_NO_WINDOW,
        timeout=8,
        check=False,
    )
    names = re.findall(r'"([^"]+)"\s+\(audio\)', probe.stderr)
    physical = [
        name for name in names
        if "microphone" in name.lower() and "vmix" not in name.lower()
    ]
    if not physical:
        physical = [name for name in names if "vmix" not in name.lower()]
    if not physical:
        raise RuntimeError(
            "Không tìm thấy microphone laptop; đặt BE_XINH_LAPTOP_MIC"
        )
    selected = physical[0]
    return [
        "-f", "dshow", "-audio_buffer_size", "50",
        "-i", f"audio={selected}",
    ], selected


def _camera_audio_reachable() -> bool:
    """Quickly verify both camera input and output transports for auto mode."""
    host = os.getenv("IMOU_IP", "").strip()
    if not host:
        return False
    try:
        talk_port = int(os.getenv("IMOU_TALK_PORT", "8086"))
    except ValueError:
        talk_port = 8086
    for port in (554, talk_port):
        try:
            with socket.create_connection((host, port), timeout=0.7):
                pass
        except OSError:
            return False
    return True


def _build_live_config(output_mode: str, voice_name: str) -> dict:
    """Build a Gemini Live config for native audio or ZeroTTS output."""
    config = {
        # The current conversational Live model only accepts AUDIO. In
        # ZeroTTS mode we discard its PCM, keep output_audio_transcription,
        # and let Hạ My read that text without making a second Gemini call.
        "response_modalities": ["AUDIO"],
        "system_instruction": LIVE_SYSTEM,
        # Bias the recognizer toward Vietnamese and project-specific names.
        # This is still Gemini Live audio understanding, not a second STT API.
        "input_audio_transcription": {
            "language_codes": ["vi-VN"],
            "custom_vocabulary": [
                "Bé Xinh", "chấm công", "Camera A", "Camera B", "IMOU",
            ],
        },
        "tools": [{"function_declarations": TOOL_DECLARATIONS}],
        "realtime_input_config": {
            "automatic_activity_detection": {"disabled": True},
        },
        "speech_config": {
            "voice_config": {
                "prebuilt_voice_config": {"voice_name": voice_name},
            },
            "language_code": "vi-VN",
        },
        "output_audio_transcription": {},
    }
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config" / "default.yaml"))
    parser.add_argument("--model", default=None,
                        help="Gemini Live model; default GEMINI_LIVE_MODEL/gemini-3.8-live")
    parser.add_argument("--wake-model", default="tiny")
    parser.add_argument(
        "--command-model", default=None,
        help=("STT cho cau hoi: zipformer (local INT8), gemini (audio truc tiep), "
              "hoac tiny/small/medium Whisper. Mac dinh "
              "BE_XINH_COMMAND_STT/zipformer."),
    )
    parser.add_argument("--conversation-idle-s", type=float, default=None,
                        help="Seconds with no follow-up before returning to wake mode.")
    parser.add_argument("--turn-wait-s", type=float, default=12.0,
                        help="Max time waiting for a follow-up utterance.")
    parser.add_argument("--full-duplex", action="store_true",
                        help="Do not suppress mic for camera-speaker echo. Only enable with proven AEC.")
    return parser.parse_args()


def _wake_transcriber(model_name: str, lang: str):
    """Load one lightweight local recognizer used only for the wake phrase."""
    if model_name == "zipformer":
        try:
            from camera_tracking.voice.sherpa_stt import SherpaZipformerSTT
            model = SherpaZipformerSTT()
            model.warmup()

            def transcribe(pcm: bytes) -> str:
                return model.transcribe_pcm(pcm)

            return transcribe
        except Exception as error:  # noqa: BLE001
            print(f"[Bé Xinh Live] Zipformer unavailable ({error}); fallback tiny Whisper.")
    try:
        from faster_whisper import WhisperModel
    except ImportError as error:
        raise RuntimeError("Cần faster-whisper hoặc sherpa-onnx cho wake phrase") from error
    model = WhisperModel("tiny", device="cpu", compute_type="int8")

    def transcribe(pcm: bytes) -> str:
        return transcribe_segment_fw(
            model, pcm, lang,
            max_no_speech_prob=0.60,
            min_avg_logprob=-1.10,
            initial_prompt=WAKE_STT_PROMPT,
        )

    return transcribe


def _command_transcriber(model_name: str, lang: str):
    """Load local command STT once; return None for Gemini audio passthrough.

    Zipformer is the production default because the Vietnamese INT8 model is
    small and fast on CPU. Keeping this separate from wake recognition lets
    us retain Whisper's prompt bias for the wake phrase while sending clean,
    explicit Vietnamese text to Gemini for the actual question.
    """
    selected = str(model_name or "zipformer").strip().lower()
    if selected in {"gemini", "native", "audio", "off", "none"}:
        print("[Bé Xinh Live] STT câu hỏi: Gemini audio trực tiếp.", flush=True)
        return None
    if selected == "zipformer":
        try:
            from camera_tracking.voice.sherpa_stt import SherpaZipformerSTT

            model = SherpaZipformerSTT(
                num_threads=max(
                    1, int(os.getenv("BE_XINH_ZIPFORMER_THREADS", "4"))
                )
            )
            load_s = model.warmup()
            print(
                f"[Bé Xinh Live] STT câu hỏi: Zipformer vi INT8 "
                f"sẵn sàng ({load_s:.2f}s).",
                flush=True,
            )
            return model.transcribe_pcm
        except Exception as error:  # noqa: BLE001
            print(
                f"[Bé Xinh Live] Zipformer chưa sẵn sàng ({error}); "
                "fallback Gemini audio.",
                flush=True,
            )
            return None
    if selected not in {"tiny", "small", "medium"}:
        print(
            f"[Bé Xinh Live] STT câu hỏi {selected!r} không hợp lệ; "
            "fallback Gemini audio.",
            flush=True,
        )
        return None
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("[Bé Xinh Live] thiếu faster-whisper; fallback Gemini audio.",
              flush=True)
        return None
    model = WhisperModel(selected, device="cpu", compute_type="int8")

    def transcribe(pcm: bytes) -> str:
        return transcribe_segment_fw(
            model, pcm, lang,
            max_no_speech_prob=0.50,
            min_avg_logprob=-1.00,
        )

    print(f"[Bé Xinh Live] STT câu hỏi: Whisper {selected} INT8.", flush=True)
    return transcribe


def _chunk_pcm(pcm: bytes, milliseconds: int = 100):
    size = max(640, int(TARGET_RATE * 2 * milliseconds / 1000))
    for offset in range(0, len(pcm), size):
        part = pcm[offset:offset + size]
        if part:
            yield part


def _agc_pcm(pcm: bytes, peak_target: float = 0.5,
             max_gain: float = 30.0, silence_peak: int = 100) -> bytes:
    """Khuếch đại mic về mức Gemini Live nghe rõ.

    Mic camera thường chỉ đạt đỉnh ~0.05 (-26dBFS); gửi thô khiến Live
    không nhận ra tiếng nói và im luôn (treo turn). Chuẩn hoá đỉnh về
    ~0.5, trần gain 30x; đoạn thật sự im (đỉnh < ~0.003) thì giữ nguyên
    để khỏi khuyếch đại nền ồn.
    """
    import array

    samples = array.array("h")
    try:
        samples.frombytes(bytes(pcm))
    except (ValueError, OverflowError):
        return pcm
    if not samples:
        return pcm
    peak = 0
    for sample in samples:
        magnitude = abs(sample)
        if magnitude > peak:
            peak = magnitude
    if peak <= silence_peak:
        return pcm
    gain = min((peak_target * 32768.0) / peak, max_gain)
    if gain <= 1.0:
        return pcm
    print(f"[Bé Xinh Live] mic gain={gain:.1f}x (peak {peak}/32768).",
          flush=True)
    return array.array(
        "h",
        (max(-32768, min(32767, int(sample * gain))) for sample in samples),
    ).tobytes()


def _safe_tool_call(name: str, args: object) -> str:
    func = TOOL_FUNCS.get(str(name))
    if not callable(func) or str(name).startswith("_"):
        return f"Tool {name} chưa được hỗ trợ."
    kwargs = dict(args or {}) if isinstance(args, dict) else {}
    try:
        return str(func(**kwargs))
    except TypeError:
        try:
            return str(func())
        except Exception as error:  # noqa: BLE001
            return f"Tool {name} lỗi: {error}"
    except Exception as error:  # noqa: BLE001
        return f"Tool {name} lỗi: {error}"


async def _send_audio_turn(session, pcm: bytes, types) -> None:
    # capture_utterance already returns one locally segmented utterance, so
    # explicit boundaries are more reliable than asking server VAD to detect
    # speech in a short burst that is uploaded faster than realtime.
    await session.send_realtime_input(activity_start=types.ActivityStart())
    for chunk in _chunk_pcm(pcm, 100):
        await session.send_realtime_input(
            audio=types.Blob(data=chunk, mime_type="audio/pcm;rate=16000")
        )
    await session.send_realtime_input(activity_end=types.ActivityEnd())


async def _receive_turn(session, speaker: CameraCheckInAnnouncer, types,
                        timeout_s: float = 18.0, *,
                        stream_native_audio: bool = True) -> tuple[str, str, float]:
    """Receive one Live turn and service native audio/text plus tool calls.

    Có timeout: mic ồn/yếu khiến Live im luôn thì không treo turn vĩnh
    viễn mà trả turn rỗng để caller về wake mode cho người dùng gọi lại.
    """
    stream = None
    user_text = ""
    assistant_text = ""
    first_output_at: float | None = None
    started = time.monotonic()
    try:
        async with asyncio.timeout(timeout_s):
            # A tool call closes one server turn. After sending its result the
            # SDK exposes the spoken answer through a new receive() iterator,
            # so keep receiving until the final non-tool turn completes.
            tool_result_pending = False
            final_turn_complete = False
            while not final_turn_complete:
                received_any = False
                async for response in session.receive():
                    received_any = True
                # The SDK exposes response.data as the native 24 kHz PCM shortcut.
                    data = getattr(response, "data", None)
                    audio_chunks = [bytes(data)] if data else []
                    content = getattr(response, "server_content", None)
                    if not audio_chunks and content is not None:
                        model_turn = getattr(content, "model_turn", None)
                        for part in getattr(model_turn, "parts", None) or []:
                            inline = getattr(part, "inline_data", None)
                            inline_data = getattr(inline, "data", None)
                            mime = str(getattr(inline, "mime_type", "") or "")
                            if inline_data and (not mime or mime.startswith("audio/")):
                                audio_chunks.append(bytes(inline_data))
                    for audio_chunk in audio_chunks if stream_native_audio else []:
                        if stream is None:
                            first_output_at = time.monotonic()
                            stream = speaker.open_pcm_stream(input_sample_rate=24_000)
                        stream.write(audio_chunk)

                    tool_call = getattr(response, "tool_call", None)
                    if tool_call:
                        function_responses = []
                        for fc in getattr(tool_call, "function_calls", None) or []:
                            result = _safe_tool_call(fc.name, getattr(fc, "args", None))
                            print(f"[Bé Xinh Live] tool {fc.name} -> {result}", flush=True)
                            function_responses.append(types.FunctionResponse(
                                id=fc.id, name=fc.name, response={"result": result}
                            ))
                        if function_responses:
                            await session.send_tool_response(
                                function_responses=function_responses
                            )
                            tool_result_pending = True

                    model_output_seen = bool(audio_chunks)
                    if content is not None:
                        in_tx = getattr(content, "input_transcription", None)
                        out_tx = getattr(content, "output_transcription", None)
                        if in_tx is not None and getattr(in_tx, "text", None):
                            user_text = str(in_tx.text).strip()
                        if out_tx is not None and getattr(out_tx, "text", None):
                            assistant_text += str(out_tx.text)
                            model_output_seen = True
                            if first_output_at is None:
                                first_output_at = time.monotonic()
                        # TEXT response mode exposes the answer through model-turn
                        # parts instead of output_audio_transcription.
                        model_turn = getattr(content, "model_turn", None)
                        for part in getattr(model_turn, "parts", None) or []:
                            part_text = getattr(part, "text", None)
                            if part_text:
                                assistant_text += str(part_text)
                                model_output_seen = True
                                if first_output_at is None:
                                    first_output_at = time.monotonic()
                        if model_output_seen and not tool_call:
                            tool_result_pending = False
                        if (bool(getattr(content, "interrupted", False))
                                and stream is not None):
                            stream.close(abort=True)
                            stream = None
                        if bool(getattr(content, "turn_complete", False)):
                            if tool_result_pending:
                                break
                            final_turn_complete = True
                            break
                if final_turn_complete:
                    break
                if not tool_result_pending or not received_any:
                    break
    except TimeoutError:
        print(f"[Bé Xinh Live] Live im lang qua {timeout_s:.0f}s "
              f"(mic on/yeu?) - ve cho goi.", flush=True)
    finally:
        if stream is not None:
            stream.close()
    latency = (
        first_output_at - started
        if first_output_at is not None
        else time.monotonic() - started
    )
    return user_text, assistant_text.strip(), latency


async def _speak_zerotts(
    text: str,
    tts,
    speaker: CameraCheckInAnnouncer,
    output_dir: Path,
    voice_name: str,
) -> tuple[float, float]:
    """Synthesize bounded ZeroTTS audio, then play the cached waveform."""
    from camera_tracking.voice.speaker_guard import mark_speaker_busy

    key = hashlib.sha1(f"{voice_name}\0{text}".encode("utf-8")).hexdigest()[:16]
    wav = output_dir / f"be_xinh_live_{key}.wav"
    speaker.gain = active_voice_volume()

    synth_started = time.monotonic()
    if not wav.is_file():
        import numpy as np
        import soundfile as sf

        model = tts._load()
        max_frames = max(28, min(90, int(len(text) * 0.95) + 12))

        def _synthesize() -> None:
            audio = model.synthesize(
                text, voice=voice_name, max_frames=max_frames
            )
            samples = np.asarray(audio, dtype=np.float32).reshape(-1)
            sf.write(str(wav), samples, int(model.sample_rate))

        await asyncio.to_thread(_synthesize)
    synth_s = time.monotonic() - synth_started
    play_started = time.monotonic()
    await asyncio.to_thread(speaker.play_file_sync, wav)
    play_s = time.monotonic() - play_started
    # Keep the next RTSP capture from consuming the camera's short echo tail.
    mark_speaker_busy(hold_s=0.8)
    return synth_s, play_s


async def _speak_edge_hamy(
    text: str,
    speaker: CameraCheckInAnnouncer,
    output_dir: Path,
    voice_name: str,
) -> tuple[float, float]:
    """Generate Microsoft's Vietnamese HoaiMy voice quickly and cache it."""
    import edge_tts
    from camera_tracking.voice.speaker_guard import mark_speaker_busy

    key = hashlib.sha1(f"{voice_name}\0{text}".encode("utf-8")).hexdigest()[:16]
    audio_path = output_dir / f"be_xinh_edge_{key}.mp3"
    synth_started = time.monotonic()
    if not audio_path.is_file():
        await edge_tts.Communicate(text, voice=voice_name).save(str(audio_path))
    synth_s = time.monotonic() - synth_started

    speaker.gain = active_voice_volume()
    play_started = time.monotonic()
    await asyncio.to_thread(speaker.play_file_sync, audio_path)
    play_s = time.monotonic() - play_started
    mark_speaker_busy(hold_s=0.8)
    return synth_s, play_s


async def run_live(args: argparse.Namespace) -> int:
    try:
        from google import genai
        from google.genai import types
    except ImportError as error:
        raise RuntimeError("Cần google-genai mới để dùng Gemini Live") from error

    api_key = os.getenv("GEMINI_API_KEY", "").strip().strip("'\"")
    if not api_key:
        raise RuntimeError("Thiếu GEMINI_API_KEY trong backend/.env")
    if not acquire_chat_lock("Bé Xinh Live"):
        raise SystemExit(0)

    config = load_config(args.config)
    voice_cfg = config.voice
    requested_audio_mode = os.getenv("BE_XINH_AUDIO_MODE", "auto").strip().lower()
    if requested_audio_mode not in {"auto", "camera", "laptop"}:
        raise RuntimeError(
            "BE_XINH_AUDIO_MODE chỉ nhận auto, camera hoặc laptop"
        )
    audio_mode = requested_audio_mode
    if audio_mode == "auto":
        audio_mode = "camera" if _camera_audio_reachable() else "laptop"
        print(
            f"[Bé Xinh Live] audio auto -> {audio_mode}",
            flush=True,
        )
    if audio_mode == "laptop":
        mic_input, microphone_name = _laptop_mic_input()
        speaker = LaptopSpeaker()
        print(
            f"[Bé Xinh Live] Laptop Test | mic={microphone_name} | loa=default",
            flush=True,
        )
    else:
        rtsp = imou_url(
            int(voice_cfg.voice_listen_channel),
            int(voice_cfg.voice_listen_subtype),
        )
        if not rtsp:
            raise RuntimeError("Thiếu IMOU_IP/USER/PASSWORD cho mic RTSP")
        mic_input = [
            "-fflags", "nobuffer", "-flags", "low_delay",
            "-rtsp_transport", "tcp", "-i", rtsp,
        ]
        speaker = CameraCheckInAnnouncer.from_env()
        if speaker is None:
            raise RuntimeError(
                "Không mở được IMOU direct speaker; kiểm tra IMOU_TALK_HELPER"
            )

    stt_lang = str(voice_cfg.voice_stt_lang)
    wake = _wake_transcriber(args.wake_model, stt_lang)
    command_model = (
        args.command_model
        or os.getenv("BE_XINH_COMMAND_STT", "zipformer")
    )
    command_stt = _command_transcriber(command_model, stt_lang)
    model = args.model or os.getenv("GEMINI_LIVE_MODEL", "").strip() or "gemini-3.8-live"
    fallback_models = [
        item.strip()
        for item in os.getenv(
            "GEMINI_LIVE_FALLBACK_MODELS",
            "gemini-2.5-flash-native-audio-latest",
        ).split(",")
        if item.strip()
    ]
    model_chain = list(dict.fromkeys([model, *fallback_models]))
    model_index = 0
    idle_s = args.conversation_idle_s or float(os.getenv("BE_XINH_CONVERSATION_IDLE_SECONDS", "45"))
    turn_wait_s = max(3.0, float(args.turn_wait_s))
    client = genai.Client(api_key=api_key)

    output_mode = os.getenv("BE_XINH_LIVE_OUTPUT", "edge").strip().lower()
    if output_mode not in {"edge", "zerotts", "native"}:
        print(
            f"[Bé Xinh Live] BE_XINH_LIVE_OUTPUT={output_mode!r} không hợp lệ; "
            "dùng edge.", flush=True,
        )
        output_mode = "edge"
    if audio_mode == "laptop" and output_mode == "native":
        print(
            "[Bé Xinh Live] Laptop Test chuyển native -> edge để phát loa local.",
            flush=True,
        )
        output_mode = "edge"
    voice_name = os.getenv("BE_XINH_LIVE_VOICE", "Leda").strip() or "Leda"
    zerotts_voice = os.getenv("BE_XINH_ZEROTTS_VOICE", "hamy").strip() or "hamy"
    edge_voice = (
        os.getenv("BE_XINH_EDGE_VOICE", "vi-VN-HoaiMyNeural").strip()
        or "vi-VN-HoaiMyNeural"
    )
    zerotts_device = os.getenv("BE_XINH_ZEROTTS_DEVICE", "cpu").strip() or "cpu"
    try:
        zerotts_threads = max(
            1, int(os.getenv("BE_XINH_ZEROTTS_THREADS", "12"))
        )
    except ValueError:
        zerotts_threads = 12
    tts = None
    answer_dir = ROOT / "output" / "be_xinh_live_answers"
    answer_dir.mkdir(parents=True, exist_ok=True)
    if output_mode == "zerotts":
        from camera_tracking.voice.zerotts_tts import ZeroTTSBackend

        print(
            f"[Bé Xinh Live] đang nạp ZeroTTS voice={zerotts_voice} "
            f"device={zerotts_device}...", flush=True,
        )
        tts = ZeroTTSBackend(
            device=zerotts_device,
            voice=zerotts_voice,
            num_threads=zerotts_threads,
        )
        await asyncio.to_thread(tts._load)

    live_config = _build_live_config(output_mode, voice_name)
    active_voice = {
        "edge": edge_voice,
        "zerotts": zerotts_voice,
        "native": voice_name,
    }[output_mode]
    print(
        f"[Bé Xinh Live] READY | model={model_chain[model_index]} "
        f"| fallback={model_chain[1:]} | output={output_mode} "
        f"| audio={audio_mode} | voice={active_voice} | command-stt="
        f"{command_model if command_stt is not None else 'gemini-audio'} "
        f"| session idle={idle_s:.0f}s"
    )

    try:
        retry_pcm: bytes | None = None
        connect_retries = 0
        while True:
            replaying_command = retry_pcm is not None
            if retry_pcm is not None:
                pcm = retry_pcm
                retry_pcm = None
                wake_text = "Bé Xinh ơi"
                print(
                    f"[Bé Xinh Live] thử kết nối lại "
                    f"({connect_retries}/2), giữ nguyên câu vừa nói...",
                    flush=True,
                )
            else:
                pcm = await asyncio.to_thread(
                    capture_utterance, mic_input, voice_cfg,
                    prompt="[Bé Xinh Live] gọi 'Bé Xinh ơi'...",
                    end_silence_ms=700, max_len_s=8.0, max_wait_s=None,
                    meter_s=4.0, respect_speaker_busy=not args.full_duplex,
                )
                if not pcm:
                    continue
                wake_text = await asyncio.to_thread(wake, pcm)
                if not wake_text or not is_wake(wake_text):
                    continue
                connect_retries = 0
            print(f"[Bé Xinh Live] WAKE: {wake_text}", flush=True)
            mark_turn()

            # A wake-only utterance must not trigger a long generated greeting:
            # it blocks the half-duplex mic and users naturally ask their real
            # question while Bé Xinh is still speaking. Use the short cached
            # Hạ My listening acknowledgement, then capture the command first.
            inline_command = (
                replaying_command
                or bool(strip_wake_command(wake_text))
            )
            if not inline_command:
                ack_path = ROOT / "output" / "voice_fillers" / "be_xinh_listening.wav"
                if ack_path.is_file():
                    try:
                        speaker.gain = active_voice_volume()
                        await asyncio.to_thread(speaker.play_file_sync, ack_path)
                        from camera_tracking.voice.speaker_guard import mark_speaker_busy

                        mark_speaker_busy(hold_s=0.35)
                    except Exception as error:  # noqa: BLE001
                        print(
                            f"[Bé Xinh Live] listening ACK lỗi: {error}",
                            flush=True,
                        )
                pcm = await asyncio.to_thread(
                    capture_utterance, mic_input, voice_cfg,
                    prompt="[Bé Xinh Live] mời bạn nói câu hỏi...",
                    end_silence_ms=1200, max_len_s=15.0,
                    max_wait_s=turn_wait_s,
                    respect_speaker_busy=not args.full_duplex,
                )
                if not pcm:
                    print(
                        "[Bé Xinh Live] chưa nghe được câu hỏi; về wake mode.",
                        flush=True,
                    )
                    clear_turn()
                    continue

            conversation_started = time.monotonic()
            session_opened = False
            try:
                active_model = model_chain[model_index]
                async with client.aio.live.connect(
                    model=active_model, config=live_config
                ) as session:
                    session_opened = True
                    # For inline wake + command this is the original recording;
                    # for wake-only this is the dedicated follow-up recording.
                    pending_pcm = pcm
                    while True:
                        if pending_pcm is None:
                            pending_pcm = await asyncio.to_thread(
                                capture_utterance, mic_input, voice_cfg,
                                prompt="[Bé Xinh Live] đang nghe...",
                                end_silence_ms=1200, max_len_s=15.0,
                                max_wait_s=turn_wait_s,
                                respect_speaker_busy=not args.full_duplex,
                            )
                        if not pending_pcm:
                            if time.monotonic() - conversation_started >= idle_s:
                                print("[Bé Xinh Live] hết conversation window; về wake mode.")
                                break
                            # Keep the same Live session/context while waiting a bit longer.
                            continue

                        sent_at = time.monotonic()
                        raw_pcm = pending_pcm
                        local_text = ""
                        if command_stt is not None:
                            local_text = (
                                await asyncio.to_thread(command_stt, raw_pcm)
                            ).strip()
                        if local_text:
                            print(
                                f"[Bé Xinh Live] Zipformer: {local_text}",
                                flush=True,
                            )
                            await session.send_client_content(
                                turns=types.Content(
                                    role="user",
                                    parts=[types.Part(text=local_text)],
                                ),
                                turn_complete=True,
                            )
                        else:
                            if command_stt is not None:
                                print(
                                    "[Bé Xinh Live] Zipformer trả rỗng; "
                                    "fallback Gemini audio.",
                                    flush=True,
                                )
                            pending_pcm = _agc_pcm(raw_pcm)
                            await _send_audio_turn(session, pending_pcm, types)
                        user_text, assistant_text, response_latency = await _receive_turn(
                            session, speaker, types,
                            stream_native_audio=output_mode == "native",
                        )
                        if local_text and not user_text:
                            user_text = local_text
                        if not local_text and not user_text and not assistant_text:
                            # Safety net for weak/noisy camera microphones:
                            # transcribe locally, then keep Gemini Live native
                            # voice output and the current conversation context.
                            fallback_text = await asyncio.to_thread(wake, raw_pcm)
                            fallback_text = (
                                strip_wake_command(fallback_text) or fallback_text
                            ).strip()
                            if fallback_text:
                                print(
                                    f"[Bé Xinh Live] audio fallback -> text: "
                                    f"{fallback_text}", flush=True,
                                )
                                await session.send_client_content(
                                    turns=types.Content(
                                        role="user",
                                        parts=[types.Part(text=fallback_text)],
                                    ),
                                    turn_complete=True,
                                )
                                user_text, assistant_text, response_latency = (
                                    await _receive_turn(
                                        session, speaker, types,
                                        stream_native_audio=output_mode == "native",
                                    )
                                )
                        if user_text:
                            print(f"[Bé Xinh Live] bạn: {user_text}", flush=True)
                        if assistant_text:
                            print(f"[Bé Xinh Live] Bé Xinh: {assistant_text}", flush=True)
                            if output_mode == "zerotts" and tts is not None:
                                try:
                                    first_hamy_s, hamy_total_s = await _speak_zerotts(
                                        assistant_text, tts, speaker,
                                        answer_dir, zerotts_voice,
                                    )
                                    print(
                                        f"[Bé Xinh Live] Hạ My first-audio="
                                        f"{first_hamy_s:.2f}s | stream="
                                        f"{hamy_total_s:.2f}s", flush=True,
                                    )
                                except Exception as error:  # noqa: BLE001
                                    print(
                                        f"[Bé Xinh Live] Hạ My/loa lỗi: "
                                        f"{type(error).__name__}: {error}",
                                        flush=True,
                                    )
                            elif output_mode == "edge":
                                try:
                                    tts_s, play_s = await _speak_edge_hamy(
                                        assistant_text, speaker, answer_dir, edge_voice
                                    )
                                    print(
                                        f"[Bé Xinh Live] HoaiMy TTS={tts_s:.2f}s "
                                        f"| loa={play_s:.2f}s", flush=True,
                                    )
                                except Exception as error:  # noqa: BLE001
                                    print(
                                        f"[Bé Xinh Live] HoaiMy/loa lỗi: "
                                        f"{type(error).__name__}: {error}",
                                        flush=True,
                                    )
                        total = time.monotonic() - sent_at
                        print(
                            f"[Bé Xinh Live] first-output={response_latency:.2f}s | "
                            f"turn={total:.2f}s", flush=True,
                        )
                        if not user_text and not assistant_text:
                            print("[Bé Xinh Live] turn rong (Gemini khong "
                                  "nghe ro) - ve cho goi.", flush=True)
                            break
                        conversation_started = time.monotonic()
                        pending_pcm = None
            except Exception as error:  # noqa: BLE001
                error_text = str(error).lower()
                quota_error = "quota" in error_text or "1011" in error_text
                if quota_error and model_index + 1 < len(model_chain):
                    failed_model = model_chain[model_index]
                    model_index += 1
                    retry_pcm = pcm
                    connect_retries = 0
                    print(
                        f"[Bé Xinh Live] {failed_model} hết quota; chuyển "
                        f"{model_chain[model_index]} và giữ nguyên câu hỏi.",
                        flush=True,
                    )
                    continue
                if (
                    not session_opened
                    and isinstance(error, (TimeoutError, OSError))
                    and connect_retries < 2
                ):
                    connect_retries += 1
                    retry_pcm = pcm
                    await asyncio.sleep(0.8 * connect_retries)
                    continue
                connect_retries = 0
                print(f"[Bé Xinh Live] session lỗi: {type(error).__name__}: {error}", flush=True)
                print("[Bé Xinh Live] về wake mode; classic assistant vẫn là fallback.", flush=True)
            finally:
                clear_turn()
    except KeyboardInterrupt:
        print("\n[Bé Xinh Live] dừng.")
    finally:
        clear_turn()
        try:
            release_chat_lock()
        except Exception:  # noqa: BLE001
            pass
        speaker.close()
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(run_live(parse_args())))


if __name__ == "__main__":
    main()
