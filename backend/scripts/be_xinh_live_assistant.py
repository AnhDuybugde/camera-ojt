"""Realtime Bé Xinh voice assistant using Gemini Live + IMOU camera audio.

Flow
----
1. A tiny/local STT loop listens only for the wake phrase "Bé Xinh ơi".
2. One Gemini Live session is opened for a conversation window.
3. Subsequent user turns are sent as 16 kHz PCM directly to Gemini; no command
   STT and no ZeroTTS are required on the realtime path.
4. Gemini's native 24 kHz PCM is forwarded to IMOU VisualTalk chunk-by-chunk.
5. Local camera/runtime tools are executed through Live function calling.

The existing ``be_xinh_assistant.py`` remains the classic fallback path.
"""
from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
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
        acquire_chat_lock,
        capture_utterance,
        is_wake,
        release_chat_lock,
        strip_wake_command,
    )
from camera_tracking.voice.rtsp_voice_listener import TARGET_RATE, transcribe_segment_fw

LIVE_SYSTEM = """Bạn là Bé Xinh, người bạn đồng hành bằng giọng nói của hệ thống Camera-OJT.
Nói tiếng Việt tự nhiên, ấm áp, trực tiếp và ngắn gọn. Mặc định trả lời 1-3 câu;
chỉ dài hơn khi người dùng thực sự cần giải thích. Hãy nhớ mạch hội thoại trong
phiên hiện tại để hiểu các câu nối tiếp như 'còn người đó?', 'vậy ngày mai?',
'hoặc tại sao?'. Khi cần dữ liệu camera, chấm công, thời gian, thời tiết hoặc
trạng thái hệ thống, hãy dùng function tool thay vì đoán. Nếu dữ liệu không đủ,
nói rõ điều chưa chắc; không tự bịa danh tính hoặc trạng thái chấm công.
Không dùng Markdown trong lời nói. Tự xưng là Bé Xinh.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config" / "default.yaml"))
    parser.add_argument("--model", default=None,
                        help="Gemini Live model; default GEMINI_LIVE_MODEL/gemini-3.8-live")
    parser.add_argument("--wake-model", default="tiny")
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


def _chunk_pcm(pcm: bytes, milliseconds: int = 100):
    size = max(640, int(TARGET_RATE * 2 * milliseconds / 1000))
    for offset in range(0, len(pcm), size):
        part = pcm[offset:offset + size]
        if part:
            yield part


def _agc_pcm(pcm: bytes, peak_target: float = 0.5,
             max_gain: float = 10.0, silence_peak: int = 100) -> bytes:
    """Khuếch đại mic về mức Gemini Live nghe rõ.

    Mic camera thường chỉ đạt đỉnh ~0.05 (-26dBFS); gửi thô khiến Live
    không nhận ra tiếng nói và im luôn (treo turn). Chuẩn hoá đỉnh về
    ~0.5, trần gain 10x; đoạn thật sự im (đỉnh < ~0.003) thì giữ nguyên
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
                        timeout_s: float = 12.0) -> tuple[str, str, float]:
    """Receive one Live turn, stream native audio, and service tool calls.

    Có timeout: mic ồn/yếu khiến Live im luôn thì không treo turn vĩnh
    viễn mà trả turn rỗng để caller về wake mode cho người dùng gọi lại.
    """
    stream = None
    user_text = ""
    assistant_text = ""
    first_audio_at: float | None = None
    started = time.monotonic()
    try:
        async with asyncio.timeout(timeout_s):
            async for response in session.receive():
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
                for audio_chunk in audio_chunks:
                    if stream is None:
                        first_audio_at = time.monotonic()
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
                        await session.send_tool_response(function_responses=function_responses)

                if content is not None:
                    in_tx = getattr(content, "input_transcription", None)
                    out_tx = getattr(content, "output_transcription", None)
                    if in_tx is not None and getattr(in_tx, "text", None):
                        user_text = str(in_tx.text).strip()
                    if out_tx is not None and getattr(out_tx, "text", None):
                        assistant_text += str(out_tx.text)
                    if bool(getattr(content, "interrupted", False)) and stream is not None:
                        stream.close(abort=True)
                        stream = None
                    if bool(getattr(content, "turn_complete", False)):
                        break
    except TimeoutError:
        print(f"[Bé Xinh Live] Live im lang qua {timeout_s:.0f}s "
              f"(mic on/yeu?) - ve cho goi.", flush=True)
    finally:
        if stream is not None:
            stream.close()
    latency = (first_audio_at - started) if first_audio_at is not None else (time.monotonic() - started)
    return user_text, assistant_text.strip(), latency


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
    rtsp = imou_url(int(voice_cfg.voice_listen_channel), int(voice_cfg.voice_listen_subtype))
    if not rtsp:
        raise RuntimeError("Thiếu IMOU_IP/USER/PASSWORD cho mic RTSP")
    mic_input = ["-fflags", "nobuffer", "-flags", "low_delay",
                 "-rtsp_transport", "tcp", "-i", rtsp]

    speaker = CameraCheckInAnnouncer.from_env()
    if speaker is None:
        raise RuntimeError("Không mở được IMOU direct speaker; kiểm tra IMOU_TALK_HELPER")

    wake = _wake_transcriber(args.wake_model, str(voice_cfg.voice_stt_lang))
    model = args.model or os.getenv("GEMINI_LIVE_MODEL", "").strip() or "gemini-3.8-live"
    idle_s = args.conversation_idle_s or float(os.getenv("BE_XINH_CONVERSATION_IDLE_SECONDS", "45"))
    turn_wait_s = max(3.0, float(args.turn_wait_s))
    client = genai.Client(api_key=api_key)

    live_config = {
        "response_modalities": ["AUDIO"],
        "system_instruction": LIVE_SYSTEM,
        "tools": [{"function_declarations": TOOL_DECLARATIONS}],
        "input_audio_transcription": {},
        "output_audio_transcription": {},
        "realtime_input_config": {
            "automatic_activity_detection": {"disabled": True},
        },
    }
    print(f"[Bé Xinh Live] READY | model={model} | session idle={idle_s:.0f}s")

    try:
        while True:
            pcm = await asyncio.to_thread(
                capture_utterance, mic_input, voice_cfg,
                prompt="[Bé Xinh Live] gọi 'Bé Xinh ơi'...",
                end_silence_ms=550, max_len_s=5.0, max_wait_s=None,
                meter_s=4.0, respect_speaker_busy=not args.full_duplex,
            )
            if not pcm:
                continue
            wake_text = await asyncio.to_thread(wake, pcm)
            if not wake_text or not is_wake(wake_text):
                continue
            print(f"[Bé Xinh Live] WAKE: {wake_text}", flush=True)
            mark_turn()
            conversation_started = time.monotonic()
            try:
                async with client.aio.live.connect(model=model, config=live_config) as session:
                    # If the wake sentence already contains a command, let Live hear
                    # the original audio so names/intonation are preserved.
                    pending_pcm = pcm if strip_wake_command(wake_text) else None
                    while True:
                        if pending_pcm is None:
                            pending_pcm = await asyncio.to_thread(
                                capture_utterance, mic_input, voice_cfg,
                                prompt="[Bé Xinh Live] đang nghe...",
                                end_silence_ms=450, max_len_s=12.0,
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
                        pending_pcm = _agc_pcm(raw_pcm)
                        await _send_audio_turn(session, pending_pcm, types)
                        user_text, assistant_text, response_latency = await _receive_turn(
                            session, speaker, types
                        )
                        if not user_text and not assistant_text:
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
                                    await _receive_turn(session, speaker, types)
                                )
                        total = time.monotonic() - sent_at
                        if user_text:
                            print(f"[Bé Xinh Live] bạn: {user_text}", flush=True)
                        if assistant_text:
                            print(f"[Bé Xinh Live] Bé Xinh: {assistant_text}", flush=True)
                        print(
                            f"[Bé Xinh Live] first-audio={response_latency:.2f}s | "
                            f"turn={total:.2f}s", flush=True,
                        )
                        if not user_text and not assistant_text:
                            print("[Bé Xinh Live] turn rong (Gemini khong "
                                  "nghe ro) - ve cho goi.", flush=True)
                            break
                        conversation_started = time.monotonic()
                        pending_pcm = None
            except Exception as error:  # noqa: BLE001
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
