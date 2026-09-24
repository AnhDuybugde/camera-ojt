"""Tro ly voice "Be Xinh" - mic camera RTSP + Gemini + loa camera P2P.

WAIT_WAKE (STT) --"Be Xinh oi"--> "Be Xinh nghe ne" (loa camera)
--> LISTEN_CMD --> STT medium --> GEMINI 1-CALL JSON --> TOOLS (59 tool)
--> TTS CUDA --> loa camera P2P --> WAIT_WAKE.

Toan bo am thanh (ready/listening/thinking/missed/dap) deu phat
qua loa camera P2P, khong phat loa may tinh.
Filler "dang suy nghi" chi phat khi xu ly > 1.2s (song song).
Half-duplex: phat loa thi khong nghe.

Can .env: IMOU_IP/USER/PASSWORD (mic), IMOU_DEVICE_ID +
IMOU_CAMERA_PASSWORD (loa), GEMINI_API_KEY.

Chay:
    python scripts/halinh_assistant.py [--rounds 0]
    --rounds 0 = chay lien tuc den Ctrl+C.
    Test luat wake khong can mic:
    python scripts/halinh_assistant.py --wake-test "be xinh oi"
"""

from __future__ import annotations

import argparse
from collections import deque
import hashlib
import io
import json
import os
from queue import Empty, Queue
import re
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from camera_tracking.config import load_config
from camera_tracking.audio import AudioEventRouter, HamyCompanion
from camera_tracking.integration import BackendStatusClient, BeXinhStatusBridge
from camera_tracking.voice.qa_tools import (
    TOOL_FUNCS,
    build_tool_catalog,
    parse_router_json,
)
from camera_tracking.voice.rtsp_voice_listener import (
    DEFAULT_AUDIO_FILTER,
    FRAME_BYTES,
    TARGET_RATE,
    NoiseFloorTracker,
    find_ffmpeg,
    rms_level,
    transcribe_segment_fw,
)
from camera_tracking.voice.speaker_guard import (
    mark_speaker_busy,
    speaker_busy,
)
from camera_tracking.voice.turn_guard import (
    clear_turn,
    mark_turn,
    turn_active,
)
from camera_tracking.voice.voice_trigger import normalize_trigger_text

BE_XINH_PROMPT = """Bạn là Bé Xinh, trợ lý giọng nói tiếng Việt thân thiện cho hệ thống Camera-OJT.

Quy tắc trả lời:
* Trả lời trực tiếp câu hỏi của người dùng.
* Chỉ trả lời tối đa 1-2 câu, ưu tiên 1 câu.
* Giữ câu trả lời dưới 40 từ nếu có thể.
* Không giải thích dài dòng, không lặp lại câu hỏi.
* Không dùng Markdown, bullet point hoặc tiêu đề.
* Dùng tiếng Việt tự nhiên, thân thiện, dễ thương và dễ nghe khi chuyển thành giọng nói.
* Luôn tự xưng là Bé Xinh; không dùng tên Hà Linh.
 * Với câu hỏi đơn giản, chỉ đưa ra thông tin cần thiết.
 * TUYỆT ĐỐI không hỏi ngược lại người dùng dưới mọi hình thức: không câu
   hỏi làm rõ, không gợi ý hỏi tiếp, không đặt nhiều câu hỏi trong một lượt
   (hệ thống chưa có memory hội thoại).
 * Nếu thiếu thông tin, trả lời ngay với giả định hợp lý nhất và nói rõ
   giả định đó trong cùng 1 câu, không hỏi lại.
* Nếu có dữ liệu từ tool/API, chỉ tóm tắt kết quả quan trọng nhất cho người dùng.
* Không mô tả quá trình suy nghĩ hoặc xử lý của bạn.
* Không nói bạn là AI, trừ khi người dùng hỏi trực tiếp.
* Khi gọi tool, chỉ dùng kết quả tool để tạo câu trả lời cuối cùng."""

# Require both words to avoid waking on the common adjective "xinh" alone.
BE_XINH_TOKENS = {"be", "xinh"}
BE_XINH_PHRASES = {
    "be xinh oi", "be xinh a", "xin chao be xinh", "chao be xinh",
    "be xinh co nghe khong", "oi be xinh", "be xinh nghe khong",
}

# Bias STT ve cum goi: giup model bat "Be Xinh oi" chuan hon.
WAKE_STT_PROMPT = "Bé Xinh ơi. Chào Bé Xinh. Bé Xinh có nghe không."
# Token thua khi boc lenh kem theo cau wake ("Be Xinh oi, may gio roi").
_WAKE_FILLER_TOKENS = BE_XINH_TOKENS | {
    "oi", "ơi", "o", "a", "à", "ạ", "e", "ê", "nha", "nhe", "nhé", "ne",
}


def strip_wake_command(wake_text: str) -> str:
    """Boc phan lenh con thua sau cum goi wake.

    "Be Xinh oi, may gio roi" -> "may gio roi". Tra ve "" neu khong
    co lenh kem (chi goi wake don thuan).
    """
    import re

    tokens = re.findall(r"[a-z0-9]+",
                        normalize_trigger_text(wake_text))
    kept = [t for t in tokens if t not in _WAKE_FILLER_TOKENS]
    command = " ".join(kept)
    if (len(kept) >= 2 and len(command) >= 6
            and any(len(t) >= 3 for t in kept)):
        return command
    return ""


# Giao thuc router JSON 1-call: Gemini CHI hieu cau hoi, KHONG viet dap an.
# - Cau tra loi truc tiep -> {"type":"direct",...,"text":"<cau tra loi>"}.
# - Can tool -> {"type":"tool","tool":"<ten>","args":{...},"text":""}.
# Backend chay tool local (tool tu format cau noi) -> TTS. 1 request/vong.
ROUTER_PROTOCOL = """Bạn là bộ định tuyến câu hỏi tiếng Việt, trả về JSON MỘT DÒNG duy nhất, không thêm bất kỳ nội dung nào ngoài JSON:
{"type":"direct|tool","tool":"<tên_tool hoặc null>","args":{},"text":"<câu trả lời ngắn nếu type=direct, ngược lại để rỗng>"}

Luật:
 * type=direct: tự trả lời ngắn gọn 1-2 câu, dưới 40 từ, tiếng Việt tự nhiên để đọc thành tiếng, không Markdown. CẤM hỏi ngược lại (không câu hỏi làm rõ, không gợi ý hỏi tiếp).
 * type=tool: chỉ dùng tool trong danh sách dưới; điền đủ tham số bắt buộc; tham số tùy chọn không biết thì bỏ qua (tool có giá trị mặc định).
 * Không bịa tham số: nghe "mấy giờ" thì gọi get_current_time không tham số; nghe thiếu thông tin bắt buộc thì type=direct với câu trả lời tốt nhất theo giả định mặc định (nêu giả định), KHÔNG hỏi lại.
* Ví dụ: "Python là gì" -> {"type":"direct","tool":null,"args":{},"text":"Python là ngôn ngữ lập trình phổ biến, dễ đọc và dùng nhiều cho AI."}
* Ví dụ: "Đà Nẵng hôm nay bao nhiêu độ" -> {"type":"tool","tool":"get_weather","args":{"city":"Đà Nẵng"},"text":""}

Danh sách tool:
"""

ROUTER_SYSTEM = BE_XINH_PROMPT + "\n\n" + ROUTER_PROTOCOL + build_tool_catalog()

FILLER_FILES = {
    "ready": "be_xinh_ready.wav",
    "listening": "be_xinh_listening.wav",
    "thinking": "be_xinh_thinking.wav",
    "got_it": "be_xinh_got_it.wav",
    "missed": "be_xinh_missed.wav",
}

FILLER_TEXTS = {
    "ready": "Bé Xinh sẵn sàng rồi. Gọi Bé Xinh ơi để trò chuyện nha.",
    "listening": "Bé Xinh nghe nè.",
    "thinking": "Bé Xinh đang nghĩ một chút nha.",
    "got_it": "Bé Xinh hiểu rồi nè.",
    "missed": "Bé Xinh nghe chưa rõ. Bạn gọi Bé Xinh ơi rồi nói lại nha.",
}


def ack_via_camera(talk, filler_dir: Path, channel: int) -> None:
    """Bao 'Be Xinh nghe ne' bang loa CAMERA (P2P, chan - half-duplex).

    Phat xong moi mo mic (mic mo SAU khi loa dut) nen khong can
    chay background. Cham hon local 1-2s nhung toan bo am thanh
    ra loa camera.
    """
    try:
        talk(filler_dir / FILLER_FILES["listening"], channel, tail_s=0.35)
    except Exception as error:
        print(f"[BeXinhChat] ack camera loi (bo qua, nghe luon): {error}",
              flush=True)


# Token thua cua chinh cau ack ("Be Xinh nghe ne") khi lot vao mic.
_ACK_ECHO_TOKENS = _WAKE_FILLER_TOKENS | {"nghe", "nè", "di"}


def strip_ack_echo(command: str) -> str:
    """Cat tien to vang cua cau ack khoi dau lenh (chong loa lot vao mic)."""
    import re

    tokens = re.findall(r"[a-z0-9]+",
                        normalize_trigger_text(command))
    cut = 0
    for token in tokens:
        if token in _ACK_ECHO_TOKENS:
            cut += 1
        else:
            break
    return " ".join(tokens[cut:])


def fast_answer(question: str) -> str | None:
    """Tra loi ngay cau thuong gap, KHONG goi Gemini (~14s).

    Tra None khi khong khop -> flow Gemini binh thuong. Chi nhan cau
    ngan, ro rang de khong cuop cau phuc tap cua router.
    """
    norm = normalize_trigger_text(question)
    if not norm:
        return None
    words = norm.split()
    n = len(words)

    if n <= 8 and ("may gio" in norm or norm in ("gio", "gio roi", "bay gio may gio")):
        try:
            from camera_tracking.voice.qa_tools import get_current_time
            return get_current_time()
        except Exception:
            return None
    if n <= 10 and ("ngay may" in norm or "ngay bao nhieu" in norm
                      or "thu may" in norm):
        try:
            from camera_tracking.voice.qa_tools import get_current_date
            return get_current_date()
        except Exception:
            return None
    # "hom nay ..." mot minh rat mo ho (thoi tiet hom nay? lich hom nay?) ->
    # chi nhan khi cuc ngan va khong co tu khoa thoi tiet.
    if (n <= 5 and "hom nay" in norm
            and not any(k in norm for k in
                        ("thoi tiet", "nhiet do", "do am", "mua", "nang",
                         "gio", "bao", "uv", "do c"))):
        try:
            from camera_tracking.voice.qa_tools import get_current_date
            return get_current_date()
        except Exception:
            return None
    if n <= 6 and (norm.startswith("cam on") or norm in ("cam on", "cam on nhe", "ok cam on")):
        return "Không có gì. Chúc bạn một ngày tốt lành."
    if n <= 6 and ("tam biet" in norm or "hen gap lai" in norm):
        return "Tạm biệt bạn. Hẹn gặp lại."
    if n <= 6 and ("ban la ai" in norm or norm in ("ten gi", "ten ban la gi")):
        return "Mình là Bé Xinh, trợ lý giọng nói của hệ thống camera."
    if norm in ("xin chao", "chao", "chao ban", "hello", "hi"):
        return "Chào bạn nha. Bé Xinh nghe đây."
    if n <= 4 and ("khoe khong" in norm or norm in ("khoe khong", "ban khoe khong")):
        return "Mình khỏe. Rất vui được nói chuyện với bạn."
    return None


def gemini_model_candidates(primary: str, fallback: str) -> list[str]:
    """Return an ordered, de-duplicated Gemini failover chain."""
    result: list[str] = []
    for item in (primary, fallback):
        name = str(item or "").strip()
        if name and name not in result:
            result.append(name)
    return result


def is_gemini_capacity_error(error: BaseException) -> bool:
    message = str(error).upper()
    return ("429" in message or "RESOURCE_EXHAUSTED" in message
            or "503" in message or "UNAVAILABLE" in message)


def pcm_to_wav_bytes(pcm: bytes, sample_rate: int = TARGET_RATE) -> bytes:
    """Wrap mono signed-16 PCM as an inline WAV for Gemini audio input."""
    destination = io.BytesIO()
    with wave.open(destination, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(int(sample_rate))
        wav.writeframes(pcm)
    return destination.getvalue()


def active_voice_volume(
    config_path: Path | None = None,
) -> float:
    """Return the single live speaker gain selected in the dashboard."""
    path = config_path or ROOT.parent / ".runtime" / "be-xinh-volume.json"
    payload: dict = {}
    try:
        candidate = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(candidate, dict):
            payload = candidate
    except (OSError, json.JSONDecodeError):
        pass

    try:
        # normal_percent keeps old settings compatible during migration.
        value = payload.get("percent", payload.get("normal_percent", 85))
        return max(10, min(100, int(value))) / 100.0
    except (TypeError, ValueError):
        return 0.85


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config" / "default.yaml"))
    parser.add_argument("--rounds", type=int, default=0,
                        help="So vong hoi-dap (0 = lien tuc den Ctrl+C).")
    parser.add_argument("--channel", type=int, default=None,
                        help="Kenh loa P2P (mac dinh p2p_channel trong config).")
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--tts-voice", default="hamy")
    parser.add_argument("--tts-device", default="cpu",
                        help="cuda (co GPU) hoac cpu.")
    parser.add_argument("--wake-model", default="tiny",
                        help="Model STT vong cho wake 'Be Xinh oi': zipformer "
                             "= nhanh ~0.1s (khuyen nghi), small/tiny = "
                             "whisper CPU.")
    parser.add_argument("--cmd-model", default="tiny",
                        help="Model STT vong nghe lenh: zipformer=Zipformer-vi "
                             "30M CPU ~0.3s (khuyen nghi), medium=chuan ~9s "
                             "CPU, small=nhanh ~2-3s.")
    parser.add_argument("--think-filler-s", type=float, default=30.0,
                        help="Xu ly lau hon nguong nay thi phat filler thinking "
                             "(cao de filler P2P ~10s khong cong them delay).")
    parser.add_argument("--think-timeout-s", type=float, default=90.0)
    parser.add_argument("--wake-miss-n", type=int, default=3,
                        help="Hut wake lien tiep bao nhieu lan co tieng thi "
                             "nhac 'nghe khong ro' 1 cau.")
    parser.add_argument("--wake-test", default=None,
                        help="Test luat wake voi text go tay, khong can mic.")
    parser.add_argument("--with-search", action="store_true",
                        help="Bat search grounding (can billing/quota cao hon).")
    return parser.parse_args()


def is_wake(text: str) -> bool:
    """True for an invocation, not merely any sentence naming Be Xinh."""
    norm = normalize_trigger_text(text)
    if not norm:
        return False
    for phrase in BE_XINH_PHRASES:
        if phrase in norm:
            return True
    import re

    tokens = re.findall(r"[a-z0-9]+", norm)
    # Accept the short standalone call "Be Xinh", but reject speaker echo
    # such as "Be Xinh xin chao ban" from the assistant's own reply.
    return tokens == ["be", "xinh"]


def capture_utterance(
    ffmpeg_input: list[str],
    voice_cfg,
    *,
    prompt: str,
    end_silence_ms: int = 1000,
    max_len_s: float = 10.0,
    max_wait_s: float | None = None,
    meter_s: float | None = None,
    respect_speaker_busy: bool = True,
) -> bytes | None:
    """Ghi 1 doan noi: doi tieng noi (vo han neu max_wait_s=None) roi cat khi im.

    ffmpeg_input: phan input tuy device, VD camera
    ["-fflags", "nobuffer", "-rtsp_transport", "tcp", "-i", rtsp_url].
    meter_s: in dong muc mic moi N giay trong luc doi. Het max_wait_s
    thi tra None; None (EOF/stream chet) thi vong goi tu mo lai stream.
    respect_speaker_busy: True = tat mic khi loa dang phat de tranh echo.
    False chi danh cho chan doan mic; production luon de True.
    """
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        print("[BeXinhChat] khong tim thay ffmpeg")
        return None
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error",
        *ffmpeg_input,
        "-map", "0:a:0?", "-vn", "-ac", "1", "-ar", str(TARGET_RATE),
        "-af", DEFAULT_AUDIO_FILTER,
        "-f", "s16le", "pipe:1",
    ]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, bufsize=0)
    except Exception as error:
        print(f"[BeXinhChat] khong mo duoc mic: {error}")
        return None
    assert proc.stdout is not None
    chunk_queue: Queue[bytes | None] = Queue()

    def _read_audio() -> None:
        try:
            while True:
                chunk = proc.stdout.read(4096)
                if not chunk:
                    break
                chunk_queue.put(chunk)
        finally:
            chunk_queue.put(None)

    reader = threading.Thread(
        target=_read_audio, name="be-xinh-rtsp-reader", daemon=True)
    reader.start()
    last_audio_at = time.monotonic()
    tracker = NoiseFloorTracker(
        static_threshold=float(voice_cfg.voice_vad_threshold),
        factor=float(voice_cfg.voice_vad_floor_factor),
        abs_min=float(voice_cfg.voice_vad_floor_min),
        ceiling=float(voice_cfg.voice_vad_ceiling),
    )
    frame_ms = 30
    heard = 0
    started = False
    buf = bytearray()
    silence_ms = 0
    wait_s = 0.0
    peak = 0.0
    last_thr = 0.0
    last_floor = 0.0
    got_frame = False
    leftover = bytearray()
    # Preserve the syllable just before VAD crosses its threshold. Without
    # pre-roll, short Vietnamese initials and names are frequently clipped.
    pre_roll: deque[bytes] = deque(maxlen=10)  # 10 x 30 ms = 300 ms
    next_meter = time.monotonic() + (meter_s or 0)
    max_bytes = int(max_len_s * TARGET_RATE * 2)
    busy = False
    busy_checks = 0
    busy_announced = False
    print(prompt, flush=True)
    try:
        while True:
            try:
                chunk = chunk_queue.get(timeout=1.0)
            except Empty:
                if time.monotonic() - last_audio_at >= 6.0:
                    print("[BeXinhChat] RTSP mic im 6s - tu mo lai luong.",
                          flush=True)
                    break
                continue
            if chunk is None:
                break
            last_audio_at = time.monotonic()
            leftover += chunk
            n_frames = 0
            while len(leftover) >= FRAME_BYTES:
                frame = bytes(leftover[:FRAME_BYTES])
                del leftover[:FRAME_BYTES]
                n_frames += 1
                # Half-duplex xuyen process: loa (greeter chao / Be Xinh
                # noi) dang phat thi TAT MIC - bo frame, huy doan dang
                # ghi do de khoi nghe lai chinh tieng loa. Check moi 8
                # frame (~4 lan/giay) cho nhe.
                # CHI ap dung trong turn (sau WAKE): luc cho wake mic
                # luon mo de khong miss tieng goi, ke ca loa dang chao.
                busy_checks += 1
                if (respect_speaker_busy and busy_checks % 8 == 1):
                    busy = speaker_busy()
                if busy and respect_speaker_busy:
                    heard = 0
                    if not busy_announced:
                        busy_announced = True
                        print("[BeXinhChat] loa dang phat - tam tat mic.",
                              flush=True)
                    if started:
                        started = False
                        buf.clear()
                        silence_ms = 0
                    pre_roll.clear()
                    continue
                level = rms_level(frame)
                thr = tracker.update(level)
                if not got_frame:
                    got_frame = True
                    print("[BeXinhChat] mic thông, gọi 'Bé Xinh ơi' đi.",
                          flush=True)
                if level > peak:
                    peak = level
                last_thr, last_floor = thr, tracker.floor
                loud = level >= thr
                if not started:
                    pre_roll.append(frame)
                    heard = heard + 1 if loud else 0
                    if heard >= 2:
                        started = True
                        buf += b"".join(pre_roll)
                        pre_roll.clear()
                else:
                    buf += frame
                    silence_ms = 0 if loud else silence_ms + frame_ms
                    if silence_ms >= end_silence_ms or len(buf) >= max_bytes:
                        return bytes(buf)
            if meter_s and time.monotonic() >= next_meter and not started:
                next_meter = time.monotonic() + meter_s
                show, peak = peak, 0.0
                if not got_frame:
                    print("[BeXinhChat:meter] chua co frame audio "
                          "(sai --mic-source?)", flush=True)
                else:
                    mark = "CO TIENG" if show >= last_thr else "yen"
                    print(f"[BeXinhChat:meter] {mark} dinh={show:.4f} "
                          f"nguong={last_thr:.4f} nen={last_floor:.4f}",
                          flush=True)
            if not started and max_wait_s is not None:
                wait_s += n_frames * frame_ms / 1000.0
                if wait_s >= max_wait_s:
                    return None
    finally:
        try:
            proc.terminate()
        except Exception:
            pass
        try:
            proc.wait(timeout=0.8)
        except subprocess.TimeoutExpired:
            proc.kill()
        reader.join(timeout=0.5)
    return bytes(buf) if started and buf else None


def think(question: str, model: str, max_tokens: int,
          use_search: bool = False,
          audio_pcm: bytes | None = None) -> tuple[str, float, list[str]]:
    """Router JSON 1-call duy nhat: Gemini chi hieu cau hoi.

    - direct -> text trong JSON la dap an cuoi (khong goi them).
    - tool   -> dispatch TOOL_FUNCS local; chinh tool format cau noi.
    Khong bao gio goi Gemini lan 2/vong. Tra (dap, giay tool, tools).
    """
    import os

    try:
        from google import genai
        from google.genai import types
    except ImportError as error:
        raise RuntimeError("chua cai google-genai: pip install google-genai") from error
    api_key = os.getenv("GEMINI_API_KEY", "").strip().strip("'\"")
    if not api_key:
        raise RuntimeError("thieu GEMINI_API_KEY trong .env")
    client = genai.Client(
        api_key=api_key,
        http_options={"retry_options": {"attempts": 1}})
    config_args = {
        "system_instruction": ROUTER_SYSTEM,
        "max_output_tokens": max(32, int(max_tokens)),
        "temperature": 0,
    }
    # Flash-Lite currently rejects an explicit thinking configuration.  It is
    # our low-latency quota fallback, so leave that field out for Lite models.
    if "lite" not in model.lower():
        config_args["thinking_config"] = types.ThinkingConfig(
            thinking_budget=0)
    config = types.GenerateContentConfig(**config_args)
    contents: object = question
    if audio_pcm:
        glossary = os.getenv(
            "BE_XINH_SPEECH_GLOSSARY",
            "Bé Xinh, Anh Quốc Ngọc, Quốc Ngọc, Anh Khoa 617, Đà Nẵng, "
            "Camera OJT, AI Mind",
        ).strip()
        contents = [
            (
                "Hãy nghe chính xác câu nói tiếng Việt trong file âm thanh. "
                "Âm thanh là nguồn chính; transcript cục bộ bên dưới chỉ là "
                "gợi ý và có thể sai. Tự phát hiện tiếng Việt, tiếng Anh hoặc "
                "câu nói trộn hai ngôn ngữ; ưu tiên tiếng Việt và hiểu được "
                "giọng miền Trung/Đà Nẵng. Không tự đổi tên riêng thành từ "
                "tiếng Anh gần âm. Trong JSON kết quả, thêm field heard chứa "
                "nguyên văn câu đã nghe và field language là vi, en hoặc "
                "mixed. Thực hiện yêu cầu theo đúng giao thức JSON trong "
                "system instruction.\n"
                f"Từ vựng/tên riêng ưu tiên nếu âm thanh phù hợp: {glossary}.\n"
                f"Transcript gợi ý: {question}"
            ),
            types.Part.from_bytes(
                data=pcm_to_wav_bytes(audio_pcm), mime_type="audio/wav"),
        ]
    response = client.models.generate_content(
        model=model, contents=contents, config=config)
    raw = (getattr(response, "text", "") or "").strip()
    if audio_pcm:
        match = re.search(r"\{.*\}", raw, re.S)
        if match:
            try:
                details = json.loads(match.group(0))
                heard = " ".join(str(details.get("heard") or "").split())
                language = str(details.get("language") or "unknown").strip()
                if heard:
                    print(f"[BeXinhChat] Gemini nghe ({language}): {heard}",
                          flush=True)
            except (TypeError, ValueError):
                pass
    plan = parse_router_json(raw)
    if plan["type"] == "direct" or not plan["tool"]:
        return plan["text"], 0.0, []
    name = str(plan["tool"])
    args = plan["args"]
    func = TOOL_FUNCS.get(name)
    if not callable(func) or name.startswith("_"):
        return f"Hàm {name} chưa hỗ trợ.", 0.0, [name]
    t0 = time.monotonic()
    try:
        answer = func(**{k: v for k, v in args.items()})
    except TypeError:
        try:
            answer = func()
        except Exception as error:
            answer = f"lỗi tool {name}: {error}"
    except Exception as error:
        answer = f"lỗi tool {name}: {error}"
    tool_s = time.monotonic() - t0
    print(f"[BeXinhChat] tool {name}({args}) -> {answer}", flush=True)
    return str(answer), tool_s, [name]


def main() -> None:
    args = parse_args()
    if args.wake_test is not None:
        print(f"{args.wake_test!r} -> wake={is_wake(args.wake_test)}")
        return
    if args.with_search:
        print("[BeXinhChat] single-call JSON mode khong ho tro search "
              "grounding, bo qua --with-search.", flush=True)
    config = load_config(args.config)
    voice_cfg = config.voice
    channel = args.channel or int(voice_cfg.p2p_channel)
    model = args.model or os.getenv("GEMINI_MODEL", "").strip() or "gemini-3.5-flash"
    if not os.getenv("GEMINI_API_KEY", "").strip():
        raise SystemExit("Thieu GEMINI_API_KEY: them vao .env roi chay lai.")
    from camera_tracking.camera.camera_imou import imou_url

    rtsp = imou_url(int(voice_cfg.voice_listen_channel),
                    int(voice_cfg.voice_listen_subtype))
    if not rtsp:
        raise SystemExit("Thieu IMOU_IP/USER/PASSWORD cho mic RTSP.")
    mic_input = ["-fflags", "nobuffer", "-flags", "low_delay",
                 "-rtsp_transport", "tcp", "-i", rtsp]

    filler_dir = ROOT / "output" / "voice_fillers"
    missing = [f for f in FILLER_FILES.values() if not (filler_dir / f).is_file()]
    if missing:
        print(f"[BeXinhChat] sẽ tạo filler lần đầu: {missing}", flush=True)

    from faster_whisper import WhisperModel

    from camera_tracking.voice.p2p_talk import (
        ImouP2PCredentials,
        ImouP2PTalkOutput,
    )
    from camera_tracking.voice.zerotts_tts import ZeroTTSBackend
    from camera_tracking.audio.announcer import CameraCheckInAnnouncer

    print("[BeXinhChat] warmup (STT wake + lenh, TTS, tunnel loa)...", flush=True)
    t0 = time.monotonic()
    # RTX 2050 4GB: STT giu CPU (CUDA tran VRAM voi ZeroTTS).
    # zipformer-vi 30M (~0.3s) cho ca wake + lenh; whisper giu lam fallback.
    # Chon qua --wake-model / --cmd-model.
    def _pick(name: str, default: str) -> str:
        val = str(name or default).strip().lower()
        if val not in ("tiny", "small", "medium", "zipformer"):
            print(f"[BeXinhChat] model {name!r} la, dung {default}.", flush=True)
            return default
        return val
    _wake_model = _pick(args.wake_model, "zipformer")
    _cmd_model = _pick(args.cmd_model, "zipformer")
    _zipformer = None
    if "zipformer" in (_wake_model, _cmd_model):
        try:
            from camera_tracking.voice.sherpa_stt import SherpaZipformerSTT

            _zipformer = SherpaZipformerSTT(
                num_threads=int(getattr(voice_cfg, "stt_threads", 6)))
            _zipformer.warmup()
            print(f"[BeXinhChat] STT zipformer CPU xong "
                  f"({time.monotonic() - t0:.1f}s, dung chung wake+lenh).",
                  flush=True)
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            # A fresh Windows setup may not have sherpa-onnx or its external
            # model yet. Keep voice chat usable with the already-supported
            # faster-whisper backend instead of terminating the process.
            print(f"[BeXinhChat] Zipformer chua san sang ({error}); "
                  "tu dong dung Whisper.", flush=True)
            if _wake_model == "zipformer":
                _wake_model = "tiny"
            if _cmd_model == "zipformer":
                _cmd_model = "tiny"
            _zipformer = None

    def _zip_text(pcm: bytes) -> str:
        # Transducer khong co cua no_speech nhu Whisper: tieng on phong
        # van ra 1-2 tu rac ("CAC", "TRONG") -> <2 tu coi nhu im lang.
        # Wake ("ha linh oi"=3 tu, "linh oi"=2 tu) va lenh that (ca cau)
        # deu di qua; chi mat case goi cut "linh" 1 tu.
        assert _zipformer is not None
        text = _zipformer.transcribe_pcm(pcm)
        if len(text.split()) < 2:
            return ""
        return text

    if _wake_model == "zipformer":
        def transcribe_wake(pcm: bytes) -> str:
            return _zip_text(pcm)
    else:
        stt_wake = WhisperModel(_wake_model, device="cpu",
                                compute_type="int8")

        def transcribe_wake(pcm: bytes) -> str:
            return transcribe_segment_fw(
                stt_wake, pcm, str(voice_cfg.voice_stt_lang),
                max_no_speech_prob=0.60,
                min_avg_logprob=-1.10,
                initial_prompt=WAKE_STT_PROMPT,
            )
        print(f"[BeXinhChat] STT wake {_wake_model} xong "
              f"({time.monotonic() - t0:.1f}s).", flush=True)
    t0 = time.monotonic()
    if _cmd_model == "zipformer":
        assert _zipformer is not None

        def transcribe_cmd(pcm: bytes) -> str:
            return _zip_text(pcm)

        print(f"[BeXinhChat] STT lenh zipformer CPU xong "
              f"({time.monotonic() - t0:.1f}s).", flush=True)
    else:
        stt_medium = WhisperModel(_cmd_model, device="cpu",
                                  compute_type="int8")

        def transcribe_cmd(pcm: bytes) -> str:
            return transcribe_segment_fw(
                stt_medium, pcm, str(voice_cfg.voice_stt_lang),
                max_no_speech_prob=float(
                    voice_cfg.voice_max_no_speech_prob),
                min_avg_logprob=float(voice_cfg.voice_min_avg_logprob),
            )

        print(f"[BeXinhChat] STT lenh {_cmd_model} CPU xong "
              f"({time.monotonic() - t0:.1f}s).", flush=True)
    tts = ZeroTTSBackend(device=args.tts_device, voice=args.tts_voice)
    tts._load()
    filler_dir.mkdir(parents=True, exist_ok=True)
    for _key, _filename in FILLER_FILES.items():
        _target = filler_dir / _filename
        if not _target.is_file():
            print(f"[BeXinhChat] tạo {_filename}...", flush=True)
            tts.save_wav(FILLER_TEXTS[_key], _target)
    # Tunnel RIENG cho voice chat (halinh_bind_port, mac dinh 18087): greeter
    # tracking giu 18086. Chung port -> bind fail -> talk 25-70s.
    _halinh_port = int(getattr(voice_cfg, "halinh_bind_port",
                               getattr(voice_cfg, "p2p_bind_port", 18086) + 1))
    p2p = ImouP2PTalkOutput(
        ImouP2PCredentials.from_env(),
        channel=channel,
        # Voice chat must never block tens of seconds on a broken talkback
        # session. Fail fast, listen to the question, reconnect next turn.
        timeout=min(6.0, float(voice_cfg.p2p_timeout_s)),
        attempts=1,
        retry_delay=float(voice_cfg.p2p_retry_delay_s),
        sample_rate=int(voice_cfg.p2p_sample_rate),
        volume=float(voice_cfg.p2p_volume),
        bind_port=_halinh_port,
        establish_timeout=10.0,
        retry_persistent=False,
    )
    direct_speaker = CameraCheckInAnnouncer.from_env()
    # Duong truyen loa phai SAN SANG THAT moi duoc noi "san sang":
    # cho dong bo, retry vai lan; khong noi thi dung (khong gia vo san sang).
    tunnel_ok = direct_speaker is not None
    if tunnel_ok:
        print("[BeXinhChat] loa LAN truc tiep san sang; P2P la fallback.",
              flush=True)
    else:
        for attempt in range(1, 4):
            try:
                p2p.warmup(raise_on_error=True)
                tunnel_ok = True
                break
            except Exception as error:
                print(f"[BeXinhChat] cho loa lan {attempt}/3: {error}", flush=True)
                time.sleep(5.0)
    if not tunnel_ok:
        raise SystemExit(
            "[BeXinhChat] khong mo duoc duong truyen loa sau 3 lan (kiem tra mang/ "
            "cloud Imou roi chay lai). Chua san sang, dung chuong trinh.")

    _last_volume_ratio: float | None = None

    def talk(path: str | Path, _channel: int | None = None, *,
             tail_s: float = 5.0) -> None:
        nonlocal _last_volume_ratio
        volume_ratio = active_voice_volume()
        if direct_speaker is not None:
            direct_speaker.gain = volume_ratio
        p2p.volume = volume_ratio
        if volume_ratio != _last_volume_ratio:
            print(f"[BeXinhChat] am luong: "
                  f"{round(volume_ratio * 100)}%.", flush=True)
            _last_volume_ratio = volume_ratio
        # Phan xu loa: greeter (hinh anh) co the dang phat do loi chao
        # da xep tu truoc WAKE -> doi loa ranh (toi da 10s) roi moi noi,
        # khong de 2 ben noi chong nhau.
        _waited = 0.0
        while speaker_busy() and _waited < 10.0:
            time.sleep(0.2)
            _waited += 0.2
        if _waited >= 10.0:
            print(f"[BeXinhChat] loa van ban sau 10s, cu phat "
                  f"{Path(path).name} (co the chong lan nhau).", flush=True)
        # Giu mic rong truoc: P2P relay cham hon file nhieu (7-28s) nen
        # duration file + margin khong du; giu thua roi that lai sau.
        from camera_tracking.voice.speaker_guard import media_seconds

        _dur = media_seconds(path) or 6.0
        try:
            mark_speaker_busy(hold_s=_dur + 30.0)
        except Exception:  # noqa: BLE001 - guard chi la phu
            pass
        t0 = time.monotonic()
        try:
            if direct_speaker is not None:
                try:
                    direct_speaker.play_file_sync(path)
                except Exception as direct_error:  # noqa: BLE001
                    print(f"[BeXinhChat] loa LAN loi, thu P2P: {direct_error}",
                          flush=True)
                    try:
                        p2p(path, channel)
                    except Exception as fallback_error:  # noqa: BLE001
                        print(f"[BeXinhChat] ca LAN va P2P deu loi; bo audio "
                              f"nay, van giu mic: {fallback_error}", flush=True)
            else:
                try:
                    p2p(path, channel)
                except Exception as fallback_error:  # noqa: BLE001
                    print(f"[BeXinhChat] P2P loi; bo audio nay, van giu mic: "
                          f"{fallback_error}", flush=True)
        finally:
            # RTSP mic co the con audio cu vai giay sau khi P2P bao phat xong.
            # Cau tra loi dung 5s de khong tu danh thuc; rieng ACK truyen
            # tail_s=0.35 de khong mat phan dau cau hoi cua nguoi dung.
            try:
                mark_speaker_busy(hold_s=max(0.0, float(tail_s)))
            except Exception:  # noqa: BLE001
                pass
        print(f"[BeXinhChat] loa xong {Path(path).name} "
              f"({time.monotonic() - t0:.1f}s).", flush=True)

    # Zone events and conversational Q&A share the same direct speaker.
    # The router drops proactive lines while turn_guard is active, so a user
    # saying "Bé Xinh ơi" always has priority and never competes with a zone
    # greeting in a second process.
    zone_bridge_stop = threading.Event()
    zone_bridge_thread: threading.Thread | None = None
    if direct_speaker is not None:
        zone_companion = HamyCompanion.from_env(direct_speaker)
        zone_router = AudioEventRouter(
            direct_speaker,
            turn_guard=turn_active,
        )
        zone_bridge = BeXinhStatusBridge(
            zone_companion,
            event_router=zone_router,
        )
        zone_client = BackendStatusClient(
            os.getenv(
                "TRACKING_STATUS_URL",
                "http://127.0.0.1:8765/status.json",
            ),
            timeout_s=2.0,
        )

        def _run_zone_bridge() -> None:
            last_error = ""
            while not zone_bridge_stop.wait(0.12):
                started = time.monotonic()
                try:
                    zone_bridge.process(
                        zone_client.fetch(),
                        now_s=started,
                        wall_time_s=time.time(),
                    )
                    last_error = ""
                except Exception as error:  # noqa: BLE001 - sidecar must stay alive
                    message = str(error)
                    if message != last_error:
                        print(
                            f"[BeXinhChat/Zone] chờ backend: {message}",
                            flush=True,
                        )
                        last_error = message

        zone_bridge_thread = threading.Thread(
            target=_run_zone_bridge,
            name="be-xinh-zone-bridge",
            daemon=True,
        )
        zone_bridge_thread.start()
        print("[BeXinhChat] Audio Zone bridge: ACTIVE", flush=True)

    # Pre-convert TAT CA filler sang AAC ngay tu warmup: moi cau
    # (ready/listening/thinking/missed) tiet kiem 1-2s spawn ffmpeg o lan
    # phat dau, quan trong khi P2P relay von da cham.
    try:
        for _filler in FILLER_FILES.values():
            p2p._convert_cached(filler_dir / _filler)
        print("[BeXinhChat] da preload AAC fillers.", flush=True)
    except Exception as preload_error:  # noqa: BLE001 - chi la toi uu
        print(f"[BeXinhChat] preload AAC fillers loi (bo qua): {preload_error}",
              flush=True)

    # Do not occupy/reopen camera talkback merely to announce startup. This
    # saves several seconds and avoids a stale camera session killing chat.
    print("[BeXinhChat] sẵn sàng. Gọi 'Bé Xinh ơi' để bắt đầu. "
          "Ctrl+C để dừng.", flush=True)

    out_dir = ROOT / "output" / "qa_cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    done_rounds = 0
    wake_miss = 0
    empty_wake = 0  # dem doan on STT ra rong (de heartbeat, khoi tuong chet)
    # A quota failure is normally project/model-scoped, not a useful signal to
    # retry on every utterance. Keep the fast Lite fallback hot for five
    # minutes before probing the primary model again.
    primary_retry_after = 0.0
    try:
        while True:
            if args.rounds and done_rounds >= args.rounds:
                break
            # -- WAIT_WAKE (tiny 0.3s, nhanh gap ~10x medium) --
            # Van ne loa khi cho wake: neu khong, camera se thu lai chinh
            # cau tra loi cua Be Xinh va tu kich hoat mot vong chao moi.
            pcm = capture_utterance(
                mic_input, voice_cfg,
                prompt="[BeXinhChat] ... nghe (gọi 'Bé Xinh ơi') ...",
                end_silence_ms=700, max_len_s=4.0, max_wait_s=None,
                meter_s=3.0, respect_speaker_busy=True)
            if not pcm:
                time.sleep(1.0)
                continue
            t0 = time.monotonic()
            wake_text = transcribe_wake(pcm)
            print(f"[BeXinhChat] STT wake ({time.monotonic() - t0:.2f}s, {_wake_model}): "
                  f"{wake_text!r}", flush=True)
            if not wake_text or not is_wake(wake_text):
                # Co tieng nhung khong phai goi Be Xinh: dem hut; du N lan thi
                # nhac 1 cau roi ve cho (STT rong/im lang thi khong dem).
                if wake_text:
                    wake_miss += 1
                    empty_wake = 0
                    print(f"[BeXinhChat] hut wake ({wake_miss}): {wake_text}",
                          flush=True)
                    if wake_miss >= max(1, args.wake_miss_n):
                        wake_miss = 0
                        talk(filler_dir / FILLER_FILES["missed"])
                else:
                    # Tieng quat/va dap phong: STT ra rong. In thua de biet
                    # mic van song (khong thi tuong chuong trinh treo).
                    empty_wake += 1
                    if empty_wake % 5 == 1:
                        print("[BeXinhChat] mic van nghe (toan tieng on phong) - "
                              "gọi 'Bé Xinh ơi' để bắt đầu.", flush=True)
                continue
            wake_miss = 0
            empty_wake = 0
            print(f"[BeXinhChat] WAKE: {wake_text}", flush=True)
            # Khoa turn-taking: pipeline tracking nhin thay thi tam ngung
            # xep chao vẫy/mặt de 2 loa khong noi chong giua hoi-dap.
            # TTL tu het neu crash giua turn.
            try:
                mark_turn()
            except Exception:  # noqa: BLE001
                pass
            question = strip_wake_command(wake_text)
            stt_s = 0.0
            if question:
                # Wake kem lenh ("Be Xinh oi, may gio roi"): tiny bat wake
                # nhanh nhung co the sai lenh dai -> chay lai model lenh
                # tren cung pcm de lay lenh chuan (hiem, chap nhan cham 1 lan).
                t0 = time.monotonic()
                refined = transcribe_cmd(pcm)
                refined_cmd = strip_wake_command(refined) if refined else ""
                print(f"[BeXinhChat] refine lenh ({_cmd_model}) "
                      f"({time.monotonic() - t0:.2f}s): {refined!r} -> "
                      f"{refined_cmd!r}", flush=True)
                if refined_cmd:
                    question = refined_cmd
                print(f"[BeXinhChat] lenh kem wake: {question}", flush=True)
                total0 = time.monotonic()
            else:
                print("[BeXinhChat] ack loa ('Bé Xinh nghe nè' xong rồi hãy nói).",
                      flush=True)
                ack_via_camera(talk, filler_dir, channel)

            # -- LISTEN_CMD (model lenh: medium/zipformer) --
            if not question:
                stt_s = 0.0
                attempt = 0
                while attempt < 2:
                    attempt += 1
                    # Trong turn (sau WAKE): mic NE loa dang phat de lenh
                    # khong dinh echo cau chao/ack.
                    pcm = capture_utterance(
                        mic_input, voice_cfg,
                        prompt="[BeXinhChat] NÓI NGAY (Bé Xinh đang nghe)...",
                        end_silence_ms=700,
                        max_len_s=7.0,
                        max_wait_s=6.0, respect_speaker_busy=True)
                    if not pcm:
                        print(f"[BeXinhChat] lan {attempt}: khong nghe thay "
                              f"cau hoi trong 6s.", flush=True)
                        continue
                    total0 = time.monotonic()
                    t0 = time.monotonic()
                    question = transcribe_cmd(pcm)
                    stt_s = time.monotonic() - t0
                    if question:
                        question = strip_ack_echo(question)
                    if question and is_wake(question) \
                            and not strip_wake_command(question):
                        print(f"[BeXinhChat] lan {attempt}: chi nghe wake "
                              f"({question}) - doi cau lenh that.",
                              flush=True)
                        question = ""
                    if question:
                        break
                    print(f"[BeXinhChat] lan {attempt}: STT rong ({stt_s:.2f}s) - "
                          f"van nghe tiep.", flush=True)
                if not question:
                    print("[BeXinhChat] het luot nghe, ve cho wake moi.",
                          flush=True)
                    clear_turn()
                    continue
            print(f"[BeXinhChat] lenh ({stt_s:.2f}s): {question}", flush=True)
            # Cau noi cho tool get_last_transcript (vong sau hoi lai van co).
            try:
                (out_dir / "last_transcript.txt").write_text(
                    question, encoding="utf-8")
            except OSError:
                pass

            # -- THINK (thread + filler thinking theo nguong) --
            def _run_think_once(target_model: str):
                one: dict = {}

                def _one() -> None:
                    try:
                        ans, tls, used = think(
                            question, target_model, args.max_tokens,
                            args.with_search, audio_pcm=pcm)
                        one.update(answer=ans, tool_s=tls, used=used)
                    except Exception as error:  # noqa: BLE001
                        one.update(error=error)

                worker = threading.Thread(target=_one, name="halinh-think",
                                          daemon=True)
                start = time.monotonic()
                worker.start()
                worker.join(timeout=float(args.think_filler_s))
                if worker.is_alive():
                    talk(filler_dir / FILLER_FILES["thinking"])
                    worker.join(timeout=float(args.think_timeout_s))
                    if worker.is_alive():
                        one.update(error=TimeoutError(
                            f"think qua {args.think_timeout_s}s chua xong"))
                return one, time.monotonic() - start

            think_ok = False
            answer = ""
            tool_s = 0.0
            llm_s = 0.0
            # Fast-path: cau thuong gap tra ngay, khoi doi Gemini ~14s.
            fast_enabled = os.getenv(
                "BE_XINH_FAST_PATH", "false").strip().lower() in (
                    "1", "true", "yes", "on")
            fast = fast_answer(question) if fast_enabled else None
            if fast is not None:
                result = {"answer": fast, "tool_s": 0.0, "used": ["fast-path"]}
                llm_s = 0.0
                think_ok = True
                print(f"[BeXinhChat] fast-path (0s): {fast}", flush=True)
            fallback_model = os.getenv(
                "GEMINI_FALLBACK_MODEL", "gemini-3.5-flash-lite").strip()
            if time.monotonic() < primary_retry_after:
                model_chain = gemini_model_candidates(fallback_model, "")
                print(f"[BeXinhChat] primary dang cooldown; dung ngay "
                      f"{fallback_model}.", flush=True)
            else:
                model_chain = gemini_model_candidates(model, fallback_model)
            for model_index, target_model in enumerate(model_chain):
                if think_ok:
                    break
                result, llm_s = _run_think_once(target_model)
                error = result.get("error")
                if error is None:
                    answer = result.get("answer", "")
                    tool_s = result.get("tool_s", 0.0)
                    think_ok = True
                    if model_index:
                        result.setdefault("used", []).insert(
                            0, f"fallback:{target_model}")
                        print(f"[BeXinhChat] Gemini fallback OK: "
                              f"{target_model}.", flush=True)
                    break
                message = str(error)
                is_capacity = is_gemini_capacity_error(error)
                has_fallback = model_index + 1 < len(model_chain)
                if is_capacity and has_fallback:
                    if target_model == model:
                        primary_retry_after = time.monotonic() + 300.0
                    print(f"[BeXinhChat] {target_model} het quota/ban; "
                          f"chuyen ngay sang {model_chain[model_index + 1]}.",
                          flush=True)
                    continue
                if not is_capacity:
                    print(f"[BeXinhChat] Gemini loi gon: "
                          f"{type(error).__name__}: "
                          f"{message[:200]}", flush=True)
                    if isinstance(error, TimeoutError):
                        try:
                            timeout_wav = (out_dir /
                                           f"be_xinh_timeout_{int(time.time())}.wav")
                            tts.save_wav(
                                "Bé Xinh nghĩ lâu quá, bạn nói lại giúp Bé Xinh nha.",
                                timeout_wav)
                            talk(timeout_wav)
                        except Exception as speak_error:  # noqa: BLE001
                            print(f"[BeXinhChat] bao timeout loi: {speak_error}",
                                  flush=True)
                    break
                print(f"[BeXinhChat] Gemini het quota ca chuoi "
                      f"{model_chain} - bao ngay, khong treo 60s.", flush=True)
                try:
                    quota_wav = out_dir / f"be_xinh_quota_{int(time.time())}.wav"
                    tts.save_wav(
                        "Bé Xinh đang hết lượt trả lời thông minh rồi. "
                        "Bạn thử lại sau một chút nha.",
                        quota_wav)
                    talk(quota_wav)
                except Exception as speak_error:  # noqa: BLE001
                    print(f"[BeXinhChat] bao quota loi: "
                          f"{speak_error}", flush=True)
                break
            if not think_ok:
                clear_turn()
                continue
            answer = result.get("answer", "")
            tool_s = result.get("tool_s", 0.0)
            if not answer:
                print(f"[BeXinhChat] Gemini tra rong ({llm_s:.2f}s), ve cho.",
                      flush=True)
                clear_turn()
                continue
            print(f"[BeXinhChat] dap ({llm_s:.2f}s, tool {tool_s:.2f}s "
                  f"{result.get('used', [])}): {answer}", flush=True)

            # -- SPEAK (loa camera P2P) --
            t0 = time.monotonic()
            answer_key = hashlib.sha1(
                f"{args.tts_voice}\0{answer}".encode("utf-8")
            ).hexdigest()[:16]
            wav = out_dir / f"be_xinh_answer_{answer_key}.wav"
            if not wav.is_file():
                tts.save_wav(answer, wav)
            tts_s = time.monotonic() - t0
            t0 = time.monotonic()
            talk(wav, channel)
            play_s = time.monotonic() - t0
            print(f"[BeXinhChat] TTS {tts_s:.2f}s | loa {play_s:.2f}s | "
                  f"VONG {time.monotonic() - total0:.1f}s.", flush=True)
            done_rounds += 1
            clear_turn()
    except KeyboardInterrupt:
        print("\n[BeXinhChat] dung.", flush=True)
    finally:
        try:
            clear_turn()
        except Exception:  # noqa: BLE001
            pass
        zone_bridge_stop.set()
        if zone_bridge_thread is not None:
            zone_bridge_thread.join(timeout=3.0)
        if direct_speaker is not None:
            direct_speaker.close()
        p2p.close()


if __name__ == "__main__":
    main()
