"""Tro ly voice "Ha Linh" - BAN CHINH THUC (mic camera RTSP + loa camera P2P).

WAIT_WAKE (STT small) --"Ha Linh oi"--> "Ha Linh nghe ne" (loa camera)
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
    python scripts/halinh_assistant.py --wake-test "ha linh oi"
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from camera_tracking.config import load_config
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
)
from camera_tracking.voice.voice_trigger import normalize_trigger_text

HA_LINH_PROMPT = """Bạn là Hà Linh, trợ lý giọng nói tiếng Việt cho hệ thống Camera-OJT.

Quy tắc trả lời:
* Trả lời trực tiếp câu hỏi của người dùng.
* Chỉ trả lời tối đa 1-2 câu, ưu tiên 1 câu.
* Giữ câu trả lời dưới 40 từ nếu có thể.
* Không giải thích dài dòng, không lặp lại câu hỏi.
* Không dùng Markdown, bullet point hoặc tiêu đề.
* Dùng tiếng Việt tự nhiên, dễ nghe khi chuyển sang giọng nói.
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

# Wake: "Ha Linh oi" lam chinh; "Ha Linh"/"Linh oi" du phong.
# Token-match tren text da normalize. "Ha" dung mot minh KHONG wake
# (de nham voi cuoi/ha ha); "Linh" mot minh chi wake khi segment ngan.
HA_LINH_TOKENS = {"ha", "linh"}
HA_LINH_PHRASES = {"ha linh oi", "ha linh a", "xin chao ha linh",
                   "chao ha linh", "ha linh co nghe khong", "linh oi",
                   "oi ha linh", "ha linh nghe khong", "chao linh",
                   "xin chao linh"}
HA_LINH_MAX_TOKENS_SINGLE = 2  # "Linh"/"Linh oi" ngan moi wake.

# Biais STT ve cum goi: giup small bat "Ha Linh oi" chuan hon.
WAKE_STT_PROMPT = "Hà Linh ơi. Linh ơi. Chào Hà Linh."
# Token thua khi boc lenh kem theo cau wake ("Ha Linh oi, may gio roi").
_WAKE_FILLER_TOKENS = HA_LINH_TOKENS | {
    "oi", "ơi", "o", "a", "à", "ạ", "e", "ê", "nha", "nhe", "nhé", "ne",
}


def strip_wake_command(wake_text: str) -> str:
    """Boc phan lenh con thua sau cum goi wake.

    "Ha Linh oi, may gio roi" -> "may gio roi". Tra ve "" neu khong
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

ROUTER_SYSTEM = HA_LINH_PROMPT + "\n\n" + ROUTER_PROTOCOL + build_tool_catalog()

FILLER_FILES = {
    "ready": "ready.wav",
    "listening": "listening.wav",
    "thinking": "thinking.wav",
    "got_it": "got_it.wav",
    "missed": "missed.wav",
}


def ack_via_camera(talk, filler_dir: Path, channel: int) -> None:
    """Bao 'Ha Linh nghe ne' bang loa CAMERA (P2P, chan - half-duplex).

    Phat xong moi mo mic (mic mo SAU khi loa dut) nen khong can
    chay background. Cham hon local 1-2s nhung toan bo am thanh
    ra loa camera.
    """
    try:
        talk(filler_dir / FILLER_FILES["listening"], channel)
    except Exception as error:
        print(f"[HaLinh] ack camera loi (bo qua, nghe luon): {error}",
              flush=True)


# Token thua cua chinh cau ack ("Ha Linh nghe ne") khi lot vao mic.
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
        return "Mình là Hà Linh, trợ lý giọng nói của hệ thống camera."
    if norm in ("xin chao", "chao", "chao ban", "hello", "hi"):
        return "Chào bạn. Hà Linh nghe đây."
    if n <= 4 and ("khoe khong" in norm or norm in ("khoe khong", "ban khoe khong")):
        return "Mình khỏe. Rất vui được nói chuyện với bạn."
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config" / "default.yaml"))
    parser.add_argument("--rounds", type=int, default=0,
                        help="So vong hoi-dap (0 = lien tuc den Ctrl+C).")
    parser.add_argument("--channel", type=int, default=None,
                        help="Kenh loa P2P (mac dinh p2p_channel trong config).")
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--tts-voice", default="maichi")
    parser.add_argument("--tts-device", default="cuda",
                        help="cuda (co GPU) hoac cpu.")
    parser.add_argument("--wake-model", default="zipformer",
                        help="Model STT vong cho wake 'Ha Linh oi': zipformer "
                             "= nhanh ~0.1s (khuyen nghi), small/tiny = "
                             "whisper CPU.")
    parser.add_argument("--cmd-model", default="zipformer",
                        help="Model STT vong nghe lenh: zipformer=Zipformer-vi "
                             "30M CPU ~0.3s (khuyen nghi), medium=chuan ~9s "
                             "CPU, small=nhanh ~2-3s.")
    parser.add_argument("--think-filler-s", type=float, default=6.0,
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
    """True khi cau STT goi Ha Linh ("Ha Linh oi" / "Ha Linh" / "Linh oi")."""
    norm = normalize_trigger_text(text)
    if not norm:
        return False
    for phrase in HA_LINH_PHRASES:
        if phrase in norm:
            return True
    import re

    tokens = re.findall(r"[a-z0-9]+", norm)
    have_ha = "ha" in tokens
    have_linh = "linh" in tokens
    if have_ha and have_linh:
        return True
    if not have_linh:
        return False  # "Ha"/"ha ha" dung mot minh thi khong wake.
    # Chi con "linh": stutter ("linh linh") hoac segment ngan ("linh oi").
    if tokens.count("linh") >= 2 and len(tokens) <= 3:
        return True
    # "Linh"/"Linh oi" ngan thi wake; "Ha" dung mot minh thi khong.
    return (len(tokens) <= HA_LINH_MAX_TOKENS_SINGLE
            and all(t in _WAKE_FILLER_TOKENS for t in tokens))


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
    respect_speaker_busy: True = tat mic khi loa dang phat (trong turn,
    tranh echo). False = mic luon mo (vong cho wake, tranh miss tieng goi).
    """
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        print("[HaLinh] khong tim thay ffmpeg")
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
        print(f"[HaLinh] khong mo duoc mic: {error}")
        return None
    assert proc.stdout is not None
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
    next_meter = time.monotonic() + (meter_s or 0)
    max_bytes = int(max_len_s * TARGET_RATE * 2)
    busy = False
    busy_checks = 0
    busy_announced = False
    print(prompt, flush=True)
    try:
        while True:
            chunk = proc.stdout.read(4096)
            if not chunk:
                break
            leftover += chunk
            n_frames = 0
            while len(leftover) >= FRAME_BYTES:
                frame = bytes(leftover[:FRAME_BYTES])
                del leftover[:FRAME_BYTES]
                n_frames += 1
                # Half-duplex xuyen process: loa (greeter chao / Ha Linh
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
                        print("[HaLinh] loa dang phat - tam tat mic.",
                              flush=True)
                    if started:
                        started = False
                        buf.clear()
                        silence_ms = 0
                    continue
                level = rms_level(frame)
                thr = tracker.update(level)
                if not got_frame:
                    got_frame = True
                    print("[HaLinh] mic thong, goi 'Hà Linh ơi' di.",
                          flush=True)
                if level > peak:
                    peak = level
                last_thr, last_floor = thr, tracker.floor
                loud = level >= thr
                if not started:
                    heard = heard + 1 if loud else 0
                    if heard >= 3:
                        started = True
                else:
                    buf += frame
                    silence_ms = 0 if loud else silence_ms + frame_ms
                    if silence_ms >= end_silence_ms or len(buf) >= max_bytes:
                        return bytes(buf)
            if meter_s and time.monotonic() >= next_meter and not started:
                next_meter = time.monotonic() + meter_s
                show, peak = peak, 0.0
                if not got_frame:
                    print("[HaLinh:meter] chua co frame audio "
                          "(sai --mic-source?)", flush=True)
                else:
                    mark = "CO TIENG" if show >= last_thr else "yen"
                    print(f"[HaLinh:meter] {mark} dinh={show:.4f} "
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
    return bytes(buf) if started and buf else None


def think(question: str, model: str, max_tokens: int,
          use_search: bool = False) -> tuple[str, float, list[str]]:
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
    config = types.GenerateContentConfig(
        system_instruction=ROUTER_SYSTEM,
        max_output_tokens=300,
        temperature=0,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )
    response = client.models.generate_content(
        model=model, contents=question, config=config)
    raw = (getattr(response, "text", "") or "").strip()
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
    print(f"[HaLinh] tool {name}({args}) -> {answer}", flush=True)
    return str(answer), tool_s, [name]


def main() -> None:
    import os

    args = parse_args()
    if args.wake_test is not None:
        print(f"{args.wake_test!r} -> wake={is_wake(args.wake_test)}")
        return
    if args.with_search:
        print("[HaLinh] single-call JSON mode khong ho tro search "
              "grounding, bo qua --with-search.", flush=True)
    config = load_config(args.config)
    voice_cfg = config.voice
    channel = args.channel or int(voice_cfg.p2p_channel)
    model = args.model or os.getenv("GEMINI_MODEL", "").strip() or "gemini-3.6-flash"
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
        raise SystemExit(f"Thieu filler WAV {missing}: gen truoc (ZeroTTS).")

    from faster_whisper import WhisperModel

    from camera_tracking.voice.p2p_talk import (
        ImouP2PCredentials,
        ImouP2PTalkOutput,
    )
    from camera_tracking.voice.zerotts_tts import ZeroTTSBackend

    print("[HaLinh] warmup (STT wake + lenh, TTS CUDA, tunnel loa)...", flush=True)
    t0 = time.monotonic()
    # RTX 2050 4GB: STT giu CPU (CUDA tran VRAM voi ZeroTTS).
    # zipformer-vi 30M (~0.3s) cho ca wake + lenh; whisper giu lam fallback.
    # Chon qua --wake-model / --cmd-model.
    def _pick(name: str, default: str) -> str:
        val = str(name or default).strip().lower()
        if val not in ("tiny", "small", "medium", "zipformer"):
            print(f"[HaLinh] model {name!r} la, dung {default}.", flush=True)
            return default
        return val
    _wake_model = _pick(args.wake_model, "zipformer")
    _cmd_model = _pick(args.cmd_model, "zipformer")
    _zipformer = None
    if "zipformer" in (_wake_model, _cmd_model):
        from camera_tracking.voice.sherpa_stt import SherpaZipformerSTT

        _zipformer = SherpaZipformerSTT(
            num_threads=int(getattr(voice_cfg, "stt_threads", 6)))
        _zipformer.warmup()
        print(f"[HaLinh] STT zipformer CPU xong "
              f"({time.monotonic() - t0:.1f}s, dung chung wake+lenh).",
              flush=True)

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
        print(f"[HaLinh] STT wake {_wake_model} xong "
              f"({time.monotonic() - t0:.1f}s).", flush=True)
    t0 = time.monotonic()
    if _cmd_model == "zipformer":
        assert _zipformer is not None

        def transcribe_cmd(pcm: bytes) -> str:
            return _zip_text(pcm)

        print(f"[HaLinh] STT lenh zipformer CPU xong "
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

        print(f"[HaLinh] STT lenh {_cmd_model} CPU xong "
              f"({time.monotonic() - t0:.1f}s).", flush=True)
    tts = ZeroTTSBackend(device=args.tts_device, voice=args.tts_voice)
    tts._load()
    # Tunnel RIENG cho Ha Linh (halinh_bind_port, mac dinh 18087): greeter
    # tracking giu 18086. Chung port -> bind fail -> talk 25-70s.
    _halinh_port = int(getattr(voice_cfg, "halinh_bind_port",
                               getattr(voice_cfg, "p2p_bind_port", 18086) + 1))
    p2p = ImouP2PTalkOutput(
        ImouP2PCredentials.from_env(),
        channel=channel,
        timeout=float(voice_cfg.p2p_timeout_s),
        attempts=int(voice_cfg.p2p_attempts),
        retry_delay=float(voice_cfg.p2p_retry_delay_s),
        sample_rate=int(voice_cfg.p2p_sample_rate),
        volume=float(voice_cfg.p2p_volume),
        bind_port=_halinh_port,
    )
    # Duong truyen loa phai SAN SANG THAT moi duoc noi "san sang":
    # cho dong bo, retry vai lan; khong noi thi dung (khong gia vo san sang).
    tunnel_ok = False
    for attempt in range(1, 4):
        try:
            p2p.warmup(raise_on_error=True)
            tunnel_ok = True
            break
        except Exception as error:
            print(f"[HaLinh] cho loa lan {attempt}/3: {error}", flush=True)
            time.sleep(5.0)
    if not tunnel_ok:
        raise SystemExit(
            "[HaLinh] khong mo duoc duong truyen loa sau 3 lan (kiem tra mang/ "
            "cloud Imou roi chay lai). Chua san sang, dung chuong trinh.")

    def talk(path: str | Path, _channel: int | None = None) -> None:
        # Phan xu loa: greeter (hinh anh) co the dang phat do loi chao
        # da xep tu truoc WAKE -> doi loa ranh (toi da 10s) roi moi noi,
        # khong de 2 ben noi chong nhau.
        _waited = 0.0
        while speaker_busy() and _waited < 10.0:
            time.sleep(0.2)
            _waited += 0.2
        if _waited >= 10.0:
            print(f"[HaLinh] loa van ban sau 10s, cu phat "
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
            p2p(path, channel)
        finally:
            # Phat xong: chi giu duoi vang 2s cho mic mo lai ngay.
            try:
                mark_speaker_busy(hold_s=2.0)
            except Exception:  # noqa: BLE001
                pass
        print(f"[HaLinh] loa xong {Path(path).name} "
              f"({time.monotonic() - t0:.1f}s).", flush=True)

    # Pre-convert TAT CA filler sang AAC ngay tu warmup: moi cau
    # (ready/listening/thinking/missed) tiet kiem 1-2s spawn ffmpeg o lan
    # phat dau, quan trong khi P2P relay von da cham.
    try:
        for _filler in FILLER_FILES.values():
            p2p._convert_cached(filler_dir / _filler)
        print("[HaLinh] da preload AAC fillers.", flush=True)
    except Exception as preload_error:  # noqa: BLE001 - chi la toi uu
        print(f"[HaLinh] preload AAC fillers loi (bo qua): {preload_error}",
              flush=True)

    talk(filler_dir / FILLER_FILES["ready"], channel)
    print("[HaLinh] sẵn sàng. Gọi 'Hà Linh ơi' để bắt đầu "
          "(dự phòng: 'Hà Linh', 'Linh ơi'). Ctrl+C dừng.", flush=True)

    out_dir = ROOT / "output" / "qa_cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    done_rounds = 0
    wake_miss = 0
    empty_wake = 0  # dem doan on STT ra rong (de heartbeat, khoi tuong chet)
    try:
        while True:
            if args.rounds and done_rounds >= args.rounds:
                break
            # -- WAIT_WAKE (tiny 0.3s, nhanh gap ~10x medium) --
            # Mic LUON MO o vong nay (ke ca loa dang chao) de khong miss
            # tieng goi "Ha Linh oi". Ne echo chi bat sau WAKE.
            pcm = capture_utterance(
                mic_input, voice_cfg,
                prompt="[HaLinh] ... nghe (goi 'Hà Linh ơi') ...",
                end_silence_ms=700, max_len_s=4.0, max_wait_s=None,
                meter_s=3.0, respect_speaker_busy=False)
            if not pcm:
                time.sleep(1.0)
                continue
            t0 = time.monotonic()
            wake_text = transcribe_wake(pcm)
            print(f"[HaLinh] STT wake ({time.monotonic() - t0:.2f}s, {_wake_model}): "
                  f"{wake_text!r}", flush=True)
            if not wake_text or not is_wake(wake_text):
                # Co tieng nhung khong phai goi Ha Linh: dem hut; du N lan thi
                # nhac 1 cau roi ve cho (STT rong/im lang thi khong dem).
                if wake_text:
                    wake_miss += 1
                    empty_wake = 0
                    print(f"[HaLinh] hut wake ({wake_miss}): {wake_text}",
                          flush=True)
                    if wake_miss >= max(1, args.wake_miss_n):
                        wake_miss = 0
                        talk(filler_dir / FILLER_FILES["missed"])
                else:
                    # Tieng quat/va dap phong: STT ra rong. In thua de biet
                    # mic van song (khong thi tuong chuong trinh treo).
                    empty_wake += 1
                    if empty_wake % 5 == 1:
                        print("[HaLinh] mic van nghe (toan tieng on phong) - "
                              "goi 'Ha Linh oi' de bat dau.", flush=True)
                continue
            wake_miss = 0
            empty_wake = 0
            print(f"[HaLinh] WAKE: {wake_text}", flush=True)
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
                # Wake kem lenh ("Ha Linh oi, may gio roi"): tiny bat wake
                # nhanh nhung co the sai lenh dai -> chay lai model lenh
                # tren cung pcm de lay lenh chuan (hiem, chap nhan cham 1 lan).
                t0 = time.monotonic()
                refined = transcribe_cmd(pcm)
                refined_cmd = strip_wake_command(refined) if refined else ""
                print(f"[HaLinh] refine lenh ({_cmd_model}) "
                      f"({time.monotonic() - t0:.2f}s): {refined!r} -> "
                      f"{refined_cmd!r}", flush=True)
                if refined_cmd:
                    question = refined_cmd
                print(f"[HaLinh] lenh kem wake: {question}", flush=True)
                total0 = time.monotonic()
            else:
                print("[HaLinh] ack loa ('Hà Linh nghe nè' xong roi hay noi).",
                      flush=True)
                ack_via_camera(talk, filler_dir, channel)

            # -- LISTEN_CMD (model lenh: medium/zipformer) --
            if not question:
                stt_s = 0.0
                attempt = 0
                while True:
                    attempt += 1
                    # Trong turn (sau WAKE): mic NE loa dang phat de lenh
                    # khong dinh echo cau chao/ack.
                    pcm = capture_utterance(
                        mic_input, voice_cfg,
                        prompt="[HaLinh] NÓI NGAY (cứ nói, Hà Linh vẫn nghe)...",
                        end_silence_ms=1300,
                        max_len_s=10.0,
                        max_wait_s=None, respect_speaker_busy=True)
                    if not pcm:
                        print(f"[HaLinh] lan {attempt}: mat stream mic, "
                              f"mo lai...", flush=True)
                        time.sleep(1.0)
                        continue
                    total0 = time.monotonic()
                    t0 = time.monotonic()
                    question = transcribe_cmd(pcm)
                    stt_s = time.monotonic() - t0
                    if question:
                        question = strip_ack_echo(question)
                    if question and is_wake(question) \
                            and not strip_wake_command(question):
                        print(f"[HaLinh] lan {attempt}: chi nghe wake "
                              f"({question}) - doi cau lenh that.",
                              flush=True)
                        question = ""
                    if question:
                        break
                    print(f"[HaLinh] lan {attempt}: STT rong ({stt_s:.2f}s) - "
                          f"van nghe tiep.", flush=True)
            print(f"[HaLinh] lenh ({stt_s:.2f}s): {question}", flush=True)
            # Cau noi cho tool get_last_transcript (vong sau hoi lai van co).
            try:
                (out_dir / "last_transcript.txt").write_text(
                    question, encoding="utf-8")
            except OSError:
                pass

            # -- THINK (thread + filler thinking theo nguong) --
            def _run_think_once():
                one: dict = {}

                def _one() -> None:
                    try:
                        ans, tls, used = think(
                            question, model, args.max_tokens,
                            args.with_search)
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
            fast = fast_answer(question)
            if fast is not None:
                result = {"answer": fast, "tool_s": 0.0, "used": ["fast-path"]}
                llm_s = 0.0
                think_ok = True
                print(f"[HaLinh] fast-path (0s): {fast}", flush=True)
            for quota_try in range(3):
                if think_ok:
                    break
                result, llm_s = _run_think_once()
                error = result.get("error")
                if error is None:
                    answer = result.get("answer", "")
                    tool_s = result.get("tool_s", 0.0)
                    think_ok = True
                    break
                message = str(error)
                is_quota = ("429" in message
                            or "RESOURCE_EXHAUSTED" in message)
                if not is_quota:
                    print(f"[HaLinh] Gemini loi gon: "
                          f"{type(error).__name__}: "
                          f"{message[:200]}", flush=True)
                    if isinstance(error, TimeoutError):
                        try:
                            timeout_wav = (out_dir /
                                           f"halinh_timeout_{int(time.time())}.wav")
                            tts.save_wav(
                                "Hà Linh nghĩ lâu quá, bạn nói lại giúp Hà Linh nhé.",
                                timeout_wav)
                            talk(timeout_wav)
                        except Exception as speak_error:  # noqa: BLE001
                            print(f"[HaLinh] bao timeout loi: {speak_error}",
                                  flush=True)
                    break
                if quota_try >= 2:
                    print("[HaLinh] van 429 sau 2 lan retry - bo cau nay, "
                          "ve cho.", flush=True)
                    break
                print(f"[HaLinh] Gemini het quota (429) lan {quota_try + 1}/3 - "
                      f"bao loa + nghi 60s roi thu lai.", flush=True)
                try:
                    quota_wav = out_dir / f"halinh_quota_{int(time.time())}.wav"
                    tts.save_wav(
                        "Hà Linh hết quota rồi, đợi một phút rồi Hà Linh trả lời.",
                        quota_wav)
                    talk(quota_wav)
                except Exception as speak_error:  # noqa: BLE001
                    print(f"[HaLinh] bao quota loi (van nghi 60s): "
                          f"{speak_error}", flush=True)
                time.sleep(60.0)
            if not think_ok:
                clear_turn()
                continue
            answer = result.get("answer", "")
            tool_s = result.get("tool_s", 0.0)
            if not answer:
                print(f"[HaLinh] Gemini tra rong ({llm_s:.2f}s), ve cho.",
                      flush=True)
                clear_turn()
                continue
            print(f"[HaLinh] dap ({llm_s:.2f}s, tool {tool_s:.2f}s "
                  f"{result.get('used', [])}): {answer}", flush=True)

            # -- SPEAK (loa camera P2P) --
            t0 = time.monotonic()
            wav = out_dir / f"halinh_{int(time.time())}.wav"
            tts.save_wav(answer, wav)
            tts_s = time.monotonic() - t0
            t0 = time.monotonic()
            talk(wav, channel)
            play_s = time.monotonic() - t0
            print(f"[HaLinh] TTS {tts_s:.2f}s | loa {play_s:.2f}s | "
                  f"VONG {time.monotonic() - total0:.1f}s.", flush=True)
            done_rounds += 1
            clear_turn()
    except KeyboardInterrupt:
        print("\n[HaLinh] dung.", flush=True)
    finally:
        try:
            clear_turn()
        except Exception:  # noqa: BLE001
            pass
        p2p.close()


if __name__ == "__main__":
    main()
