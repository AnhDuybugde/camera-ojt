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
* Nếu thiếu thông tin quan trọng để trả lời chính xác, hỏi lại bằng một câu ngắn.
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
* type=direct: tự trả lời ngắn gọn 1-2 câu, dưới 40 từ, tiếng Việt tự nhiên để đọc thành tiếng, không Markdown.
* type=tool: chỉ dùng tool trong danh sách dưới; điền đủ tham số bắt buộc; tham số tùy chọn không biết thì bỏ qua (tool có giá trị mặc định).
* Không bịa tham số: nghe "mấy giờ" thì gọi get_current_time không tham số; nghe thiếu thông tin bắt buộc thì hỏi lại bằng type=direct.
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
    parser.add_argument("--cmd-model", default="medium",
                        help="DEPRECATED (medium-only, small da bo): giu de tuong "
                             "thich CLI.")
    parser.add_argument("--think-filler-s", type=float, default=1.2,
                        help="Xu ly lau hon nguong nay thi phat filler thinking.")
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
) -> bytes | None:
    """Ghi 1 doan noi: doi tieng noi (vo han neu max_wait_s=None) roi cat khi im.

    ffmpeg_input: phan input tuy device, VD camera
    ["-fflags", "nobuffer", "-rtsp_transport", "tcp", "-i", rtsp_url].
    meter_s: in dong muc mic moi N giay trong luc doi. Het max_wait_s
    thi tra None; None (EOF/stream chet) thi vong goi tu mo lai stream.
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

    print("[HaLinh] warmup (STT tiny-wake + medium-lenh, TTS CUDA, tunnel loa)...", flush=True)
    t0 = time.monotonic()
    stt_wake = WhisperModel("tiny", device="cpu", compute_type="int8")
    print(f"[HaLinh] STT wake tiny xong ({time.monotonic() - t0:.1f}s).", flush=True)
    t0 = time.monotonic()
    stt_medium = WhisperModel("medium", device="cpu", compute_type="int8")
    stt_cmd = stt_medium
    print(f"[HaLinh] STT lenh medium xong ({time.monotonic() - t0:.1f}s).", flush=True)
    tts = ZeroTTSBackend(device=args.tts_device, voice=args.tts_voice)
    tts._load()
    p2p = ImouP2PTalkOutput(
        ImouP2PCredentials.from_env(),
        channel=channel,
        timeout=float(voice_cfg.p2p_timeout_s),
        attempts=int(voice_cfg.p2p_attempts),
        retry_delay=float(voice_cfg.p2p_retry_delay_s),
        sample_rate=int(voice_cfg.p2p_sample_rate),
        volume=float(voice_cfg.p2p_volume),
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
        t0 = time.monotonic()
        p2p(path, channel)
        print(f"[HaLinh] loa xong {Path(path).name} "
              f"({time.monotonic() - t0:.1f}s).", flush=True)

    # Pre-convert filler ack sang AAC ngay tu warmup de lan ack dau
    # khong mat them 1-2s spawn ffmpeg convert.
    try:
        p2p._convert_cached(filler_dir / FILLER_FILES["listening"])
        print("[HaLinh] da preload AAC ack.", flush=True)
    except Exception as preload_error:  # noqa: BLE001 - chi la toi uu
        print(f"[HaLinh] preload AAC ack loi (bo qua): {preload_error}",
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
            pcm = capture_utterance(
                mic_input, voice_cfg,
                prompt="[HaLinh] ... nghe (goi 'Hà Linh ơi') ...",
                end_silence_ms=700, max_len_s=4.0, max_wait_s=None,
                meter_s=3.0)
            if not pcm:
                time.sleep(1.0)
                continue
            t0 = time.monotonic()
            wake_text = transcribe_segment_fw(
                stt_wake, pcm, str(voice_cfg.voice_stt_lang),
                max_no_speech_prob=0.60,
                min_avg_logprob=-1.10,
                initial_prompt=WAKE_STT_PROMPT,
            )
            print(f"[HaLinh] STT wake ({time.monotonic() - t0:.2f}s, tiny): "
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
            question = strip_wake_command(wake_text)
            stt_s = 0.0
            if question:
                # Wake kem lenh ("Ha Linh oi, may gio roi"): tiny bat wake
                # nhanh nhung co the sai lenh dai -> chay lai medium tren
                # cung pcm de lay lenh chuan (hiem, chap nhan cham 1 lan).
                t0 = time.monotonic()
                refined = transcribe_segment_fw(
                    stt_cmd, pcm, str(voice_cfg.voice_stt_lang),
                    max_no_speech_prob=float(
                        voice_cfg.voice_max_no_speech_prob),
                    min_avg_logprob=float(voice_cfg.voice_min_avg_logprob),
                )
                refined_cmd = strip_wake_command(refined) if refined else ""
                print(f"[HaLinh] refine lenh medium "
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

            # -- LISTEN_CMD (medium, chuan) --
            if not question:
                stt_s = 0.0
                attempt = 0
                while True:
                    attempt += 1
                    pcm = capture_utterance(
                        mic_input, voice_cfg,
                        prompt="[HaLinh] NÓI NGAY (cứ nói, Hà Linh vẫn nghe)...",
                        end_silence_ms=2000,
                        max_len_s=15.0,
                        max_wait_s=None)
                    if not pcm:
                        print(f"[HaLinh] lan {attempt}: mat stream mic, "
                              f"mo lai...", flush=True)
                        time.sleep(1.0)
                        continue
                    total0 = time.monotonic()
                    t0 = time.monotonic()
                    question = transcribe_segment_fw(
                        stt_cmd, pcm, str(voice_cfg.voice_stt_lang),
                        max_no_speech_prob=float(
                            voice_cfg.voice_max_no_speech_prob),
                        min_avg_logprob=float(voice_cfg.voice_min_avg_logprob),
                    )
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
            for quota_try in range(3):
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
                continue
            answer = result.get("answer", "")
            tool_s = result.get("tool_s", 0.0)
            if not answer:
                print(f"[HaLinh] Gemini tra rong ({llm_s:.2f}s), ve cho.",
                      flush=True)
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
    except KeyboardInterrupt:
        print("\n[HaLinh] dung.", flush=True)
    finally:
        p2p.close()


if __name__ == "__main__":
    main()
