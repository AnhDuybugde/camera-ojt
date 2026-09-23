"""Tro ly voice "Ha Linh" - BAN TEST LOCAL (mic + loa may tinh).

2 che do:
  QA (mac dinh): WAIT_WAKE --"Ha Linh oi"--> LISTEN --> GEMINI 1-CALL
    --> TOOLS --> TTS --> loa. Moi vong can goi ten.
  TRO CHUYEN: "toi muon tam su lau dai" -> Ha Linh hoi xac nhan
    ("...doi sang che do tro chuyen chu?") -> co: nghe lien tuc,
    mien wake, 1 cau rep 1 cau; khong: o lai QA. Treo vo thoi han
    den khi tra loi ro; im lang lau thi nhac "Ha Linh van dang
    nghe ne". Noi "dung tro chuyen" de ve QA.

Lo trinh GIU NGUYEN nhu scripts/halinh_assistant.py (ban camera):
    WAIT_WAKE (STT small) --"Ha Linh oi"--> "Ha Linh nghe ne" (loa laptop)
    --> LISTEN_CMD --> STT medium --> GEMINI 1-CALL JSON --> TOOLS
    --> TTS CUDA --> loa laptop --> WAIT_WAKE.

Khac biet duy nhat so voi ban camera:
    * Mic:  ffmpeg pulse (mic laptop) thay vi RTSP camera.
    * Loa:  paplay/aplay/ffplay (loa laptop) thay vi P2P camera.
    * Khong can IMOU_*. Chi can GEMINI_API_KEY trong .env.

Chay:
    python scripts/halinh_assistant_local.py [--rounds 0]
    --rounds 0 = chay lien tuc den Ctrl+C.
    Test luat wake khong can mic:
    python scripts/halinh_assistant_local.py --wake-test "ha linh oi"
    Liet ke nguon mic pulse:
    pactl list short sources
"""

from __future__ import annotations

import argparse
import shutil
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
from camera_tracking.voice.query import HA_LINH_PROMPT, think, fast_answer
from camera_tracking.voice.qa_tools import (
    SEARCH_TOTAL_CHARS,
    TOOL_FUNCS,
    build_search_context,
    build_tool_catalog,
    parse_router_json,
    search_web_raw,
    with_scene,
)
from camera_tracking.voice.speaker_guard import (
    mark_speaker_busy,
    speaker_busy,
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


# Wake: "Ha Linh oi" lam chinh; "Ha Linh"/"Linh oi" du phong.
# Token-match tren text da normalize. "Ha" dung mot minh KHONG wake
# (de nham voi cuoi/ha ha); "Linh" mot minh chi wake khi segment ngan.
HA_LINH_TOKENS = {"ha", "linh"}
HA_LINH_PHRASES = {"ha linh oi", "ha linh a", "xin chao ha linh",
                   "chao ha linh", "ha linh co nghe khong", "linh oi",
                   "oi ha linh", "ha linh nghe khong", "chao linh",
                   "xin chao linh"}
HA_LINH_MAX_TOKENS_SINGLE = 2  # "Linh"/"Linh oi" ngan moi wake.

# Biais STT ve cum goi (thay prompt "Hello imou" mac dinh): giup
# small bat "Ha Linh oi" chuan hon, it be thanh chu la.
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


# Giao thuc router JSON: Gemini CHI hieu cau hoi, KHONG viet dap an.
# - Cau tra loi truc tiep -> {"type":"direct",...,"text":"<cau tra loi>"}.
# - Can tool -> {"type":"tool","tool":"<ten>","args":{...},"text":""}.
# Backend chay tool local (tool tu format cau noi) -> TTS.
# Rieng web_search chay 2-call: router tao query -> backend goi Search API
# ngoai (Tavily/Serper/Brave/Exa/SearXNG) lay JSON -> Gemini lan 2 tom tat
# 1-2 cau de TTS (xem think()). Khong dung Grounding Search cua Gemini
# nen khong vuong gioi han Free Tier.

# Prompt tong hop ket qua search (Gemini lan 2): chi tom tat, khong router.


FILLER_FILES = {
    "ready": "ready.wav",
    "listening": "listening.wav",
    "thinking": "thinking.wav",
    "got_it": "got_it.wav",
    "missed": "missed.wav",
}


def play_wav(path: str | Path, timeout_s: float = 60.0) -> None:
    """Phat 1 file WAV ra loa laptop (chan den khi xong - half-duplex).

    Thu theo thu tu: paplay (pulse, khuyen nghi) -> aplay (alsa)
    -> ffplay. Raise RuntimeError neu khong co player nao.
    """
    wav = str(path)
    if shutil.which("paplay"):
        subprocess.run(["paplay", wav], check=True,
                       timeout=timeout_s,
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        return
    if shutil.which("aplay"):
        subprocess.run(["aplay", "-q", wav], check=True,
                       timeout=timeout_s,
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        return
    ffmpeg = find_ffmpeg()
    if ffmpeg:
        ffplay = shutil.which("ffplay") or str(
            Path(ffmpeg).parent / "ffplay")
        if shutil.which(ffplay) or Path(ffplay).is_file():
            subprocess.run(
                [ffplay, "-nodisp", "-autoexit", "-loglevel", "error", wav],
                check=True, timeout=timeout_s,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
    raise RuntimeError("khong tim thay paplay/aplay/ffplay de phat loa")


def ack_via_speaker(talk, filler_dir: Path, channel: int | None = None) -> None:
    """Bao 'Ha Linh nghe ne' bang loa LAPTOP (chan - half-duplex).

    Tuong duong ack_via_camera() ban camera: phat xong moi mo mic
    (mic mo SAU khi loa dut) nen khong can chay background.
    """
    try:
        talk(filler_dir / FILLER_FILES["listening"], channel)
    except Exception as error:
        print(f"[HaLinh-local] ack loa loi (bo qua, nghe luon): {error}",
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


# ---------------------------------------------------------------------------
# Che do TRO CHUYEN (tam su lau dai) vs HOI DAP (QA).
# QA: moi vong can wake "Ha Linh oi". TRO CHUYEN: nghe lien tuc, mien wake.
# ---------------------------------------------------------------------------

# "lau dai" + 1 trong cac cum nay moi tinh la muon tro chuyen
# (tranh "du bao lau dai", "bao hanh lau dai"... hieu nham).
CHAT_CONTEXT_WORDS = ("tam su", "tro chuyen", "noi chuyen", "chuyen tro")

_CONFIRM_NO_PHRASES = (
    "khong chuyen", "khong can", "khong doi", "khong muon", "o lai",
    "dung lai", "dung chuyen", "thoi dung", "thoi khoi", "de sau",
    "chua muon", "de khi khac",
)
_CONFIRM_YES_PHRASES = (
    "dong y", "chuyen di", "cu chuyen", "dung roi", "dung vay",
    "ok luon", "cho chuyen", "bat dau di",
)
_CONFIRM_NO_TOKENS = {"khong", "thoi", "chua", "no"}
_CONFIRM_YES_TOKENS = {"co", "u", "uh", "um", "ok", "oke", "okay",
                       "yes", "yep", "duoc", "di", "nha", "nhe"}
# Cau dai: tu dong y chi tinh khi DUNG DAU cau ("co, chuyen di" thi co;
# "gi co" / "noi lai di" thi khong). "di/nha/nhe" dung dau de nham nen loai.
_CONFIRM_YES_STARTERS = {"co", "u", "uh", "um", "ok", "oke", "okay",
                         "yes", "yep", "duoc"}

_CHAT_EXIT_VERBS = {"dung", "thoi", "ket thuc", "ngung", "thoat", "huy"}
_CHAT_EXIT_PHRASES = (
    "dung lai", "ngung lai", "ket thuc o day", "thoi khong noi nua",
    "dung noi nua", "tro chuyen the thoi",
)


def is_chat_request(text: str) -> bool:
    """True khi user muon tam su lau dai ('... lau dai' + ngu canh)."""
    norm = normalize_trigger_text(text)
    if "lau dai" not in norm:
        return False
    return any(word in norm for word in CHAT_CONTEXT_WORDS)


def parse_confirm(text: str) -> bool | None:
    """Tra loi co/khong cho cau xin xac nhan: True/False/None (chua ro)."""
    import re

    norm = normalize_trigger_text(text)
    if not norm:
        return None
    for phrase in _CONFIRM_NO_PHRASES:
        if phrase in norm:
            return False
    for phrase in _CONFIRM_YES_PHRASES:
        if phrase in norm:
            return True
    tokens = set(re.findall(r"[a-z0-9]+", norm))
    if tokens & _CONFIRM_NO_TOKENS:
        return False
    listed = re.findall(r"[a-z0-9]+", norm)
    if len(listed) == 1:
        return True if listed[0] in _CONFIRM_YES_TOKENS else None
    if listed and listed[0] in _CONFIRM_YES_STARTERS:
        return True
    return None


def is_chat_exit(text: str) -> bool:
    """True khi user muon dung tro chuyen, ve che do QA."""
    import re

    norm = normalize_trigger_text(text)
    if not norm:
        return False
    if "che do qa" in norm or "che do hoi dap" in norm:
        return True
    for phrase in _CHAT_EXIT_PHRASES:
        if phrase in norm:
            return True
    tokens = set(re.findall(r"[a-z0-9]+", norm))
    if tokens & _CHAT_EXIT_VERBS:
        return any(word in norm for word in CHAT_CONTEXT_WORDS)
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config" / "default.yaml"))
    parser.add_argument("--rounds", type=int, default=0,
                        help="So vong hoi-dap (0 = lien tuc den Ctrl+C).")
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-tokens", type=int, default=300,
                        help="Token toi da cho cau tong hop search (lan 2). "
                             "Router JSON lan 1 luon 300.")
    parser.add_argument("--router-tokens", type=int, default=300,
                        help="Token toi da cho router JSON (lan 1).")
    parser.add_argument("--search-tokens", type=int, default=0,
                        help="Token toi da cho tong hop search (lan 2, 0 = "
                             "theo --max-tokens).")
    parser.add_argument("--search-results", type=int, default=5,
                        help="So ket qua search toi da moi vong (1-10).")
    import os as _os

    try:
        _search_chars_default = max(
            1000, int((_os.getenv("SEARCH_MAX_CHARS") or "").strip() or 0)
        ) if (_os.getenv("SEARCH_MAX_CHARS") or "").strip() else SEARCH_TOTAL_CHARS
    except (TypeError, ValueError):
        _search_chars_default = SEARCH_TOTAL_CHARS
    parser.add_argument("--search-chars", type=int,
                        default=_search_chars_default,
                        help="Context search toi da (ky tu) nap cho Gemini "
                             "lan 2 (mac dinh 6000 ~ 1500 token, hoac "
                             "SEARCH_MAX_CHARS trong .env).")
    parser.add_argument("--tts-voice", default="maichi")
    parser.add_argument("--tts-device", default="cuda",
                        help="cuda (co GPU) hoac cpu.")
    parser.add_argument("--cmd-model", default="medium",
                        help="DEPRECATED (medium-only, small da bo): giu de tuong "
                             "thich CLI.")
    parser.add_argument("--tail-s", type=float, default=0.3,
                        help="Nghi sau moi lan phat loa cho vang tat han "
                             "truoc khi mo mic (chong echo).")
    parser.add_argument("--mic-source",
                        default="alsa_input.pci-0000_00_1f.3-platform-skl_hda_dsp_generic.HiFi__hw_sofhdadsp_6__source",
                        help="Nguon pulse cho ffmpeg (-f pulse -i ...). "
                             "Mac dinh = DMIC (mic laptop). Xem: "
                             "pactl list short sources")
    parser.add_argument("--think-filler-s", type=float, default=1.2,
                        help="Xu ly lau hon nguong nay thi phat filler thinking.")
    parser.add_argument("--think-timeout-s", type=float, default=90.0)
    parser.add_argument("--wake-miss-n", type=int, default=3,
                        help="Hut wake lien tiep bao nhieu lan co tieng thi "
                             "nhac 'nghe khong ro' 1 cau.")
    parser.add_argument("--chat-idle-s", type=float, default=60.0,
                        help="Che do tro chuyen: im lang qua N giay thi nhac "
                             "'Ha Linh van dang nghe ne' roi nghe tiep.")
    parser.add_argument("--confirm-idle-s", type=float, default=45.0,
                        help="Cho xac nhan doi che do: im lang qua N giay thi "
                             "nhac roi doi tiep (vo thoi han).")
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
    if (len(tokens) <= HA_LINH_MAX_TOKENS_SINGLE
            and all(t in _WAKE_FILLER_TOKENS for t in tokens)):
        return True
    # STT noi khong chuan ("hen linh oi", "ha nin oi"): so do giong chuoi.
    import difflib

    for cand in ("ha linh oi", "ha linh", "chao ha linh",
                 "xin chao ha linh"):
        if difflib.SequenceMatcher(None, norm, cand).ratio() >= 0.80:
            return True
    return False


def capture_utterance(
    mic_source: str,
    voice_cfg,
    *,
    prompt: str,
    end_silence_ms: int = 1000,
    max_len_s: float = 10.0,
    max_wait_s: float | None = None,
    meter_s: float | None = None,
) -> bytes | None:
    """Ghi 1 doan noi tu MIC LAPTOP (pulse): doi tieng noi roi cat khi im.

    Giong het ban camera, chi khac input ffmpeg:
      camera: -rtsp_transport tcp -i <rtsp_url>
      local:  -f pulse -i <mic_source>
    """
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        print("[HaLinh-local] khong tim thay ffmpeg")
        return None
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error",
        "-f", "pulse", "-i", mic_source,
        "-ac", "1", "-ar", str(TARGET_RATE),
        "-af", DEFAULT_AUDIO_FILTER,
        "-f", "s16le", "pipe:1",
    ]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, bufsize=0)
    except Exception as error:
        print(f"[HaLinh-local] khong mo duoc mic pulse "
              f"({mic_source!r}): {error}")
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
    busy = False
    busy_checks = 0
    busy_announced = False
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
                # Half-duplex xuyen process: loa (greeter chao / TTS) dang
                # phat thi TAT MIC - bo frame, huy doan dang ghi do.
                # Check moi 8 frame (~4 lan/giay) cho nhe.
                busy_checks += 1
                if busy_checks % 8 == 1:
                    busy = speaker_busy()
                if busy:
                    heard = 0
                    if not busy_announced:
                        busy_announced = True
                        print("[HaLinh-local] loa dang phat - tam tat mic.",
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
                    print("[HaLinh-local] mic thong, goi 'Hà Linh ơi' di.",
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
                    print("[HaLinh-local:meter] chua co frame audio "
                          "(sai --mic-source?)", flush=True)
                else:
                    mark = "CO TIENG" if show >= last_thr else "yen"
                    print(f"[HaLinh-local:meter] {mark} dinh={show:.4f} "
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






def main() -> None:
    import os

    args = parse_args()
    if args.wake_test is not None:
        print(f"{args.wake_test!r} -> wake={is_wake(args.wake_test)}")
        return
    if args.with_search:
        print("[HaLinh-local] single-call JSON mode khong ho tro search "
              "grounding, bo qua --with-search.", flush=True)
    config = load_config(args.config)
    voice_cfg = config.voice
    model = args.model or os.getenv("GEMINI_MODEL", "").strip() or "gemini-3.6-flash"
    if not os.getenv("GEMINI_API_KEY", "").strip():
        raise SystemExit("Thieu GEMINI_API_KEY: them vao .env roi chay lai.")

    filler_dir = ROOT / "output" / "voice_fillers"
    missing = [f for f in FILLER_FILES.values() if not (filler_dir / f).is_file()]
    if missing:
        raise SystemExit(f"Thieu filler WAV {missing}: gen truoc (ZeroTTS).")

    from faster_whisper import WhisperModel

    from camera_tracking.voice.zerotts_tts import ZeroTTSBackend

    print("[HaLinh-local] warmup (STT tiny-wake + medium-lenh, TTS)...", flush=True)
    t0 = time.monotonic()
    stt_wake = WhisperModel("tiny", device="cpu", compute_type="int8")
    print(f"[HaLinh-local] STT wake tiny xong ({time.monotonic() - t0:.1f}s).", flush=True)
    t0 = time.monotonic()
    stt_medium = WhisperModel("medium", device="cpu", compute_type="int8")
    stt_cmd = stt_medium
    print(f"[HaLinh-local] STT lenh medium xong ({time.monotonic() - t0:.1f}s).", flush=True)
    tts = ZeroTTSBackend(device=args.tts_device, voice=args.tts_voice)
    tts._load()

    def talk(path: str | Path, _channel: int | None = None) -> None:
        # Bao mic (vong sau / greeter) biet loa dang phat (half-duplex).
        try:
            mark_speaker_busy(path)
        except Exception:  # noqa: BLE001 - guard chi la phu
            pass
        play_wav(path)
        # Cho vang loa tat han truoc khi mo mic (chong echo lot vao mic).
        time.sleep(max(0.0, float(args.tail_s)))

    # Kiem tra loa that truoc khi noi "san sang" (giong ban camera
    # kiem tra tunnel P2P): loa hong thi dung som, khong gia vo san sang.
    try:
        talk(filler_dir / FILLER_FILES["ready"])
    except Exception as error:
        raise SystemExit(f"[HaLinh-local] khong phat duoc loa laptop: {error} "
                         f"(kiem tra paplay/aplay + loa).")
    print("[HaLinh-local] sẵn sàng. Gọi 'Hà Linh ơi' để bắt đầu "
          "(dự phòng: 'Hà Linh', 'Linh ơi'). Ctrl+C dừng.", flush=True)
    print(f"[HaLinh-local] mic pulse={args.mic_source!r} "
          f"(doi nguon? pactl list short sources).", flush=True)

    out_dir = ROOT / "output" / "qa_cache"
    out_dir.mkdir(parents=True, exist_ok=True)

    def say_cached(filename: str, text: str) -> None:
        """Phat cau thong bao doi che do (gen 1 lan, tai su dung)."""
        path = out_dir / filename
        if not (path.is_file() and path.stat().st_size > 0):
            tts.save_wav(text, path)
        talk(path)

    done_rounds = 0
    mode = "qa"  # "qa" (can wake) | "chat" (tam su, mien wake)
    wake_miss = 0
    empty_wake = 0  # dem doan on STT ra rong (de heartbeat, khoi tuong chet)
    try:
        while True:
            if args.rounds and done_rounds >= args.rounds:
                break
            # -- WAIT_WAKE (tiny 0.3s, nhanh gap ~10x medium) --
            pcm = capture_utterance(
                args.mic_source, voice_cfg,
                prompt="[HaLinh-local] ... nghe (goi 'Hà Linh ơi') ...",
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
            print(f"[HaLinh-local] STT wake ({time.monotonic() - t0:.2f}s, tiny): "
                  f"{wake_text!r}", flush=True)
            if not wake_text or not is_wake(wake_text):
                # Co tieng nhung khong phai goi Ha Linh: dem hut; du N lan thi
                # nhac 1 cau roi ve cho (STT rong/im lang thi khong dem).
                if wake_text:
                    wake_miss += 1
                    empty_wake = 0
                    print(f"[HaLinh-local] hut wake ({wake_miss}): {wake_text}",
                          flush=True)
                    if wake_miss >= max(1, args.wake_miss_n):
                        wake_miss = 0
                        talk(filler_dir / FILLER_FILES["missed"])
                else:
                    # Tieng quat/va dap phong: STT ra rong. In thua de biet
                    # mic van song (khong thi tuong chuong trinh treo).
                    empty_wake += 1
                    if empty_wake % 5 == 1:
                        print("[HaLinh-local] mic van nghe (toan tieng on phong) - "
                              "goi 'Ha Linh oi' de bat dau.", flush=True)
                continue
            wake_miss = 0
            empty_wake = 0
            print(f"[HaLinh-local] WAKE: {wake_text}", flush=True)
            question = strip_wake_command(wake_text)
            stt_s = 0.0
            if question:
                # tiny bat wake nhanh nhung co the sai lenh dai -> chay lai
                # medium tren cung pcm de lay lenh chuan (hiem, chap nhan cham).
                t0 = time.monotonic()
                refined = transcribe_segment_fw(
                    stt_cmd, pcm, str(voice_cfg.voice_stt_lang),
                    max_no_speech_prob=float(
                        voice_cfg.voice_max_no_speech_prob),
                    min_avg_logprob=float(voice_cfg.voice_min_avg_logprob),
                )
                refined_cmd = strip_wake_command(refined) if refined else ""
                print(f"[HaLinh-local] refine lenh medium "
                      f"({time.monotonic() - t0:.2f}s): {refined!r} -> "
                      f"{refined_cmd!r}", flush=True)
                if refined_cmd:
                    question = refined_cmd
                print(f"[HaLinh-local] lenh kem wake: {question}", flush=True)
                total0 = time.monotonic()
            else:
                print("[HaLinh-local] ack loa ('Hà Linh nghe nè' xong roi hay noi).",
                      flush=True)
                ack_via_speaker(talk, filler_dir)

            # -- LISTEN_CMD (medium, chuan) --
            if not question:
                stt_s = 0.0
                attempt = 0
                while True:
                    attempt += 1
                    pcm = capture_utterance(
                        args.mic_source, voice_cfg,
                        prompt="[HaLinh-local] NÓI NGAY (cứ nói, Hà Linh vẫn nghe)...",
                        end_silence_ms=2000,
                        max_len_s=15.0,
                        max_wait_s=None)
                    if not pcm:
                        print(f"[HaLinh-local] lan {attempt}: mat stream mic, "
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
                        print(f"[HaLinh-local] lan {attempt}: chi nghe wake "
                              f"({question}) - doi cau lenh that.",
                              flush=True)
                        question = ""
                    if question:
                        break
                    print(f"[HaLinh-local] lan {attempt}: STT rong ({stt_s:.2f}s) - "
                          f"van nghe tiep.", flush=True)
            print(f"[HaLinh-local] lenh ({stt_s:.2f}s): {question}", flush=True)
            # Cau noi cho tool get_last_transcript (vong sau hoi lai van co).
            try:
                (out_dir / "last_transcript.txt").write_text(
                    question, encoding="utf-8")
            except OSError:
                pass

            # -- DOI CHE DO: QA <-> TRO CHUYEN --
            # "toi muon tam su lau dai" -> hoi xac nhan -> co: chat, khong: o lai.
            # Treo vo thoi han den khi tra loi ro; im lang lau thi nhac
            # "Ha Linh van dang nghe ne" roi doi tiep.
            if mode == "qa" and is_chat_request(question):
                print("[HaLinh-local] xin xac nhan doi sang TRO CHUYEN.",
                      flush=True)
                say_cached(
                    "halinh_ask_chat.wav",
                    "Bạn muốn đổi từ chế độ hỏi đáp sang chế độ trò chuyện chứ?")
                unclear = 0
                while True:
                    pcm = capture_utterance(
                        args.mic_source, voice_cfg,
                        prompt="[HaLinh-local] TRA LOI (co / khong)...",
                        end_silence_ms=1500, max_len_s=8.0,
                        max_wait_s=float(args.confirm_idle_s) or None)
                    if pcm is None:
                        say_cached("halinh_still_here.wav",
                                   "Hà Linh vẫn đang nghe nè.")
                        continue
                    ans = transcribe_segment_fw(
                        stt_cmd, pcm, str(voice_cfg.voice_stt_lang),
                        max_no_speech_prob=float(
                            voice_cfg.voice_max_no_speech_prob),
                        min_avg_logprob=float(
                            voice_cfg.voice_min_avg_logprob),
                    )
                    if not ans:
                        continue
                    print(f"[HaLinh-local] xac nhan: {ans}", flush=True)
                    if "noi lai" in normalize_trigger_text(ans):
                        talk(out_dir / "halinh_ask_chat.wav")
                        unclear = 0
                        continue
                    verdict = parse_confirm(ans)
                    if verdict is True:
                        say_cached(
                            "halinh_to_chat.wav",
                            "Đã chuyển sang chế độ trò chuyện. Bạn cứ nói, "
                            "không cần gọi Hà Linh ơi nữa. Muốn dừng thì nói "
                            "dừng trò chuyện nhé.")
                        mode = "chat"
                        print("[HaLinh-local] che do: TRO CHUYEN.", flush=True)
                        break
                    if verdict is False:
                        say_cached(
                            "halinh_stay_qa.wav",
                            "Rồi, mình vẫn ở chế độ hỏi đáp nhé. "
                            "Gọi Hà Linh ơi khi cần.")
                        print("[HaLinh-local] che do: HOI DAP (giu nguyen).",
                              flush=True)
                        break
                    unclear += 1
                    # Phản hồi NGAY mỗi lần nghe không rõ (đừng im lặng treo
                    # máy - user tưởng chết): hỏi gọn "Có hay không?".
                    say_cached("halinh_yesno.wav", "Có hay không?")
                if mode != "chat":
                    continue
            if mode == "chat":
                # -- TRO CHUYEN: nghe lien tuc, mien wake, 1 cau rep 1 cau.
                # STT rong/rac (khong thanh text) thi khong rep, nghe tiep.
                print("[HaLinh-local] TAM SU: cu noi tu nhien; "
                      "noi 'dung tro chuyen' de ve hoi dap.", flush=True)
                while True:
                    pcm = capture_utterance(
                        args.mic_source, voice_cfg,
                        prompt="[HaLinh-local] TAM SU (cu noi, khong can goi)...",
                        end_silence_ms=1500, max_len_s=15.0,
                        max_wait_s=float(args.chat_idle_s) or None)
                    if pcm is None:
                        say_cached("halinh_still_here.wav",
                                   "Hà Linh vẫn đang nghe nè.")
                        continue
                    total0 = time.monotonic()
                    t0 = time.monotonic()
                    utter = transcribe_segment_fw(
                        stt_cmd, pcm, str(voice_cfg.voice_stt_lang),
                        max_no_speech_prob=float(
                            voice_cfg.voice_max_no_speech_prob),
                        min_avg_logprob=float(
                            voice_cfg.voice_min_avg_logprob),
                    )
                    if not utter:
                        continue
                    print(f"[HaLinh-local] tam su: {utter}", flush=True)
                    # Quen mieng goi ten: boc wake, chi con wake suong thi bo.
                    if is_wake(utter):
                        utter = strip_wake_command(utter)
                        if not utter:
                            continue
                    if is_chat_exit(utter):
                        say_cached(
                            "halinh_to_qa.wav",
                            "Đã về chế độ hỏi đáp. Gọi Hà Linh ơi khi cần nhé.")
                        mode = "qa"
                        print("[HaLinh-local] che do: HOI DAP.", flush=True)
                        break
                    try:
                        answer, tool_s, used = think(
                            utter, model, args.max_tokens, args.with_search,
                            args.router_tokens, args.search_tokens,
                            args.search_results, args.search_chars)
                    except Exception as error:  # noqa: BLE001
                        print(f"[HaLinh-local] loi: {type(error).__name__}: "
                              f"{str(error)[:200]}", flush=True)
                        continue
                    if not answer:
                        continue
                    print(f"[HaLinh-local] dap {used}: {answer}", flush=True)
                    wav = out_dir / f"halinh_chat_{int(time.time())}.wav"
                    tts.save_wav(answer, wav)
                    talk(wav)
                continue

            # -- THINK (thread + filler thinking theo nguong) --
            def _run_think_once():
                one: dict = {}

                def _one() -> None:
                    try:
                        ans, tls, used = think(
                            question, model, args.max_tokens,
                            args.with_search, args.router_tokens,
                            args.search_tokens, args.search_results,
                            args.search_chars)
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
                    print(f"[HaLinh-local] Gemini loi gon: "
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
                            print(f"[HaLinh-local] bao timeout loi: {speak_error}",
                                  flush=True)
                    break
                if quota_try >= 2:
                    print("[HaLinh-local] van 429 sau 2 lan retry - bo cau nay, "
                          "ve cho.", flush=True)
                    break
                print(f"[HaLinh-local] Gemini het quota (429) lan {quota_try + 1}/3 - "
                      f"bao loa + nghi 60s roi thu lai.", flush=True)
                try:
                    quota_wav = out_dir / f"halinh_quota_{int(time.time())}.wav"
                    tts.save_wav(
                        "Hà Linh hết quota rồi, đợi một phút rồi Hà Linh trả lời.",
                        quota_wav)
                    talk(quota_wav)
                except Exception as speak_error:  # noqa: BLE001
                    print(f"[HaLinh-local] bao quota loi (van nghi 60s): "
                          f"{speak_error}", flush=True)
                time.sleep(60.0)
            if not think_ok:
                continue
            answer = result.get("answer", "")
            tool_s = result.get("tool_s", 0.0)
            if not answer:
                print(f"[HaLinh-local] Gemini tra rong ({llm_s:.2f}s), ve cho.",
                      flush=True)
                continue
            print(f"[HaLinh-local] dap ({llm_s:.2f}s, tool {tool_s:.2f}s "
                  f"{result.get('used', [])}): {answer}", flush=True)

            # -- SPEAK (loa laptop) --
            t0 = time.monotonic()
            wav = out_dir / f"halinh_{int(time.time())}.wav"
            tts.save_wav(answer, wav)
            tts_s = time.monotonic() - t0
            t0 = time.monotonic()
            talk(wav)
            play_s = time.monotonic() - t0
            print(f"[HaLinh-local] TTS {tts_s:.2f}s | loa {play_s:.2f}s | "
                  f"VONG {time.monotonic() - total0:.1f}s.", flush=True)
            done_rounds += 1
    except KeyboardInterrupt:
        print("\n[HaLinh-local] dung.", flush=True)


if __name__ == "__main__":
    main()
