"""Shared query routing for camera voice, laptop voice and text."""
import time
from camera_tracking.voice.voice_trigger import normalize_trigger_text
from camera_tracking.voice.qa_tools import (SEARCH_TOTAL_CHARS, TOOL_FUNCS, build_search_context, build_tool_catalog, parse_router_json, search_web_raw, with_scene)

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

ROUTER_PROTOCOL = """Bạn là bộ định tuyến câu hỏi tiếng Việt, trả về JSON MỘT DÒNG duy nhất, không thêm bất kỳ nội dung nào ngoài JSON:
{"type":"direct|tool","tool":"<tên_tool hoặc null>","args":{},"text":"<câu trả lời ngắn nếu type=direct, ngược lại để rỗng>"}

Luật:
 * type=direct: tự trả lời ngắn gọn 1-2 câu, dưới 40 từ, tiếng Việt tự nhiên để đọc thành tiếng, không Markdown. CẤM hỏi ngược lại (không câu hỏi làm rõ, không gợi ý hỏi tiếp).
 * type=tool: chỉ dùng tool trong danh sách dưới; điền đủ tham số bắt buộc; tham số tùy chọn không biết thì bỏ qua (tool có giá trị mặc định).
 * Câu hỏi cần dữ liệu MỚI (giá realtime, tỷ giá hôm nay, tin tức, sự kiện sau 2024, kiến thức ngoài tầm có sẵn) thì BẮT BUỘC dùng web_search với query tiếng Anh ngắn gọn, không tự bịa số liệu.
 * Không bịa tham số: nghe "mấy giờ" thì gọi get_current_time không tham số; nghe thiếu thông tin bắt buộc thì type=direct với câu trả lời tốt nhất theo giả định mặc định (nêu giả định), KHÔNG hỏi lại.
* Ví dụ: "Python là gì" -> {"type":"direct","tool":null,"args":{},"text":"Python là ngôn ngữ lập trình phổ biến, dễ đọc và dùng nhiều cho AI."}
* Ví dụ: "Đà Nẵng hôm nay bao nhiêu độ" -> {"type":"tool","tool":"get_weather","args":{"city":"Đà Nẵng"},"text":""}
* Ví dụ: "Giá Bitcoin hôm nay bao nhiêu" -> {"type":"tool","tool":"web_search","args":{"query":"Bitcoin price today USD"},"text":""}

Danh sách tool:
"""

SEARCH_SYNTH_PROMPT = """Bạn là Hà Linh, trợ lý giọng nói tiếng Việt. Tóm tắt KẾT QUẢ TÌM KIẾM dưới đây thành câu trả lời cho câu hỏi của người dùng.

Quy tắc: 1-2 câu, dưới 40 từ, tiếng Việt tự nhiên để đọc thành tiếng, không Markdown, không bullet. Chỉ dùng thông tin có trong kết quả, không bịa số liệu. Nếu kết quả không đủ thì nói rõ là chưa có dữ liệu mới. TUYỆT ĐỐI không hỏi ngược lại."""

ROUTER_SYSTEM = HA_LINH_PROMPT + "\n\n" + ROUTER_PROTOCOL + build_tool_catalog()

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

def _synthesize_search(client, types, model: str, question: str,
                       context: str, max_tokens: int) -> str:
    """Gemini lan 2: tom tat JSON search thanh 1-2 cau noi (<40 tu)."""
    budget = max(250, int(max_tokens or 300))
    config = types.GenerateContentConfig(
        system_instruction=SEARCH_SYNTH_PROMPT,
        max_output_tokens=budget,
        temperature=0,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )
    response = client.models.generate_content(
        model=model,
        contents=f"Câu hỏi: {question}\n\nKẾT QUẢ TÌM KIẾM:\n{context}",
        config=config)
    return ((getattr(response, "text", "") or "").strip()
            or "Hà Linh chưa tổng hợp được kết quả tìm kiếm.")

def think(question: str, model: str, max_tokens: int,
          use_search: bool = False, router_tokens: int = 300,
          search_tokens: int = 0, search_results: int = 5,
          search_chars: int = SEARCH_TOTAL_CHARS
          ) -> tuple[str, float, list[str]]:
    """Router JSON + tool local; rieng web_search chay 2-call.

    - direct -> text trong JSON la dap an cuoi (khong goi them).
    - tool thuong -> dispatch TOOL_FUNCS local; chinh tool format cau noi.
    - web_search -> backend goi Search API ngoai lay JSON (Tavily/Serper/
      Brave/Exa/SearXNG, xem qa_tools.search_web_raw) -> Gemini lan 2 tom
      tat 1-2 cau de TTS. Hong lan 2 thi fallback cau format san.
    Tra (dap, giay tool, tools).
    """
    import os

    direct = fast_answer(question)
    if direct is not None:
        return direct, 0.0, []
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
        http_options={"timeout": 20000, "retry_options": {"attempts": 1}})
    try:
        router_budget = max(150, int(router_tokens or 300))
    except (TypeError, ValueError):
        router_budget = 300
    config = types.GenerateContentConfig(
        system_instruction=ROUTER_SYSTEM,
        max_output_tokens=router_budget,
        temperature=0,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )
    response = client.models.generate_content(
        model=model, contents=with_scene(question), config=config)
    raw = (getattr(response, "text", "") or "").strip()
    plan = parse_router_json(raw)
    if plan["type"] == "direct" or not plan["tool"]:
        return plan["text"], 0.0, []
    name = str(plan["tool"])
    args = plan["args"]
    if name == "web_search":
        # 2-call: search ngoai lay JSON roi tong hop (mo rong context).
        t0 = time.monotonic()
        try:
            num = int(args.get("num_results", search_results or 5))
        except (TypeError, ValueError):
            num = 5
        num = max(1, min(10, num or 5))
        try:
            chars = int(args.get("max_chars", search_chars
                                 or SEARCH_TOTAL_CHARS))
        except (TypeError, ValueError):
            chars = SEARCH_TOTAL_CHARS
        try:
            hits = search_web_raw(args.get("query", ""), num)
        except Exception as error:
            return f"lỗi tool web_search: {error}", 0.0, [name]
        context = build_search_context(hits, chars)
        print(f"[HaLinh] web_search JSON ({len(hits)} kq, "
              f"{len(context)} ky tu)", flush=True)
        if not context.strip():
            fallback = TOOL_FUNCS["web_search"](
                args.get("query", ""), num)
            return str(fallback), time.monotonic() - t0, [name]
        synth_budget = search_tokens or max_tokens or 300
        try:
            answer = _synthesize_search(client, types, model, question,
                                        context, int(synth_budget))
        except Exception as error:
            print(f"[HaLinh] tong hop search loi "
                  f"(fallback cau gon): {error}", flush=True)
            answer = TOOL_FUNCS["web_search"](args.get("query", ""), num)
        tool_s = time.monotonic() - t0
        print(f"[HaLinh] web_search tom tat -> {answer}", flush=True)
        return str(answer), tool_s, [name]
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
