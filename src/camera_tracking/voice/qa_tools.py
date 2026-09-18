"""Tools goi tu Gemini function calling (chay local, khong can key).

Hien co: get_weather (Open-Meteo, mien phi) + ~60 tool local/API-free
khac (tien te, don vi, tin tuc, trang thai may, file whitelist, so lieu
tracking truc tiep...). Them tool moi = them 1 ham + 1 function
declaration trong router prompt cua scripts/halinh_assistant*.py.
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request

DEFAULT_CITY = "Đà Nẵng"

# Ma WMO weather_code -> tieng Viet noi.
WEATHER_WORDS = {
    0: "trời quang",
    1: "trời ít mây",
    2: "trời nhiều mây",
    3: "trời âm u",
    45: "có sương mù",
    48: "có sương mù đóng băng",
    51: "mưa phùn nhẹ",
    53: "mưa phùn",
    55: "mưa phùn dày",
    56: "mưa phùn lạnh nhẹ",
    57: "mưa phùn lạnh",
    61: "mưa nhẹ",
    63: "mưa vừa",
    65: "mưa to",
    66: "mưa lạnh nhẹ",
    67: "mưa lạnh",
    71: "tuyết nhẹ",
    73: "tuyết vừa",
    75: "tuyết dày",
    77: "mưa tuyết",
    80: "mưa rào nhẹ",
    81: "mưa rào vừa",
    82: "mưa rào to",
    85: "tuyết rào nhẹ",
    86: "tuyết rào dày",
    95: "có dông",
    96: "có dông kèm mưa đá nhẹ",
    99: "có dông kèm mưa đá to",
}


def _http_json(url: str, timeout: float = 10.0) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "camera-ojt-qa/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def geocode_city(city: str) -> tuple[str, float, float]:
    """Ten thanh pho -> (ten chuan, lat, lon). Raise RuntimeError khi khong tim."""
    query = urllib.parse.urlencode(
        {"name": city, "count": 1, "language": "vi", "format": "json"})
    try:
        data = _http_json(f"https://geocoding-api.open-meteo.com/v1/search?{query}")
    except Exception as error:
        raise RuntimeError(f"khong tra cuu duoc dia danh: {error}") from error
    results = data.get("results") or []
    if not results:
        raise RuntimeError(f"khong tim thay dia danh '{city}'")
    top = results[0]
    return str(top.get("name", city)), float(top["latitude"]), float(top["longitude"])


def get_weather(city: str | None = None) -> str:
    """Thoi tiet hien tai (noi, 1-2 cau) cho thanh pho, mac dinh TP.HCM."""
    city = (city or DEFAULT_CITY).strip() or DEFAULT_CITY
    name, lat, lon = geocode_city(city)
    query = urllib.parse.urlencode({
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,apparent_temperature,relative_humidity_2m,"
                   "weather_code,wind_speed_10m",
        "timezone": "Asia/Ho_Chi_Minh",
    })
    try:
        data = _http_json(f"https://api.open-meteo.com/v1/forecast?{query}")
    except Exception as error:
        raise RuntimeError(f"khong lay duoc thoi tiet: {error}") from error
    current = data.get("current") or {}
    try:
        temp = float(current["temperature_2m"])
        feels = float(current.get("apparent_temperature", temp))
        humidity = int(float(current.get("relative_humidity_2m", 0)))
        wind = float(current.get("wind_speed_10m", 0))
        words = WEATHER_WORDS.get(int(float(current.get("weather_code", -1))), "")
    except (TypeError, ValueError, KeyError) as error:
        raise RuntimeError(f"du lieu thoi tiet la: {error}") from error
    sky = f", {words}" if words else ""
    return (f"{name} hiện tại {temp:.0f} độ{sky}, cảm giác như {feels:.0f} độ, "
            f"độ ẩm {humidity} phần trăm, gió {wind:.0f} km một giờ.")


__all__ = [
    "DEFAULT_CITY",
    "STUB_TOOLS",
    "TOOL_DECLARATIONS",
    "TOOL_FUNCS",
    "WEATHER_WORDS",
    "build_tool_catalog",
    "calculate",
    "geocode_city",
    "get_current_date",
    "get_current_time",
    "get_weather",
    "parse_router_json",
]


# ---------------------------------------------------------------------------
# Tien ich dung chung cho tool noi (khong phat sinh API call)
# ---------------------------------------------------------------------------

def _unaccent(text: str) -> str:
    import re
    import unicodedata

    decomposed = unicodedata.normalize("NFD", (text or "").lower())
    stripped = "".join(ch for ch in decomposed
                       if unicodedata.category(ch) != "Mn")
    stripped = stripped.replace("đ", "d")
    return " ".join(re.findall(r"[a-z0-9]+", stripped))


def _fmt_num(value: object, max_dec: int = 2) -> str:
    """So dang noi tieng Viet: nghin cach '.', thap phan dau ','."""
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return str(value)
    if abs(number - round(number)) < 1e-9 and abs(number) < 1e15:
        return f"{int(round(number)):,}".replace(",", ".")
    text = f"{number:,.{max(0, int(max_dec))}f}"
    return text.replace(",", "X").replace(".", ",").replace("X", ".")


def _fmt_big(value: object, unit: str = "") -> str:
    """101000000 -> 'khoang 101 trieu' (kem don vi)."""
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return str(value)
    unit = f" {unit}".rstrip()
    for limit, word in ((1e9, "tỷ"), (1e6, "triệu"), (1e3, "nghìn")):
        if abs(number) >= limit:
            return f"khoảng {_fmt_num(number / limit)} {word}{unit}".replace(
                "  ", " ")
    return f"{_fmt_num(number)}{unit}"


def _speak_hour_min(iso_or_hm: str) -> str:
    """'2026-09-18T05:45' -> '5 gio 45 phut sang'."""
    import re

    match = re.search(r"(\d{1,2}):(\d{2})", iso_or_hm or "")
    if not match:
        return str(iso_or_hm)
    hour, minute = int(match.group(1)), int(match.group(2))
    if 5 <= hour < 11:
        part = "sáng"
    elif 11 <= hour < 13:
        part = "trưa"
    elif 13 <= hour < 18:
        part = "chiều"
    elif 18 <= hour < 22:
        part = "tối"
    else:
        part = "đêm"
    hour12 = hour % 12 or 12
    if minute == 0:
        return f"{hour12} giờ {part}"
    return f"{hour12} giờ {minute} phút {part}"


def _resolve_day(day: str | None, daily_dates: list) -> tuple[int, str]:
    """Ten ngay tu nhien -> (index trong daily[], nhan tieng Viet)."""
    import re
    from datetime import date

    key = _unaccent(day or "")
    if not key or key in {"today", "hom nay", "nay", "hnay"}:
        return 0, "hôm nay"
    if key in {"tomorrow", "mai", "ngay mai", "ng mai"}:
        return 1, "ngày mai"
    if key in {"kia", "ngay kia", "mot"}:  # "ngày kia"
        return 2, "ngày kia"
    iso = ""
    match = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})$", (day or "").strip())
    if match:
        iso = f"{int(match.group(1)):04d}-{int(match.group(2)):02d}-{int(match.group(3)):02d}"
    else:
        match = re.match(r"^(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?$",
                         (day or "").strip())
        if match:
            year = int(match.group(3) or date.today().year)
            year += 2000 if year < 100 else 0
            iso = f"{year:04d}-{int(match.group(2)):02d}-{int(match.group(1)):02d}"
    if iso and iso in list(daily_dates):
        idx = list(daily_dates).index(iso)
        parts = iso.split("-")
        return idx, f"ngày {int(parts[2])} tháng {int(parts[1])}"
    raise RuntimeError(f"không hiểu ngày '{day}' (thử 'hôm nay', 'ngày mai', "
                       f"'ngày kia' hoặc 20/9)")


def _openmeteo_daily(city: str | None) -> tuple[str, dict]:
    """Geocode + daily 7 ngay (max/min, code, mua, UV, moc/mac)."""
    from urllib.parse import urlencode

    name, lat, lon = geocode_city((city or "").strip() or DEFAULT_CITY)
    query = urlencode({
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max,temperature_2m_min,weathercode,"
                 "precipitation_probability_max,uv_index_max,sunrise,sunset",
        "timezone": "auto",
        "forecast_days": 7,
    })
    try:
        data = _http_json(f"https://api.open-meteo.com/v1/forecast?{query}")
    except Exception as error:
        raise RuntimeError(f"không lấy được dự báo: {error}") from error
    daily = data.get("daily") or {}
    if not (daily.get("time") or []):
        raise RuntimeError("dữ liệu dự báo trống")
    return name, daily


# ---------------------------------------------------------------------------
# Thoi tiet mo rong (Open-Meteo, free)
# ---------------------------------------------------------------------------

def get_weather_forecast(city: str | None = None,
                         day: str | None = None) -> str:
    """Du bao nhiet do + troi may ngay chi dinh (mac dinh ngay mai)."""
    name, daily = _openmeteo_daily(city)
    idx, label = _resolve_day(day or "ngày mai", daily["time"])
    tmax = float(daily["temperature_2m_max"][idx])
    tmin = float(daily["temperature_2m_min"][idx])
    words = WEATHER_WORDS.get(int(float(daily["weathercode"][idx])), "")
    sky = f", {words}" if words else ""
    return (f"{name} {label} {tmin:.0f} đến {tmax:.0f} độ{sky}.")


def get_rain_probability(city: str | None = None,
                         day: str | None = None) -> str:
    """Kha nang mua (%) ngay chi dinh ('Mai co mua khong?')."""
    name, daily = _openmeteo_daily(city)
    idx, label = _resolve_day(day or "ngày mai", daily["time"])
    prob = daily["precipitation_probability_max"][idx]
    prob = 0 if prob is None else int(prob)
    if prob < 20:
        words = "khó mưa"
    elif prob < 50:
        words = "có thể mưa nhỏ"
    elif prob < 80:
        words = "dễ mưa"
    else:
        words = "mưa to"
    return (f"{name} {label} khả năng mưa {prob} phần trăm, {words}.")


def get_air_quality(city: str | None = None) -> str:
    """AQI (My), PM2.5, PM10 hien tai."""
    from urllib.parse import urlencode

    name, lat, lon = geocode_city((city or "").strip() or DEFAULT_CITY)
    query = urlencode({"latitude": lat, "longitude": lon,
                       "current": "us_aqi,pm2_5,pm10", "timezone": "auto"})
    try:
        data = _http_json(
            f"https://air-quality-api.open-meteo.com/v1/air-quality?{query}")
    except Exception as error:
        raise RuntimeError(f"không lấy được chất lượng không khí: {error}") from error
    current = data.get("current") or {}
    aqi = current.get("us_aqi")
    if aqi is None:
        return f"{name} hiện chưa có số liệu chất lượng không khí."
    aqi = int(float(aqi))
    if aqi <= 50:
        words = "tốt"
    elif aqi <= 100:
        words = "trung bình"
    elif aqi <= 150:
        words = "kém với người nhạy cảm"
    elif aqi <= 200:
        words = "xấu"
    elif aqi <= 300:
        words = "rất xấu"
    else:
        words = "nguy hại"
    pm25 = current.get("pm2_5")
    extra = "" if pm25 is None else f", bụi mịn {_fmt_num(pm25, 1)} microgam một mét khối"
    return (f"{name} hiện tại AQI {aqi}, chất lượng không khí {words}{extra}.")


def get_uv_index(city: str | None = None) -> str:
    """Chi so UV hien tai + cao nhat hom nay."""
    from urllib.parse import urlencode

    name, lat, lon = geocode_city((city or "").strip() or DEFAULT_CITY)
    query = urlencode({"latitude": lat, "longitude": lon,
                       "current": "uv_index", "daily": "uv_index_max",
                       "timezone": "auto", "forecast_days": 1})
    try:
        data = _http_json(f"https://api.open-meteo.com/v1/forecast?{query}")
    except Exception as error:
        raise RuntimeError(f"không lấy được chỉ số UV: {error}") from error
    now = (data.get("current") or {}).get("uv_index")
    peak = ((data.get("daily") or {}).get("uv_index_max") or [None])[0]
    if now is None:
        return f"{name} hiện chưa có số liệu UV."
    uv = float(now)

    def _words(v: float) -> str:
        if v < 3:
            return "thấp"
        if v < 6:
            return "trung bình"
        if v < 8:
            return "cao"
        if v < 11:
            return "rất cao"
        return "cực cao, hạn chế ra nắng"

    extra = "" if peak is None else f", cao nhất hôm nay {_fmt_num(peak, 1)}"
    return (f"{name} hiện tại UV {_fmt_num(uv, 1)}, mức {_words(uv)}{extra}.")


def get_sunrise_sunset(city: str | None = None,
                       day: str | None = None) -> str:
    """Gio mat troi moc / lan ngay chi dinh."""
    name, daily = _openmeteo_daily(city)
    idx, label = _resolve_day(day, daily["time"])
    rise = _speak_hour_min(str(daily["sunrise"][idx]))
    set_ = _speak_hour_min(str(daily["sunset"][idx]))
    return f"{name} {label} mặt trời mọc lúc {rise}, lặn lúc {set_}."


# ---------------------------------------------------------------------------
# Tien te (Frankfurter v2, free, khong key)
# ---------------------------------------------------------------------------

_CURRENCY_NAMES = {
    "dong": "VND", "vnd": "VND", "viet nam dong": "VND",
    "do la": "USD", "usd": "USD", "do la my": "USD",
    "euro": "EUR", "eur": "EUR",
    "bang anh": "GBP", "gbp": "GBP",
    "yen": "JPY", "jpy": "JPY", "yen nhat": "JPY",
    "nhan dan te": "CNY", "cny": "CNY", "te trung quoc": "CNY",
    "won": "KRW", "krw": "KRW", "won han": "KRW",
    "do sing": "SGD", "sgd": "SGD",
    "bat thai": "THB", "thb": "THB", "bath": "THB",
    "do uc": "AUD", "aud": "AUD",
    "do canada": "CAD", "cad": "CAD",
}


def _currency_code(raw: str | None) -> str:
    import re

    key = _unaccent(raw or "")
    if key in _CURRENCY_NAMES:
        return _CURRENCY_NAMES[key]
    code = re.sub(r"[^a-z]", "", key).upper()
    if len(code) == 3:
        return code
    raise RuntimeError(f"không hiểu đơn vị tiền '{raw}' (thử USD, VND, Euro...)")


def convert_currency(amount: float | str | None = None,
                     from_currency: str | None = None,
                     to_currency: str | None = None) -> str:
    """Doi tien theo ty gia Frankfurter (mid-market, cap nhat hang ngay)."""
    try:
        value = float(str(amount).replace(",", "."))
    except (TypeError, ValueError):
        raise RuntimeError(f"số tiền '{amount}' không hợp lệ") from None
    if value <= 0 or value > 1e15:
        raise RuntimeError("số tiền phải lớn hơn 0")
    base = _currency_code(from_currency or "USD")
    quote = _currency_code(to_currency or "VND")
    url = f"https://api.frankfurter.dev/v2/rate/{base}/{quote}"
    try:
        data = _http_json(url)
        rate = float(data["rate"])
    except Exception as error:
        raise RuntimeError(f"không đổi được tiền: {error}") from error
    out = value * rate
    dec = 0 if quote in {"VND", "JPY", "KRW"} else 2
    return (f"{_fmt_num(value, 2)} {base} bằng khoảng "
            f"{_fmt_num(out, dec)} {quote}.")


# ---------------------------------------------------------------------------
# Doi don vi / ngay gio (local)
# ---------------------------------------------------------------------------

_UNIT_TABLE: dict[str, tuple[str, float]] = {}
for _aliases, _to_base in [
    (("mm", "milimet"), ("length", 0.001)),
    (("cm", "xentimet", "centimet"), ("length", 0.01)),
    (("dm", "deximet"), ("length", 0.1)),
    (("m", "met"), ("length", 1.0)),
    (("km", "kilomet", "cay so"), ("length", 1000.0)),
    (("in", "inch"), ("length", 0.0254)),
    (("ft", "foot", "feet"), ("length", 0.3048)),
    (("yd", "yard"), ("length", 0.9144)),
    (("mi", "mile", "dam"), ("length", 1609.344)),
    (("mg", "miligam"), ("mass", 1e-6)),
    (("g", "gam", "gram"), ("mass", 0.001)),
    (("kg", "kilogam"), ("mass", 1.0)),
    (("yen",), ("mass", 10.0)),
    (("ta",), ("mass", 100.0)),
    (("tan",), ("mass", 1000.0)),
    (("lb", "pound"), ("mass", 0.45359237)),
    (("oz", "ounce"), ("mass", 0.028349523125)),
    (("ml", "mililit"), ("volume", 0.001)),
    (("l", "lit"), ("volume", 1.0)),
    (("m3", "met khoi"), ("volume", 1000.0)),
]:
    for _alias in _aliases:
        _UNIT_TABLE[_alias] = _to_base


def convert_unit(value: float | str | None = None,
                 from_unit: str | None = None,
                 to_unit: str | None = None) -> str:
    """Doi don vi do dai / khoi luong / nhiet do / the tich (local)."""
    try:
        number = float(str(value).replace(",", "."))
    except (TypeError, ValueError):
        raise RuntimeError(f"giá trị '{value}' không hợp lệ") from None
    frm = _unaccent(f"{from_unit or ''}".replace("độ", "").replace("do", "").strip())
    to = _unaccent(f"{to_unit or ''}".replace("độ", "").replace("do", "").strip())
    frm = {"celsius": "c", "fahrenheit": "f", "kelvin": "k"}.get(frm, frm)
    to = {"celsius": "c", "fahrenheit": "f", "kelvin": "k"}.get(to, to)
    if frm in {"c", "f", "k"} or to in {"c", "f", "k"}:
        if frm not in {"c", "f", "k"} or to not in {"c", "f", "k"}:
            raise RuntimeError("nhiệt độ chỉ đổi giữa độ C, F, K")
        c = number if frm == "c" else ((number - 32) * 5 / 9 if frm == "f"
                                      else number - 273.15)
        out = c if to == "c" else (c * 9 / 5 + 32 if to == "f" else c + 273.15)
        names = {"c": "độ C", "f": "độ F", "k": "độ K"}
        return f"{_fmt_num(number)} {names[frm]} bằng {_fmt_num(out)} {names[to]}."
    if frm not in _UNIT_TABLE or to not in _UNIT_TABLE:
        raise RuntimeError(f"chưa hỗ trợ đổi '{from_unit}' sang '{to_unit}'")
    kind_a, rate_a = _UNIT_TABLE[frm]
    kind_b, rate_b = _UNIT_TABLE[to]
    if kind_a != kind_b:
        raise RuntimeError(f"không đổi {kind_a} sang {kind_b} được")
    return (f"{_fmt_num(number)} {from_unit} bằng "
            f"{_fmt_num(number * rate_a / rate_b)} {to_unit}.")


def _parse_vi_date(raw: str | None):
    import re
    from datetime import date

    text = (raw or "").strip()
    key = _unaccent(text)
    if key in {"today", "hom nay", "nay", ""}:
        return date.today()
    if key in {"tomorrow", "mai", "ngay mai"}:
        from datetime import timedelta

        return date.today() + timedelta(days=1)
    match = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})$", text)
    if match:
        return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    match = re.match(r"^(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?$", text)
    if match:
        year = int(match.group(3)) if match.group(3) else date.today().year
        year += 2000 if year < 100 else 0
        return date(year, int(match.group(2)), int(match.group(1)))
    raise RuntimeError(f"không hiểu ngày '{raw}' (thử 20/9 hoặc 2026-09-20)")


def date_difference(date1: str | None = None,
                    date2: str | None = None) -> str:
    """So ngay giua 2 moc (VD 'con bao nhieu ngay toi 1/1/2027')."""
    from datetime import date

    first, second = _parse_vi_date(date1), _parse_vi_date(date2)
    delta = (second - first).days
    fmt = lambda d: f"{d.day}/{d.month}/{d.year}"  # noqa: E731
    if delta == 0:
        return f"{fmt(first)} và {fmt(second)} là cùng một ngày."
    if delta > 0:
        return f"Từ {fmt(first)} đến {fmt(second)} là {delta} ngày."
    return f"Từ {fmt(first)} đến {fmt(second)} đã qua {-delta} ngày."


_TZ_NAMES = {
    "viet nam": "Asia/Ho_Chi_Minh", "ha noi": "Asia/Ho_Chi_Minh",
    "sai gon": "Asia/Ho_Chi_Minh", "tokyo": "Asia/Tokyo", "nhat": "Asia/Tokyo",
    "seoul": "Asia/Seoul", "han quoc": "Asia/Seoul",
    "bac kinh": "Asia/Shanghai", "trung quoc": "Asia/Shanghai",
    "singapore": "Asia/Singapore", "bangkok": "Asia/Bangkok",
    "thai lan": "Asia/Bangkok", "london": "Europe/London",
    "anh": "Europe/London", "paris": "Europe/Paris", "phap": "Europe/Paris",
    "berlin": "Europe/Berlin", "duc": "Europe/Berlin",
    "new york": "America/New_York", "my": "America/New_York",
    "los angeles": "America/Los_Angeles", "sydney": "Australia/Sydney",
    "uc": "Australia/Sydney", "utc": "UTC", "gmt": "UTC",
}


def convert_timezone(time: str | None = None,
                     from_tz: str | None = None,
                     to_tz: str | None = None) -> str:
    """Doi gio 'HH:MM' hom nay giua 2 mui gio (mac dinh tu Viet Nam)."""
    import re
    from datetime import datetime
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

    def _zone(raw: str | None, default: str) -> ZoneInfo:
        key = _unaccent(raw or "")
        name = _TZ_NAMES.get(key) or (raw or default).strip()
        try:
            return ZoneInfo(name)
        except (ZoneInfoNotFoundError, ValueError):
            raise RuntimeError(f"không hiểu múi giờ '{raw}'") from None

    match = re.match(r"^(\d{1,2})(?::(\d{1,2}))?$", (time or "").strip())
    if not match:
        raise RuntimeError(f"không hiểu giờ '{time}' (thử 14:30)")
    hour, minute = int(match.group(1)), int(match.group(2) or 0)
    if not (0 <= hour < 24 and 0 <= minute < 60):
        raise RuntimeError(f"giờ '{time}' không hợp lệ")
    if not (to_tz or "").strip():
        raise RuntimeError("thiếu nơi cần đổi đến (ví dụ Tokyo, London)")
    try:
        from datetime import date as _date

        here = datetime(_date.today().year, _date.today().month,
                        _date.today().day, hour, minute,
                        tzinfo=_zone(from_tz, "Asia/Ho_Chi_Minh"))
        there = here.astimezone(_zone(to_tz, "UTC"))
    except RuntimeError:
        raise
    except Exception as error:
        raise RuntimeError(f"không đổi được múi giờ: {error}") from error
    return (f"{hour} giờ {minute} phút giờ Việt Nam là "
            f"{there.hour} giờ {there.minute} phút giờ {(to_tz or '').strip()}.")


# ---------------------------------------------------------------------------
# He thong may local (psutil / nvidia-smi)
# ---------------------------------------------------------------------------

def _psutil():
    try:
        import psutil
    except ImportError as error:
        raise RuntimeError("chưa cài psutil") from error
    return psutil


def get_cpu_usage() -> str:
    """% CPU hien tai (do 0.5 giay)."""
    psutil = _psutil()
    return f"CPU hiện dùng {_fmt_num(psutil.cpu_percent(interval=0.5), 0)} phần trăm."


def get_ram_usage() -> str:
    """RAM da dung / tong."""
    psutil = _psutil()
    mem = psutil.virtual_memory()
    return (f"RAM đã dùng {_fmt_num(mem.percent, 0)} phần trăm, "
            f"còn trống {_fmt_num(mem.available / 1e9, 1)} GB.")


def get_disk_usage() -> str:
    """O dia / con trong bao nhieu."""
    psutil = _psutil()
    disk = psutil.disk_usage("/")
    return (f"Ổ đĩa đã dùng {_fmt_num(disk.percent, 0)} phần trăm, "
            f"còn trống {_fmt_num(disk.free / 1e9, 1)} GB.")


def get_system_status() -> str:
    """Tom tat CPU + RAM + dia 1 cau."""
    psutil = _psutil()
    cpu = psutil.cpu_percent(interval=0.5)
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    return (f"Máy đang dùng {_fmt_num(cpu, 0)} phần trăm CPU, "
            f"{_fmt_num(mem.percent, 0)} phần trăm RAM, "
            f"{_fmt_num(disk.percent, 0)} phần trăm ổ đĩa.")


def _nvidia_query() -> list[str]:
    import subprocess

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,temperature.gpu,"
             "memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
    except (FileNotFoundError, subprocess.SubprocessError):
        raise RuntimeError("máy này không có GPU NVIDIA") from None
    if out.returncode != 0:
        raise RuntimeError("máy này không có GPU NVIDIA")
    return [line.strip() for line in out.stdout.strip().splitlines() if line.strip()]


def get_gpu_usage() -> str:
    """% GPU + VRAM (NVIDIA)."""
    parts = [x.strip() for x in _nvidia_query()[0].split(",")]
    if len(parts) < 4:
        raise RuntimeError("không đọc được thông số GPU")
    util, _temp, mem_used, mem_total = parts[:4]
    return (f"GPU đang dùng {util} phần trăm, "
            f"VRAM {mem_used} trên {mem_total} MB.")


def get_gpu_temperature() -> str:
    """Nhiet do GPU (NVIDIA)."""
    try:
        temp = _nvidia_query()[0].split(",")[1].strip()
    except Exception:
        raise RuntimeError("máy này không có GPU NVIDIA") from None
    level = "mát" if float(temp) < 60 else ("ấm" if float(temp) < 80 else "nóng")
    return f"GPU hiện {temp} độ, mức {level}."


def get_system_uptime() -> str:
    """May da chay bao lau."""
    psutil = _psutil()
    import time

    seconds = int(time.time() - psutil.boot_time())
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, _ = divmod(seconds, 60)
    parts = []
    if days:
        parts.append(f"{days} ngày")
    if hours:
        parts.append(f"{hours} giờ")
    if minutes or not parts:
        parts.append(f"{minutes} phút")
    return f"Máy đã chạy được {' '.join(parts)}."


def get_network_status() -> str:
    """Co internet khong + IP local."""
    import socket

    try:
        socket.create_connection(("8.8.8.8", 53), timeout=3).close()
        internet = "có internet"
    except OSError:
        internet = "mất internet"
    try:
        ip = socket.gethostbyname(socket.gethostname())
    except OSError:
        ip = "không rõ"
    return f"Máy {internet}, IP nội bộ {ip}."


def ping_host(host: str | None = None) -> str:
    """Ping 1 goi, tra do tre ms."""
    import re
    import subprocess

    target = (host or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9.\-]{1,253}", target):
        raise RuntimeError(f"địa chỉ '{host}' không hợp lệ")
    try:
        out = subprocess.run(["ping", "-c", "1", "-W", "2", target],
                             capture_output=True, text=True, timeout=10)
    except (FileNotFoundError, subprocess.SubprocessError):
        raise RuntimeError("máy thiếu lệnh ping") from None
    match = re.search(r"time=([\d.]+)\s*ms", out.stdout)
    if out.returncode == 0 and match:
        return f"Ping {target} mất {_fmt_num(float(match.group(1)), 1)} mili giây."
    return f"Không ping được {target} (mất mạng hoặc sai địa chỉ)."


def check_port(host: str | None = None, port: int | str | None = None) -> str:
    """Cổng TCP co mo khong (kiem tra camera/RTSP/backend)."""
    import re
    import socket

    target = (host or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9.\-]{1,253}", target):
        raise RuntimeError(f"địa chỉ '{host}' không hợp lệ")
    try:
        number = int(port)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise RuntimeError(f"cổng '{port}' không hợp lệ") from None
    if not 1 <= number <= 65535:
        raise RuntimeError(f"cổng '{port}' không hợp lệ")
    try:
        socket.create_connection((target, number), timeout=3).close()
        return f"Cổng {number} trên {target} đang mở."
    except OSError:
        return (f"Cổng {number} trên {target} đang đóng "
                f"hoặc không kết nối được.")


# ---------------------------------------------------------------------------
# File local (gioi han trong whitelist de an toan)
# ---------------------------------------------------------------------------

_FILE_ROOTS = ("output", "data", "config")


def _repo_root():
    from pathlib import Path

    return Path(__file__).resolve().parents[3]


def _safe_path(raw: str | None, must_exist: bool = True):
    from pathlib import Path

    root = _repo_root()
    candidate = (root / (raw or "")).resolve() if not str(raw or "").startswith("/") \
        else Path(str(raw)).resolve()
    allowed = [(root / name).resolve() for name in _FILE_ROOTS]
    if not any(candidate == base or base in candidate.parents for base in allowed):
        raise RuntimeError("chỉ được xem trong thư mục output, data, config")
    if must_exist and not candidate.exists():
        raise RuntimeError(f"không tìm thấy '{raw}'")
    return candidate


def list_files(folder: str | None = None) -> str:
    """Liet ke file trong thu muc whitelist (mac dinh output)."""
    path = _safe_path(folder or "output")
    if not path.is_dir():
        raise RuntimeError(f"'{folder}' không phải thư mục")
    entries = sorted(path.iterdir())
    if not entries:
        return f"Thư mục {path.name} đang trống."
    shown = [e.name + ("/" if e.is_dir() else "") for e in entries[:20]]
    more = f", còn {len(entries) - 20} mục nữa" if len(entries) > 20 else ""
    return (f"Thư mục {path.name} có {len(entries)} mục: "
            f"{', '.join(shown)}{more}.")


def find_file(name: str | None = None) -> str:
    """Tim file theo ten trong output/data/config."""
    keyword = (name or "").strip()
    if len(keyword) < 2:
        raise RuntimeError("tên file cần ít nhất 2 ký tự")
    hits: list[str] = []
    for base in _FILE_ROOTS:
        root = (_repo_root() / base).resolve()
        if not root.is_dir():
            continue
        for found in root.rglob(f"*{keyword}*"):
            hits.append(str(found.relative_to(_repo_root())))
            if len(hits) >= 10:
                break
    if not hits:
        return f"Không tìm thấy file nào tên giống '{keyword}'."
    return f"Tìm thấy {len(hits)} file: {', '.join(hits[:5])}."


def get_latest_file(folder: str | None = None) -> str:
    """File moi nhat trong thu muc."""
    path = _safe_path(folder or "output")
    if not path.is_dir():
        raise RuntimeError(f"'{folder}' không phải thư mục")
    files = [e for e in path.iterdir() if e.is_file()]
    if not files:
        return f"Thư mục {path.name} chưa có file nào."
    latest = max(files, key=lambda e: e.stat().st_mtime)
    size_kb = latest.stat().st_size / 1024
    return (f"File mới nhất trong {path.name} là {latest.name}, "
            f"{_fmt_num(size_kb, 0)} KB.")


def read_text_file(path: str | None = None) -> str:
    """Doc nhanh txt/log/md (toi da ~8000 ky tu)."""
    target = _safe_path(path)
    if target.suffix.lower() not in {".txt", ".log", ".md", ".json",
                                     ".csv", ".yaml", ".yml"}:
        raise RuntimeError("chỉ đọc file chữ txt, log, json, csv, md, yaml")
    if target.stat().st_size > 200_000:
        raise RuntimeError("file quá to, không đọc hết được")
    try:
        text = target.read_text(encoding="utf-8", errors="replace")[:8000]
    except OSError as error:
        raise RuntimeError(f"không đọc được file: {error}") from error
    if not text.strip():
        return f"File {target.name} đang trống."
    return f"Nội dung {target.name}: {text.strip()[:1500]}"


def read_json_file(path: str | None = None) -> str:
    """Tom tat cau truc file JSON."""
    import json as _json

    target = _safe_path(path)
    try:
        data = _json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RuntimeError(f"file JSON lỗi: {error}") from error
    if isinstance(data, dict):
        keys = list(data)[:10]
        return (f"File {target.name} có {len(data)} khóa: "
                f"{', '.join(str(k) for k in keys)}.")
    if isinstance(data, list):
        return f"File {target.name} là danh sách {len(data)} phần tử."
    return f"File {target.name} chứa một giá trị đơn."


def read_csv_file(path: str | None = None) -> str:
    """Dem hang/cot + ten cot file CSV."""
    import csv as _csv

    target = _safe_path(path)
    try:
        with open(target, newline="", encoding="utf-8-sig") as handle:
            rows = list(_csv.reader(handle))[:501]
    except OSError as error:
        raise RuntimeError(f"không đọc được file: {error}") from error
    if not rows:
        return f"File {target.name} đang trống."
    header = ", ".join(rows[0][:8])
    return (f"File {target.name} có {len(rows) - 1} dòng dữ liệu, "
            f"{len(rows[0])} cột: {header}.")


def get_folder_size(path: str | None = None) -> str:
    """Dung luong thu muc."""
    target = _safe_path(path or "output")
    if not target.is_dir():
        raise RuntimeError(f"'{path}' không phải thư mục")
    total = sum(entry.stat().st_size for entry in target.rglob("*")
                if entry.is_file())
    if total >= 1e9:
        return f"Thư mục {target.name} nặng {_fmt_num(total / 1e9, 1)} GB."
    if total >= 1e6:
        return f"Thư mục {target.name} nặng {_fmt_num(total / 1e6, 1)} MB."
    return f"Thư mục {target.name} nặng {_fmt_num(total / 1e3, 0)} KB."


# ---------------------------------------------------------------------------
# API public (free, khong key): le, kinh te, dong dat, sach
# ---------------------------------------------------------------------------

_COUNTRY_NAMES = {
    "viet nam": "VNM", "vnm": "VNM", "vn": "VNM",
    "my": "USA", "hoa ky": "USA", "usa": "USA", "us": "USA",
    "trung quoc": "CHN", "chn": "CHN", "cn": "CHN",
    "nhat ban": "JPN", "nhat": "JPN", "jpn": "JPN", "jp": "JPN",
    "han quoc": "KOR", "kor": "KOR", "kr": "KOR",
    "anh": "GBR", "gbr": "GBR", "gb": "GBR",
    "phap": "FRA", "fra": "FRA", "fr": "FRA",
    "duc": "DEU", "deu": "DEU", "de": "DEU",
    "uc": "AUS", "aus": "AUS", "au": "AUS",
    "canada": "CAN", "can": "CAN", "ca": "CAN",
    "nga": "RUS", "rus": "RUS", "ru": "RUS",
    "an do": "IND", "ind": "IND", "in": "IND",
    "thai lan": "THA", "tha": "THA", "th": "THA",
    "singapore": "SGP", "sgp": "SGP", "sg": "SGP",
    "lao": "LAO", "lao": "LAO", "campuchia": "KHM", "khm": "KHM",
    "malaysia": "MYS", "mys": "MYS", "indonesia": "IDN", "idn": "IDN",
    "philippines": "PHL", "phl": "PHL", "dai loan": "TWN", "twn": "TWN",
}


def _country_code(raw: str | None) -> str:
    import re

    key = _unaccent(raw or "")
    if key in _COUNTRY_NAMES:
        return _COUNTRY_NAMES[key]
    code = re.sub(r"[^a-z]", "", key).upper()
    if len(code) in (2, 3):
        return code
    raise RuntimeError(f"không hiểu nước '{raw}'")


def get_public_holidays(year: int | str | None = None,
                        country: str | None = None) -> str:
    """Ngay le nam chi dinh (Nager.Date, mac dinh Viet Nam)."""
    from datetime import date

    code = _country_code(country or "Việt Nam")
    # Nager.Date dung ma alpha-2 (VN), World Bank dung alpha-3 (VNM).
    code = {"VNM": "VN", "USA": "US", "CHN": "CN", "JPN": "JP", "KOR": "KR",
            "GBR": "GB", "FRA": "FR", "DEU": "DE", "AUS": "AU", "CAN": "CA",
            "RUS": "RU", "IND": "IN", "THA": "TH", "SGP": "SG", "LAO": "LA",
            "KHM": "KH", "MYS": "MY", "IDN": "ID", "PHL": "PH",
            "TWN": "TW"}.get(code, code)
    try:
        number = int(year) if year else date.today().year  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise RuntimeError(f"năm '{year}' không hợp lệ") from None
    if not 1900 <= number <= 2100:
        raise RuntimeError(f"năm '{year}' không hợp lệ")
    try:
        data = _http_json(
            f"https://date.nager.at/api/v3/PublicHolidays/{number}/{code}")
    except Exception as error:
        raise RuntimeError(f"không lấy được ngày lễ: {error}") from error
    if not data:
        return f"Năm {number} nước {code} chưa có lịch nghỉ lễ."
    first = [f"{h.get('localName', h.get('name'))} "
             f"({h.get('date', '')[8:10]}/{h.get('date', '')[5:7]})"
             for h in data[:5]]
    return (f"Năm {number} nước {code} có {len(data)} ngày lễ, "
            f"gần nhất: {', '.join(first)}.")


def _worldbank(country: str | None, indicator: str, label: str,
               money: bool = False, percent: bool = False) -> str:
    code = _country_code(country or "Việt Nam")
    url = (f"https://api.worldbank.org/v2/country/{code}/indicator/"
           f"{indicator}?format=json&per_page=10")
    try:
        data = _http_json(url)
        rows = data[1] or []
    except Exception as error:
        raise RuntimeError(f"không lấy được dữ liệu: {error}") from error
    picked = next((r for r in rows if r.get("value") is not None), None)
    if picked is None:
        return f"Chưa có số liệu {label} của nước {code}."
    value = float(picked["value"])
    year = picked.get("date", "")
    if money:
        shown = _fmt_big(value, "đô la")
    elif percent:
        shown = f"{_fmt_num(value)} phần trăm"
    else:
        shown = _fmt_big(value, "người")
    return f"{label} của {code} năm {year} là {shown}."


def get_country_population(country: str | None = None) -> str:
    """Dan so moi nhat (World Bank)."""
    return _worldbank(country, "SP.POP.TOTL", "Dân số")


def get_country_gdp(country: str | None = None) -> str:
    """GDP USD moi nhat (World Bank)."""
    return _worldbank(country, "NY.GDP.MKTP.CD", "GDP", money=True)


def get_country_inflation(country: str | None = None) -> str:
    """Lam phat % moi nhat (World Bank)."""
    return _worldbank(country, "FP.CPI.TOTL.ZG", "Lạm phát", percent=True)


def get_recent_earthquakes(region: str | None = None) -> str:
    """Dong dat >=5.5 do 30 ngay qua (USGS), loc theo vung neu co."""
    from datetime import date, timedelta
    from urllib.parse import urlencode

    since = (date.today() - timedelta(days=30)).isoformat()
    query = urlencode({"format": "geojson", "starttime": since,
                       "minmagnitude": 5.5, "limit": 10, "orderby": "time"})
    try:
        data = _http_json(
            f"https://earthquake.usgs.gov/fdsnws/event/1/query?{query}")
    except Exception as error:
        raise RuntimeError(f"không lấy được dữ liệu động đất: {error}") from error
    feats = (data.get("features") or [])
    want = _unaccent(region or "")
    if want:
        feats = [f for f in feats
                 if want in _unaccent((f.get("properties") or {}).get("place", ""))]
        if not feats:
            return (f"30 ngày qua không có động đất lớn nào ở vùng {region}.")
    if not feats:
        return "30 ngày qua không ghi nhận động đất trên 5,5 độ."
    top = []
    for feat in feats[:3]:
        props = feat.get("properties") or {}
        mag = props.get("mag", "?")
        place = props.get("place", "chưa rõ vị trí")
        top.append(f"{mag} độ ở {place}")
    scope = f" vùng {region}" if want else ""
    return f"Động đất lớn nhất 30 ngày qua{scope}: {'; '.join(top)}."


def search_book(title: str | None = None) -> str:
    """Tim sach theo ten (Open Library)."""
    from urllib.parse import urlencode

    keyword = (title or "").strip()
    if len(keyword) < 2:
        raise RuntimeError("tên sách cần ít nhất 2 ký tự")
    try:
        data = _http_json(
            "https://openlibrary.org/search.json?"
            + urlencode({"title": keyword, "limit": 3}))
    except Exception as error:
        raise RuntimeError(f"không tìm được sách: {error}") from error
    docs = data.get("docs") or []
    if not docs:
        return f"Không tìm thấy sách nào tên giống '{keyword}'."
    first = docs[0]
    name = first.get("title", "chưa rõ tên")
    authors = ", ".join((first.get("author_name") or ["khuyết danh"])[:2])
    year = first.get("first_publish_year", "không rõ năm")
    return f"Sách '{name}' của {authors}, xuất bản đầu năm {year}."


def get_book_info(isbn: str | None = None) -> str:
    """Tra sach theo ma ISBN (Open Library)."""
    import re

    digits = re.sub(r"[^0-9Xx]", "", (isbn or ""))
    if len(digits) not in (10, 13):
        raise RuntimeError(f"mã ISBN '{isbn}' không hợp lệ")
    try:
        data = _http_json(f"https://openlibrary.org/isbn/{digits}.json")
    except Exception as error:
        raise RuntimeError(f"không tra được ISBN: {error}") from error
    name = data.get("title", "chưa rõ tên")
    year = data.get("publish_date", "không rõ năm")
    return f"ISBN {digits} là sách '{name}', xuất bản {year}."


# ---------------------------------------------------------------------------
# Tien ich he thong local (uuid, hash, json, timestamp)
# ---------------------------------------------------------------------------

def generate_uuid() -> str:
    """Sinh 1 UUID v4."""
    import uuid

    return f"UUID mới là {uuid.uuid4()}."


def hash_text(text: str | None = None) -> str:
    """SHA256 cua doan text (noi 12 ky tu dau)."""
    import hashlib

    if not (text or "").strip():
        raise RuntimeError("chưa có chữ nào để băm")
    digest = hashlib.sha256(text.strip().encode("utf-8")).hexdigest()
    return f"Mã SHA256 bắt đầu bằng {digest[:12]}."


def validate_json(text: str | None = None) -> str:
    """Kiem tra chuoi co phai JSON hop le."""
    import json as _json

    try:
        data = _json.loads(text or "")
    except (ValueError, TypeError) as error:
        return f"Chuỗi này không phải JSON hợp lệ: {error}."
    extra = (f" Có {len(data)} khóa." if isinstance(data, dict)
             else (f" Có {len(data)} phần tử."
                   if isinstance(data, list) else ""))
    return f"Chuỗi này là JSON hợp lệ.{extra}"


def timestamp_to_datetime(ts: int | str | float | None = None) -> str:
    """Timestamp -> ngay gio Viet Nam."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    try:
        number = float(str(ts).strip())  # type: ignore[union-attr]
    except (TypeError, ValueError, AttributeError):
        raise RuntimeError(f"timestamp '{ts}' không hợp lệ") from None
    number = number / 1000.0 if number > 1e12 else number
    try:
        moment = datetime.fromtimestamp(number, ZoneInfo("Asia/Ho_Chi_Minh"))
    except (OverflowError, OSError, ValueError):
        raise RuntimeError(f"timestamp '{ts}' không hợp lệ") from None
    weekdays = ["thứ Hai", "thứ Ba", "thứ Tư", "thứ Năm",
                "thứ Sáu", "thứ Bảy", "Chủ nhật"]
    return (f"Là {weekdays[moment.weekday()]}, {moment.day} tháng {moment.month} "
            f"năm {moment.year} lúc {moment.hour} giờ {moment.minute} phút.")


def datetime_to_timestamp(dt: str | None = None) -> str:
    """'YYYY-MM-DD [HH:MM:SS]' gio Viet Nam -> timestamp."""
    import re
    from datetime import datetime
    from zoneinfo import ZoneInfo

    text = (dt or "").strip()
    match = re.match(r"^(\d{4}-\d{1,2}-\d{1,2})(?:[ T](\d{1,2}):(\d{1,2})"
                     r"(?::(\d{1,2}))?)?$", text)
    if not match:
        match = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{4})(?:[ T](\d{1,2}):(\d{1,2})"
                         r"(?::(\d{1,2}))?)?$", text)
        if not match:
            raise RuntimeError(f"không hiểu ngày giờ '{dt}' "
                               f"(thử 2026-09-20 14:30)")
        year, month, day = (int(match.group(3)), int(match.group(2)),
                            int(match.group(1)))
        hms = [int(match.group(i) or 0) for i in (4, 5, 6)]
    else:
        year, month, day = (int(x) for x in match.group(1).split("-"))
        hms = [int(match.group(i) or 0) for i in (2, 3, 4)]
    try:
        moment = datetime(year, month, day, *hms,
                          tzinfo=ZoneInfo("Asia/Ho_Chi_Minh"))
    except ValueError:
        raise RuntimeError(f"ngày giờ '{dt}' không hợp lệ") from None
    return f"Timestamp là {int(moment.timestamp())}."


# ---------------------------------------------------------------------------
# Cau noi that voi chuong trinh (doc file cau noi / kiem tra that)
# ---------------------------------------------------------------------------

def get_last_transcript() -> str:
    """Cau STT gan nhat nguoi dung da noi (do halinh_assistant_local.py ghi lai)."""
    target = _repo_root() / "output" / "qa_cache" / "last_transcript.txt"
    try:
        text = target.read_text(encoding="utf-8").strip()
    except OSError:
        return "Chưa ghi nhận câu nói nào trong phiên này."
    if not text:
        return "Chưa ghi nhận câu nói nào trong phiên này."
    return f"Câu gần nhất bạn nói là: {text[:300]}"


def get_audio_status() -> str:
    """Mic pulse mac dinh con song khong."""
    import subprocess

    try:
        out = subprocess.run(["pactl", "get-default-source"],
                             capture_output=True, text=True, timeout=5)
    except (FileNotFoundError, subprocess.SubprocessError):
        return "Máy này không có hệ thống âm thanh PulseAudio."
    if out.returncode != 0:
        return "Không tìm thấy mic nào trên máy."
    return f"Mic {out.stdout.strip()} đang sẵn sàng."


def get_latest_error() -> str:
    """Dong ERROR moi nhat trong log output (neu co)."""
    root = (_repo_root() / "output").resolve()
    logs = sorted(root.glob("*.log"),
                  key=lambda e: e.stat().st_mtime, reverse=True)[:3]
    for log in logs:
        try:
            lines = log.read_text(encoding="utf-8",
                                  errors="replace").splitlines()[-300:]
        except OSError:
            continue
        hits = [ln.strip()[:200] for ln in lines
                if "error" in ln.lower() or "traceback" in ln.lower()]
        if hits:
            return f"Lỗi mới nhất trong {log.name}: {hits[-1][:300]}"
    return "Không thấy lỗi nào trong log output."


# ---------------------------------------------------------------------------
# STUB - chua lien ket: hen gio/bao thuc can scheduler nen tam tra loi
# xac nhan gia dinh. Gi chu ky ham chuan de sau thay ruot la xong.
# ---------------------------------------------------------------------------


def set_timer(seconds: int | str | None = None) -> str:
    """STUB hen gio (chua co scheduler) -> tra loi xac nhan gia dinh."""
    try:
        number = int(seconds)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise RuntimeError(f"số giây '{seconds}' không hợp lệ") from None
    if not 1 <= number <= 86400:
        raise RuntimeError("hẹn giờ từ 1 giây đến 24 giờ")
    import random as _random

    _random.seed()
    if number < 60:
        return f"Đã hẹn giờ {number} giây nữa. (Bản thử: chưa reo thật.)"
    mins, secs = divmod(number, 60)
    tail = f" {secs} giây" if secs else ""
    return (f"Đã hẹn giờ {mins} phút{tail} nữa. "
            f"(Bản thử: chưa reo thật.)")


def set_alarm(time: str | None = None) -> str:
    """STUB bao thuc (chua co scheduler) -> tra loi xac nhan gia dinh."""
    import re

    match = re.match(r"^(\d{1,2})(?::(\d{1,2}))?$", (time or "").strip())
    if not match:
        raise RuntimeError(f"không hiểu giờ '{time}' (thử 6:30)")
    hour, minute = int(match.group(1)), int(match.group(2) or 0)
    if not (0 <= hour < 24 and 0 <= minute < 60):
        raise RuntimeError(f"giờ '{time}' không hợp lệ")
    return (f"Đã đặt báo thức lúc {hour} giờ {minute} phút. "
            f"(Bản thử: chưa reo thật.)")


# ---------------------------------------------------------------------------
# Cau runtime THAT tu pipeline tracking (run_workstate_local ghi
# output/qa_cache/runtime_status.json moi 2s). Cac tool duoi doc file nay;
# pipeline chua chay -> tra loi that la "chua co du lieu" (khong random).
# ---------------------------------------------------------------------------

_RUNTIME_PARTS = ("output", "qa_cache", "runtime_status.json")
_RUNTIME_MAX_AGE_S = 15.0


def read_runtime_status(max_age_s: float = _RUNTIME_MAX_AGE_S) -> dict | None:
    """Doc runtime_status.json; None neu thieu/cu/hong."""
    import json as _json
    import time as _time

    try:
        target = _repo_root().joinpath(*_RUNTIME_PARTS)
        data = _json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    try:
        age = _time.time() - float(data.get("ts", 0))
    except (TypeError, ValueError):
        return None
    if age < 0 or age > float(max_age_s):
        return None
    return data


def _runtime_or_none() -> dict | None:
    return read_runtime_status()


def _fmt_ago(seconds: object) -> str:
    """0.5 -> 'vua xong'; 90 -> '1 phut truoc' (noi)."""
    try:
        value = float(seconds)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "không rõ lúc nào"
    if value < 0:
        return "không rõ lúc nào"
    if value < 10:
        return "vừa xong"
    if value < 60:
        return f"{int(value)} giây trước"
    minutes = int(value // 60)
    if minutes < 60:
        return f"{minutes} phút trước"
    hours = minutes // 60
    if hours < 24:
        rest = minutes % 60
        return f"{hours} giờ {rest} phút trước" if rest else f"{hours} giờ trước"
    return f"{hours // 24} ngày trước"


def _fmt_dur(seconds: object) -> str:
    """3660 -> '1 gio 1 phut' (noi thoi luong)."""
    try:
        total = int(float(seconds))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "không rõ"
    if total < 0:
        return "không rõ"
    days, total = divmod(total, 86400)
    hours, total = divmod(total, 3600)
    minutes, _ = divmod(total, 60)
    parts = []
    if days:
        parts.append(f"{days} ngày")
    if hours:
        parts.append(f"{hours} giờ")
    if minutes or not parts:
        parts.append(f"{minutes} phút")
    return " ".join(parts)


def _match_name(want: str | None, names: list[str]) -> str | None:
    """Khop ten bo dau/khong phan biet hoa thuong, tra ten goc."""
    key = _unaccent(want or "")
    if not key:
        return None
    for name in names:
        norm = _unaccent(name)
        if key and (key in norm or norm in key):
            return name
    return None


def get_camera_status() -> str:
    """Trang thai camera that (doc runtime tracking)."""
    data = _runtime_or_none()
    if data is None:
        return "Pipeline tracking chưa chạy nên chưa có trạng thái camera."

    def _words(cam: object, label: str) -> str:
        cam = cam if isinstance(cam, dict) else {}
        if not cam.get("enabled", True):
            return f"Camera {label} đang tắt"
        if cam.get("open") and not cam.get("exhausted"):
            return f"Camera {label} trực tuyến"
        return f"Camera {label} mất tín hiệu"

    cam_a = data.get("camera_a")
    if data.get("single_channel", True):
        return f"{_words(cam_a, 'A')} (chạy 1 kênh webcam)."
    return f"{_words(cam_a, 'A')} {_words(data.get('camera_b'), 'B')}."


def get_stream_status() -> str:
    """Luong video co on khong (doc runtime tracking)."""
    data = _runtime_or_none()
    if data is None:
        return "Pipeline tracking chưa chạy nên chưa có trạng thái luồng."
    cam_a = data.get("camera_a") if isinstance(
        data.get("camera_a"), dict) else {}
    fps = data.get("loop_fps")
    extra = (f" Tốc độ vòng lặp {_fmt_num(fps, 0)} khung một giây."
             if isinstance(fps, (int, float)) and fps > 0 else "")
    if not cam_a.get("open") or cam_a.get("exhausted"):
        return "Luồng camera A đang gián đoạn."
    if data.get("single_channel", True):
        return f"Luồng webcam A ổn định.{extra}"
    cam_b = data.get("camera_b") if isinstance(
        data.get("camera_b"), dict) else {}
    tail = (" Kênh B ổn định." if cam_b.get("open")
            and not cam_b.get("exhausted") else " Kênh B đang gián đoạn.")
    return f"Luồng camera A ổn định.{extra}{tail}"


def get_camera_fps() -> str:
    """FPS vong lap that (doc runtime tracking)."""
    data = _runtime_or_none()
    fps = (data or {}).get("loop_fps")
    if not isinstance(fps, (int, float)) or fps <= 0:
        return "Chưa đo được FPS (pipeline chưa chạy?)."
    return f"Camera đang chạy {_fmt_num(fps, 0)} khung hình một giây."


def get_person_count() -> str:
    """So nguoi tracking thay (doc runtime tracking)."""
    data = _runtime_or_none()
    if data is None:
        return "Pipeline tracking chưa chạy nên chưa đếm được người."
    try:
        number = int(data.get("count", 0))
    except (TypeError, ValueError):
        number = 0
    if number <= 0:
        return "Hiện không thấy ai trong phòng."
    return f"Hiện có {number} người trong phòng."


def _room_people(data: dict) -> tuple[list[str], int]:
    """(ten da dinh danh trong phong, so nguoi la) tu runtime."""
    known: list[str] = []
    strangers = 0
    people = data.get("people") or []
    if not isinstance(people, list):
        return known, strangers
    for person in people:
        if not isinstance(person, dict) or not person.get("in_room"):
            continue
        name = str(person.get("name") or "").strip()
        if name and "nknown" not in name.lower():
            if name not in known:
                known.append(name)
        else:
            strangers += 1
    return known, strangers


def get_current_people() -> str:
    """Ai dang co mat (doc runtime tracking)."""
    data = _runtime_or_none()
    if data is None:
        return "Pipeline tracking chưa chạy nên chưa biết ai có mặt."
    known, strangers = _room_people(data)
    if not known and strangers <= 0:
        return "Hiện chưa thấy ai trong phòng."
    bits = []
    if known:
        bits.append(", ".join(known))
    if strangers > 0:
        bits.append(f"{strangers} người lạ")
    return "Trong phòng đang có " + " và ".join(bits) + "."


def get_last_seen(name: str | None = None) -> str:
    """Ai do xuat hien lan cuoi khi nao (doc runtime tracking)."""
    data = _runtime_or_none()
    if data is None:
        return "Pipeline tracking chưa chạy nên chưa có lịch sử xuất hiện."
    people = data.get("people") or []
    if not isinstance(people, list) or not people:
        return "Trong phiên này chưa thấy ai."
    if (name or "").strip():
        names = [str(p.get("name") or "") for p in people
                 if isinstance(p, dict) and p.get("name")]
        matched = _match_name(name, names)
        if matched is None:
            return f"Trong phiên này chưa thấy {name}."
        entry = next(p for p in people
                     if isinstance(p, dict) and p.get("name") == matched)
        return (f"{matched} xuất hiện lần cuối "
                f"{_fmt_ago(entry.get('last_seen_ago_s'))}.")
    dated = [p for p in people if isinstance(p, dict)]
    dated.sort(key=lambda p: (p.get("last_seen_ago_s") is None,
                              p.get("last_seen_ago_s") or 0))
    top = dated[0]
    label = str(top.get("name") or "một người lạ").strip()
    return (f"Gần nhất là {label}, xuất hiện lần cuối "
            f"{_fmt_ago(top.get('last_seen_ago_s'))}.")


def get_attendance_today(name: str | None = None) -> str:
    """Cham cong hom nay (doc runtime tracking)."""
    data = _runtime_or_none()
    if data is None:
        return "Pipeline tracking chưa chạy nên chưa có dữ liệu chấm công."
    rows = data.get("attendance_today") or []
    if not isinstance(rows, list):
        rows = []
    rows = [r for r in rows if isinstance(r, dict)]
    if (name or "").strip():
        names = [str(r.get("name") or "") for r in rows]
        matched = _match_name(name, names)
        if matched is None:
            return f"{name} hôm nay chưa chấm công."
        entry = next(r for r in rows if r.get("name") == matched)
        return (f"{matched} đã chấm công lúc "
                f"{_speak_hour_min(str(entry.get('time') or ''))}.")
    if not rows:
        return "Hôm nay chưa ai chấm công."
    first = [f"{r.get('name')} lúc "
             f"{_speak_hour_min(str(r.get('time') or ''))}" for r in rows[:3]]
    more = f", còn {len(rows) - 3} người nữa" if len(rows) > 3 else ""
    return (f"Hôm nay có {len(rows)} người chấm công: "
            f"{', '.join(first)}{more}.")


def get_recent_events() -> str:
    """Su kien gan nhat (doc runtime tracking)."""
    data = _runtime_or_none()
    if data is None:
        return "Pipeline tracking chưa chạy nên chưa có sự kiện nào."
    events = [str(e)[:200] for e in (data.get("recent_events") or [])
              if str(e).strip()]
    if not events:
        return "Trong phiên này chưa có sự kiện nào."
    if len(events) == 1:
        return f"Sự kiện gần nhất: {events[0]}"
    return f"Sự kiện gần nhất: {events[-1]} Trước đó: {events[-2]}"


def get_app_status() -> str:
    """Pipeline co dang chay khong (doc runtime tracking)."""
    data = _runtime_or_none()
    if data is None:
        return "Pipeline tracking hiện không chạy."
    try:
        number = int(data.get("count", 0))
    except (TypeError, ValueError):
        number = 0
    return (f"Pipeline đang chạy bình thường, đã chạy {_fmt_dur(data.get('uptime_s'))}, "
            f"đang thấy {number} người.")


def get_model_status() -> str:
    """Model AI nao da load (doc runtime + voice)."""
    data = _runtime_or_none()
    models = (data or {}).get("models") if isinstance(data, dict) else None
    if not isinstance(models, dict):
        return ("Pipeline tracking chưa chạy. Phía voice thì Whisper "
                "và ZeroTTS đã sẵn sàng.")
    from pathlib import Path as _Path

    yolo = _Path(str(models.get("yolo", "?"))).name
    reid = str(models.get("reid", "?"))
    face = str(models.get("face", "?"))
    return (f"YOLO {yolo}, ReID {reid} và Face {face} đã load xong. "
            f"Whisper và ZeroTTS phía voice sẵn sàng.")


def get_inference_latency() -> str:
    """Do tre YOLO that (doc runtime tracking)."""
    data = _runtime_or_none()
    ms = (data or {}).get("inference_ms")
    if not isinstance(ms, (int, float)) or ms <= 0:
        return "Chưa đo được độ trễ (pipeline chưa chạy?)."
    return (f"YOLO suy luận mất khoảng {_fmt_num(ms, 1)} "
            f"mili giây một khung hình.")


def get_stt_latency() -> str:
    """Do tre Whisper that (trung binh cac lan gan nhat)."""
    from camera_tracking.voice.rtsp_voice_listener import stt_latency_stats

    stats = stt_latency_stats()
    if stats is None:
        return "Trong phiên này chưa có lượt nhận diện nào."
    count, mean = stats
    return (f"Whisper nhận diện trung bình {_fmt_num(mean, 1)} giây một câu "
            f"({count} câu gần nhất).")


# ---------------------------------------------------------------------------
# Ket hop HINH ANH + AM THANH (fusion 2 chieu voi pipeline tracking).
# - Mat thay gi -> mieng biet do: describe_scene/with_scene nhet ngu canh
#   camera vao cau hoi (van 1 call Gemini).
# - Mieng ra lenh -> mat lam theo: request_greet/take_snapshot ghi file
#   voice_command.json, tracking poll + thuc thi (chao qua loa / luu hinh).
# ---------------------------------------------------------------------------

_SCENE_LABELS = {
    "working": "đang làm việc",
    "returning": "đang quay lại",
    "away": "vắng mặt",
    "away_temp": "vắng tạm thời",
    "possibly_out": "có thể đã ra ngoài",
    "unknown": "",
}


def describe_scene() -> str | None:
    """1 cau mo ta truoc camera (de nhet vao prompt voice)."""
    data = _runtime_or_none()
    if data is None:
        return None
    bits = []
    people = data.get("people") or []
    if not isinstance(people, list):
        return "Trước camera chưa có ai."
    for person in people:
        if not isinstance(person, dict) or not person.get("in_room"):
            continue
        name = str(person.get("name") or "").strip()
        label = _SCENE_LABELS.get(
            str(person.get("label") or "").strip().lower(), "")
        if name and "nknown" not in name.lower():
            bits.append(f"{name} ({label})" if label else name)
        else:
            bits.append(f"một người lạ ({label})" if label else "một người lạ")
    if not bits:
        return "Trước camera chưa có ai."
    if len(bits) == 1:
        return f"Trước camera có {bits[0]}."
    return f"Trước camera có {', '.join(bits[:-1])} và {bits[-1]}."


def with_scene(question: str) -> str:
    """Ghep ngu canh camera vao cau hoi (voice hieu 'nguoi truoc mat')."""
    scene = describe_scene()
    if not scene:
        return question
    return f"[Ngữ cảnh camera lúc hỏi: {scene}]\nCâu hỏi: {question}"


_VOICE_CMD_PARTS = ("output", "qa_cache", "voice_command.json")


def _write_voice_cmd(action: str, **fields) -> None:
    import json as _json
    import time as _time

    target = _repo_root().joinpath(*_VOICE_CMD_PARTS)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"action": action, "ts": _time.time(), "status": "pending"}
    payload.update(fields)
    tmp = target.with_suffix(".tmp")
    tmp.write_text(_json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.replace(target)


def request_greet(name: str | None = None) -> str:
    """Nho camera chao ai do dang trong phong (tracking thuc thi)."""
    data = _runtime_or_none()
    if data is None:
        return "Pipeline tracking chưa chạy nên chưa nhờ chào được."
    want = (name or "").strip()
    if not want:
        return "Chưa nghe rõ tên, bạn nói lại tên người cần chào nhé."
    known, _ = _room_people(data)
    matched = _match_name(want, known)
    if matched is None:
        return f"Không thấy {want} trong phòng nên chưa chào được."
    try:
        _write_voice_cmd("greet", name=matched)
    except OSError as error:
        return f"Không gửi được lệnh chào: {error}"
    return f"Đã nhờ camera chào {matched}."


def take_snapshot(camera: str | None = None) -> str:
    """Nho camera chup 1 tam hinh (tracking luu vao output/snapshots)."""
    data = _runtime_or_none()
    if data is None:
        return "Pipeline tracking chưa chạy nên chưa chụp được."
    cam = (camera or "A").strip().upper()
    if cam not in ("A", "B"):
        cam = "A"
    try:
        _write_voice_cmd("snapshot", camera=cam)
    except OSError as error:
        return f"Không gửi được lệnh chụp: {error}"
    return f"Đã chụp ảnh camera {cam}, lưu vào thư mục snapshots."


# ---------------------------------------------------------------------------
# Registry: ten tool -> ham + khai bao function-calling + danh sach stub.
# halinh_assistant_local.py import tu day (1 nguon duy nhat, khoi lech nhau).
# ---------------------------------------------------------------------------

def _decl(name: str, description: str, properties: dict | None = None,
          required: list | None = None) -> dict:
    return {"name": name, "description": description,
            "parameters": {"type": "object",
                           "properties": properties or {},
                           "required": required or []}}


def _str_prop(description: str) -> dict:
    return {"type": "string", "description": description}


TOOL_DECLARATIONS: list[dict] = [
    _decl("get_weather", "Thời tiết hiện tại của một thành phố.",
          {"city": _str_prop("Tên thành phố, bỏ trống = Đà Nẵng.")}),
    _decl("get_current_time", "Giờ hiện tại (giờ Việt Nam)."),
    _decl("get_current_date", "Ngày hôm nay."),
    _decl("calculate", "Tính biểu thức số học.",
          {"expression": _str_prop("Biểu thức, ví dụ '15*37'.")},
          ["expression"]),
    _decl("get_weather_forecast", "Dự báo nhiệt độ ngày chỉ định.",
          {"city": _str_prop("Thành phố, bỏ trống = Đà Nẵng."),
           "day": _str_prop("Hôm nay, ngày mai, ngày kia hoặc 20/9. "
                            "Bỏ trống = ngày mai.")}),
    _decl("get_rain_probability", "Khả năng mưa ngày chỉ định.",
          {"city": _str_prop("Thành phố, bỏ trống = Đà Nẵng."),
           "day": _str_prop("Hôm nay, ngày mai hoặc 20/9. Bỏ trống = ngày mai.")}),
    _decl("get_air_quality", "Chất lượng không khí AQI, bụi mịn hiện tại.",
          {"city": _str_prop("Thành phố, bỏ trống = Đà Nẵng.")}),
    _decl("get_uv_index", "Chỉ số UV hiện tại.",
          {"city": _str_prop("Thành phố, bỏ trống = Đà Nẵng.")}),
    _decl("get_sunrise_sunset", "Giờ mặt trời mọc và lặn.",
          {"city": _str_prop("Thành phố, bỏ trống = Đà Nẵng."),
           "day": _str_prop("Hôm nay, ngày mai hoặc 20/9. Bỏ trống = hôm nay.")}),
    _decl("convert_currency", "Đổi tiền theo tỷ giá Frankfurter.",
          {"amount": _str_prop("Số tiền, ví dụ 100."),
           "from_currency": _str_prop("Từ tiền tệ: USD, VND, Euro..."),
           "to_currency": _str_prop("Sang tiền tệ: VND, USD...")},
          ["amount", "from_currency", "to_currency"]),
    _decl("convert_unit", "Đổi đơn vị dài, nặng, nhiệt độ, thể tích.",
          {"value": _str_prop("Giá trị số."),
           "from_unit": _str_prop("Từ đơn vị: km, kg, độ C..."),
           "to_unit": _str_prop("Sang đơn vị: m, g, độ F...")},
          ["value", "from_unit", "to_unit"]),
    _decl("date_difference", "Số ngày giữa hai mốc.",
          {"date1": _str_prop("Mốc đầu: 1/9/2026 hoặc hôm nay."),
           "date2": _str_prop("Mốc sau: 1/1/2027.")},
          ["date1", "date2"]),
    _decl("convert_timezone", "Đổi giờ giữa hai nơi.",
          {"time": _str_prop("Giờ cần đổi: 14:30."),
           "from_tz": _str_prop("Từ nơi nào, bỏ trống = Việt Nam."),
           "to_tz": _str_prop("Sang nơi nào: Tokyo, London...")},
          ["time", "to_tz"]),
    _decl("get_system_status", "Tóm tắt CPU, RAM, ổ đĩa máy này."),
    _decl("get_cpu_usage", "Phần trăm CPU hiện tại."),
    _decl("get_ram_usage", "RAM đã dùng và còn trống."),
    _decl("get_disk_usage", "Ổ đĩa đã dùng và còn trống."),
    _decl("get_gpu_usage", "Phần trăm GPU và VRAM NVIDIA."),
    _decl("get_gpu_temperature", "Nhiệt độ GPU NVIDIA."),
    _decl("get_system_uptime", "Máy đã chạy bao lâu."),
    _decl("get_network_status", "Có internet không, IP nội bộ."),
    _decl("ping_host", "Ping một địa chỉ, đo độ trễ.",
          {"host": _str_prop("Tên miền hoặc IP: google.com.")}, ["host"]),
    _decl("check_port", "Cổng TCP có mở không.",
          {"host": _str_prop("Tên miền hoặc IP."),
           "port": _str_prop("Số cổng: 554, 80...")}, ["host", "port"]),
    _decl("list_files", "Liệt kê file trong thư mục cho phép.",
          {"folder": _str_prop("output, data hoặc config. Bỏ trống = output.")}),
    _decl("find_file", "Tìm file theo tên trong output, data, config.",
          {"name": _str_prop("Từ khóa tên file.")}, ["name"]),
    _decl("get_latest_file", "File mới nhất trong thư mục cho phép.",
          {"folder": _str_prop("output, data hoặc config. Bỏ trống = output.")}),
    _decl("read_text_file", "Đọc nhanh file chữ txt, log, md.",
          {"path": _str_prop("Đường dẫn trong output, data, config.")},
          ["path"]),
    _decl("read_json_file", "Tóm tắt cấu trúc file JSON.",
          {"path": _str_prop("Đường dẫn trong output, data, config.")},
          ["path"]),
    _decl("read_csv_file", "Đếm dòng, cột và tên cột file CSV.",
          {"path": _str_prop("Đường dẫn trong output, data, config.")},
          ["path"]),
    _decl("get_folder_size", "Dung lượng thư mục.",
          {"path": _str_prop("output, data hoặc config. Bỏ trống = output.")}),
    _decl("get_public_holidays", "Ngày lễ trong năm của một nước.",
          {"year": _str_prop("Năm: 2026. Bỏ trống = năm nay."),
           "country": _str_prop("Nước: Việt Nam. Bỏ trống = Việt Nam.")}),
    _decl("get_country_population", "Dân số mới nhất (World Bank).",
          {"country": _str_prop("Tên nước: Việt Nam, Mỹ, Nhật...")},
          ["country"]),
    _decl("get_country_gdp", "GDP đô la mới nhất (World Bank).",
          {"country": _str_prop("Tên nước.")}, ["country"]),
    _decl("get_country_inflation", "Lạm phát phần trăm mới nhất (World Bank).",
          {"country": _str_prop("Tên nước.")}, ["country"]),
    _decl("get_recent_earthquakes", "Động đất lớn 30 ngày qua (USGS).",
          {"region": _str_prop("Vùng cần lọc: Nhật Bản... Bỏ trống = toàn cầu.")}),
    _decl("search_book", "Tìm sách theo tên (Open Library).",
          {"title": _str_prop("Tên sách.")}, ["title"]),
    _decl("get_book_info", "Tra sách theo mã ISBN (Open Library).",
          {"isbn": _str_prop("Mã ISBN 10 hoặc 13 số.")}, ["isbn"]),
    _decl("generate_uuid", "Sinh một mã UUID mới."),
    _decl("hash_text", "Băm SHA256 một đoạn chữ.",
          {"text": _str_prop("Chữ cần băm.")}, ["text"]),
    _decl("validate_json", "Kiểm tra chuỗi có phải JSON hợp lệ.",
          {"text": _str_prop("Chuỗi cần kiểm tra.")}, ["text"]),
    _decl("timestamp_to_datetime", "Timestamp thành ngày giờ Việt Nam.",
          {"ts": _str_prop("Số timestamp giây.")}, ["ts"]),
    _decl("datetime_to_timestamp", "Ngày giờ Việt Nam thành timestamp.",
          {"dt": _str_prop("2026-09-20 14:30.")}, ["dt"]),
    _decl("get_last_transcript", "Câu nói gần nhất của người dùng."),
    _decl("get_audio_status", "Mic máy tính còn hoạt động không."),
    _decl("get_latest_error", "Lỗi mới nhất trong log output."),
    _decl("set_timer", "Hẹn giờ (bản thử, chưa reo thật).",
          {"seconds": _str_prop("Số giây: 300.")}, ["seconds"]),
    _decl("set_alarm", "Đặt báo thức (bản thử, chưa reo thật).",
          {"time": _str_prop("Giờ báo: 6:30.")}, ["time"]),
    _decl("get_camera_status", "Trạng thái camera (số liệu trực tiếp)."),
    _decl("get_stream_status", "Luồng RTSP có ổn không (số liệu trực tiếp)."),
    _decl("get_camera_fps", "FPS thực của camera (số liệu trực tiếp)."),
    _decl("get_person_count", "Bao nhiêu người trong phòng (số liệu trực tiếp)."),
    _decl("get_current_people", "Những ai đang có mặt (số liệu trực tiếp)."),
    _decl("get_last_seen", "Ai đó xuất hiện lần cuối khi nào (số liệu trực tiếp).",
          {"name": _str_prop("Tên người. Bỏ trống = người bất kỳ.")}),
    _decl("get_attendance_today", "Ai đó chấm công hôm nay chưa (số liệu trực tiếp).",
          {"name": _str_prop("Tên người. Bỏ trống = người bất kỳ.")}),
    _decl("get_recent_events", "Sự kiện gần nhất (số liệu trực tiếp)."),
    _decl("get_app_status", "Pipeline có đang chạy không (số liệu trực tiếp)."),
    _decl("get_model_status", "Model AI đã load chưa (số liệu trực tiếp)."),
    _decl("get_inference_latency", "YOLO mất bao nhiêu ms (số liệu trực tiếp)."),
    _decl("get_stt_latency", "Whisper mất bao lâu (số liệu trực tiếp)."),
    _decl("request_greet", "Nhờ camera chào ai đó đang trong phòng.",
          {"name": _str_prop("Tên người cần chào.")}, ["name"]),
    _decl("take_snapshot", "Nhờ camera chụp 1 tấm hình.",
          {"camera": _str_prop("A hoặc B. Bỏ trống = A.")}),
]

STUB_TOOLS = frozenset({
    "set_timer", "set_alarm",
})


def build_tool_catalog() -> str:
    """Danh sach tool gon nhe de nhet vao system prompt router JSON."""
    rows = []
    for item in TOOL_DECLARATIONS:
        props = (item.get("parameters") or {}).get("properties") or {}
        required = set((item.get("parameters") or {}).get("required") or [])
        args = ", ".join(k if k in required else f"{k}?"
                         for k in props) or "không tham số"
        rows.append(f"- {item['name']}({args}): {item['description']}")
    return "\n".join(rows)


def parse_router_json(text: str) -> dict:
    """Parse JSON 1 dong cua router Gemini -> {type, tool, args, text}.

    Fallback an toan: khong phai JSON -> direct voi text goc (khong crash).
    """
    import re as _re

    raw = text or ""
    match = _re.search(r"\{.*\}", raw, _re.S)
    if not match:
        return {"type": "direct", "tool": None, "args": {},
                "text": raw.strip()}
    try:
        data = json.loads(match.group(0))
    except ValueError:
        return {"type": "direct", "tool": None, "args": {},
                "text": raw.strip()}
    if not isinstance(data, dict):
        return {"type": "direct", "tool": None, "args": {},
                "text": raw.strip()}
    kind = str(data.get("type", "direct")).strip().lower()
    if kind not in ("direct", "tool"):
        kind = "direct"
    tool = data.get("tool") or None
    args = data.get("args") or {}
    if not isinstance(args, dict):
        args = {}
    reply = str(data.get("text") or "").strip()
    if kind == "tool" and not tool:
        return {"type": "direct", "tool": None, "args": {},
                "text": reply or "Tôi chưa hiểu, bạn nói lại giúp nhé."}
    return {"type": kind, "tool": tool, "args": args, "text": reply}


def _now() -> object:
    from datetime import datetime

    try:
        from zoneinfo import ZoneInfo

        return datetime.now(ZoneInfo("Asia/Ho_Chi_Minh"))
    except Exception:
        return datetime.now()


def get_current_time() -> str:
    """Gio hien tai, noi 1 cau ngan."""
    now = _now()
    hour, minute = now.hour, now.minute
    if 5 <= hour < 11:
        part = "sáng"
    elif 11 <= hour < 13:
        part = "trưa"
    elif 13 <= hour < 18:
        part = "chiều"
    elif 18 <= hour < 22:
        part = "tối"
    else:
        part = "đêm"
    hour12 = hour % 12 or 12
    if minute == 0:
        return f"Bây giờ là {hour12} giờ {part}."
    return f"Bây giờ là {hour12} giờ {minute} phút {part}."


def get_current_date() -> str:
    """Ngay hom nay, noi 1 cau ngan."""
    now = _now()
    weekdays = ["thứ Hai", "thứ Ba", "thứ Tư", "thứ Năm",
                "thứ Sáu", "thứ Bảy", "Chủ nhật"]
    return (f"Hôm nay là {weekdays[now.weekday()]}, "
            f"ngày {now.day} tháng {now.month} năm {now.year}.")


def calculate(expression: str) -> str:
    """Tinh bieu thuc so hoc don gian (+ - * / % ** va ngoac)."""
    import ast
    import operator

    allowed = {
        ast.Expression: None,
        ast.BinOp: None,
        ast.UnaryOp: None,
        ast.Constant: None,
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    def _eval(node: object) -> float:
        node_type = type(node)
        if node_type not in allowed:
            raise ValueError(f"khong ho tro: {node_type.__name__}")
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(
                    node.value, (int, float)):
                raise ValueError("chi tinh so")
            return float(node.value)
        if isinstance(node, ast.BinOp):
            return allowed[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp):
            return allowed[type(node.op)](_eval(node.operand))
        raise ValueError("bieu thuc la")

    cleaned = (expression or "").replace(",", ".").replace("x", "*").strip()
    if not cleaned or len(cleaned) > 60:
        raise ValueError("bieu thuc rong hoac qua dai")
    try:
        tree = ast.parse(cleaned, mode="eval")
        value = _eval(tree)
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError) as error:
        raise RuntimeError(f"khong tinh duoc '{expression}': {error}") from error
    if value == int(value) and abs(value) < 1e12:
        spoken = str(int(value))
    else:
        spoken = f"{value:.2f}".rstrip("0").rstrip(".")
    return f"Kết quả là {spoken}."


# Registry ten tool -> ham (CUOI file: moi def da chay xong o tren).
# halinh_assistant_local.py import TOOL_FUNCS tu day (1 nguon duy nhat).
TOOL_FUNCS = {
    "get_weather": get_weather,
    "get_current_time": get_current_time,
    "get_current_date": get_current_date,
    "calculate": calculate,
    "get_weather_forecast": get_weather_forecast,
    "get_rain_probability": get_rain_probability,
    "get_air_quality": get_air_quality,
    "get_uv_index": get_uv_index,
    "get_sunrise_sunset": get_sunrise_sunset,
    "convert_currency": convert_currency,
    "convert_unit": convert_unit,
    "date_difference": date_difference,
    "convert_timezone": convert_timezone,
    "get_system_status": get_system_status,
    "get_cpu_usage": get_cpu_usage,
    "get_ram_usage": get_ram_usage,
    "get_disk_usage": get_disk_usage,
    "get_gpu_usage": get_gpu_usage,
    "get_gpu_temperature": get_gpu_temperature,
    "get_system_uptime": get_system_uptime,
    "get_network_status": get_network_status,
    "ping_host": ping_host,
    "check_port": check_port,
    "list_files": list_files,
    "find_file": find_file,
    "get_latest_file": get_latest_file,
    "read_text_file": read_text_file,
    "read_json_file": read_json_file,
    "read_csv_file": read_csv_file,
    "get_folder_size": get_folder_size,
    "get_public_holidays": get_public_holidays,
    "get_country_population": get_country_population,
    "get_country_gdp": get_country_gdp,
    "get_country_inflation": get_country_inflation,
    "get_recent_earthquakes": get_recent_earthquakes,
    "search_book": search_book,
    "get_book_info": get_book_info,
    "generate_uuid": generate_uuid,
    "hash_text": hash_text,
    "validate_json": validate_json,
    "timestamp_to_datetime": timestamp_to_datetime,
    "datetime_to_timestamp": datetime_to_timestamp,
    "get_last_transcript": get_last_transcript,
    "get_audio_status": get_audio_status,
    "get_latest_error": get_latest_error,
    "set_timer": set_timer,
    "set_alarm": set_alarm,
    "get_camera_status": get_camera_status,
    "get_stream_status": get_stream_status,
    "get_camera_fps": get_camera_fps,
    "get_person_count": get_person_count,
    "get_current_people": get_current_people,
    "get_last_seen": get_last_seen,
    "get_attendance_today": get_attendance_today,
    "get_recent_events": get_recent_events,
    "get_app_status": get_app_status,
    "get_model_status": get_model_status,
    "get_inference_latency": get_inference_latency,
    "get_stt_latency": get_stt_latency,
    "request_greet": request_greet,
    "take_snapshot": take_snapshot,
}
