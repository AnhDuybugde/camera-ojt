"""Minimal ONVIF RTSP audio-backchannel client.

Ranger Dual Pro advertises an AAC MPEG4-GENERIC send-only track. The client
sets up only that track over RTSP/TCP and sends one AAC access unit per RTP
packet, keeping camera audio independent from the browser dashboard.
"""

from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
import os
import random
import re
import socket
import struct
import time
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener
from xml.etree import ElementTree


BACKCHANNEL_REQUIREMENT = "www.onvif.org/ver20/backchannel"
SOAP_NS = "http://www.w3.org/2003/05/soap-envelope"
DEVICE_NS = "http://www.onvif.org/ver10/device/wsdl"
MEDIA_NS = "http://www.onvif.org/ver10/media/wsdl"
WSSE_NS = (
    "http://docs.oasis-open.org/wss/2004/01/"
    "oasis-200401-wss-wssecurity-secext-1.0.xsd"
)
WSU_NS = (
    "http://docs.oasis-open.org/wss/2004/01/"
    "oasis-200401-wss-wssecurity-utility-1.0.xsd"
)


class CameraAudioError(RuntimeError):
    """Camera rejected or interrupted an audio-backchannel operation."""


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _wsse_security(username: str, password: str) -> str:
    nonce_bytes = os.urandom(16)
    created = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )
    digest = base64.b64encode(
        hashlib.sha1(nonce_bytes + created.encode() + password.encode()).digest()
    ).decode()
    nonce = base64.b64encode(nonce_bytes).decode()
    return (
        f'<wsse:Security s:mustUnderstand="1" xmlns:wsse="{WSSE_NS}" '
        f'xmlns:wsu="{WSU_NS}"><wsse:UsernameToken>'
        f"<wsse:Username>{username}</wsse:Username>"
        '<wsse:Password Type="http://docs.oasis-open.org/wss/2004/01/'
        'oasis-200401-wss-username-token-profile-1.0#PasswordDigest">'
        f"{digest}</wsse:Password>"
        '<wsse:Nonce EncodingType="http://docs.oasis-open.org/wss/2004/01/'
        'oasis-200401-wss-soap-message-security-1.0#Base64Binary">'
        f"{nonce}</wsse:Nonce><wsu:Created>{created}</wsu:Created>"
        "</wsse:UsernameToken></wsse:Security>"
    )


def _soap(
    url: str,
    action: str,
    body: str,
    username: str,
    password: str,
) -> ElementTree.Element:
    envelope = (
        f'<?xml version="1.0" encoding="UTF-8"?>'
        f'<s:Envelope xmlns:s="{SOAP_NS}"><s:Header>'
        f"{_wsse_security(username, password)}"
        f"</s:Header><s:Body>{body}</s:Body></s:Envelope>"
    ).encode()
    request = Request(
        url,
        data=envelope,
        headers={
            "Content-Type": f'application/soap+xml; charset=utf-8; action="{action}"'
        },
        method="POST",
    )
    try:
        # Camera endpoints are on the LAN. Bypass process-level HTTP proxies;
        # CI/dev shells may define one that cannot route private addresses.
        with build_opener(ProxyHandler({})).open(request, timeout=8.0) as response:
            return ElementTree.fromstring(response.read())
    except HTTPError as error:
        raise CameraAudioError(f"ONVIF HTTP {error.code}") from error


def discover_backchannel_uri(host: str, username: str, password: str) -> str:
    """Return a stream URI whose ONVIF profile includes speaker output."""
    device_url = f"http://{host}/onvif/device_service"
    capabilities = _soap(
        device_url,
        f"{DEVICE_NS}/GetCapabilities",
        f'<tds:GetCapabilities xmlns:tds="{DEVICE_NS}"><tds:Category>All</tds:Category>'
        "</tds:GetCapabilities>",
        username,
        password,
    )
    media_url = next(
        (
            node.text
            for node in capabilities.iter()
            if _local_name(node.tag) == "XAddr"
            and node.text
            and "/media" in node.text.lower()
        ),
        None,
    )
    if not media_url:
        raise CameraAudioError("Camera did not expose an ONVIF media service")
    profiles = _soap(
        media_url,
        f"{MEDIA_NS}/GetProfiles",
        f'<trt:GetProfiles xmlns:trt="{MEDIA_NS}"/>',
        username,
        password,
    )
    profile_token: str | None = None
    for profile in profiles.iter():
        if _local_name(profile.tag) != "Profiles":
            continue
        names = {_local_name(node.tag) for node in profile.iter()}
        if {"AudioOutputConfiguration", "AudioDecoderConfiguration"} <= names:
            profile_token = profile.attrib.get("token")
            break
    if not profile_token:
        raise CameraAudioError("No ONVIF profile contains audio output and decoder")
    stream = _soap(
        media_url,
        f"{MEDIA_NS}/GetStreamUri",
        f'<trt:GetStreamUri xmlns:trt="{MEDIA_NS}" '
        'xmlns:tt="http://www.onvif.org/ver10/schema">'
        "<trt:StreamSetup><tt:Stream>RTP-Unicast</tt:Stream>"
        "<tt:Transport><tt:Protocol>RTSP</tt:Protocol></tt:Transport>"
        f"</trt:StreamSetup><trt:ProfileToken>{profile_token}</trt:ProfileToken>"
        "</trt:GetStreamUri>",
        username,
        password,
    )
    uri = next(
        (node.text for node in stream.iter() if _local_name(node.tag) == "Uri"),
        None,
    )
    if not uri:
        raise CameraAudioError("ONVIF profile returned no RTSP URI")
    return uri


class _RtspConnection:
    def __init__(self, uri: str, username: str, password: str) -> None:
        parsed = urlsplit(uri)
        self.uri = uri
        self.host = parsed.hostname or ""
        self.port = parsed.port or 554
        self.username = username
        self.password = password
        self.sock: socket.socket | None = None
        self.cseq = 0
        self.challenge: str | None = None
        self.nonce_count = 0
        self.cnonce = os.urandom(8).hex()
        self.session: str | None = None

    def __enter__(self) -> _RtspConnection:
        self.sock = socket.create_connection((self.host, self.port), timeout=6.0)
        self.sock.settimeout(6.0)
        return self

    def __exit__(self, *_exc: object) -> None:
        if self.sock is not None:
            self.sock.close()
            self.sock = None

    def _authorization(self, method: str, uri: str) -> str | None:
        if not self.challenge:
            return None
        fields = {
            key.lower(): quoted or plain
            for key, quoted, plain in re.findall(
                r'(\w+)=(?:"([^"]*)"|([^,\s]+))',
                self.challenge.removeprefix("Digest "),
            )
        }
        realm, nonce = fields.get("realm", ""), fields.get("nonce", "")
        ha1 = hashlib.md5(
            f"{self.username}:{realm}:{self.password}".encode()
        ).hexdigest()
        ha2 = hashlib.md5(f"{method}:{uri}".encode()).hexdigest()
        qop = "auth" if "auth" in fields.get("qop", "").split(",") else ""
        parts = [
            f'username="{self.username}"',
            f'realm="{realm}"',
            f'nonce="{nonce}"',
            f'uri="{uri}"',
        ]
        if qop:
            self.nonce_count += 1
            nc = f"{self.nonce_count:08x}"
            response = hashlib.md5(
                f"{ha1}:{nonce}:{nc}:{self.cnonce}:{qop}:{ha2}".encode()
            ).hexdigest()
            parts.extend([f"qop={qop}", f"nc={nc}", f'cnonce="{self.cnonce}"'])
        else:
            response = hashlib.md5(f"{ha1}:{nonce}:{ha2}".encode()).hexdigest()
        parts.append(f'response="{response}"')
        if opaque := fields.get("opaque"):
            parts.append(f'opaque="{opaque}"')
        return "Digest " + ", ".join(parts)

    def request(
        self,
        method: str,
        uri: str,
        headers: dict[str, str] | None = None,
        *,
        retry_auth: bool = True,
    ) -> tuple[int, dict[str, str], bytes]:
        if self.sock is None:
            raise CameraAudioError("RTSP socket is not connected")
        self.cseq += 1
        lines = [f"{method} {uri} RTSP/1.0", f"CSeq: {self.cseq}"]
        authorization = self._authorization(method, uri)
        if authorization:
            lines.append(f"Authorization: {authorization}")
        for name, value in (headers or {}).items():
            lines.append(f"{name}: {value}")
        self.sock.sendall(("\r\n".join(lines) + "\r\n\r\n").encode())
        status, response_headers, body = self._read_response()
        if status == 401 and retry_auth:
            challenge = response_headers.get("www-authenticate", "")
            if not challenge.lower().startswith("digest "):
                raise CameraAudioError("Camera did not offer RTSP Digest authentication")
            self.challenge = challenge
            return self.request(method, uri, headers, retry_auth=False)
        return status, response_headers, body

    def _read_response(self) -> tuple[int, dict[str, str], bytes]:
        if self.sock is None:
            raise CameraAudioError("RTSP socket is not connected")
        data = bytearray()
        while b"\r\n\r\n" not in data:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise CameraAudioError("Camera closed the RTSP connection")
            data.extend(chunk)
        raw_headers, _, body = bytes(data).partition(b"\r\n\r\n")
        lines = raw_headers.decode("iso-8859-1").split("\r\n")
        status = int(lines[0].split()[1])
        headers: dict[str, str] = {}
        for line in lines[1:]:
            if ":" in line:
                name, value = line.split(":", 1)
                headers[name.strip().lower()] = value.strip()
        length = int(headers.get("content-length", "0"))
        while len(body) < length:
            body += self.sock.recv(length - len(body))
        return status, headers, body[:length] if length else body


def _backchannel_track(sdp: str) -> tuple[str, int, int]:
    current: dict[str, str] | None = None
    tracks: list[dict[str, str]] = []
    for raw_line in sdp.splitlines():
        line = raw_line.strip()
        if line.startswith("m="):
            if current:
                tracks.append(current)
            parts = line[2:].split()
            current = {
                "media": parts[0],
                "payload": parts[-1],
                "direction": "",
                "control": "",
                "codec": "",
            }
        elif current is not None and line in {"a=sendonly", "a=recvonly"}:
            current["direction"] = line[2:]
        elif current is not None and line.startswith("a=control:"):
            current["control"] = line.split(":", 1)[1]
        elif current is not None and line.startswith("a=rtpmap:"):
            current["codec"] = line.split(None, 1)[-1]
    if current:
        tracks.append(current)
    for track in tracks:
        if track["media"] == "audio" and track["direction"] == "sendonly":
            codec_parts = track["codec"].split("/")
            if not codec_parts or codec_parts[0].upper() != "MPEG4-GENERIC":
                raise CameraAudioError(f"Unsupported backchannel codec: {track['codec']}")
            return track["control"], int(track["payload"]), int(codec_parts[1])
    raise CameraAudioError("RTSP SDP did not advertise an audio send-only track")


def _control_uri(base_uri: str, content_base: str | None, control: str) -> str:
    if control.startswith("rtsp://"):
        return control
    base = content_base or base_uri
    if control.startswith("/"):
        parsed = urlsplit(base_uri)
        return f"rtsp://{parsed.hostname}:{parsed.port or 554}{control}"
    return base.rstrip("/") + "/" + control


def _interleaved_data_channel(transport: str) -> int:
    """Return the RTP channel selected by the RTSP server."""
    match = re.search(r"(?:^|;)\s*interleaved=(\d+)(?:-(\d+))?", transport, re.I)
    if not match:
        raise CameraAudioError(
            "Camera SETUP response did not select an interleaved RTP channel"
        )
    channel = int(match.group(1))
    if not 0 <= channel <= 255:
        raise CameraAudioError(f"Invalid RTSP interleaved channel: {channel}")
    return channel


def play_aac(
    uri: str,
    username: str,
    password: str,
    packets: list[bytes],
) -> dict[str, int | str]:
    """Send AAC access units to the camera speaker over RTP/RTSP/TCP."""
    with _RtspConnection(uri, username, password) as rtsp:
        status, headers, body = rtsp.request(
            "DESCRIBE",
            uri,
            {
                "Accept": "application/sdp",
                "Require": BACKCHANNEL_REQUIREMENT,
                "User-Agent": "CameraTracking/1.0",
            },
        )
        if status != 200:
            raise CameraAudioError(f"RTSP DESCRIBE failed with {status}")
        control, payload_type, sample_rate = _backchannel_track(
            body.decode("utf-8", errors="replace")
        )
        track_uri = _control_uri(uri, headers.get("content-base"), control)
        status, setup_headers, _ = rtsp.request(
            "SETUP",
            track_uri,
            {
                "Transport": "RTP/AVP/TCP;unicast;interleaved=0-1",
                "Require": BACKCHANNEL_REQUIREMENT,
            },
        )
        if status != 200:
            raise CameraAudioError(f"RTSP SETUP failed with {status}")
        rtsp.session = setup_headers.get("session", "").split(";", 1)[0]
        if not rtsp.session:
            raise CameraAudioError("Camera returned no RTSP session")
        transport = setup_headers.get("transport", "")
        data_channel = _interleaved_data_channel(transport)
        status, _, _ = rtsp.request(
            "PLAY",
            uri,
            {"Session": rtsp.session, "Require": BACKCHANNEL_REQUIREMENT},
        )
        if status != 200:
            raise CameraAudioError(f"RTSP PLAY failed with {status}")
        if rtsp.sock is None:
            raise CameraAudioError("RTSP socket closed before audio playback")
        # Embedded camera decoders often need a moment after PLAY before the
        # first backchannel access unit arrives.
        time.sleep(0.35)
        sequence = random.randrange(0, 65_536)
        timestamp = random.randrange(0, 2**32)
        ssrc = random.randrange(1, 2**32)
        started = time.monotonic()
        for index, aac in enumerate(packets):
            target = started + index * (1024 / sample_rate)
            delay = target - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            rtp_header = struct.pack(
                "!BBHII",
                0x80,
                0x80 | payload_type,
                sequence,
                timestamp,
                ssrc,
            )
            au_header = b"\x00\x10" + (len(aac) << 3).to_bytes(2, "big")
            packet = rtp_header + au_header + aac
            rtsp.sock.sendall(
                b"$" + bytes([data_channel]) + len(packet).to_bytes(2, "big") + packet
            )
            sequence = (sequence + 1) & 0xFFFF
            timestamp = (timestamp + 1024) & 0xFFFFFFFF
        time.sleep(0.35)
        rtsp.request("TEARDOWN", uri, {"Session": rtsp.session})
        return {
            "payload_type": payload_type,
            "sample_rate": sample_rate,
            "interleaved_channel": data_channel,
            "packet_count": len(packets),
        }
