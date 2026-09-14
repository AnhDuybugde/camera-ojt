import json
from urllib.request import Request, urlopen

from camera_tracking.streaming.mjpeg import MjpegStreamer


def test_enrollment_management_endpoints() -> None:
    deleted: list[str] = []
    streamer = MjpegStreamer(host="127.0.0.1", port=0)
    streamer.set_enrollment_actions(
        lambda _payload: {"ok": True},
        lambda: [{
            "person_id": "NV001",
            "display_name": "Nguyen Van A",
            "has_image": True,
        }],
        lambda person_id: deleted.append(person_id) or {"ok": True},
    )
    assert streamer.start()
    assert streamer._server is not None
    port = streamer._server.server_address[1]

    try:
        with urlopen(f"http://127.0.0.1:{port}/enrollment/persons") as response:
            payload = json.loads(response.read())
        assert payload["people"][0]["person_id"] == "NV001"

        request = Request(
            f"http://127.0.0.1:{port}/enrollment/persons/NV001",
            method="DELETE",
        )
        with urlopen(request) as response:
            result = json.loads(response.read())
        assert result["ok"] is True
        assert deleted == ["NV001"]
    finally:
        streamer.stop()
