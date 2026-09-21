"""MediaPipe Hands sidecar used when the main app runs on Python 3.13.

The protocol is newline-delimited JSON over stdin/stdout. Keeping MediaPipe in
one persistent Python 3.12 process avoids both an unsupported dependency in the
main runtime and the latency of starting a process for every video frame.
"""

from __future__ import annotations

import base64
import json
import sys

import cv2
import mediapipe as mp
import numpy as np


def _reply(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def main() -> int:
    hands = mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.50,
        min_tracking_confidence=0.50,
    )
    _reply({"ready": True})
    try:
        for line in sys.stdin:
            request_id = None
            try:
                request = json.loads(line)
                request_id = request.get("id")
                encoded = base64.b64decode(request["image"], validate=True)
                image = cv2.imdecode(
                    np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR
                )
                if image is None:
                    raise ValueError("invalid JPEG image")
                result = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                detected = []
                for hand in result.multi_hand_landmarks or []:
                    detected.append([[point.x, point.y] for point in hand.landmark])
                _reply({"id": request_id, "hands": detected})
            except Exception as error:
                _reply({"id": request_id, "hands": [], "error": str(error)})
    finally:
        hands.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
