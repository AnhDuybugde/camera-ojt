"""Single isolated UI/backend smoke check; uses a temporary DB, no camera/cloud."""
import os
from pathlib import Path
import tempfile
import threading

from backend.app.api.server import make_server
from backend.app.api.service import ApplicationAPI
from camera_tracking.application.backend import load_services


def main():
    with tempfile.TemporaryDirectory(prefix="camera-ui-smoke-") as temporary:
        os.environ["CAMERA_DATABASE_PATH"] = str(Path(temporary) / "test.db")
        os.environ["CAMERA_ADMIN_PASSWORD"] = "isolated-smoke-password"
        db, auth, attendance, admin, sync = load_services()
        from backend.app.api.operations import Operations
        api = ApplicationAPI(db, auth, admin, sync,
                             Operations(db, attendance, db.path), None)
        server = make_server(api, port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        os.environ["CAMERA_API_URL"] = f"http://127.0.0.1:{server.server_port}"
        try:
            from streamlit.testing.v1 import AppTest
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "apps/attendance/app.py"))
            session = api.login("ADMIN", "isolated-smoke-password")
            app.session_state["authenticated"] = True
            app.session_state["role"] = "ADMIN"
            app.session_state["api_token"] = session["token"]
            app.run(timeout=20)
            if app.exception:
                print("UI/backend smoke failed:", app.exception[0].message)
                return 1
            print("UI/backend smoke: dashboard rendered through authenticated API")
            return 0
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    raise SystemExit(main())
