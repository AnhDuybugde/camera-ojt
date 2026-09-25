"""Small JSON server with bounded requests and explicit application methods."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .codec import decode, encode


def make_server(api, host="127.0.0.1", port=8767):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            self.connection.settimeout(15)
            try:
                parts = self.path.split("/")
                if len(parts) != 5 or parts[1:3] != ["api", "v1"]:
                    self.send_error(404)
                    return
                size = int(self.headers.get("Content-Length", 0))
                if not 0 < size <= 8 * 1024 * 1024:
                    self.send_error(413)
                    return
                body = decode(json.loads(self.rfile.read(size)))
                result = api.dispatch(parts[3], parts[4], body.get("args", []),
                    body.get("kwargs", {}), self.headers.get("Authorization", "").removeprefix("Bearer "))
                status, payload = 200, {"result": encode(result)}
            except PermissionError as error:
                status, payload = 403, {"error": str(error)}
            except (ValueError, TypeError, KeyError):
                status, payload = 400, {"error": "Invalid request"}
            except Exception:
                status, payload = 500, {"error": "Backend operation failed"}
            data = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return ThreadingHTTPServer((host, port), Handler)
