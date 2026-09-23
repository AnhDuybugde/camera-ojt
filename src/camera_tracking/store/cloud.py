"""Supabase transport with bounded requests; credentials never enter payloads."""
import json
import os
from urllib.request import Request, urlopen


class CloudEventWriter:
    def __init__(self, url, key):
        self.url, self.key = url.rstrip("/"), key
        if not self.url.startswith("https://"):
            raise ValueError("Cloud synchronization requires HTTPS")

    @classmethod
    def from_env(cls):
        return cls(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])

    def __call__(self, event):
        request = Request(self.url + "/rest/v1/camera_events?on_conflict=event_id",
            data=json.dumps(event).encode(), headers={
                "apikey": self.key, "Authorization": "Bearer " + self.key,
                "Content-Type": "application/json",
                "Prefer": "resolution=ignore-duplicates,return=minimal"})
        with urlopen(request, timeout=10) as response:
            if response.status not in (200, 201, 204):
                raise RuntimeError("Cloud did not acknowledge event")

    def request(self, path, payload=None):
        request = Request(self.url + "/rest/v1/" + path,
            data=None if payload is None else json.dumps(payload).encode(),
            headers={"apikey": self.key, "Authorization": "Bearer " + self.key,
                     "Content-Type": "application/json"})
        with urlopen(request, timeout=10) as response:
            return json.load(response)

    def compare_and_swap(self, row):
        return self.request("rpc/camera_record_cas", {
            "p_kind": row["kind"], "p_key": row["record_key"],
            "p_payload": json.loads(row["payload"]), "p_deleted": bool(row["deleted"]),
            "p_expected": row["base_revision"],
        })

    def changes(self, cursor):
        return self.request(f"camera_records?change_seq=gt.{int(cursor)}&order=change_seq.asc&limit=100")

    def compare_batch(self, rows):
        return self.request("rpc/camera_records_batch", {"items": [
            {"kind": row["kind"], "key": row["record_key"], "payload": json.loads(row["payload"]),
             "deleted": bool(row["deleted"]), "expected": row["base_revision"]} for row in rows]})
