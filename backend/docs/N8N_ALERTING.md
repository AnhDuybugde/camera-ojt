# Alerting n8n

Pipeline chi gui webhook sau khi backend da xac nhan `CHECK_IN` hoac
`LEAVE_OFFICE`. Detect, nhan dien mat va xac dinh event van nam hoan toan trong
pipeline camera. Alert la mot worker nen nho; neu n8n loi thi ghi log va bo qua
event, khong lam cham detect.

## Cau hinh backend

Copy cac bien `ALERTING_*` va `N8N_ALERT_*` trong `backend/.env.example` vao
`backend/.env`, sau do restart `run_workstate.py` hoac
`run_workstate_local.py`.

```env
ALERTING_ENABLED=true
N8N_ALERT_WEBHOOK_URL=https://your-instance.app.n8n.cloud/webhook/camera-alert
N8N_ALERT_WEBHOOK_TOKEN=<mot-chuoi-random-dai>
ALERTING_EVENTS=CHECK_IN,LEAVE_OFFICE
ALERTING_TIMEOUT_SECONDS=3
```

Backend van doc `N8N_ALERT_WEBHOOK_SECRET` de tuong thich cau hinh cu, nhung
nen dung bien `N8N_ALERT_WEBHOOK_TOKEN` moi. Khong commit token.

## Request contract

Backend `POST` JSON voi header `X-Camera-Token`. Payload luon co:

```json
{
  "event_type": "CHECK_IN",
  "person_name": "Van Dai",
  "occurred_at": "2026-09-21T08:00:00+07:00",
  "channel": "A"
}
```

## Workflow n8n

1. **Webhook**: POST `camera-alert`, doc header `X-Camera-Token`.
2. **IF**: chi chap nhan token dung va event `CHECK_IN`/`LEAVE_OFFICE`.
3. **Switch**: tao noi dung check-in hoac roi van phong.
4. Goi node Zalo/chatbot co san de gui vao mot nhom Zalo co dinh.
5. **Respond to Webhook**: HTTP 200 JSON `{"ok": true}`.

Khong can HMAC, raw body, Data Table, dedupe database hay mapping
`person_id -> zalo_user_id`.
