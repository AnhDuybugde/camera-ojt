# Security and biometric privacy

This system processes camera images, voice and biometric identifiers. A working
demo is not automatically safe for production.

## Supported deployment boundary

- Keep MJPEG, status and control ports on `127.0.0.1`.
- Give remote users access only through authenticated TLS/VPN.
- Never commit `.env`, API keys, RTSP credentials, face images, embeddings,
  database files or runtime logs.
- Run `backend/scripts/preflight_production.py --strict` before release.

## Biometric safety

Automatic attendance is disabled by default in production. Enable it only after
the backend has a real liveness/anti-spoof gate and a deployment-specific pilot
has measured false accepts and false rejects. Always provide a visible manual
correction path; never use this project as the sole basis for payroll,
discipline, physical access or another high-impact decision.

Obtain informed consent, publish the purpose and retention period, minimize
stored crops/embeddings, restrict access, encrypt backups and honor deletion
requests. Restroom/WC zone events must not speak or persist a person's name.

## Secret incident response

If a key is pasted into chat, a screenshot, a log or Git history, treat it as
compromised: revoke it at the provider, issue a new key, update the local `.env`
and restart the affected process. Removing the text later is not sufficient.

## Reporting

Report a vulnerability privately to the repository owner. Include affected
commit, reproduction steps, impact and the smallest safe evidence set; do not
include real employee images, credentials or biometric records.
