import { useEffect, useState } from 'react'

const CONTROL_URL = (import.meta.env.VITE_CONTROL_URL as string | undefined)?.replace(/\/$/, '')
  ?? 'http://localhost:8766'

type PipeStatus = {
  running: boolean
  pid: number | null
  uptime_s: number
  stream_port: number | null
  args: string[]
  log_tail: string[]
}

async function api(path: string, body?: unknown): Promise<unknown> {
  const res = await fetch(`${CONTROL_URL}${path}`, {
    method: body === undefined ? 'GET' : 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  })
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json()
}

export default function PipelineControl() {
  const [status, setStatus] = useState<PipeStatus | null>(null)
  const [online, setOnline] = useState(false)
  const [busy, setBusy] = useState(false)
  const [imgsz, setImgsz] = useState('800')
  const [noFace, setNoFace] = useState(false)
  const [script, setScript] = useState('scripts/run_workstate.py')
  const [msg, setMsg] = useState('')

  async function refresh() {
    try {
      const s = (await api('/api/status')) as PipeStatus
      setStatus(s)
      setOnline(true)
    } catch {
      setOnline(false)
    }
  }

  useEffect(() => {
    refresh()
    const timer = setInterval(refresh, 3000)
    return () => clearInterval(timer)
  }, [])

  async function start() {
    setBusy(true)
    setMsg('')
    try {
      const r = (await api('/api/start', { script, imgsz: Number(imgsz) || 800, no_face: noFace })) as {
        started: boolean
        reason?: string
        pid?: number
      }
      setMsg(r.started ? `Pipeline started (pid ${r.pid}). Live feed in ~10s.` : `Not started: ${r.reason ?? 'unknown'}`)
    } catch (e) {
      setMsg(`Start failed: ${(e as Error).message}`)
    } finally {
      setBusy(false)
      refresh()
    }
  }

  async function stop() {
    if (!window.confirm('Stop the camera pipeline? Live feed will go offline.')) return
    setBusy(true)
    try {
      await api('/api/stop', {})
      setMsg('Pipeline stopped.')
    } catch (e) {
      setMsg(`Stop failed: ${(e as Error).message}`)
    } finally {
      setBusy(false)
      refresh()
    }
  }

  return (
    <section className="card">
      <h2>
        Pipeline{' '}
        {online && status && (
          status.running
            ? <span className="badge badge-working">RUNNING{status.pid ? ` · pid ${status.pid}` : ''}</span>
            : <span className="badge badge-unknown">STOPPED</span>
        )}
      </h2>
      {!online ? (
        <p className="footnote">
          Supervisor offline — run <code>python scripts\pipeline_supervisor.py</code> on the camera PC
          (or double-click <code>start-all.bat</code>), then cameras can be started from here.
        </p>
      ) : (
        <>
          <div className="control-row">
            <label>
              Pipeline:{' '}
              <select value={script} onChange={(e) => setScript(e.target.value)} disabled={status?.running || busy}>
                <option value="scripts/run_workstate.py">run_workstate.py — 2× Imou</option>
                <option value="scripts/run_workstate_local.py">run_workstate_local.py — webcam + Hà Linh</option>
              </select>
            </label>
            <label>
              Model size:{' '}
              <select value={imgsz} onChange={(e) => setImgsz(e.target.value)} disabled={status?.running || busy}>
                <option value="640">640 — faster</option>
                <option value="800">800 — default</option>
                <option value="960">960 — sharper</option>
              </select>
            </label>
            <label className="scope-check">
              <input type="checkbox" checked={noFace} onChange={(e) => setNoFace(e.target.checked)}
                disabled={status?.running || busy} /> tracking only (no face)
            </label>
            {status?.running ? (
              <button className="danger-btn" disabled={busy} onClick={stop}>Stop cameras</button>
            ) : (
              <button disabled={busy} onClick={start}>{busy ? 'Starting...' : 'Start cameras'}</button>
            )}
            <button className="ghost" onClick={refresh}>Refresh</button>
          </div>
          {status?.running && status.uptime_s > 0 && (
            <p className="footnote">Uptime: {Math.round(status.uptime_s)}s · args: <code>{status.args.join(' ')}</code></p>
          )}
          {msg && <p className="footnote">{msg}</p>}
          {status && status.log_tail.length > 0 && (
            <pre className="log-box">{status.log_tail.slice(-20).join('\n')}</pre>
          )}
        </>
      )}
    </section>
  )
}
