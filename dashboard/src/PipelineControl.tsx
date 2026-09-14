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
      const r = (await api('/api/start', { imgsz: Number(imgsz) || 800, no_face: noFace })) as {
        started: boolean
        reason?: string
        pid?: number
      }
      setMsg(r.started ? `Pipeline đã khởi động (PID ${r.pid}). Camera sẽ hiển thị sau khoảng 10 giây.` : `Không thể khởi động: ${r.reason ?? 'không rõ nguyên nhân'}`)
    } catch (e) {
      setMsg(`Khởi động thất bại: ${(e as Error).message}`)
    } finally {
      setBusy(false)
      refresh()
    }
  }

  async function stop() {
    if (!window.confirm('Dừng pipeline camera? Hình ảnh trực tiếp sẽ bị ngắt.')) return
    setBusy(true)
    try {
      await api('/api/stop', {})
      setMsg('Pipeline đã dừng.')
    } catch (e) {
      setMsg(`Dừng pipeline thất bại: ${(e as Error).message}`)
    } finally {
      setBusy(false)
      refresh()
    }
  }

  return (
    <section className="card">
      <h2>
        Pipeline camera{' '}
        {online && status && (
          status.running
            ? <span className="badge badge-working">ĐANG CHẠY{status.pid ? ` · PID ${status.pid}` : ''}</span>
            : <span className="badge badge-unknown">ĐÃ DỪNG</span>
        )}
      </h2>
      {!online ? (
        <p className="footnote">
          Supervisor đang ngoại tuyến. Chạy <code>python scripts\pipeline_supervisor.py</code> trên máy camera
          (hoặc mở <code>start-all.bat</code>) để điều khiển camera tại đây.
        </p>
      ) : (
        <>
          <div className="control-row">
            <label>
              Kích thước xử lý:{' '}
              <select value={imgsz} onChange={(e) => setImgsz(e.target.value)} disabled={status?.running || busy}>
                <option value="640">640 — nhanh hơn</option>
                <option value="800">800 — mặc định</option>
                <option value="960">960 — rõ hơn</option>
              </select>
            </label>
            <label className="scope-check">
              <input type="checkbox" checked={noFace} onChange={(e) => setNoFace(e.target.checked)}
                disabled={status?.running || busy} /> chỉ theo dõi, không nhận diện khuôn mặt
            </label>
            {status?.running ? (
              <button className="danger-btn" disabled={busy} onClick={stop}>Dừng camera</button>
            ) : (
              <button disabled={busy} onClick={start}>{busy ? 'Đang khởi động...' : 'Khởi động camera'}</button>
            )}
            <button className="ghost" onClick={refresh}>Làm mới</button>
          </div>
          {status?.running && status.uptime_s > 0 && (
            <p className="footnote">Thời gian hoạt động: {Math.round(status.uptime_s)} giây · tham số: <code>{status.args.join(' ')}</code></p>
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
