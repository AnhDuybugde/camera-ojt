import { useEffect, useState } from 'react'
import KioskCheckIn, { type KioskEvent } from './KioskCheckIn'
import PersonRegistration from './PersonRegistration'
import PipelineControl from './PipelineControl'
import { supabase, supabaseConfigured } from './lib/supabase'
import {
  deleteAllForDate,
  deleteAttendance,
  deleteEvents,
  deleteRoomStatus,
  type DeleteScope,
} from './lib/history'

type AttendanceRow = {
  date: string
  person_id: string
  person_name: string
  global_id: number | null
  attended: boolean
  check_in_at: string | null
  face_score: number | null
  needs_review: boolean
}

type RoomRow = {
  date: string
  global_id: number
  person_id: string | null
  person_name: string | null
  in_room: boolean
  label: string
  last_leave_at: string | null
  last_enter_at: string | null
}

type RoomEventRow = {
  id: number
  date: string
  global_id: number
  person_id: string | null
  event: string
  channel: string
  at: string
}

type LivePerson = {
  gid: number
  person_id?: string | null
  name: string | null
  label: string
  in_room: boolean
  face_score?: number | null
  camera?: string | null
  tracking_state?: string | null
}

type PendingAttendance = {
  date: string
  person_id: string
  person_name: string
  global_id: number
  check_in_at: string
  face_score: number
}

const STREAM_URL = (import.meta.env.VITE_STREAM_URL as string | undefined)?.replace(/\/$/, '')
  ?? 'http://localhost:8765'
const KIOSK_CHANNEL = (
  (import.meta.env.VITE_KIOSK_CHANNEL as string | undefined)?.toUpperCase() === 'B'
    ? 'B'
    : 'A'
) as 'A' | 'B'

// Old rows in the DB may still carry Vietnamese labels; normalize to EN.
const LABEL_MAP: Record<string, string> = {
  'Đang làm việc': 'Working',
  'Rời khỏi chỗ': 'Away',
  'Rời khỏi văn phòng': 'Out of office',
  'Đang quay lại': 'Returning',
  'Chưa xác định': 'Unknown',
}

function displayLabel(label: string): string {
  return LABEL_MAP[label] ?? label
}

function labelClass(label: string): string {
  const en = displayLabel(label).toLowerCase()
  if (en === 'working') return 'badge badge-working'
  if (en === 'away') return 'badge badge-away'
  if (en.includes('near')) return 'badge badge-near'
  if (en.includes('out')) return 'badge badge-out'
  if (en.includes('return')) return 'badge badge-returning'
  return 'badge badge-unknown'
}

const EVENT_VI: Record<string, string> = {
  CHECK_IN: 'Điểm danh',
  LEAVE_OFFICE: 'Rời văn phòng',
  RETURN: 'Quay lại',
}

function eventClass(event: string): string {
  if (event === 'CHECK_IN') return 'badge badge-working'
  if (event === 'LEAVE_OFFICE') return 'badge badge-out'
  if (event === 'RETURN') return 'badge badge-returning'
  return 'badge badge-unknown'
}

function todayISO(): string {
  return new Date().toISOString().slice(0, 10)
}

export default function App() {
  const [screen, setScreen] = useState<'kiosk' | 'admin' | 'registration'>('kiosk')
  const [day, setDay] = useState(todayISO())
  const [attendance, setAttendance] = useState<AttendanceRow[]>([])
  const [room, setRoom] = useState<RoomRow[]>([])
  const [events, setEvents] = useState<RoomEventRow[]>([])
  const [selAttend, setSelAttend] = useState<string[]>([])
  const [selRoom, setSelRoom] = useState<number[]>([])
  const [selEvents, setSelEvents] = useState<number[]>([])
  const [scopes, setScopes] = useState<DeleteScope[]>(['attendance', 'room', 'events', 'faces'])
  const [pendingDelete, setPendingDelete] = useState<'selected' | 'all' | null>(null)
  const [deleting, setDeleting] = useState(false)
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [user, setUser] = useState<string | null>(null)
  const [msg, setMsg] = useState('')
  const [isError, setIsError] = useState(false)
  const [livePeople, setLivePeople] = useState<LivePerson[]>([])
  const [pendingAttendance, setPendingAttendance] = useState<PendingAttendance[]>([])
  const [liveEvents, setLiveEvents] = useState<KioskEvent[]>([])
  const [sendingAttendance, setSendingAttendance] = useState(false)
  const [liveOk, setLiveOk] = useState(false)

  useEffect(() => {
    if (!supabaseConfigured) return
    supabase.auth.getSession().then(({ data }) => {
      setUser(data.session?.user?.email ?? null)
    })
    const { data: sub } = supabase.auth.onAuthStateChange((_e, s) => {
      setUser(s?.user?.email ?? null)
    })
    return () => sub.subscription.unsubscribe()
  }, [])

  function note(text: string, error = false) {
    setMsg(text)
    setIsError(error)
  }

  async function load(showLoading = true) {
    if (!supabaseConfigured) {
      if (showLoading) note('Supabase is not configured.')
      return
    }
    if (showLoading) note('Loading...')
    const [a, r, e] = await Promise.all([
      supabase.from('attendance_daily').select('*').eq('date', day).order('check_in_at'),
      // Hide rows merged into a canonical Global ID by face reconcile.
      supabase.from('room_status_daily').select('*').eq('date', day).is('merged_into', null).order('global_id'),
      supabase.from('room_events').select('*').eq('date', day).order('at', { ascending: false }).limit(100),
    ])
    if (a.error) note(`Attendance error: ${a.error.message}`, true)
    else if (r.error) note(`Room status error: ${r.error.message}`, true)
    else if (e.error) note(`Events error: ${e.error.message}`, true)
    else note('')
    setAttendance((a.data ?? []) as AttendanceRow[])
    setRoom((r.data ?? []) as RoomRow[])
    setEvents((e.data ?? []) as RoomEventRow[])
    setSelAttend([])
    setSelRoom([])
    setSelEvents([])
    setPendingDelete(null)
  }

  useEffect(() => {
    load()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [day])

  useEffect(() => {
    const timer = setInterval(() => {
      void load(false)
    }, 5000)
    return () => clearInterval(timer)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [day])

  useEffect(() => {
    if (!STREAM_URL) return
    let alive = true
    async function poll() {
      try {
        const res = await fetch(`${STREAM_URL}/status.json`)
        if (!res.ok) throw new Error('offline')
        const data = await res.json()
        if (!alive) return
        setLivePeople((data.people ?? []) as LivePerson[])
        setPendingAttendance((data.pending_attendance ?? []) as PendingAttendance[])
        if (Array.isArray(data.recent_events)) {
          setLiveEvents(data.recent_events as KioskEvent[])
        }
        if (Array.isArray(data.attendance_today)) {
          setAttendance(data.attendance_today as AttendanceRow[])
        }
        setLiveOk(true)
      } catch {
        if (alive) setLiveOk(false)
      }
    }
    poll()
    const timer = setInterval(poll, 2000)
    return () => { alive = false; clearInterval(timer) }
  }, [])

  async function sendPendingAttendance() {
    if (!STREAM_URL || pendingAttendance.length === 0) return
    setSendingAttendance(true)
    try {
      const res = await fetch(STREAM_URL + '/attendance/send')
      const data = await res.json()
      note('Sent ' + (data.sent ?? 0) + ' attendance record(s).' +
        (data.failed ? ' Failed: ' + data.failed + '.' : ''))
      await load(false)
    } catch {
      note('Could not send attendance to Supabase.', true)
    } finally {
      setSendingAttendance(false)
    }
  }

  async function login(e: React.FormEvent) {
    e.preventDefault()
    const { error } = await supabase.auth.signInWithPassword({ email, password })
    note(error ? `Login failed: ${error.message}` : 'Signed in (only admins can edit).', !!error)
  }

  async function toggleAttend(row: AttendanceRow) {
    const { error } = await supabase
      .from('attendance_daily')
      .update({ attended: !row.attended })
      .eq('date', row.date)
      .eq('person_id', row.person_id)
    note(error ? `Update failed (admin role required): ${error.message}` : 'Attendance updated.', !!error)
    load()
  }

  async function toggleInRoom(row: RoomRow) {
    const { error } = await supabase
      .from('room_status_daily')
      .update({ in_room: !row.in_room })
      .eq('date', row.date)
      .eq('global_id', row.global_id)
    note(error ? `Update failed (admin role required): ${error.message}` : 'In-room updated.', !!error)
    load()
  }

  function toggleSel<T>(list: T[], setList: (v: T[]) => void, key: T) {
    setList(list.includes(key) ? list.filter((k) => k !== key) : [...list, key])
  }

  function toggleScope(scope: DeleteScope) {
    setScopes(scopes.includes(scope) ? scopes.filter((s) => s !== scope) : [...scopes, scope])
  }

  const selectedCount = selAttend.length + selRoom.length + selEvents.length

  async function confirmDelete() {
    if (!pendingDelete) return
    const summary = pendingDelete === 'all'
      ? `ALL history for ${day} (scopes: ${scopes.join(', ') || 'none'})`
      : `${selectedCount} selected record(s) for ${day}`
    if (!window.confirm(`Permanently delete ${summary}?\nThis cannot be undone.`)) return
    setDeleting(true)
    try {
      if (pendingDelete === 'all') {
        const counts = await deleteAllForDate(day, scopes)
        const total = Object.values(counts).reduce((s, n) => s + n, 0)
        note(`Deleted ${total} record(s) for ${day}.`)
      } else {
        if (selAttend.length > 0) await deleteAttendance(day, selAttend)
        if (selRoom.length > 0) await deleteRoomStatus(day, selRoom)
        if (selEvents.length > 0) await deleteEvents(selEvents)
        note(`Deleted ${selectedCount} record(s).`)
      }
    } catch (err) {
      note(`Delete failed (admin role required): ${(err as Error).message}`, true)
    } finally {
      setDeleting(false)
      load()
    }
  }

  if (screen === 'registration') {
    return <PersonRegistration streamUrl={STREAM_URL} onBack={() => setScreen('admin')} />
  }

  if (screen === 'kiosk') {
    return (
      <KioskCheckIn
        streamUrl={STREAM_URL}
        cameraChannel={KIOSK_CHANNEL}
        liveOk={liveOk}
        people={livePeople}
        attendance={attendance}
        pendingAttendance={pendingAttendance}
        events={liveEvents}
        onOpenAdmin={() => setScreen('admin')}
      />
    )
  }

  return (
    <div className="app">
      <header className="app-header">
        <div>
          <h1>Attendance &amp; Room Status</h1>
          <p>Face check-in once per person per day, live room presence per Global ID.</p>
        </div>
        <div className="app-header-actions">
          {(user || !supabaseConfigured) && (
            <button type="button" className="header-action" onClick={() => setScreen('registration')}>
              Đăng ký nhân viên
            </button>
          )}
          <button type="button" className="header-action" onClick={() => setScreen('kiosk')}>
            Màn hình điểm danh
          </button>
        </div>
      </header>

      <div className="toolbar">
        <label>
          Date:{' '}
          <input type="date" value={day} onChange={(e) => setDay(e.target.value)} />
        </label>
        <button onClick={() => load()}>Reload</button>
        <span className="spacer">
          {user ? (
            <span className="user-chip">
              {user} <button className="ghost" onClick={() => supabase.auth.signOut()}>Sign out</button>
            </span>
          ) : (
            <form onSubmit={login} className="login-form">
              <input type="email" placeholder="admin email" value={email} onChange={(e) => setEmail(e.target.value)} />
              <input type="password" placeholder="password" value={password} onChange={(e) => setPassword(e.target.value)} />
              <button type="submit">Login</button>
            </form>
          )}
        </span>
      </div>
      <p className={`status-msg${isError ? ' error' : ''}`}>{msg}</p>

      <section className="card">
        <h2>Live cameras {liveOk && <span className="badge badge-working">LIVE</span>}</h2>
        {!STREAM_URL || !liveOk ? (
          <p className="footnote">
            Stream offline — start the pipeline on the camera PC
            (<code>python scripts\run_workstate.py</code>, default port 8765)
            and set <code>VITE_STREAM_URL</code> if the dashboard runs elsewhere.
          </p>
        ) : (
          <>
            <div className="live-grid">
              <figure>
                <img src={`${STREAM_URL}/cam_a.mjpg`} alt="Channel A live" />
                <figcaption>Channel A — room (YOLO + Global ID overlay)</figcaption>
              </figure>
              <figure>
                <img src={`${STREAM_URL}/cam_b.mjpg`} alt="Channel B live" />
                <figcaption>Channel B — door + face check-in</figcaption>
              </figure>
            </div>
            <div className="live-chips">
              {livePeople.length === 0 && <span className="footnote">No one tracked right now.</span>}
              {livePeople.map((p) => (
                <span key={p.gid} className="live-chip">
                  G{p.gid}{p.name ? ` · ${p.name}` : ''}{' '}
                  <span className={labelClass(p.label)}>{displayLabel(p.label)}</span>
                </span>
              ))}
            </div>
          </>
        )}
      </section>

      <section className="card">
        <h2>Sự kiện trực tiếp {liveOk && <span className="badge badge-working">LIVE</span>}</h2>
        {!liveOk ? (
          <p className="footnote">Pipeline offline — sự kiện mới sẽ hiện tại đây khi camera chạy lại.</p>
        ) : liveEvents.length === 0 ? (
          <p className="footnote">Chưa có sự kiện nào trong phiên chạy này.</p>
        ) : (
          <ul className="live-events">
            {liveEvents.slice().reverse().map((e, i) => (
              <li key={`${e.at ?? ''}-${e.global_id ?? ''}-${e.event}-${i}`}>
                <span className={eventClass(e.event)}>{EVENT_VI[e.event] ?? e.event}</span>
                <strong>{e.person_name ?? `G${e.global_id ?? '?'}`}</strong>
                <span className="footnote">Cam {e.channel ?? '?'} · {e.at ?? '—'}</span>
              </li>
            ))}
          </ul>
        )}
      </section>

      <section className="card">
        <h2>Attendance ({attendance.length})</h2>
        {pendingAttendance.length > 0 && (
          <div className="pending-attendance">
            <p>
              Pending local attendance: {pendingAttendance.length}
              {' '}
              <button onClick={sendPendingAttendance} disabled={sendingAttendance}>
                {sendingAttendance ? 'Sending...' : 'Send to Supabase'}
              </button>
            </p>
            <ul>
              {pendingAttendance.map((r) => (
                <li key={r.date + '-' + r.person_id}>
                  G{r.global_id} — {r.person_name} — score {r.face_score.toFixed(2)}
                </li>
              ))}
            </ul>
          </div>
        )}
        <table>
          <thead>
            <tr>
              <th><input type="checkbox" aria-label="Select all attendance"
                checked={attendance.length > 0 && selAttend.length === attendance.length}
                onChange={(e) => setSelAttend(e.target.checked ? attendance.map((r) => r.person_id) : [])} /></th>
              <th>Person</th><th>GID</th><th>Check-in</th><th>Score</th><th>Present</th><th>Admin</th>
            </tr>
          </thead>
          <tbody>
            {attendance.length === 0 && (
              <tr className="empty-row"><td colSpan={7}>No records for this date.</td></tr>
            )}
            {attendance.map((r) => (
              <tr key={r.person_id}>
                <td><input type="checkbox" aria-label={`Select ${r.person_id}`}
                  checked={selAttend.includes(r.person_id)}
                  onChange={() => toggleSel(selAttend, setSelAttend, r.person_id)} /></td>
                <td>{r.person_name}</td>
                <td>{r.global_id ?? '—'}</td>
                <td>{r.check_in_at ?? '—'}</td>
                <td>{r.face_score?.toFixed(2) ?? '—'}</td>
                <td className={r.attended ? 'pill-yes' : 'pill-no'}>{r.attended ? 'Yes' : 'No'}</td>
                <td><button onClick={() => toggleAttend(r)}>Toggle</button></td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      <section className="card">
        <h2>Room status ({room.length})</h2>
        <table>
          <thead>
            <tr>
              <th><input type="checkbox" aria-label="Select all room rows"
                checked={room.length > 0 && selRoom.length === room.length}
                onChange={(e) => setSelRoom(e.target.checked ? room.map((r) => r.global_id) : [])} /></th>
              <th>GID</th><th>Name</th><th>Status</th><th>In room</th><th>Left at</th><th>Entered at</th><th>Admin</th>
            </tr>
          </thead>
          <tbody>
            {room.length === 0 && (
              <tr className="empty-row"><td colSpan={8}>No records for this date.</td></tr>
            )}
            {room.map((r) => (
              <tr key={r.global_id}>
                <td><input type="checkbox" aria-label={`Select G${r.global_id}`}
                  checked={selRoom.includes(r.global_id)}
                  onChange={() => toggleSel(selRoom, setSelRoom, r.global_id)} /></td>
                <td>G{r.global_id}</td>
                <td>{r.person_name ?? '—'}</td>
                <td><span className={labelClass(r.label)}>{displayLabel(r.label)}</span></td>
                <td className={r.in_room ? 'pill-yes' : 'pill-no'}>{r.in_room ? 'Yes' : 'No'}</td>
                <td>{r.last_leave_at ?? '—'}</td>
                <td>{r.last_enter_at ?? '—'}</td>
                <td><button onClick={() => toggleInRoom(r)}>Toggle</button></td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="footnote">
          Pipeline ticks attendance once per person per day and flips in-room at most once per hour.
          Manual admin edits go through RLS (role = admin in the roles table).
        </p>
      </section>

      <section className="card">
        <h2>Room events ({events.length}, latest 100)</h2>
        <table>
          <thead>
            <tr>
              <th><input type="checkbox" aria-label="Select all events"
                checked={events.length > 0 && selEvents.length === events.length}
                onChange={(e) => setSelEvents(e.target.checked ? events.map((r) => r.id) : [])} /></th>
              <th>At</th><th>GID</th><th>Person</th><th>Event</th><th>Ch</th>
            </tr>
          </thead>
          <tbody>
            {events.length === 0 && (
              <tr className="empty-row"><td colSpan={6}>No events for this date.</td></tr>
            )}
            {events.map((r) => (
              <tr key={r.id}>
                <td><input type="checkbox" aria-label={`Select event ${r.id}`}
                  checked={selEvents.includes(r.id)}
                  onChange={() => toggleSel(selEvents, setSelEvents, r.id)} /></td>
                <td>{r.at}</td>
                <td>G{r.global_id}</td>
                <td>{r.person_id ?? '—'}</td>
                <td>{r.event}</td>
                <td>{r.channel}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      {user && <PipelineControl />}

      {user && (
        <section className="card danger">
          <h2>History management (admin, demo reset)</h2>
          <p className="footnote">
            Selected: {selAttend.length} attendance · {selRoom.length} room · {selEvents.length} events.
            Delete-all scopes for {day}:{' '}
            {(['attendance', 'room', 'events', 'faces'] as DeleteScope[]).map((s) => (
              <label key={s} className="scope-check">
                <input type="checkbox" checked={scopes.includes(s)} onChange={() => toggleScope(s)} /> {s}
              </label>
            ))}
          </p>
          <div className="danger-row">
            <button className="danger-btn" disabled={selectedCount === 0 || deleting}
              onClick={() => setPendingDelete('selected')}>
              Delete selected ({selectedCount})
            </button>
            <button className="danger-btn" disabled={scopes.length === 0 || deleting}
              onClick={() => setPendingDelete('all')}>
              Delete all for {day}
            </button>
          </div>
          {pendingDelete && (
            <div className="warn-box">
              <strong>Warning: this permanently deletes{' '}
                {pendingDelete === 'all'
                  ? `ALL ${scopes.join(', ')} history for ${day}`
                  : `${selectedCount} selected record(s)`}.</strong>
              <span> Face-crop files in Storage are removed too. This cannot be undone.</span>
              <div className="danger-row">
                <button className="danger-btn" disabled={deleting} onClick={confirmDelete}>
                  {deleting ? 'Deleting...' : 'Yes, delete permanently'}
                </button>
                <button className="ghost" disabled={deleting} onClick={() => setPendingDelete(null)}>
                  Cancel
                </button>
              </div>
            </div>
          )}
        </section>
      )}
    </div>
  )
}
