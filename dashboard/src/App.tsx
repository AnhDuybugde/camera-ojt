import { useEffect, useState } from 'react'
import Brand from './Brand'
import EmployeeManagement from './EmployeeManagement'
import KioskCheckIn from './KioskCheckIn'
import PersonRegistration, { type EnrolledPerson } from './PersonRegistration'
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

function displayLabel(label: string, identified = false): string {
  const normalized = LABEL_MAP[label] ?? label
  return identified && normalized === 'Unknown' ? 'Chưa xác định trạng thái' : normalized
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

function labelText(label: string, identified = false): string {
  const normalized = displayLabel(label, identified)
  const translations: Record<string, string> = {
    Working: 'Đang làm việc',
    Away: 'Rời vị trí',
    'Out of office': 'Ngoài văn phòng',
    Returning: 'Đang quay lại',
    Unknown: 'Chưa xác định',
  }
  return translations[normalized] ?? normalized
}

function todayISO(): string {
  return new Date().toISOString().slice(0, 10)
}

function formatDateTime(value: string | null): string {
  if (!value) return '—'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return new Intl.DateTimeFormat('vi-VN', {
    day: '2-digit',
    month: '2-digit',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  }).format(date)
}

type AdminTab = 'attendance' | 'room' | 'events' | 'maintenance'

export default function App() {
  const [screen, setScreen] = useState<'kiosk' | 'admin' | 'registration' | 'employees'>('kiosk')
  const [registrationPerson, setRegistrationPerson] = useState<EnrolledPerson | null>(null)
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
  const [sendingAttendance, setSendingAttendance] = useState(false)
  const [liveOk, setLiveOk] = useState(false)
  const [adminTab, setAdminTab] = useState<AdminTab>('attendance')

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
      if (showLoading) note('')
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
        if (Array.isArray(data.attendance_today)) {
          setAttendance(data.attendance_today as AttendanceRow[])
        }
        const kioskCameraName = KIOSK_CHANNEL === 'A' ? 'cam_a' : 'cam_b'
        const kioskCamera = Array.isArray(data.cameras)
          ? data.cameras.find((camera: { name?: string }) => camera.name === kioskCameraName)
          : null
        setLiveOk(Boolean(kioskCamera?.live))
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
    return (
      <PersonRegistration
        streamUrl={STREAM_URL}
        initialPerson={registrationPerson}
        onBack={() => setScreen('employees')}
      />
    )
  }

  if (screen === 'employees') {
    return (
      <EmployeeManagement
        streamUrl={STREAM_URL}
        onBack={() => setScreen('admin')}
        onRegister={() => {
          setRegistrationPerson(null)
          setScreen('registration')
        }}
        onReplace={(person) => {
          setRegistrationPerson(person)
          setScreen('registration')
        }}
      />
    )
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
        onOpenAdmin={() => setScreen('admin')}
      />
    )
  }

  const canManage = Boolean(user || !supabaseConfigured)
  const inRoomCount = room.filter((item) => item.in_room).length

  return (
    <div className="app admin-app">
      <header className="app-header admin-header">
        <Brand section="Bảng điều hành chấm công theo thời gian thực" />
        <div className="app-header-actions">
          {canManage && (
            <button type="button" className="header-action" onClick={() => setScreen('employees')}>
              Nhân viên
            </button>
          )}
          <button type="button" className="header-action header-action-primary" onClick={() => setScreen('kiosk')}>
            Màn hình điểm danh
          </button>
        </div>
      </header>

      <section className="admin-controls" aria-label="Bộ lọc và tài khoản">
        <div className="date-control">
          <label htmlFor="dashboard-date">Ngày làm việc</label>
          <input id="dashboard-date" type="date" value={day} onChange={(e) => setDay(e.target.value)} />
          <button type="button" onClick={() => load()} aria-label="Làm mới dữ liệu">Làm mới</button>
        </div>
        <div className="admin-session">
          {!supabaseConfigured ? (
            <span className="mode-indicator"><i /> Chế độ dữ liệu cục bộ</span>
          ) : user ? (
            <span className="user-chip">
              {user} <button className="ghost" onClick={() => supabase.auth.signOut()}>Đăng xuất</button>
            </span>
          ) : (
            <form onSubmit={login} className="login-form">
              <input type="email" aria-label="Email quản trị" placeholder="Email quản trị" value={email} onChange={(e) => setEmail(e.target.value)} />
              <input type="password" aria-label="Mật khẩu" placeholder="Mật khẩu" value={password} onChange={(e) => setPassword(e.target.value)} />
              <button type="submit">Đăng nhập</button>
            </form>
          )}
        </div>
      </section>
      {msg && <p className={`status-msg${isError ? ' error' : ''}`}>{msg}</p>}

      <section className="overview-strip" aria-label="Tổng quan hệ thống">
        <div><span>Kết nối camera</span><strong className={liveOk ? 'metric-ok' : 'metric-error'}>{liveOk ? 'Trực tuyến' : 'Ngoại tuyến'}</strong></div>
        <div><span>Đang theo dõi</span><strong>{livePeople.length}</strong></div>
        <div><span>Đã điểm danh</span><strong>{attendance.filter((item) => item.attended).length}</strong></div>
        <div><span>Trong phòng</span><strong>{inRoomCount}</strong></div>
        <div><span>Chờ đồng bộ</span><strong className={pendingAttendance.length ? 'metric-warning' : ''}>{pendingAttendance.length}</strong></div>
      </section>

      <section className="camera-workspace">
        <div className="section-heading">
          <div><h2>Camera trực tiếp</h2><p>Quan sát nhận diện và Global ID trên hai khu vực</p></div>
          <span className={`camera-state ${liveOk ? 'online' : 'offline'}`}><i />{liveOk ? 'LIVE' : 'OFFLINE'}</span>
        </div>
        {!STREAM_URL || !liveOk ? (
          <div className="stream-empty">
            <strong>Chưa nhận được tín hiệu camera</strong>
            <span>Pipeline sẽ tự kết nối lại. Kiểm tra cổng 8765 nếu trạng thái này kéo dài.</span>
          </div>
        ) : (
          <>
            <div className="live-grid">
              <figure>
                <img src={`${STREAM_URL}/cam_a.mjpg`} alt="Camera A trực tiếp" />
                <figcaption><strong>Camera A</strong><span>Phòng làm việc · YOLO + Global ID</span></figcaption>
              </figure>
              <figure>
                <img src={`${STREAM_URL}/cam_b.mjpg`} alt="Camera B trực tiếp" />
                <figcaption><strong>Camera B</strong><span>Cửa ra vào · Xác thực khuôn mặt</span></figcaption>
              </figure>
            </div>
            <div className="live-chips" aria-label="Người đang được theo dõi">
              {livePeople.length === 0 && <span className="footnote">Chưa phát hiện người trong khung hình.</span>}
              {livePeople.map((person) => (
                <span key={person.gid} className="live-chip">
                  <strong>G{person.gid}</strong>{person.name ? ` · ${person.name}` : ''}
                  <span className={labelClass(person.label)}>{labelText(person.label, Boolean(person.person_id || person.name))}</span>
                </span>
              ))}
            </div>
          </>
        )}
      </section>

      <nav className="admin-tabs" aria-label="Dữ liệu quản trị">
        {([
          ['attendance', 'Điểm danh', attendance.length],
          ['room', 'Trạng thái phòng', room.length],
          ['events', 'Nhật ký di chuyển', events.length],
          ['maintenance', 'Vận hành', null],
        ] as [AdminTab, string, number | null][]).map(([tab, label, count]) => (
          <button key={tab} type="button" className={adminTab === tab ? 'active' : ''} onClick={() => setAdminTab(tab)}>
            {label}{count !== null && <span>{count}</span>}
          </button>
        ))}
      </nav>

      <main className="admin-panel">
        {adminTab === 'attendance' && (
          <section aria-labelledby="attendance-title">
            <div className="section-heading"><div><h2 id="attendance-title">Danh sách điểm danh</h2><p>Dữ liệu theo ngày đã chọn</p></div></div>
            {pendingAttendance.length > 0 && (
              <div className="sync-notice">
                <div><strong>{pendingAttendance.length} bản ghi chờ đồng bộ</strong><span>Dữ liệu đã được lưu an toàn trên máy này.</span></div>
                <button onClick={sendPendingAttendance} disabled={sendingAttendance}>{sendingAttendance ? 'Đang gửi...' : 'Gửi lên Supabase'}</button>
              </div>
            )}
            <div className="table-scroll"><table>
              <thead><tr>
                <th><input type="checkbox" aria-label="Chọn tất cả bản ghi điểm danh" checked={attendance.length > 0 && selAttend.length === attendance.length} onChange={(e) => setSelAttend(e.target.checked ? attendance.map((row) => row.person_id) : [])} /></th>
                <th>Nhân viên</th><th>Global ID</th><th>Thời gian vào</th><th>Độ tin cậy</th><th>Kết quả</th>{user && <th>Thao tác</th>}
              </tr></thead>
              <tbody>
                {attendance.length === 0 && <tr className="empty-row"><td colSpan={user ? 7 : 6}>Chưa có bản ghi trong ngày này.</td></tr>}
                {attendance.map((row) => <tr key={row.person_id}>
                  <td><input type="checkbox" aria-label={`Chọn ${row.person_name}`} checked={selAttend.includes(row.person_id)} onChange={() => toggleSel(selAttend, setSelAttend, row.person_id)} /></td>
                  <td><strong>{row.person_name}</strong><small>{row.person_id}</small></td><td>G{row.global_id ?? '—'}</td><td>{formatDateTime(row.check_in_at)}</td><td>{row.face_score?.toFixed(2) ?? '—'}</td>
                  <td><span className={row.attended ? 'status-pill success' : 'status-pill muted'}>{row.attended ? 'Có mặt' : 'Vắng'}</span></td>
                  {user && <td><button className="table-action" onClick={() => toggleAttend(row)}>{row.attended ? 'Đánh dấu vắng' : 'Đánh dấu có mặt'}</button></td>}
                </tr>)}
              </tbody>
            </table></div>
          </section>
        )}

        {adminTab === 'room' && (
          <section aria-labelledby="room-title">
            <div className="section-heading"><div><h2 id="room-title">Trạng thái phòng</h2><p>Vị trí gần nhất của nhân viên trong ngày</p></div></div>
            <div className="table-scroll"><table><thead><tr>
              <th><input type="checkbox" aria-label="Chọn tất cả trạng thái" checked={room.length > 0 && selRoom.length === room.length} onChange={(e) => setSelRoom(e.target.checked ? room.map((row) => row.global_id) : [])} /></th>
              <th>Global ID</th><th>Nhân viên</th><th>Trạng thái</th><th>Trong phòng</th><th>Vào lúc</th><th>Rời lúc</th>{user && <th>Thao tác</th>}
            </tr></thead><tbody>
              {room.length === 0 && <tr className="empty-row"><td colSpan={user ? 8 : 7}>Chưa có dữ liệu trạng thái trong ngày này.</td></tr>}
              {room.map((row) => <tr key={row.global_id}>
                <td><input type="checkbox" aria-label={`Chọn G${row.global_id}`} checked={selRoom.includes(row.global_id)} onChange={() => toggleSel(selRoom, setSelRoom, row.global_id)} /></td>
                <td>G{row.global_id}</td><td><strong>{row.person_name ?? 'Chưa xác định'}</strong><small>{row.person_id ?? '—'}</small></td><td><span className={labelClass(row.label)}>{labelText(row.label, Boolean(row.person_id || row.person_name))}</span></td>
                <td><span className={row.in_room ? 'status-pill success' : 'status-pill muted'}>{row.in_room ? 'Có' : 'Không'}</span></td><td>{formatDateTime(row.last_enter_at)}</td><td>{formatDateTime(row.last_leave_at)}</td>
                {user && <td><button className="table-action" onClick={() => toggleInRoom(row)}>Cập nhật</button></td>}
              </tr>)}
            </tbody></table></div>
          </section>
        )}

        {adminTab === 'events' && (
          <section aria-labelledby="events-title">
            <div className="section-heading"><div><h2 id="events-title">Nhật ký di chuyển</h2><p>Tối đa 100 sự kiện gần nhất trong ngày</p></div></div>
            <div className="table-scroll"><table><thead><tr>
              <th><input type="checkbox" aria-label="Chọn tất cả sự kiện" checked={events.length > 0 && selEvents.length === events.length} onChange={(e) => setSelEvents(e.target.checked ? events.map((row) => row.id) : [])} /></th>
              <th>Thời gian</th><th>Global ID</th><th>Mã nhân viên</th><th>Sự kiện</th><th>Camera</th>
            </tr></thead><tbody>
              {events.length === 0 && <tr className="empty-row"><td colSpan={6}>Chưa có sự kiện trong ngày này.</td></tr>}
              {events.map((row) => <tr key={row.id}>
                <td><input type="checkbox" aria-label={`Chọn sự kiện ${row.id}`} checked={selEvents.includes(row.id)} onChange={() => toggleSel(selEvents, setSelEvents, row.id)} /></td>
                <td>{formatDateTime(row.at)}</td><td>G{row.global_id}</td><td>{row.person_id ?? '—'}</td><td><span className="status-pill info">{row.event}</span></td><td>{row.channel}</td>
              </tr>)}
            </tbody></table></div>
          </section>
        )}

        {adminTab === 'maintenance' && (
          <section className="maintenance-panel" aria-labelledby="maintenance-title">
            <div className="section-heading"><div><h2 id="maintenance-title">Vận hành hệ thống</h2><p>Kiểm soát pipeline và dữ liệu quản trị</p></div></div>
            {canManage && <PipelineControl />}
            {user && <div className="danger-zone">
              <h3>Xóa lịch sử</h3><p>Chọn phạm vi dữ liệu cần xóa cho ngày {day}. Thao tác này không thể hoàn tác.</p>
              <div className="scope-list">{(['attendance', 'room', 'events', 'faces'] as DeleteScope[]).map((scope) => <label key={scope}><input type="checkbox" checked={scopes.includes(scope)} onChange={() => toggleScope(scope)} /> {scope}</label>)}</div>
              <div className="danger-row"><button className="danger-btn" disabled={selectedCount === 0 || deleting} onClick={() => setPendingDelete('selected')}>Xóa {selectedCount} mục đã chọn</button><button className="danger-btn" disabled={scopes.length === 0 || deleting} onClick={() => setPendingDelete('all')}>Xóa dữ liệu ngày này</button></div>
              {pendingDelete && <div className="warn-box"><strong>Xác nhận xóa vĩnh viễn {pendingDelete === 'all' ? `dữ liệu ${scopes.join(', ')}` : `${selectedCount} mục đã chọn`}.</strong><div className="danger-row"><button className="danger-btn" disabled={deleting} onClick={confirmDelete}>{deleting ? 'Đang xóa...' : 'Xác nhận xóa'}</button><button className="ghost" disabled={deleting} onClick={() => setPendingDelete(null)}>Hủy</button></div></div>}
            </div>}
          </section>
        )}
      </main>
    </div>
  )
}
