import { useEffect, useMemo, useState } from 'react'

export type KioskPerson = {
  gid: number
  person_id?: string | null
  name: string | null
  label: string
  in_room: boolean
  face_score?: number | null
  identity_state?: 'employee' | 'unknown'
  identity_confidence?: number | null
  employee_confidence?: number | null
  camera?: string | null
  cameras?: string[]
  tracking_state?: string | null
  face_candidates?: Array<{ name: string; score: number }>
}

export type KioskAttendance = {
  person_id: string
  person_name: string
  global_id: number | null
  attended?: boolean
  check_in_at: string | null
  face_score: number | null
}

export type KioskEvent = {
  event: 'CHECK_IN' | 'LEAVE_OFFICE' | 'RETURN' | string
  global_id: number | null
  channel: string | null
  at: string | null
  person_id?: string | null
  person_name?: string | null
}

type RecognitionState =
  | 'idle'
  | 'recognizing'
  | 'recognized'
  | 'not-recognized'
  | 'success'
  | 'duplicate'
  | 'offline'

type Props = {
  streamUrl: string
  cameraChannel: 'A' | 'B'
  liveOk: boolean
  people: KioskPerson[]
  attendance: KioskAttendance[]
  pendingAttendance: KioskAttendance[]
  events: KioskEvent[]
  onOpenAdmin: () => void
}

const STATE_COPY: Record<RecognitionState, { title: string; detail: string }> = {
  idle: {
    title: 'Sẵn sàng điểm danh',
    detail: 'Vui lòng đứng trước camera',
  },
  recognizing: {
    title: 'Đang nhận diện...',
    detail: 'Giữ khuôn mặt trong khung hình',
  },
  recognized: {
    title: 'Đã nhận diện',
    detail: 'Đang xác nhận dữ liệu chấm công',
  },
  'not-recognized': {
    title: 'Không nhận diện được',
    detail: 'Vui lòng nhìn thẳng vào camera và thử lại',
  },
  success: {
    title: 'Check-in thành công',
    detail: 'Dữ liệu chấm công đã được ghi nhận',
  },
  duplicate: {
    title: 'Đã check-in trước đó',
    detail: 'Hệ thống không ghi nhận trùng lượt',
  },
  offline: {
    title: 'Camera đang ngoại tuyến',
    detail: 'Đang chờ kết nối lại pipeline',
  },
}

const EVENT_COPY: Record<string, { label: string; className: string }> = {
  CHECK_IN: { label: 'Điểm danh', className: 'event-checkin' },
  LEAVE_OFFICE: { label: 'Rời văn phòng', className: 'event-leave' },
  RETURN: { label: 'Quay lại', className: 'event-return' },
}

function eventCopy(event: string): { label: string; className: string } {
  return EVENT_COPY[event] ?? { label: event, className: 'event-other' }
}

function initials(name: string | null | undefined): string {
  if (!name) return 'ID'
  return name
    .trim()
    .split(/\s+/)
    .slice(-2)
    .map((part) => part[0]?.toUpperCase() ?? '')
    .join('')
}

function formatTime(value: string | null | undefined, fallback: Date): string {
  if (!value) return fallback.toLocaleTimeString('vi-VN', { hour12: false })
  const parsed = new Date(value)
  if (Number.isNaN(parsed.getTime())) return value
  return parsed.toLocaleTimeString('vi-VN', { hour12: false })
}

export default function KioskCheckIn({
  streamUrl,
  cameraChannel,
  liveOk,
  people,
  attendance,
  pendingAttendance,
  events,
  onOpenAdmin,
}: Props) {
  const [now, setNow] = useState(new Date())
  const [unknownSince, setUnknownSince] = useState<number | null>(null)
  const [recentSuccess, setRecentSuccess] = useState<{
    record: KioskAttendance
    seenAt: number
  } | null>(null)

  useEffect(() => {
    const timer = window.setInterval(() => setNow(new Date()), 1000)
    return () => window.clearInterval(timer)
  }, [])

  const cameraPeople = useMemo(() => {
    const active = people.filter((person) =>
      !person.tracking_state || person.tracking_state === 'ACTIVE',
    )
    const hasCameraMetadata = active.some((person) =>
      Boolean(person.camera || person.cameras?.length),
    )
    return hasCameraMetadata
      ? active.filter((person) =>
        person.cameras?.includes(cameraChannel) || person.camera === cameraChannel,
      )
      : active
  }, [cameraChannel, people])

  const countA = useMemo(
    () => people.filter((p) => (p.cameras?.includes('A') || p.camera === 'A') && (!p.tracking_state || p.tracking_state === 'ACTIVE')).length,
    [people],
  )
  const countB = useMemo(
    () => people.filter((p) => (p.cameras?.includes('B') || p.camera === 'B') && (!p.tracking_state || p.tracking_state === 'ACTIVE')).length,
    [people],
  )

  const subject = useMemo(
    () => cameraPeople.find((person) => person.name) ?? cameraPeople[0] ?? null,
    [cameraPeople],
  )

  useEffect(() => {
    if (subject && !subject.name) setUnknownSince(Date.now())
    else setUnknownSince(null)
  }, [subject?.gid, subject?.name])

  const latestPending = pendingAttendance[pendingAttendance.length - 1]
  useEffect(() => {
    if (!latestPending) return
    setRecentSuccess({ record: latestPending, seenAt: Date.now() })
  }, [latestPending?.person_id, latestPending?.check_in_at])

  const matchedAttendance = subject
    ? attendance.find((record) =>
      record.attended !== false && (
        (subject.person_id && record.person_id === subject.person_id)
        || record.global_id === subject.gid
        || (!!subject.name && record.person_name === subject.name)
      ))
    : undefined

  const successActive = recentSuccess && now.getTime() - recentSuccess.seenAt < 8000
  let state: RecognitionState = 'idle'
  if (!liveOk) state = 'offline'
  else if (successActive) state = 'success'
  else if (subject?.name && matchedAttendance) state = 'duplicate'
  else if (subject?.name) state = 'recognized'
  else if (subject && unknownSince && now.getTime() - unknownSince >= 4500) {
    state = 'not-recognized'
  } else if (subject) state = 'recognizing'

  const record = successActive ? recentSuccess.record : matchedAttendance
  const displayName = record?.person_name ?? subject?.name ?? null
  const employeeId = record?.person_id ?? subject?.person_id ?? null
  const confidence = record?.face_score ?? subject?.face_score ?? null
  const copy = STATE_COPY[state]

  return (
    <main className="kiosk" data-state={state}>
      <header className="kiosk-header">
        <div className="kiosk-brand">
          <span className="brand-mark">FPT</span>
          <div>
            <strong>Hệ thống chấm công</strong>
            <span>FPT University</span>
          </div>
        </div>
        <div className="kiosk-header-actions">
          <span className={`connection-state${liveOk ? ' online' : ''}`}>
            <i />{liveOk ? 'Camera trực tuyến' : 'Mất kết nối'}
          </span>
          <time dateTime={now.toISOString()}>
            <strong>{now.toLocaleTimeString('vi-VN', { hour12: false })}</strong>
            <span>{now.toLocaleDateString('vi-VN')}</span>
          </time>
          <button type="button" className="admin-link" onClick={onOpenAdmin}>
            Quản trị
          </button>
        </div>
      </header>

      <div className="kiosk-content">
        <section className="camera-panel" aria-label="Camera trực tiếp">
          <div className="camera-bar">
            <div>
              <strong>Camera trực tiếp</strong>
              <span>A · Phòng ({countA}) — B · Cửa ({countB}) · Điểm danh theo Channel {cameraChannel}</span>
            </div>
            <span className="camera-mode">LIVE</span>
          </div>
          <div className="camera-duo">
            <figure className="camera-cell">
              <div className="camera-viewport">
                <img src={`${streamUrl}/cam_a.mjpg`} alt="Camera A — phòng" />
              </div>
              <figcaption>Camera A — phòng</figcaption>
            </figure>
            <figure className="camera-cell">
              <div className="camera-viewport">
                <img src={`${streamUrl}/cam_b.mjpg`} alt="Camera B — cửa" />
              </div>
              <figcaption>Camera B — cửa</figcaption>
            </figure>
          </div>
          {!liveOk && (
            <div className="camera-offline">
              <strong>Không có tín hiệu</strong>
              <span>Pipeline sẽ tự kết nối lại</span>
            </div>
          )}
        </section>

        <aside className="recognition-panel" aria-live="polite">
          <div className="recognition-status">
            <span className="status-symbol" aria-hidden="true" />
            <span>{copy.title}</span>
          </div>

          <div className="identity-block">
            <div className="avatar" aria-label="Ảnh đại diện nhân viên">
              {initials(displayName)}
            </div>
            <div>
              <p className="identity-kicker">Nhân viên</p>
              <h1>{displayName ?? 'Chưa xác định'}</h1>
              <p>{copy.detail}</p>
            </div>
          </div>

          <dl className="recognition-details">
            <div>
              <dt>Mã nhân viên</dt>
              <dd>{employeeId ?? '—'}</dd>
            </div>
            <div>
              <dt>Thời gian</dt>
              <dd>{formatTime(record?.check_in_at, now)}</dd>
            </div>
            <div>
              <dt>Camera</dt>
              <dd>{subject?.camera ? `Camera ${subject.camera}` : `Camera ${cameraChannel}`}</dd>
            </div>
            <div>
              <dt>Độ tin cậy</dt>
              <dd>{confidence == null ? '—' : `${Math.round(confidence * 100)}%`}</dd>
            </div>
          </dl>

          <div className="privacy-note">
            Dữ liệu khuôn mặt chỉ được sử dụng để xác thực chấm công.
          </div>
        </aside>
      </div>

      <div className="kiosk-bottom">
      <section className="recent-checkins" aria-label="Lượt điểm danh gần đây">
        <div className="recent-heading">
          <strong>Lượt điểm danh gần đây</strong>
          <span>{attendance.length} nhân viên hôm nay</span>
        </div>
        <div className="recent-list">
          {attendance.length === 0 && <span className="recent-empty">Chưa có lượt điểm danh</span>}
          {attendance.slice(-4).reverse().map((item) => (
            <div className="recent-person" key={item.person_id}>
              <span className="recent-avatar">{initials(item.person_name)}</span>
              <span>
                <strong>{item.person_name}</strong>
                <small>{formatTime(item.check_in_at, now)}</small>
              </span>
            </div>
          ))}
        </div>
      </section>

      <section className="activity-feed" aria-label="Hoạt động gần đây">
        <div className="recent-heading">
          <strong>Hoạt động</strong>
          <span>{liveOk ? 'Trực tiếp từ pipeline' : 'Ngoại tuyến'}</span>
        </div>
        <div className="identity-match-list" aria-label="Độ tương đồng khuôn mặt">
          {cameraPeople.map((person) => (
            <div className="identity-match" key={`match-${person.gid}`}>
              <strong>G{person.gid}</strong>
              <span>
                {person.face_candidates?.length
                  ? person.face_candidates.slice(0, 2).map((candidate, index) =>
                    `${index + 1}. ${candidate.name} ${Math.round(Math.max(0, candidate.score) * 100)}%`,
                  ).join(' · ')
                  : 'Đang chờ khuôn mặt'}
              </span>
            </div>
          ))}
        </div>
        <ul className="activity-list">
          {events.length === 0 && (
            <li className="recent-empty">Chưa có hoạt động nào</li>
          )}
          {events.slice(-8).reverse().map((item, index) => {
            const copy = eventCopy(item.event)
            return (
              <li key={`${item.at ?? ''}-${item.global_id ?? ''}-${item.event}-${index}`} className="activity-item">
                <span className={`activity-dot ${copy.className}`} aria-hidden="true" />
                <span className="activity-body">
                  <strong>{item.person_name ?? `G${item.global_id ?? '?'}`}</strong>
                  <small>{copy.label}{item.channel ? ` · Cam ${item.channel}` : ''}</small>
                </span>
                <time>{formatTime(item.at, now)}</time>
              </li>
            )
          })}
        </ul>
      </section>
      </div>
    </main>
  )
}
