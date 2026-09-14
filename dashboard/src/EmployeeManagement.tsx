import { useCallback, useEffect, useMemo, useState } from 'react'
import Brand from './Brand'
import type { EnrolledPerson } from './PersonRegistration'

type Props = {
  streamUrl: string
  onBack: () => void
  onRegister: () => void
  onReplace: (person: EnrolledPerson) => void
}

type PeopleResponse = {
  people?: EnrolledPerson[]
  message?: string
}

function initials(name: string): string {
  const parts = name.trim().split(/\s+/).filter(Boolean)
  return parts.slice(-2).map((part) => part[0]?.toUpperCase()).join('') || 'NV'
}

export default function EmployeeManagement({
  streamUrl,
  onBack,
  onRegister,
  onReplace,
}: Props) {
  const [people, setPeople] = useState<EnrolledPerson[]>([])
  const [loading, setLoading] = useState(true)
  const [message, setMessage] = useState('')
  const [error, setError] = useState(false)
  const [pendingDelete, setPendingDelete] = useState<EnrolledPerson | null>(null)
  const [confirmId, setConfirmId] = useState('')
  const [deleting, setDeleting] = useState(false)
  const [query, setQuery] = useState('')

  const filteredPeople = useMemo(() => {
    const keyword = query.trim().toLocaleLowerCase('vi')
    if (!keyword) return people
    return people.filter((person) => (
      person.person_id.toLocaleLowerCase('vi').includes(keyword)
      || person.display_name.toLocaleLowerCase('vi').includes(keyword)
    ))
  }, [people, query])

  const readyCount = people.filter((person) => person.has_image).length

  const loadPeople = useCallback(async () => {
    setLoading(true)
    try {
      const response = await fetch(`${streamUrl}/enrollment/persons`)
      const payload = await response.json() as PeopleResponse
      if (!response.ok) throw new Error(payload.message || 'Không tải được danh sách.')
      setPeople(payload.people ?? [])
      setMessage('')
      setError(false)
    } catch (reason) {
      setMessage(reason instanceof Error ? reason.message : 'Không kết nối được pipeline camera.')
      setError(true)
    } finally {
      setLoading(false)
    }
  }, [streamUrl])

  useEffect(() => {
    void loadPeople()
  }, [loadPeople])

  function requestDelete(person: EnrolledPerson) {
    setPendingDelete(person)
    setConfirmId('')
    setMessage('')
  }

  async function deletePerson() {
    if (!pendingDelete || confirmId !== pendingDelete.person_id) return
    setDeleting(true)
    try {
      const response = await fetch(
        `${streamUrl}/enrollment/persons/${encodeURIComponent(pendingDelete.person_id)}`,
        { method: 'DELETE' },
      )
      const payload = await response.json() as { ok?: boolean; message?: string }
      if (!response.ok || !payload.ok) throw new Error(payload.message || 'Không thể xóa hồ sơ.')
      setPendingDelete(null)
      setConfirmId('')
      await loadPeople()
      setMessage(payload.message || 'Đã gỡ nhân viên khỏi nhận diện.')
      setError(false)
    } catch (reason) {
      setMessage(reason instanceof Error ? reason.message : 'Không thể xóa hồ sơ.')
      setError(true)
    } finally {
      setDeleting(false)
    }
  }

  return (
    <main className="employee-page">
      <header className="registration-header">
        <Brand section="Quản lý hồ sơ nhận diện" />
        <div className="employee-header-actions">
          <button type="button" className="registration-back" onClick={onRegister}>
            Thêm nhân viên
          </button>
          <button type="button" className="registration-back" onClick={onBack}>
            Quay lại quản trị
          </button>
        </div>
      </header>

      <section className="employee-workspace">
        <div className="employee-titlebar">
          <div>
            <h1>Hồ sơ nhận diện</h1>
            <p>{people.length} nhân viên đang hoạt động trong gallery cục bộ</p>
          </div>
          <div className="employee-toolbar">
            <input
              type="search"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Tìm theo tên hoặc mã..."
              aria-label="Tìm nhân viên"
            />
            <button type="button" className="secondary-action" onClick={() => void loadPeople()}>
              Tải lại
            </button>
          </div>
        </div>

        <dl className="employee-stats">
          <div><dt>Tổng hồ sơ</dt><dd>{people.length}</dd></div>
          <div><dt>Ảnh sẵn sàng</dt><dd>{readyCount}</dd></div>
          <div><dt>Cần bổ sung ảnh</dt><dd>{people.length - readyCount}</dd></div>
        </dl>

        {message && (
          <div className={`employee-message ${error ? 'error' : 'success'}`} role="status">
            {message}
          </div>
        )}

        {loading ? (
          <p className="employee-empty">Đang tải danh sách...</p>
        ) : people.length === 0 ? (
          <div className="employee-empty">
            <strong>Chưa có nhân viên</strong>
            <span>Đăng ký khuôn mặt đầu tiên để bắt đầu chấm công.</span>
          </div>
        ) : filteredPeople.length === 0 ? (
          <div className="employee-empty">
            <strong>Không tìm thấy nhân viên</strong>
            <span>Thử tìm bằng tên hoặc mã khác.</span>
          </div>
        ) : (
          <div className="employee-table-wrap">
            <table className="employee-table">
              <thead>
                <tr>
                  <th>Mã nhân viên</th>
                  <th>Họ và tên</th>
                  <th>Ảnh nhận diện</th>
                  <th aria-label="Thao tác" />
                </tr>
              </thead>
              <tbody>
                {filteredPeople.map((person) => (
                  <tr key={person.person_id}>
                    <td><code>{person.person_id}</code></td>
                    <td>
                      <div className="employee-identity">
                        <span className="employee-avatar" aria-hidden="true">
                          {initials(person.display_name)}
                        </span>
                        <strong>{person.display_name}</strong>
                      </div>
                    </td>
                    <td>
                      <span className={`image-state ${person.has_image ? 'ready' : 'missing'}`}>
                        {person.has_image ? 'Sẵn sàng' : 'Thiếu ảnh'}
                      </span>
                    </td>
                    <td className="employee-actions">
                      <button type="button" className="secondary-action" onClick={() => onReplace(person)}>
                        Thay ảnh
                      </button>
                      <button type="button" className="danger-action" onClick={() => requestDelete(person)}>
                        Gỡ nhận diện
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      {pendingDelete && (
        <div className="employee-dialog-backdrop" role="presentation">
          <section className="employee-dialog" role="dialog" aria-modal="true" aria-labelledby="delete-title">
            <h2 id="delete-title">Gỡ {pendingDelete.display_name}?</h2>
            <p>
              Ảnh sẽ bị xóa khỏi hệ thống nhận diện. Lịch sử điểm danh vẫn được giữ lại.
              Nhập mã <code>{pendingDelete.person_id}</code> để xác nhận.
            </p>
            <input
              type="text"
              value={confirmId}
              onChange={(event) => setConfirmId(event.target.value)}
              placeholder={pendingDelete.person_id}
              autoFocus
            />
            <div className="employee-dialog-actions">
              <button type="button" className="secondary-action" onClick={() => setPendingDelete(null)}>
                Hủy
              </button>
              <button
                type="button"
                className="danger-action"
                disabled={confirmId !== pendingDelete.person_id || deleting}
                onClick={() => void deletePerson()}
              >
                {deleting ? 'Đang gỡ...' : 'Xác nhận gỡ'}
              </button>
            </div>
          </section>
        </div>
      )}
    </main>
  )
}
