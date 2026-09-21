import { useRef, useState } from 'react'

type Props = {
  streamUrl: string
  onBack: () => void
}

type RegistrationResponse = {
  ok: boolean
  person_id?: string
  display_name?: string
  message?: string
}

export default function PersonRegistration({ streamUrl, onBack }: Props) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [employeeId, setEmployeeId] = useState('')
  const [displayName, setDisplayName] = useState('')
  const [imageData, setImageData] = useState('')
  const [fileName, setFileName] = useState('')
  const [consent, setConsent] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [result, setResult] = useState<RegistrationResponse | null>(null)

  function selectImage(file: File | undefined) {
    setResult(null)
    if (!file) return
    if (!file.type.startsWith('image/')) {
      setResult({ ok: false, message: 'Vui lòng chọn đúng định dạng ảnh.' })
      return
    }
    if (file.size > 5 * 1024 * 1024) {
      setResult({ ok: false, message: 'Ảnh phải nhỏ hơn 5 MB.' })
      return
    }
    const reader = new FileReader()
    reader.onload = () => {
      setImageData(String(reader.result ?? ''))
      setFileName(file.name)
    }
    reader.onerror = () => setResult({ ok: false, message: 'Không thể đọc ảnh đã chọn.' })
    reader.readAsDataURL(file)
  }

  async function submit(event: React.FormEvent) {
    event.preventDefault()
    setSubmitting(true)
    setResult(null)
    try {
      const response = await fetch(`${streamUrl}/enrollment/register`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          employee_id: employeeId.trim(),
          display_name: displayName.trim(),
          image: imageData,
          consent,
        }),
      })
      const payload = await response.json() as RegistrationResponse
      setResult(payload)
      if (response.ok && payload.ok) {
        setEmployeeId('')
        setDisplayName('')
        setImageData('')
        setFileName('')
        setConsent(false)
        if (inputRef.current) inputRef.current.value = ''
      }
    } catch {
      setResult({
        ok: false,
        message: 'Không kết nối được pipeline camera. Hãy khởi động lại backend.',
      })
    } finally {
      setSubmitting(false)
    }
  }

  const ready = Boolean(employeeId.trim() && displayName.trim() && imageData && consent)

  return (
    <main className="registration-page">
      <header className="registration-header">
        <div className="kiosk-brand">
          <span className="brand-mark">FPT</span>
          <div>
            <strong>Đăng ký nhân viên</strong>
            <span>Face enrollment</span>
          </div>
        </div>
        <button type="button" className="registration-back" onClick={onBack}>
          Quay lại điểm danh
        </button>
      </header>

      <form className="registration-workspace" onSubmit={submit}>
        <section className="registration-fields">
          <div className="section-heading">
            <span>01</span>
            <div>
              <h1>Thông tin nhân viên</h1>
              <p>Thông tin định danh nội bộ</p>
            </div>
          </div>

          <label className="field-label" htmlFor="employee-id">Mã nhân viên</label>
          <input
            id="employee-id"
            type="text"
            value={employeeId}
            onChange={(event) => setEmployeeId(event.target.value)}
            placeholder="Ví dụ: NV001"
            maxLength={64}
            autoComplete="off"
            required
          />

          <label className="field-label" htmlFor="display-name">Họ và tên</label>
          <input
            id="display-name"
            type="text"
            value={displayName}
            onChange={(event) => setDisplayName(event.target.value)}
            placeholder="Nguyễn Văn A"
            maxLength={100}
            autoComplete="name"
            required
          />

          <label className="consent-row">
            <input
              type="checkbox"
              checked={consent}
              onChange={(event) => setConsent(event.target.checked)}
            />
            <span>
              Nhân viên đồng ý sử dụng dữ liệu khuôn mặt cho mục đích chấm công.
            </span>
          </label>

          {result && (
            <div className={`registration-result ${result.ok ? 'success' : 'error'}`} role="status">
              <strong>{result.ok ? 'Đăng ký thành công' : 'Chưa thể đăng ký'}</strong>
              <span>{result.message}</span>
            </div>
          )}

          <button type="submit" className="register-submit" disabled={!ready || submitting}>
            {submitting ? 'Đang kiểm tra khuôn mặt...' : 'Đăng ký nhân viên'}
          </button>
        </section>

        <section className="registration-photo">
          <div className="section-heading">
            <span>02</span>
            <div>
              <h2>Ảnh khuôn mặt</h2>
              <p>Một người, chính diện, đủ sáng</p>
            </div>
          </div>

          <button
            type="button"
            className={`photo-picker${imageData ? ' has-image' : ''}`}
            onClick={() => inputRef.current?.click()}
          >
            {imageData ? (
              <img src={imageData} alt="Ảnh khuôn mặt được chọn" />
            ) : (
              <span className="photo-placeholder" aria-hidden="true">
                <i />
              </span>
            )}
            <span className="photo-picker-copy">
              <strong>{imageData ? 'Chọn ảnh khác' : 'Chọn ảnh khuôn mặt'}</strong>
              <small>{fileName || 'JPG, PNG hoặc WEBP · tối đa 5 MB'}</small>
            </span>
          </button>
          <input
            ref={inputRef}
            className="visually-hidden"
            type="file"
            accept="image/jpeg,image/png,image/webp"
            onChange={(event) => selectImage(event.target.files?.[0])}
          />

          <dl className="photo-checklist">
            <div><dt>Khuôn mặt</dt><dd>Chỉ một người</dd></div>
            <div><dt>Góc nhìn</dt><dd>Nhìn thẳng</dd></div>
            <div><dt>Ánh sáng</dt><dd>Không ngược sáng</dd></div>
          </dl>
        </section>
      </form>
    </main>
  )
}
