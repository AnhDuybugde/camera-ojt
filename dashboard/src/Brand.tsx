type Props = {
  section?: string
}

export default function Brand({ section }: Props) {
  return (
    <div className="kiosk-brand">
      <img className="brand-logo" src="/assets/ai-mind-logo.png" alt="AI Mind JSC" />
      <div>
        <strong>Hệ thống chấm công AI Mind JSC</strong>
        <span>{section ?? 'AI-powered attendance'}</span>
      </div>
    </div>
  )
}
