import { createClient } from '@supabase/supabase-js'

const configuredUrl = import.meta.env.VITE_SUPABASE_URL as string | undefined
const configuredAnon = import.meta.env.VITE_SUPABASE_ANON_KEY as string | undefined

export const supabaseConfigured = Boolean(configuredUrl && configuredAnon)

if (!supabaseConfigured) {
  console.warn('Thieu VITE_SUPABASE_URL / VITE_SUPABASE_ANON_KEY (copy .env.example).')
}

// A valid placeholder keeps the public kiosk usable before Supabase is configured.
export const supabase = createClient(
  configuredUrl ?? 'http://127.0.0.1:54321',
  configuredAnon ?? 'kiosk-not-configured',
)
