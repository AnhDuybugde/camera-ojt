import { supabase } from './supabase'

function chunk<T>(arr: T[], size = 100): T[][] {
  const out: T[][] = []
  for (let i = 0; i < arr.length; i += size) out.push(arr.slice(i, i + size))
  return out
}

async function throwIfError(promise: Promise<{ error: unknown }>, what: string) {
  const { error } = await promise
  if (error) throw new Error(`${what}: ${(error as Error).message ?? error}`)
}

/** Delete selected attendance rows for a date. */
export async function deleteAttendance(date: string, personIds: string[]) {
  for (const part of chunk([...new Set(personIds)])) {
    await throwIfError(
      supabase.from('attendance_daily').delete().eq('date', date).in('person_id', part) as never,
      'Delete attendance failed',
    )
  }
}

/** Delete selected room-status rows for a date (by Global ID). */
export async function deleteRoomStatus(date: string, gids: number[]) {
  for (const part of chunk([...new Set(gids)])) {
    await throwIfError(
      supabase.from('room_status_daily').delete().eq('date', date).in('global_id', part) as never,
      'Delete room status failed',
    )
  }
}

/** Delete room events by row id. */
export async function deleteEvents(ids: number[]) {
  for (const part of chunk([...new Set(ids)])) {
    await throwIfError(
      supabase.from('room_events').delete().in('id', part) as never,
      'Delete events failed',
    )
  }
}

export type DeleteScope = 'attendance' | 'room' | 'events' | 'faces'

/** Delete ALL rows of the selected scopes for one date (demo reset). */
export async function deleteAllForDate(date: string, scopes: DeleteScope[]) {
  const counts: Record<string, number> = {}
  if (scopes.includes('attendance')) {
    const { error, count } = await supabase
      .from('attendance_daily').delete({ count: 'exact' }).eq('date', date)
    if (error) throw new Error(`Delete attendance failed: ${error.message}`)
    counts.attendance = count ?? 0
  }
  if (scopes.includes('room')) {
    const { error, count } = await supabase
      .from('room_status_daily').delete({ count: 'exact' }).eq('date', date)
    if (error) throw new Error(`Delete room status failed: ${error.message}`)
    counts.room = count ?? 0
  }
  if (scopes.includes('events')) {
    const { error, count } = await supabase
      .from('room_events').delete({ count: 'exact' }).eq('date', date)
    if (error) throw new Error(`Delete events failed: ${error.message}`)
    counts.events = count ?? 0
  }
  if (scopes.includes('faces')) {
    counts.faces = await deleteFaceCropsForDate(date)
  }
  return counts
}

/** Delete face-crop rows for a date plus their storage files. */
export async function deleteFaceCropsForDate(date: string): Promise<number> {
  const { data, error } = await supabase
    .from('face_crops').select('id, storage_path').eq('date', date)
  if (error) throw new Error(`List face crops failed: ${error.message}`)
  const rows = (data ?? []) as { id: number; storage_path: string }[]
  const paths = [...new Set(rows.map((r) => r.storage_path).filter(Boolean))]
  // Remove files first so no orphan binaries remain (admin storage policy).
  for (const part of chunk(paths)) {
    const { error: rmError } = await supabase.storage.from('face-crops').remove(part)
    if (rmError) throw new Error(`Delete crop files failed: ${rmError.message}`)
  }
  if (rows.length > 0) {
    for (const part of chunk(rows.map((r) => r.id))) {
      await throwIfError(
        supabase.from('face_crops').delete().in('id', part) as never,
        'Delete face crops failed',
      )
    }
  }
  return rows.length
}
