-- Production hardening migration. Safe to run repeatedly.

alter table public.attendance_daily
  add column if not exists check_out_at timestamptz;

alter table public.room_status_daily
  add column if not exists merged_into integer;

-- Existing CHECK constraint must be replaced to permit lifecycle events.
do $$
declare
  constraint_name text;
begin
  for constraint_name in
    select distinct constraint_row.conname
    from pg_constraint as constraint_row
    join pg_attribute as column_row
      on column_row.attrelid = constraint_row.conrelid
     and column_row.attnum = any (constraint_row.conkey)
    where constraint_row.conrelid = 'public.room_events'::regclass
      and constraint_row.contype = 'c'
      and column_row.attname = 'event'
  loop
    execute format('alter table public.room_events drop constraint %I', constraint_name);
  end loop;
end $$;

alter table public.room_events
  add constraint room_events_event_check check (event in (
    'ENTER_ROOM','LEAVE_SEAT','LEAVE_OFFICE','RETURN',
    'CHECK_IN','CHECK_OUT','TEMP_OUT'
  ));

create index if not exists room_status_daily_merged_idx
  on public.room_status_daily (date, merged_into);
