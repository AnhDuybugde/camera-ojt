-- Attendance lifecycle and auditable biometric verification.
-- Kept as a new migration because production_hardening may already be applied.

alter table public.attendance_daily
  add column if not exists liveness_score double precision,
  add column if not exists verification_method text,
  add column if not exists automatic boolean not null default true,
  add column if not exists status text not null default 'PRESENT';

do $$
begin
  if not exists (
    select 1 from pg_constraint
    where conrelid = 'public.attendance_daily'::regclass
      and conname = 'attendance_daily_status_check'
  ) then
    alter table public.attendance_daily
      add constraint attendance_daily_status_check
      check (status in ('PRESENT','TEMP_OUT','CHECKED_OUT'));
  end if;
end $$;

create table if not exists public.attendance_audit_log (
  id bigint generated always as identity primary key,
  date date not null,
  person_id text not null,
  action text not null,
  before_value jsonb,
  after_value jsonb,
  reason text,
  actor text not null default 'camera-system',
  created_at timestamptz not null default now()
);

create index if not exists attendance_audit_day_person_idx
  on public.attendance_audit_log (date, person_id);

alter table public.attendance_audit_log enable row level security;

do $$
begin
  if not exists (
    select 1 from pg_policies
    where schemaname = 'public'
      and tablename = 'attendance_audit_log'
      and policyname = 'admin read audit'
  ) then
    create policy "admin read audit" on public.attendance_audit_log
      for select to authenticated using (
        exists (
          select 1 from public.roles r
          where r.user_id = auth.uid() and r.role = 'admin'
        )
      );
  end if;
end $$;

-- Pipeline writes with SUPABASE_SERVICE_KEY and therefore bypasses RLS.
