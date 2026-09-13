-- Supabase schema cho he diem danh + trang thai phong.
-- Chay file nay trong Supabase SQL Editor (1 lan).
-- Buckets: tao 2 bucket public-read (hoac signed): 'face-crops', 'enrolled-faces'.

create table if not exists public.persons (
  person_id text primary key,
  display_name text not null,
  photo_url text,
  active boolean not null default true,
  created_at timestamptz not null default now()
);

-- Initial enrolled employee for data/images/LeHoAnhDuy.jpg.
insert into public.persons (person_id, display_name, photo_url, active)
values ('1', 'Le Ho Anh Duy', 'data/images/LeHoAnhDuy.jpg', true)
on conflict (person_id) do update set
  display_name = excluded.display_name,
  photo_url = excluded.photo_url,
  active = excluded.active;

create table if not exists public.attendance_daily (
  date date not null,
  person_id text not null references public.persons(person_id) on delete cascade,
  person_name text not null,
  global_id integer,
  attended boolean not null default true,
  check_in_at timestamptz,
  face_score double precision,
  needs_review boolean not null default false,
  updated_at timestamptz not null default now(),
  edited_by text,
  primary key (date, person_id)
);

create table if not exists public.room_status_daily (
  date date not null,
  global_id integer not null,
  person_id text references public.persons(person_id) on delete set null,
  person_name text,
  in_room boolean not null default true,
  label text not null default 'Unknown',
  last_leave_at timestamptz,
  last_enter_at timestamptz,
  updated_at timestamptz not null default now(),
  edited_by text,
  primary key (date, global_id)
);

create table if not exists public.employee_current_state (
  employee_id text primary key references public.persons(person_id) on delete cascade,
  state text not null,
  since timestamptz not null default now(),
  camera_id text,
  global_id integer,
  confidence double precision,
  updated_at timestamptz not null default now()
);

create table if not exists public.room_events (
  id bigint generated always as identity primary key,
  date date not null,
  global_id integer not null,
  person_id text,
  event text not null check (event in (
    'ENTER_ROOM','LEAVE_SEAT','LEAVE_OFFICE','RETURN','CHECK_IN')),
  channel text not null default 'A',
  at timestamptz not null default now()
);
create index if not exists room_events_date_idx on public.room_events (date);
create index if not exists room_events_gid_idx on public.room_events (global_id);

create table if not exists public.face_crops (
  id bigint generated always as identity primary key,
  date date not null,
  owner_key text not null,
  global_id integer,
  is_known boolean not null default false,
  storage_path text not null,
  score double precision,
  created_at timestamptz not null default now()
);
create index if not exists face_crops_date_idx on public.face_crops (date);

create table if not exists public.roles (
  user_id uuid primary key references auth.users(id) on delete cascade,
  role text not null default 'viewer' check (role in ('admin','viewer'))
);

-- Row Level Security: authenticated duoc doc; chi admin duoc ghi.
alter table public.persons enable row level security;
alter table public.attendance_daily enable row level security;
alter table public.room_status_daily enable row level security;
alter table public.employee_current_state enable row level security;
alter table public.room_events enable row level security;
alter table public.face_crops enable row level security;
alter table public.roles enable row level security;

create policy "read all" on public.persons for select to authenticated using (true);
create policy "read all" on public.attendance_daily for select to authenticated using (true);
create policy "read all" on public.room_status_daily for select to authenticated using (true);
create policy "read all" on public.employee_current_state for select to authenticated using (true);
create policy "read all" on public.room_events for select to authenticated using (true);
create policy "read all" on public.face_crops for select to authenticated using (true);
create policy "self role" on public.roles for select to authenticated
  using (auth.uid() = user_id);

create policy "admin write persons" on public.persons
  for all to authenticated using (
    exists (select 1 from public.roles r where r.user_id = auth.uid() and r.role = 'admin')
  ) with check (
    exists (select 1 from public.roles r where r.user_id = auth.uid() and r.role = 'admin')
  );
create policy "admin write attendance" on public.attendance_daily
  for all to authenticated using (
    exists (select 1 from public.roles r where r.user_id = auth.uid() and r.role = 'admin')
  ) with check (
    exists (select 1 from public.roles r where r.user_id = auth.uid() and r.role = 'admin')
  );
create policy "admin write room status" on public.room_status_daily
  for all to authenticated using (
    exists (select 1 from public.roles r where r.user_id = auth.uid() and r.role = 'admin')
  ) with check (
    exists (select 1 from public.roles r where r.user_id = auth.uid() and r.role = 'admin')
  );
create policy "admin write current state" on public.employee_current_state
  for all to authenticated using (
    exists (select 1 from public.roles r where r.user_id = auth.uid() and r.role = 'admin')
  ) with check (
    exists (select 1 from public.roles r where r.user_id = auth.uid() and r.role = 'admin')
  );

-- Ghi bang service_key (pipeline) bypass RLS nen khong can policy them.

-- Storage (2 bucket private 'face-crops' + 'enrolled-faces'):
-- dashboard doc anh qua anon key + policy duoi; pipeline upload bang
-- service_key (bypass RLS) nen khong can policy insert.
create policy "auth read face-crops" on storage.objects
  for select to authenticated using (bucket_id = 'face-crops');
create policy "auth read enrolled" on storage.objects
  for select to authenticated using (bucket_id = 'enrolled-faces');
