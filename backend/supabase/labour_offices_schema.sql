-- Supabase schema for labour office directory lookup
-- Run this in Supabase SQL editor.

create table if not exists public.labour_offices (
    id bigint generated always as identity primary key,
    office_name text not null,
    country_code text not null default 'MY',
    state_region text,
    district text,
    city text,
    address text not null,
    phone text,
    email text,
    website text,
    open_hours text,
    latitude double precision,
    longitude double precision,
    is_national boolean not null default false,
    is_active boolean not null default true,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create index if not exists idx_labour_offices_active
    on public.labour_offices (is_active);

create index if not exists idx_labour_offices_national
    on public.labour_offices (is_national)
    where is_active = true;

create index if not exists idx_labour_offices_city_lower
    on public.labour_offices (lower(city));

create index if not exists idx_labour_offices_state_lower
    on public.labour_offices (lower(state_region));

create index if not exists idx_labour_offices_district_lower
    on public.labour_offices (lower(district));

create or replace function public.set_labour_offices_updated_at()
returns trigger
language plpgsql
as $$
begin
    new.updated_at = now();
    return new;
end;
$$;

drop trigger if exists trg_labour_offices_updated_at on public.labour_offices;
create trigger trg_labour_offices_updated_at
before update on public.labour_offices
for each row
execute function public.set_labour_offices_updated_at();

-- Seed starter entries (edit with your verified local contacts)
insert into public.labour_offices (
    office_name,
    country_code,
    state_region,
    district,
    city,
    address,
    phone,
    email,
    website,
    open_hours,
    is_national,
    is_active
)
values
(
    'Jabatan Tenaga Kerja Semenanjung Malaysia (HQ)',
    'MY',
    'Wilayah Persekutuan',
    'Kuala Lumpur',
    'Kuala Lumpur',
    'Aras 5, Blok D3, Kompleks D, 62530 Putrajaya',
    '+603-8000 8000',
    'jtksm@mohr.gov.my',
    'https://jtksm.mohr.gov.my',
    'Mon-Fri 8:00-17:00',
    true,
    true
),
(
    'Pejabat Tenaga Kerja Kuala Lumpur',
    'MY',
    'Wilayah Persekutuan',
    'Kuala Lumpur',
    'Kuala Lumpur',
    'Wisma Perkeso, Jalan Tun Razak, 50400 Kuala Lumpur',
    '+603-9213 1600',
    null,
    'https://jtksm.mohr.gov.my',
    'Mon-Fri 8:00-17:00',
    false,
    true
),
(
    'Pejabat Tenaga Kerja Selangor',
    'MY',
    'Selangor',
    'Petaling',
    'Shah Alam',
    'Persiaran Perbandaran, Seksyen 14, 40000 Shah Alam',
    '+603-5544 7400',
    null,
    'https://jtksm.mohr.gov.my',
    'Mon-Fri 8:00-17:00',
    false,
    true
)
on conflict do nothing;
