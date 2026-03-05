-- Supabase SQL Editor で実行するスキーマ

-- イベントテーブル
create table events (
  id uuid default gen_random_uuid() primary key,
  event_code text unique not null,
  title text not null default '飲み会',
  created_at timestamptz default now()
);

-- 参加者テーブル
create table participants (
  id uuid default gen_random_uuid() primary key,
  event_id uuid references events(id) on delete cascade not null,
  name text not null,
  pattern text not null default '職場→飲み会→自宅',
  work_location text default '',
  home_location text default '',
  created_at timestamptz default now()
);

-- RLS (Row Level Security) を有効にしつつ、匿名アクセスを許可
alter table events enable row level security;
alter table participants enable row level security;

create policy "Anyone can read events" on events for select using (true);
create policy "Anyone can create events" on events for insert with check (true);

create policy "Anyone can read participants" on participants for select using (true);
create policy "Anyone can create participants" on participants for insert with check (true);
create policy "Anyone can update participants" on participants for update using (true);
create policy "Anyone can delete participants" on participants for delete using (true);

-- インデックス
create index idx_events_code on events(event_code);
create index idx_participants_event on participants(event_id);
