# 飲み会スポットファインダー

## プロジェクト概要
参加者の職場・自宅の最寄り駅を入力すると、全員にとって公平な集合駅を提案し、周辺の飲食店も検索できるWebアプリ。

## 技術スタック
- **バックエンド**: FastAPI (`nomikai_api.py`) - 全APIエンドポイントを提供
- **フロントエンド**: 単一HTML SPA (`static/index.html`) - vanilla JS、Leaflet地図、CSS全て1ファイル
- **データベース**: Supabase (PostgreSQL) - `events` / `participants` テーブル (`schema.sql`)
- **駅データ**: `precompute.py` で CSV → pickle (`ekidata_cache.pkl`) にビルド時変換
- **外部API**: ホットペッパーグルメ API (飲食店検索)
- **デプロイ**: Render (Docker) - GitHubにpushで自動デプロイ

## ファイル構成
```
nomikai_api.py        # FastAPI バックエンド (API + SPA配信)
nomikai_spot.py       # 旧Streamlit版 (未使用、参考用)
precompute.py         # 駅データCSV → pickle変換スクリプト
schema.sql            # Supabaseテーブル定義
Dockerfile.nomikai    # Dockerビルド定義
render.yaml           # Renderデプロイ設定
requirements-nomikai.txt  # Python依存パッケージ
statione.csv / join.csv / line.csv  # 駅データCSV
static/
  index.html          # フロントエンド SPA (HTML + CSS + JS 全て)
  logo-icon.png       # ヘッダーロゴ
  hero-hand.png       # ヒーロー画像 (スマホ持ち手)
  privacy.html        # プライバシーポリシー
  terms.html          # 利用規約
  llms.txt / ads.txt  # SEO/広告用
```

## アーキテクチャ
- ルーティング: Dijkstraアルゴリズムで駅間の最短経路・所要時間を計算
- スコアリング: 職場→駅、駅→自宅の移動時間の加重合計 + 公平性(標準偏差)ペナルティ
- 地図: Leaflet + CARTO タイル、ピン型カスタムマーカー (職場=ビル、自宅=家、候補駅=ビール)
- イベント共有: 6文字ランダムコードでURL共有、複数人が同時編集可能

## デプロイ方法
```bash
git push origin main
```
GitHubへのpushでRenderが自動デプロイ。ポート8501、uvicornで起動。

## 環境変数 (Render側で設定済み)
- `SUPABASE_URL` / `SUPABASE_KEY` - Supabase接続
- `HOTPEPPER_API_KEY` - ホットペッパーAPI
- `GA_ID` - Google Analytics

## 開発時の注意
- フロントエンドは `static/index.html` 1ファイルに全て含まれる (CSS + HTML + JS)
- ファイルが大きいため、編集時はGrep/Readで該当箇所を特定してからEditすること
- `nomikai_spot.py` は旧Streamlit版で現在未使用 (参考用に残してある)
- 駅データのpickleはDockerビルド時に生成されるため、ローカルになくてもデプロイは可能

## 本番URL
https://www.nomikai-spot-finder.com/
