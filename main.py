#!/usr/bin/env python3
import os
import time
import sqlite3
import requests
import statistics
import numpy as np
from datetime import datetime, timedelta, timezone
from scipy.stats import poisson
from dotenv import load_dotenv

# -------------------- CONFIG --------------------
load_dotenv()
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise SystemExit("❌ Missing API_KEY in environment (.env file).")

BASE_URL = "https://api.football-data.org/v4/"
HEADERS = {"X-Auth-Token": API_KEY}
BATCH_SIZE = 10
RECENT_MATCHES = 10
RATE_LIMIT_WAIT = 5  # ⏱️ Hardcoded wait (in seconds) between API calls
UTC_OFFSET = 2       # Convert UTC -> UTC+2 for local match time display

session = requests.Session()
session.headers.update(HEADERS)

# -------------------- API UTILS --------------------

def safe_request(url: str, params: dict | None = None) -> dict | None:
    """Simple, stable HTTP GET with retry and fixed rate limit."""
    for attempt in range(3):
        try:
            res = session.get(url, params=params, timeout=15)
            if res.status_code == 200:
                time.sleep(RATE_LIMIT_WAIT)
                return res.json()
            elif res.status_code in (429, 503):
                print(f"⚠️ Rate limit hit ({res.status_code}), waiting {RATE_LIMIT_WAIT}s...")
                time.sleep(RATE_LIMIT_WAIT)
            else:
                print(f"HTTP {res.status_code} on attempt {attempt+1}")
        except requests.RequestException as e:
            print(f"Network error: {e}")
        time.sleep(RATE_LIMIT_WAIT)
    return None


def get_competition_matches(competition_code: str, offset=0, limit=10):
    """Fetch scheduled matches for a competition."""
    url = f"{BASE_URL}competitions/{competition_code}/matches"
    params = {"status": "SCHEDULED"}
    data = safe_request(url, params=params)
    matches = data.get("matches", []) if data else []
    return matches[offset:offset + limit], len(matches)


# -------------------- TEAM STATS --------------------

def get_team_stats(team_id: int):
    """Compute weighted average goals for/against for recent matches."""
    url = f"{BASE_URL}teams/{team_id}/matches"
    params = {"status": "FINISHED", "limit": RECENT_MATCHES}
    data = safe_request(url, params=params)
    matches = data.get("matches", []) if data else []

    if not matches:
        return {"avg_gf": 1.0, "avg_ga": 1.0}

    # Sort by date ascending (oldest -> newest)
    matches.sort(key=lambda m: m["utcDate"])

    n = len(matches)
    weights = [0.5 + (i / (n - 1)) for i in range(n)] if n > 1 else [1.0]

    gf_sum, ga_sum, total_weight = 0.0, 0.0, sum(weights)
    for w, m in zip(weights, matches):
        home = m["homeTeam"]["id"] == team_id
        gf = m["score"]["fullTime"]["home"] if home else m["score"]["fullTime"]["away"]
        ga = m["score"]["fullTime"]["away"] if home else m["score"]["fullTime"]["home"]
        gf, ga = gf or 0, ga or 0
        gf_sum += gf * w
        ga_sum += ga * w

    return {
        "avg_gf": gf_sum / total_weight if total_weight else 1.0,
        "avg_ga": ga_sum / total_weight if total_weight else 1.0
    }


# -------------------- PREDICTION MODEL --------------------

def predict_outcome(home_stats, away_stats):
    """Calculate Poisson-based match outcome probabilities."""
    league_avg = max((home_stats["avg_gf"] + away_stats["avg_gf"]) / 2, 0.5)
    lambda_home = home_stats["avg_gf"] * (away_stats["avg_ga"] / league_avg)
    lambda_away = away_stats["avg_gf"] * (home_stats["avg_ga"] / league_avg)

    max_goals = 10
    home_win = draw = away_win = 0.0

    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            p = poisson.pmf(h, lambda_home) * poisson.pmf(a, lambda_away)
            if h > a:
                home_win += p
            elif h == a:
                draw += p
            else:
                away_win += p

    total = home_win + draw + away_win
    if total == 0:
        return {"1": 0.33, "X": 0.33, "2": 0.34}

    return {"1": home_win / total, "X": draw / total, "2": away_win / total}


def best_double_chance(preds):
    """Find best double chance and its probability."""
    p1, px, p2 = preds["1"], preds["X"], preds["2"]
    dc = {"1X": p1 + px, "12": p1 + p2, "2X": p2 + px}
    best = max(dc, key=dc.get)
    return best, dc[best]


# -------------------- DB UTILS --------------------

def store_to_db(conn, match, preds, best_dc, dc_prob, corners):
    """Insert or update prediction data in SQLite DB."""
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO matches (team_home, team_away, match_time, best_double_chance, best_dc_prob,
                             prediction, prob_1, prob_x, prob_2, corners_avg)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(team_home, team_away, match_time) DO UPDATE SET
            best_double_chance=excluded.best_double_chance,
            best_dc_prob=excluded.best_dc_prob,
            prediction=excluded.prediction,
            prob_1=excluded.prob_1,
            prob_x=excluded.prob_x,
            prob_2=excluded.prob_2,
            corners_avg=excluded.corners_avg
    """, (
        match['homeTeam']['name'],
        match['awayTeam']['name'],
        match['utcDate'],
        best_dc,
        dc_prob,
        max(preds, key=preds.get),
        preds["1"],
        preds["X"],
        preds["2"],
        corners
    ))
    conn.commit()


# -------------------- HELPERS --------------------

def countdown(seconds):
    """Show a visual countdown in terminal."""
    for remaining in range(seconds, 0, -1):
        print(f"\rWaiting {remaining}s before next batch...", end="", flush=True)
        time.sleep(1)
    print("\n")


def scrape_corners(team_home, team_away):
    """Placeholder: simulate expected corners."""
    return round(float(np.random.uniform(6, 12)), 1)


# -------------------- MAIN --------------------

def main():
    conn = sqlite3.connect("football_predictions.db")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS matches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            team_home TEXT,
            team_away TEXT,
            match_time TEXT,
            best_double_chance TEXT,
            best_dc_prob REAL,
            prediction TEXT,
            prob_1 REAL,
            prob_x REAL,
            prob_2 REAL,
            corners_avg REAL,
            UNIQUE(team_home, team_away, match_time)
        )
    """)

    competition_code = input("Enter competition code (e.g. 'PL', 'SA'): ").strip().upper()
    matches_batch, total_matches = get_competition_matches(competition_code, 0, BATCH_SIZE)
    if not matches_batch:
        print("No scheduled matches found.")
        return

    first_match_date = datetime.fromisoformat(matches_batch[0]['utcDate'].replace("Z", "+00:00")).date()
    week_end_date = first_match_date + timedelta(days=7)
    offset = 0

    while True:
        matches_batch, total_matches = get_competition_matches(competition_code, offset, BATCH_SIZE)
        if not matches_batch:
            break

        matches_batch = [
            m for m in matches_batch
            if first_match_date <= datetime.fromisoformat(m['utcDate'].replace("Z","+00:00")).date() < week_end_date
        ]
        if not matches_batch:
            break

        for match in matches_batch:
            home_stats = get_team_stats(match["homeTeam"]["id"])
            away_stats = get_team_stats(match["awayTeam"]["id"])

            preds = predict_outcome(home_stats, away_stats)
            best_dc, dc_prob = best_double_chance(preds)
            corners = scrape_corners(match["homeTeam"]["name"], match["awayTeam"]["name"])
            store_to_db(conn, match, preds, best_dc, dc_prob, corners)

            match_time = datetime.fromisoformat(match["utcDate"].replace("Z","+00:00")) + timedelta(hours=UTC_OFFSET)
            print(f"\n📅 {match_time.strftime('%Y-%m-%d %H:%M')} (UTC+{UTC_OFFSET}) - {match['homeTeam']['name']} vs {match['awayTeam']['name']}")
            print(f"   → 1: {preds['1']*100:.1f}%, X: {preds['X']*100:.1f}%, 2: {preds['2']*100:.1f}%")
            print(f"   🧠 Best Double Chance: {best_dc} ({dc_prob*100:.1f}%)")
            print(f"   ⚽ Corners avg: {corners}")

            countdown(RATE_LIMIT_WAIT)

        offset += BATCH_SIZE

    conn.close()


if __name__ == "__main__":
    main()
