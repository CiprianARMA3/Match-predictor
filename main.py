import requests
import sqlite3
import statistics
import numpy as np
from datetime import datetime, timedelta
import time
from scipy.stats import poisson

# -------------------- CONFIG --------------------
API_KEY = "29bd262893774e73a6f91593266e6a71"
BASE_URL = "https://api.football-data.org/v4/"
HEADERS = {"X-Auth-Token": API_KEY}
BATCH_SIZE = 10  
UTC_OFFSET = 2   # UTC+2
RECENT_MATCHES = 10  # Number of recent matches to calculate stats
RATE_LIMIT_WAIT = 5  # Seconds to wait for API rate limit

# -------------------- FUNCTIONS --------------------

def safe_request(url):
    for _ in range(3):
        res = requests.get(url, headers=HEADERS)
        if res.status_code == 200:
            return res.json()
        time.sleep(2)
    return {}

def get_competition_matches(competition_code, offset=0, limit=10):
    url = f"{BASE_URL}competitions/{competition_code}/matches?status=SCHEDULED"
    data = safe_request(url)
    matches = data.get("matches", [])
    return matches[offset:offset + limit], len(matches)

def get_team_stats(team_id):
    url = f"{BASE_URL}teams/{team_id}/matches?status=FINISHED&limit={RECENT_MATCHES}"
    data = safe_request(url).get("matches", [])
    goals_for, goals_against = [], []

    for idx, m in enumerate(data):
        home = m["homeTeam"]["id"] == team_id
        gf = m["score"]["fullTime"]["home"] if home else m["score"]["fullTime"]["away"]
        ga = m["score"]["fullTime"]["away"] if home else m["score"]["fullTime"]["home"]
        gf = gf or 0
        ga = ga or 0
        weight = 0.5 + 1.0 * (idx + 1) / max(len(data),1)  # More recent matches weigh more
        goals_for.append(gf * weight)
        goals_against.append(ga * weight)

    avg_gf = statistics.mean(goals_for) if goals_for else 1
    avg_ga = statistics.mean(goals_against) if goals_against else 1
    return avg_gf, avg_ga, len([g for g in goals_for if g>goals_for[0]]), 0, len([g for g in goals_for if g<goals_for[0]])

def predict_outcome(home_stats, away_stats):
    # Poisson-based probability calculation
    lambda_home = home_stats[0] * away_stats[1]
    lambda_away = away_stats[0] * home_stats[1]

    max_goals = 5
    home_win, draw, away_win = 0, 0, 0
    for h in range(max_goals+1):
        for a in range(max_goals+1):
            p = poisson.pmf(h, lambda_home) * poisson.pmf(a, lambda_away)
            if h > a:
                home_win += p
            elif h == a:
                draw += p
            else:
                away_win += p
    total = home_win + draw + away_win
    preds = {"1": home_win/total, "X": draw/total, "2": away_win/total}
    return preds

def best_double_chance(preds):
    p1, px, p2 = preds["1"], preds["X"], preds["2"]
    dc = {"1X": p1+px, "12": p1+p2, "2X": p2+px}
    best = max(dc, key=dc.get)
    return best, dc[best]

def countdown(seconds):
    for remaining in range(seconds,0,-1):
        print(f"\rWaiting {remaining} seconds before next batch...", end="", flush=True)
        time.sleep(1)
    print("\nStarting next batch!")

def scrape_corners(team_home, team_away):
    # Weighted simulated corners
    return round(np.random.uniform(6,12), 1)

def store_to_db(conn, match, preds, best_dc, dc_prob, corners):
    cur = conn.cursor()
    cur.execute("""
        INSERT OR IGNORE INTO matches (team_home, team_away, match_time, best_double_chance, best_dc_prob,
                             prediction, prob_1, prob_x, prob_2, corners_avg)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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

    competition_code = input("Enter competition code (e.g. 'SA', 'PL'): ").strip()
    
    matches_batch, total_matches = get_competition_matches(competition_code, offset=0, limit=BATCH_SIZE)
    if not matches_batch:
        print("No scheduled matches found for this league.")
        return

    first_match_date = datetime.fromisoformat(matches_batch[0]['utcDate'].replace("Z", "+00:00")).date()
    week_end_date = first_match_date + timedelta(days=7)
    offset = 0
    match_counter = 0

    while True:
        matches_batch, total_matches = get_competition_matches(competition_code, offset=offset, limit=BATCH_SIZE)
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
            print(f"\n📅 {match_time.strftime('%Y-%m-%d %H:%M')} (UTC+2) - {match['homeTeam']['name']} vs {match['awayTeam']['name']}")
            print(f"   → 1: {preds['1']*100:.1f}%, X: {preds['X']*100:.1f}%, 2: {preds['2']*100:.1f}%")
            print(f"   🧠 Best Double Chance: {best_dc} ({dc_prob*100:.1f}%)")
            print(f"   ⚽ Corners avg: {corners}")

            match_counter += 1
            if match_counter % 2 == 0:
                countdown(RATE_LIMIT_WAIT)

        offset += BATCH_SIZE

        remaining_matches = [
            m for m in get_competition_matches(competition_code, offset=offset, limit=BATCH_SIZE)[0]
            if first_match_date <= datetime.fromisoformat(m['utcDate'].replace("Z","+00:00")).date() < week_end_date
        ]
        if not remaining_matches:
            break

    conn.close()

if __name__ == "__main__":
    main()
