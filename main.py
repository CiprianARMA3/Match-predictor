import requests
import sqlite3
import statistics
import numpy as np
from datetime import datetime, timedelta
import time

# -------------------- CONFIG --------------------
API_KEY = "29bd262893774e73a6f91593266e6a71"
BASE_URL = "https://api.football-data.org/v4/"
HEADERS = {"X-Auth-Token": API_KEY}
BATCH_SIZE = 10  
UTC_OFFSET = 2   # UTC+2

# -------------------- FUNCTIONS --------------------

def get_competition_matches(competition_code, offset=0, limit=10):
    url = f"{BASE_URL}competitions/{competition_code}/matches?status=SCHEDULED"
    res = requests.get(url, headers=HEADERS)
    data = res.json()
    matches = data.get("matches", [])
    return matches[offset:offset + limit], len(matches)

def get_team_stats(team_id):
    url = f"{BASE_URL}teams/{team_id}/matches?status=FINISHED&limit=10"
    res = requests.get(url, headers=HEADERS)
    data = res.json().get("matches", [])
    goals_for, goals_against, wins, draws, losses = [], [], 0, 0, 0

    for m in data:
        if m["homeTeam"]["id"] == team_id:
            gf = m["score"]["fullTime"]["home"]
            ga = m["score"]["fullTime"]["away"]
        else:
            gf = m["score"]["fullTime"]["away"]
            ga = m["score"]["fullTime"]["home"]

        goals_for.append(gf or 0)
        goals_against.append(ga or 0)
        if gf > ga:
            wins += 1
        elif gf == ga:
            draws += 1
        else:
            losses += 1

    avg_gf = statistics.mean(goals_for) if goals_for else 1
    avg_ga = statistics.mean(goals_against) if goals_against else 1
    return avg_gf, avg_ga, wins, draws, losses

def predict_outcome(home_stats, away_stats):
    home_strength = home_stats[0] - away_stats[1]
    away_strength = away_stats[0] - home_stats[1]
    draw_strength = abs(home_strength - away_strength)

    probs = np.array([
        max(0.1, home_strength / (abs(home_strength) + abs(away_strength) + 1)),
        max(0.1, draw_strength / (abs(home_strength) + abs(away_strength) + 1)),
        max(0.1, away_strength / (abs(home_strength) + abs(away_strength) + 1))
    ])
    probs = np.clip(probs, 0, 1)
    probs /= probs.sum()

    return {"1": probs[0], "X": probs[1], "2": probs[2]}

def best_double_chance(preds):
    p1, px, p2 = preds["1"], preds["X"], preds["2"]
    dc = {
        "1X": p1 + px,
        "12": p1 + p2,
        "2X": p2 + px
    }
    best = max(dc, key=dc.get)
    return best, dc[best]

def countdown(seconds):
    for remaining in range(seconds, 0, -1):
        print(f"\rWaiting {remaining} seconds before next batch...", end="", flush=True)
        time.sleep(1)
    print("\nStarting next batch!")

def scrape_corners(team_home, team_away):
    # Simulated corners (replace with real scraping if needed)
    return round(np.random.uniform(6, 12), 1)

def store_to_db(conn, match, preds, best_dc, dc_prob, corners):
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO matches (team_home, team_away, match_time, best_double_chance, best_dc_prob,
                             prediction, prob_1, prob_x, prob_2, corners_avg)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        match['homeTeam']['name'],
        match['awayTeam']['name'],
        match['utcDate'],  # store original UTC time
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
            corners_avg REAL
        )
    """)

    competition_code = input("Enter competition code (e.g. 'SA' for Serie A, 'PL' for Premier League): ").strip()
    
    # Fetch first batch to determine current week
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

        # Only process matches within the current week
        matches_batch = [
            m for m in matches_batch
            if first_match_date <= datetime.fromisoformat(m['utcDate'].replace("Z", "+00:00")).date() < week_end_date
        ]

        if not matches_batch:
            print("\nEnd of current round/week reached. Exiting...")
            break

        for match in matches_batch:
            home_stats = get_team_stats(match["homeTeam"]["id"])
            away_stats = get_team_stats(match["awayTeam"]["id"])
            preds = predict_outcome(home_stats, away_stats)
            best_dc, dc_prob = best_double_chance(preds)
            corners = scrape_corners(match["homeTeam"]["name"], match["awayTeam"]["name"])
            store_to_db(conn, match, preds, best_dc, dc_prob, corners)

            # Adjust UTC+2
            match_time = datetime.fromisoformat(match["utcDate"].replace("Z", "+00:00")) + timedelta(hours=UTC_OFFSET)
            print(f"\n📅 {match_time.strftime('%Y-%m-%d %H:%M')} (UTC+2) - {match['homeTeam']['name']} vs {match['awayTeam']['name']}")
            print(f"   → 1: {preds['1']*100:.1f}%, X: {preds['X']*100:.1f}%, 2: {preds['2']*100:.1f}%")
            print(f"   🧠 Best Double Chance: {best_dc} ({dc_prob*100:.1f}%)")
            print(f"   ⚽ Corners avg: {corners}")

            match_counter += 1
            if match_counter % 2 == 0:
                countdown(60)  # Countdown after every 2 matches

        offset += BATCH_SIZE

        # Check if more matches exist in the current week
        remaining_matches = [
            m for m in get_competition_matches(competition_code, offset=offset, limit=BATCH_SIZE)[0]
            if first_match_date <= datetime.fromisoformat(m['utcDate'].replace("Z", "+00:00")).date() < week_end_date
        ]
        if not remaining_matches:
            print("\nEnd of current round/week reached. Exiting...")
            break

    conn.close()

if __name__ == "__main__":
    main()
