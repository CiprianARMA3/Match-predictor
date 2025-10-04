#!/usr/bin/env python3
import os
import time
import sqlite3
import requests
import numpy as np
from datetime import datetime, timedelta
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
RATE_LIMIT_WAIT = 5  # Hardcoded wait between API calls
UTC_OFFSET = 2       # Local time adjustment

session = requests.Session()
session.headers.update(HEADERS)

# -------------------- API UTILS --------------------

def safe_request(url: str, params: dict | None = None) -> dict | None:
    """HTTP GET with fixed wait time."""
    try:
        res = session.get(url, params=params, timeout=15)
        if res.status_code == 200:
            time.sleep(RATE_LIMIT_WAIT)
            return res.json()
        print(f"⚠️ HTTP {res.status_code} for URL {url}. Skipping.")
        time.sleep(RATE_LIMIT_WAIT)
        return None
    except requests.RequestException as e:
        print(f"Network error: {e}. Skipping.")
        time.sleep(RATE_LIMIT_WAIT)
        return None

def get_competition_matches(competition_code: str, offset=0, limit=10):
    url = f"{BASE_URL}competitions/{competition_code}/matches"
    params = {"status": "SCHEDULED"}
    data = safe_request(url, params=params)
    matches = data.get("matches", []) if data else []
    return matches[offset:offset + limit], len(matches)

# -------------------- TEAM STATS --------------------

def get_team_stats(team_id: int):
    url = f"{BASE_URL}teams/{team_id}/matches"
    params = {"status": "FINISHED", "limit": RECENT_MATCHES}
    data = safe_request(url, params=params)
    matches = data.get("matches", []) if data else []

    if not matches:
        return {"avg_gf_home":1.0, "avg_ga_home":1.0, "avg_gf_away":1.0, "avg_ga_away":1.0}

    matches.sort(key=lambda m: m["utcDate"])
    n = len(matches)
    weights = [0.5 + (i/(n-1)) for i in range(n)] if n > 1 else [1.0]

    gf_home=ga_home=gf_away=ga_away=total_weight_home=total_weight_away=0
    for w,m in zip(weights,matches):
        home = m["homeTeam"]["id"]==team_id
        gf = m["score"]["fullTime"]["home"] if home else m["score"]["fullTime"]["away"]
        ga = m["score"]["fullTime"]["away"] if home else m["score"]["fullTime"]["home"]
        gf,ga = gf or 0, ga or 0
        if home:
            gf_home += gf*w
            ga_home += ga*w
            total_weight_home += w
        else:
            gf_away += gf*w
            ga_away += ga*w
            total_weight_away += w
    return {
        "avg_gf_home": gf_home/total_weight_home if total_weight_home else 1.0,
        "avg_ga_home": ga_home/total_weight_home if total_weight_home else 1.0,
        "avg_gf_away": gf_away/total_weight_away if total_weight_away else 1.0,
        "avg_ga_away": ga_away/total_weight_away if total_weight_away else 1.0
    }

def get_head_to_head(home_id, away_id, matches=10):
    url = f"{BASE_URL}teams/{home_id}/matches"
    params = {"status":"FINISHED","limit":matches}
    data = safe_request(url, params=params)
    if not data:
        return {"home_win":0.33,"draw":0.34,"away_win":0.33}
    
    h_wins=d_wins=a_wins=0
    for m in data.get("matches",[]):
        if m["homeTeam"]["id"]==home_id and m["awayTeam"]["id"]==away_id:
            home_score = m["score"]["fullTime"]["home"] or 0
            away_score = m["score"]["fullTime"]["away"] or 0
            if home_score>away_score: h_wins+=1
            elif home_score==away_score: d_wins+=1
            else: a_wins+=1
    total = h_wins+d_wins+a_wins or 1
    return {"home_win":h_wins/total,"draw":d_wins/total,"away_win":a_wins/total}

# -------------------- PREDICTION MODEL --------------------

def monte_carlo_poisson(lambda_home, lambda_away, simulations=10000):
    outcomes = {"1":0,"X":0,"2":0}
    for _ in range(simulations):
        h_goals = np.random.poisson(lambda_home)
        a_goals = np.random.poisson(lambda_away)
        if h_goals > a_goals: outcomes["1"]+=1
        elif h_goals==a_goals: outcomes["X"]+=1
        else: outcomes["2"]+=1
    total = sum(outcomes.values())
    return {k:v/total for k,v in outcomes.items()}

def predict_outcome(home_stats, away_stats, home_adv=1.1):
    # Scale lambda by league-average to prevent skewed probabilities
    league_avg = max((home_stats["avg_gf_home"] + away_stats["avg_gf_away"])/2,0.5)
    lambda_home = home_stats["avg_gf_home"] * (away_stats["avg_ga_away"]/league_avg) * home_adv
    lambda_away = away_stats["avg_gf_away"] * (home_stats["avg_ga_home"]/league_avg)
    return monte_carlo_poisson(lambda_home, lambda_away)

def best_double_chance(preds):
    p1,px,p2 = preds["1"], preds["X"], preds["2"]
    dc = {"1X":p1+px, "12":p1+p2, "2X":p2+px}
    best = max(dc,key=dc.get)
    return best, dc[best]

# -------------------- CORNERS --------------------

def estimate_corners(home_id, away_id):
    home_stats = get_team_stats(home_id)
    away_stats = get_team_stats(away_id)
    # Scaled formula for realistic corners
    home_corners = home_stats["avg_gf_home"]*3 + home_stats["avg_ga_home"]*2 + 2
    away_corners = away_stats["avg_gf_away"]*3 + away_stats["avg_ga_away"]*2 + 2
    return round((home_corners + away_corners)/2, 1)

# -------------------- DATABASE --------------------

def store_to_db(conn, match, preds, best_dc, dc_prob, corners):
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
        max(preds,key=preds.get),
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

    competition_code = input("Enter competition code (e.g. 'PL', 'SA'): ").strip().upper()
    offset = 0

    while True:
        matches_batch, total_matches = get_competition_matches(competition_code, offset, BATCH_SIZE)
        if not matches_batch:
            break

        for match in matches_batch:
            home_id = match["homeTeam"]["id"]
            away_id = match["awayTeam"]["id"]

            home_stats = get_team_stats(home_id)
            away_stats = get_team_stats(away_id)
            h2h = get_head_to_head(home_id, away_id)

            preds = predict_outcome(home_stats, away_stats)
            # Blend 80% statistical + 20% head-to-head
            preds["1"] = 0.8*preds["1"] + 0.2*h2h["home_win"]
            preds["X"] = 0.8*preds["X"] + 0.2*h2h["draw"]
            preds["2"] = 0.8*preds["2"] + 0.2*h2h["away_win"]

            best_dc, dc_prob = best_double_chance(preds)
            corners = estimate_corners(home_id, away_id)
            store_to_db(conn, match, preds, best_dc, dc_prob, corners)

            match_time = datetime.fromisoformat(match["utcDate"].replace("Z","+00:00")) + timedelta(hours=UTC_OFFSET)
            print(f"\n📅 {match_time.strftime('%Y-%m-%d %H:%M')} - {match['homeTeam']['name']} vs {match['awayTeam']['name']}")
            print(f"   → 1: {preds['1']*100:.1f}%, X: {preds['X']*100:.1f}%, 2: {preds['2']*100:.1f}%")
            print(f"   🧠 Best Double Chance: {best_dc} ({dc_prob*100:.1f}%)")
            print(f"   ⚽ Estimated Corners: {corners}")

            time.sleep(RATE_LIMIT_WAIT)  # hardcoded wait

        offset += BATCH_SIZE

    conn.close()

if __name__ == "__main__":
    main()
