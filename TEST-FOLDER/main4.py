#!/usr/bin/env python3 improved ma non ancora perfetto
import os
import time
import sqlite3
import requests
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
from datetime import timezone

# -------------------- CONFIG --------------------
load_dotenv()
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise SystemExit("❌ Missing API_KEY in environment (.env file).")

BASE_URL = "https://api.football-data.org/v4/"
HEADERS = {"X-Auth-Token": API_KEY}
BATCH_SIZE = 10
RECENT_MATCHES = 10
RATE_LIMIT_WAIT = 5
UTC_OFFSET = 2
K_FACTOR = 20  # Elo sensitivity

session = requests.Session()
session.headers.update(HEADERS)

# -------------------- API UTILITIES --------------------
def safe_request(url: str, params: dict | None = None):
    try:
        res = session.get(url, params=params, timeout=15)
        if res.status_code == 200:
            time.sleep(RATE_LIMIT_WAIT)
            return res.json()
        print(f"⚠️ HTTP {res.status_code} for {url}. Skipping.")
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
    return matches[offset:offset+limit], len(matches)

# -------------------- TEAM STATISTICS --------------------
def get_team_stats(team_id: int):
    """
    Fetch and compute team stats using only:
      - matches from the current season (preferred)
      - OR matches from the last 365 days if the season is new or data is sparse
    """
    from datetime import timezone

    url = f"{BASE_URL}teams/{team_id}/matches"
    params = {"status": "FINISHED", "limit": 50}
    data = safe_request(url, params=params)
    matches = data.get("matches", []) if data else []

    if not matches:
        return {"gf_home":1.0,"ga_home":1.0,"gf_away":1.0,"ga_away":1.0}

    # Time anchors — timezone-aware (UTC)
    now = datetime.now(timezone.utc)
    one_year_ago = now - timedelta(days=365)

    # Determine current season start: earliest scheduled match in same competition
    season_start = None
    try:
        comp_id = matches[0]["competition"]["id"]
        comp_url = f"{BASE_URL}competitions/{comp_id}/matches"
        comp_data = safe_request(comp_url, params={"status": "SCHEDULED"})
        scheduled = comp_data.get("matches", []) if comp_data else []
        if scheduled:
            first_match_date = min(datetime.fromisoformat(m["utcDate"].replace("Z", "+00:00")) for m in scheduled)
            season_start = first_match_date - timedelta(days=7)  # small buffer before season start
    except Exception:
        pass

    # Filter matches: current season preferred, else last 1 year
    filtered = []
    for m in matches:
        m_date = datetime.fromisoformat(m["utcDate"].replace("Z", "+00:00"))
        if (season_start and m_date >= season_start) or (m_date >= one_year_ago):
            filtered.append(m)

    if not filtered:
        filtered = matches[-RECENT_MATCHES:]  # fallback to most recent N matches

    # Sort chronologically
    filtered.sort(key=lambda m: m["utcDate"])
    n = len(filtered)

    # Exponential decay weighting (recent matches matter more)
    decay = 0.8
    weights = np.array([decay**(n-1-i) for i in range(n)])
    weights /= weights.sum()

    gf_home = ga_home = gf_away = ga_away = 0
    w_home = w_away = 0

    for w, m in zip(weights, filtered):
        home = m["homeTeam"]["id"] == team_id
        gf = m["score"]["fullTime"]["home"] if home else m["score"]["fullTime"]["away"]
        ga = m["score"]["fullTime"]["away"] if home else m["score"]["fullTime"]["home"]
        gf, ga = gf or 0, ga or 0

        if home:
            gf_home += gf * w
            ga_home += ga * w
            w_home += w
        else:
            gf_away += gf * w
            ga_away += ga * w
            w_away += w

    return {
        "gf_home": gf_home / w_home if w_home else 1.0,
        "ga_home": ga_home / w_home if w_home else 1.0,
        "gf_away": gf_away / w_away if w_away else 1.0,
        "ga_away": ga_away / w_away if w_away else 1.0,
    }


# -------------------- HEAD TO HEAD --------------------
def get_head_to_head(home_id, away_id, matches=10):
    url = f"{BASE_URL}teams/{home_id}/matches"
    params = {"status":"FINISHED","limit":matches}
    data = safe_request(url, params=params)
    if not data:
        return {"home_win":0.33,"draw":0.34,"away_win":0.33}

    h_win=d_win=a_win=0
    for m in data.get("matches",[]):
        if (m["homeTeam"]["id"]==home_id and m["awayTeam"]["id"]==away_id) or \
           (m["homeTeam"]["id"]==away_id and m["awayTeam"]["id"]==home_id):
            home_score = m["score"]["fullTime"]["home"] or 0
            away_score = m["score"]["fullTime"]["away"] or 0
            if home_score>away_score:
                h_win +=1 if m["homeTeam"]["id"]==home_id else 0
                a_win +=1 if m["homeTeam"]["id"]==away_id else 0
            elif home_score==away_score:
                d_win +=1
    total = h_win+d_win+a_win or 1
    return {"home_win":h_win/total,"draw":d_win/total,"away_win":a_win/total}

# -------------------- ELO SYSTEM --------------------
team_elos = {}

def get_elo(team_id, default=1500):
    return team_elos.get(team_id, default)

def set_elo(team_id, rating):
    team_elos[team_id] = rating

def update_elo(home_id, away_id, home_goals, away_goals):
    home_elo = get_elo(home_id)
    away_elo = get_elo(away_id)

    expected_home = 1 / (1 + 10 ** ((away_elo - home_elo)/400))
    expected_away = 1 - expected_home

    if home_goals > away_goals:
        score_home, score_away = 1, 0
    elif home_goals == away_goals:
        score_home, score_away = 0.5, 0.5
    else:
        score_home, score_away = 0, 1

    home_elo += K_FACTOR * (score_home - expected_home)
    away_elo += K_FACTOR * (score_away - expected_away)

    set_elo(home_id, home_elo)
    set_elo(away_id, away_elo)

# -------------------- MONTE CARLO (VECTORIZED) --------------------
def monte_carlo(lambda_home, lambda_away, sims=10000):
    goals = np.random.poisson([lambda_home, lambda_away], size=(sims, 2))
    h = goals[:,0]
    a = goals[:,1]
    outcomes = {
        "1": np.mean(h > a),
        "X": np.mean(h == a),
        "2": np.mean(h < a)
    }
    return outcomes

# -------------------- PREDICTION --------------------
def predict(home_stats, away_stats, home_id, away_id, home_adv=1.1):
    league_avg = max((home_stats["gf_home"] + away_stats["gf_away"]) / 2, 0.5)
    lambda_home = home_stats["gf_home"] * (away_stats["ga_away"]/league_avg) * home_adv
    lambda_away = away_stats["gf_away"] * (home_stats["ga_home"]/league_avg)

    # Elo adjustment
    home_elo = get_elo(home_id)
    away_elo = get_elo(away_id)
    elo_diff = home_elo - away_elo
    elo_factor = 10 ** (elo_diff/400)
    lambda_home *= elo_factor/(elo_factor+1)
    lambda_away *= 1/(elo_factor+1)

    return monte_carlo(lambda_home, lambda_away)

def blend_predictions(pred_mc, pred_h2h, alpha=0.8):
    """Normalize weights so total stays 1.0"""
    alpha = np.clip(alpha, 0, 1)
    blended = {k: alpha*pred_mc[k] + (1-alpha)*pred_h2h[k2] 
               for k,k2 in zip(["1","X","2"], ["home_win","draw","away_win"])}
    total = sum(blended.values())
    return {k: v/total for k,v in blended.items()}

def best_double_chance(preds):
    p1, px, p2 = preds["1"], preds["X"], preds["2"]
    dc = {"1X":p1+px, "12":p1+p2, "2X":p2+px}
    best = max(dc,key=dc.get)
    return best, dc[best]

# -------------------- CORNERS --------------------
def estimate_corners(home_id, away_id):
    h_stats = get_team_stats(home_id)
    a_stats = get_team_stats(away_id)
    h_c = h_stats["gf_home"]*3 + h_stats["ga_home"]*2 + 2
    a_c = a_stats["gf_away"]*3 + a_stats["ga_away"]*2 + 2
    return round((h_c+a_c)/2,1)

# -------------------- DATABASE --------------------
def store_db(conn, match, preds, best_dc, dc_prob, corners):
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO matches (team_home, team_away, match_time, best_double_chance, best_dc_prob,
                             prediction, prob_1, prob_x, prob_2, corners_avg)
        VALUES (?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(team_home, team_away, match_time) DO UPDATE SET
            best_double_chance=excluded.best_double_chance,
            best_dc_prob=excluded.best_dc_prob,
            prediction=excluded.prediction,
            prob_1=excluded.prob_1,
            prob_x=excluded.prob_x,
            prob_2=excluded.prob_2,
            corners_avg=excluded.corners_avg
    """, (
        match["homeTeam"]["name"],
        match["awayTeam"]["name"],
        match["utcDate"],
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

    comp = input("Enter competition code (PL, SA, etc.): ").strip().upper()
    offset = 0

    while True:
        matches, total = get_competition_matches(comp, offset, BATCH_SIZE)
        if not matches:
            break

        for m in matches:
            h_id = m["homeTeam"]["id"]
            a_id = m["awayTeam"]["id"]

            h_stats = get_team_stats(h_id)
            a_stats = get_team_stats(a_id)
            h2h = get_head_to_head(h_id, a_id)

            preds_mc = predict(h_stats, a_stats, h_id, a_id)
            preds = blend_predictions(preds_mc, h2h, alpha=0.8)

            best_dc, dc_prob = best_double_chance(preds)
            corners = estimate_corners(h_id, a_id)
            store_db(conn, m, preds, best_dc, dc_prob, corners)

            match_time = datetime.fromisoformat(m["utcDate"].replace("Z","+00:00")) + timedelta(hours=UTC_OFFSET)
            print(f"\n📅 {match_time.strftime('%Y-%m-%d %H:%M')} - {m['homeTeam']['name']} vs {m['awayTeam']['name']}")
            print(f"   → 1: {preds['1']*100:.1f}%, X: {preds['X']*100:.1f}%, 2: {preds['2']*100:.1f}%")
            print(f"   🧠 Best Double Chance: {best_dc} ({dc_prob*100:.1f}%)")
            print(f"   ⚽ Estimated Corners: {corners}")

            time.sleep(RATE_LIMIT_WAIT)

        offset += BATCH_SIZE

    conn.close()

if __name__ == "__main__":
    main()
