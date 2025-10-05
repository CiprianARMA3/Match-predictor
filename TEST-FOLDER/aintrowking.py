#!/usr/bin/env python3
"""
Rigorous Football Predictor
- Elo with goal-diff multiplier and home advantage
- Home/away attack/defense stats with exponential weighting
- Expected goals using multiplicative model + Elo
- Monte Carlo simulation (vectorized, reproducible)
- Head-to-head blending
- Corner estimation
- SQLite persistence
"""

import os
import time
import math
import sqlite3
import logging
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from typing import Dict, Optional, Tuple

import requests
import numpy as np
from dotenv import load_dotenv

# -------------------- CONFIG --------------------
load_dotenv()
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise SystemExit("❌ Missing API_KEY in environment (.env file).")

BASE_URL = "https://api.football-data.org/v4/"
HEADERS = {"X-Auth-Token": API_KEY}
RATE_LIMIT = 1.0
RECENT_MATCHES = 50
DECAY = 0.85
MIN_MATCHES = 6
K_FACTOR = 24
HOME_ADV = 35
MC_SIMS = 15000
RANDOM_SEED = 42
DB_PATH = "football_predictions.db"
UTC_DISPLAY_ZONE = ZoneInfo("Europe/Rome")

# -------------------- LOGGING --------------------
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-5s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("predictor")

# -------------------- API CLIENT --------------------
session = requests.Session()
session.headers.update(HEADERS)


def safe_request(url: str, params: Optional[dict] = None, retry: int = 2):
    for _ in range(retry):
        try:
            res = session.get(url, params=params, timeout=15)
            if res.status_code == 200:
                time.sleep(RATE_LIMIT)
                return res.json()
            elif res.status_code == 429:
                wait = int(res.headers.get("Retry-After", 5))
                logger.warning("429 rate limit, sleeping %ds", wait)
                time.sleep(wait)
            else:
                logger.warning("HTTP %d for %s", res.status_code, url)
                time.sleep(RATE_LIMIT)
        except requests.RequestException as e:
            logger.warning("Network error: %s", e)
            time.sleep(RATE_LIMIT)
    return None


def get_competition_matches(competition: str, status="SCHEDULED"):
    url = f"{BASE_URL}competitions/{competition}/matches"
    data = safe_request(url, params={"status": status})
    return data.get("matches", []) if data else []


def get_team_matches(team_id: int, status="FINISHED", limit=RECENT_MATCHES):
    url = f"{BASE_URL}teams/{team_id}/matches"
    data = safe_request(url, params={"status": status, "limit": limit})
    return data.get("matches", []) if data else []


# -------------------- DATABASE --------------------
class DBManager:
    def __init__(self, path=DB_PATH):
        self.conn = sqlite3.connect(path)
        self._init_schema()

    def _init_schema(self):
        cur = self.conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS matches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            home_team TEXT,
            away_team TEXT,
            match_time TEXT,
            best_double_chance TEXT,
            best_dc_prob REAL,
            prediction TEXT,
            prob_1 REAL,
            prob_x REAL,
            prob_2 REAL,
            corners_avg REAL,
            UNIQUE(home_team, away_team, match_time)
        )
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS elos (
            team_id INTEGER PRIMARY KEY,
            rating REAL,
            last_updated TEXT
        )
        """)
        self.conn.commit()

    def get_elo(self, team_id: int, default=1500):
        cur = self.conn.cursor()
        cur.execute("SELECT rating FROM elos WHERE team_id=?", (team_id,))
        r = cur.fetchone()
        return float(r[0]) if r else default

    def set_elo(self, team_id: int, rating: float):
        cur = self.conn.cursor()
        cur.execute("""
        INSERT INTO elos(team_id,rating,last_updated)
        VALUES(?,?,?)
        ON CONFLICT(team_id) DO UPDATE SET rating=excluded.rating,last_updated=excluded.last_updated
        """, (team_id, rating, datetime.utcnow().isoformat()))
        self.conn.commit()

    def store_match(self, match, preds: dict, best_dc: str, dc_prob: float, corners: float):
        cur = self.conn.cursor()
        cur.execute("""
        INSERT INTO matches (home_team, away_team, match_time, best_double_chance, best_dc_prob,
                             prediction, prob_1, prob_x, prob_2, corners_avg)
        VALUES (?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(home_team, away_team, match_time) DO UPDATE SET
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
            max(preds, key=preds.get),
            preds["1"],
            preds["X"],
            preds["2"],
            corners
        ))
        self.conn.commit()

    def close(self):
        self.conn.close()


# -------------------- ELO SYSTEM --------------------
class EloManager:
    def __init__(self, db: DBManager, k_factor=K_FACTOR, home_adv=HOME_ADV):
        self.db = db
        self.k = k_factor
        self.home_adv = home_adv

    def get(self, team_id: int):
        return self.db.get_elo(team_id)

    def set(self, team_id: int, rating: float):
        self.db.set_elo(team_id, rating)

    def update(self, home_id: int, away_id: int, home_goals: int, away_goals: int, importance=1.0):
        home_elo = self.get(home_id)
        away_elo = self.get(away_id)
        adj_home = home_elo + self.home_adv

        expected_home = 1 / (1 + 10 ** ((away_elo - adj_home)/400))
        expected_away = 1 - expected_home

        if home_goals > away_goals:
            s_home, s_away = 1, 0
        elif home_goals == away_goals:
            s_home, s_away = 0.5, 0.5
        else:
            s_home, s_away = 0, 1

        gd = max(1, abs(home_goals - away_goals))
        multiplier = 1 + math.log(gd, 2)

        k = self.k * importance * multiplier
        new_home = home_elo + k * (s_home - expected_home)
        new_away = away_elo + k * (s_away - expected_away)

        self.set(home_id, new_home)
        self.set(away_id, new_away)


# -------------------- TEAM STATS --------------------
def compute_team_stats(team_id: int):
    matches = get_team_matches(team_id)
    if not matches:
        return {"att_home":1.2,"def_home":1.2,"att_away":1.0,"def_away":1.0}

    matches.sort(key=lambda m: m["utcDate"])
    n = len(matches)
    weights = np.array([DECAY**(n-1-i) for i in range(n)])
    weights /= weights.sum()

    gf_h = ga_h = gf_a = ga_a = 0
    w_h = w_a = 0

    for w, m in zip(weights, matches):
        home = m["homeTeam"]["id"] == team_id
        hf = m["score"]["fullTime"]["home"] or 0
        af = m["score"]["fullTime"]["away"] or 0
        if home:
            gf_h += hf * w
            ga_h += af * w
            w_h += w
        else:
            gf_a += af * w
            ga_a += hf * w
            w_a += w

    alpha = max(MIN_MATCHES/2, 1)
    att_home = (gf_h + alpha*1.4) / (w_h + alpha)
    def_home = (ga_h + alpha*1.4) / (w_h + alpha)
    att_away = (gf_a + alpha*1.1) / (w_a + alpha)
    def_away = (ga_a + alpha*1.1) / (w_a + alpha)

    return {"att_home":att_home, "def_home":def_home, "att_away":att_away, "def_away":def_away}


# -------------------- HEAD-TO-HEAD --------------------
def get_head_to_head(home_id, away_id, limit=50):
    home_matches = get_team_matches(home_id, limit=limit)
    away_matches = get_team_matches(away_id, limit=limit)

    relevant = []
    seen = set()
    for m in home_matches + away_matches:
        hid = m["homeTeam"]["id"]
        aid = m["awayTeam"]["id"]
        if {hid, aid} == {home_id, away_id}:
            key = m.get("id") or (m["utcDate"], hid, aid)
            if key not in seen:
                seen.add(key)
                relevant.append(m)

    wins_h = wins_a = draws = 0
    for m in relevant:
        hf = m["score"]["fullTime"]["home"] or 0
        af = m["score"]["fullTime"]["away"] or 0
        if hf > af:
            if m["homeTeam"]["id"] == home_id:
                wins_h += 1
            else:
                wins_a += 1
        elif hf < af:
            if m["homeTeam"]["id"] == away_id:
                wins_a += 1
            else:
                wins_h += 1
        else:
            draws += 1

    total = max(1, wins_h + draws + wins_a)
    return {"home_win": wins_h/total, "draw": draws/total, "away_win": wins_a/total}


# -------------------- PREDICTOR --------------------
def expected_goals(home_stats, away_stats, home_id, away_id, elo_mgr: EloManager, home_adv=1.08):
    baseline = np.mean([
        home_stats["att_home"], home_stats["def_home"],
        home_stats["att_away"], home_stats["def_away"],
        away_stats["att_home"], away_stats["def_home"],
        away_stats["att_away"], away_stats["def_away"]
    ])

    elo_h = elo_mgr.get(home_id)
    elo_a = elo_mgr.get(away_id)
    elo_diff = elo_h - elo_a
    elo_factor_h = 1 + elo_diff/4000
    elo_factor_a = 1 - elo_diff/4000

    lambda_h = baseline * (home_stats["att_home"]/away_stats["def_away"]) * home_adv * elo_factor_h
    lambda_a = baseline * (away_stats["att_away"]/home_stats["def_home"]) * (1/home_adv) * elo_factor_a

    return max(lambda_h,0.05), max(lambda_a,0.05)


def monte_carlo(lambda_h, lambda_a, sims=MC_SIMS, rng=None):
    rng = rng or np.random.default_rng(RANDOM_SEED)
    goals = rng.poisson([lambda_h, lambda_a], size=(sims,2))
    h, a = goals[:,0], goals[:,1]
    return {"1": np.mean(h>a), "X": np.mean(h==a), "2": np.mean(h<a)}


def blend_predictions(mc, h2h, alpha=0.85):
    alpha = np.clip(alpha,0,1)
    blended = {
        "1": alpha*mc["1"] + (1-alpha)*h2h["home_win"],
        "X": alpha*mc["X"] + (1-alpha)*h2h["draw"],
        "2": alpha*mc["2"] + (1-alpha)*h2h["away_win"]
    }
    total = sum(blended.values())
    return {k: v/total for k,v in blended.items()}


def best_double_chance(preds):
    dc = {"1X": preds["1"]+preds["X"], "12": preds["1"]+preds["2"], "2X": preds["2"]+preds["X"]}
    best = max(dc, key=dc.get)
    return best, dc[best]


def estimate_corners(home_stats, away_stats):
    est = (home_stats["att_home"]*2.4 + home_stats["def_home"]*0.6 +
           away_stats["att_away"]*2.0 + away_stats["def_away"]*0.5)/2
    return round(max(6.0, est),1)


# -------------------- MAIN --------------------
def main():
    db = DBManager()
    elo_mgr = EloManager(db)

    comp = input("Enter competition code (PL, SA, etc.): ").strip().upper()
    matches = get_competition_matches(comp)

    for m in matches:
        try:
            h_id = m["homeTeam"]["id"]
            a_id = m["awayTeam"]["id"]

            h_stats = compute_team_stats(h_id)
            a_stats = compute_team_stats(a_id)
            h2h = get_head_to_head(h_id, a_id)

            lam_h, lam_a = expected_goals(h_stats, a_stats, h_id, a_id, elo_mgr)
            mc_pred = monte_carlo(lam_h, lam_a)
            blended = blend_predictions(mc_pred, h2h)

            best_dc, dc_prob = best_double_chance(blended)
            corners = estimate_corners(h_stats, a_stats)

            db.store_match(m, blended, best_dc, dc_prob, corners)

            match_time = datetime.fromisoformat(m["utcDate"].replace("Z","+00:00")).astimezone(UTC_DISPLAY_ZONE)
            print(f"\n📅 {match_time.strftime('%Y-%m-%d %H:%M')} - {m['homeTeam']['name']} vs {m['awayTeam']['name']}")
            print(f"   → 1: {blended['1']*100:.1f}%, X: {blended['X']*100:.1f}%, 2: {blended['2']*100:.1f}%")
            print(f"   🧠 Best Double Chance: {best_dc} ({dc_prob*100:.1f}%)")
            print(f"   ⚽ Estimated Corners: {corners}")

            time.sleep(RATE_LIMIT)

        except Exception as e:
            logger.exception("Error processing match: %s", e)
            continue

    db.close()
    logger.info("Done.")


if __name__ == "__main__":
    main()
