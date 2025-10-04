import sqlite3
from datetime import datetime, timedelta

# -------------------- CONFIG --------------------
DB_FILE = "football_predictions.db"
OUTPUT_HTML ="viewer2.html"
UTC_OFFSET = 2  

# -------------------- CONNECT TO DB --------------------
conn = sqlite3.connect(DB_FILE)
cur = conn.cursor()

# Fetch all matches
cur.execute("""
    SELECT team_home, team_away, match_time, best_double_chance, best_dc_prob,
           prediction, prob_1, prob_x, prob_2, corners_avg
    FROM matches
    ORDER BY match_time ASC
""")
rows = cur.fetchall()
conn.close()

# -------------------- GENERATE HTML --------------------
html = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Foottball predicts</title>
<style>
    body { font-family: Arial, sans-serif; margin: 20px; }
    table { border-collapse: collapse; width: 100%; }
    th, td { border: 1px solid #ccc; padding: 8px; text-align: center; }
    th { background-color: #f2f2f2; }
    tr:nth-child(even) { background-color: #fafafa; }
</style>
</head>
<body>
<h1>Football Predictions</h1>
<table>
<thead>
<tr>
<th>Home Team</th>
<th>Away Team</th>
<th>Match Time (UTC+1)</th>
<th>Best Double Chance</th>
<th>Best DC Prob</th>
<th>Prediction</th>
<th>1</th>
<th>X</th>
<th>2</th>
<th>Corners Avg</th>
</tr>
</thead>
<tbody>
"""

for row in rows:
    (team_home, team_away, match_time_utc, best_dc, best_dc_prob,
     prediction, prob_1, prob_x, prob_2, corners) = row
    
    # Convert UTC to UTC+1
    match_time = datetime.fromisoformat(match_time_utc) + timedelta(hours=UTC_OFFSET)
    match_time_str = match_time.strftime("%Y-%m-%d %H:%M")
    
    html += f"""
    <tr>
        <td>{team_home}</td>
        <td>{team_away}</td>
        <td>{match_time_str}</td>
        <td>{best_dc}</td>
        <td>{best_dc_prob*100:.1f}%</td>
        <td>{prediction}</td>
        <td>{prob_1*100:.1f}%</td>
        <td>{prob_x*100:.1f}%</td>
        <td>{prob_2*100:.1f}%</td>
        <td>{corners}</td>
    </tr>
    """

html += """
</tbody>
</table>
</body>
</html>
"""

# -------------------- SAVE HTML FILE --------------------
with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
    f.write(html)

print(f"HTML file generated: {OUTPUT_HTML}")
