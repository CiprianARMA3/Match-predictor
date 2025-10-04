import requests
from dotenv import load_dotenv
import os

# Load API key from .env
load_dotenv()
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("Please set FOOTBALL_API_KEY in your .env file")

BASE_URL = "https://api.football-data.org/v4/competitions/"
HEADERS = {"X-Auth-Token": API_KEY}

def fetch_competitions():
    response = requests.get(BASE_URL, headers=HEADERS)
    if response.status_code == 200:
        data = response.json()
        return data.get("competitions", [])
    else:
        print(f"Failed to retrieve competitions: {response.status_code}")
        return []

# Fetch and display all competitions with their codes
competitions = fetch_competitions()
for comp in competitions:
    print(f"{comp['name']} ({comp['code']})")
