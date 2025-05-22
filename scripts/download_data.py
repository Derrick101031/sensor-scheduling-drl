# scripts/download_data.py
import requests
import datetime
import csv
import os

CHANNEL_ID = "2968337"
READ_API_KEY = "JKW1ZRWYQDM0IEVM"

url = (f"https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds.json"
       f"?api_key={READ_API_KEY}&start={start_str}&end={end_str}")
response = requests.get(url)
data = response.json()

feeds = data.get("feeds", [])
if not feeds:
    print("No data fetched from ThingSpeak. Check API key and channel.")
    exit()

csv_file = "data/sensor_data.csv"
write_header = not os.path.exists(csv_file)

with open(csv_file, mode='a', newline='') as f:
    writer = csv.writer(f)
    if write_header:
        headers = ["timestamp"] + [f"field{i}" for i in range(1, 3)]
        writer.writerow(headers)
    for entry in feeds:
        timestamp = entry.get("created_at")
        fields = [entry.get(f"field{i}") for i in range(1, 3)]
        writer.writerow([timestamp] + fields)

print(f"Fetched {len(feeds)} records from ThingSpeak and appended to sensor_data.csv.")
