# scripts/download_data.py
import requests
import datetime
import csv

# ThingSpeak channel details (replace with your channel ID and Read API key)
CHANNEL_ID = "2968337"
READ_API_KEY = "JKW1ZRWYQDM0IEVM"

# Prepare the URL to fetch last 1 day of data (ThingSpeak allows querying by timespan or results count)
end_time = datetime.datetime.utcnow()
start_time = end_time - datetime.timedelta(days=1)
# Format times as required by ThingSpeak API (e.g., YYYY-MM-DDTHH:MM:SSZ in UTC)
end_str = end_time.strftime("%Y-%m-%dT%H:%M:%SZ")
start_str = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")

url = (f"https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds.json"
       f"?api_key={READ_API_KEY}&start={start_str}&end={end_str}")
response = requests.get(url)
data = response.json()

# Extract feeds (list of data points)
feeds = data.get("feeds", [])
if not feeds:
    print("No data fetched from ThingSpeak. Check API key and channel.")
    exit()

# Define the CSV file to save data
csv_file = "data/sensor_data.csv"
# Write header if file is new
write_header = not os.path.exists(csv_file)

with open(csv_file, mode='a', newline='') as f:
    writer = csv.writer(f)
    if write_header:
        # Assuming the ThingSpeak fields: field1, field2, ... etc., plus a timestamp
        headers = ["timestamp"] + [f"field{i}" for i in range(1, 3)]  # adjust number of fields
        writer.writerow(headers)
    # Write each new record
    for entry in feeds:
        timestamp = entry.get("created_at")
        # Collect field values in order (ThingSpeak field1..8)
        fields = [entry.get(f"field{i}") for i in range(1, 3)]
        writer.writerow([timestamp] + fields)

print(f"Fetched {len(feeds)} records from ThingSpeak and appended to sensor_data.csv.")
