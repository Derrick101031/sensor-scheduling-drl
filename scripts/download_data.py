import os
import requests
import csv

# ThingSpeak configuration
CHANNEL_ID = "2968337"
READ_API_KEY = "JKW1ZRWYQDM0IEVM"
CSV_DIR = "data"
CSV_PATH = os.path.join(CSV_DIR, "sensor_data.csv")

# Ensure data folder exists
os.makedirs(CSV_DIR, exist_ok=True)

# URL to fetch all data from ThingSpeak in CSV format
url = f"https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds.csv?api_key={READ_API_KEY}"

print("🔄 Fetching data from ThingSpeak...")
response = requests.get(url)

if response.status_code != 200:
    print(f"❌ Failed to download data. HTTP Status Code: {response.status_code}")
    exit(1)

# Save response text directly to CSV
with open(CSV_PATH, "w", newline='') as f:
    f.write(response.text)

print(f"✅ Data saved to {CSV_PATH}")

# Check if required fields are present
print("🔍 Validating downloaded CSV...")

with open(CSV_PATH, "r") as f:
    reader = csv.reader(f)
    header = next(reader)

required_fields = ["field1", "field2", "field3"]
missing = [field for field in required_fields if field not in header]

if missing:
    print(f"❌ Missing required fields in CSV: {', '.join(missing)}")
    print("ℹ️  Make sure your ThingSpeak channel has field1 (moisture), field2 (temp), field3 (humidity)")
    exit(1)

print("✅ Required fields found: field1, field2, field3")
print("📈 Ready for model training.")
