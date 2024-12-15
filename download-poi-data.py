import requests
import json
url = "https://overpass-api.de/api/interpreter"
query = """
[out:json][timeout:25];
(
  node["amenity"](48.8156, 2.2241, 48.9021, 2.4699);
  way["amenity"](48.8156, 2.2241, 48.9021, 2.4699);
  relation["amenity"](48.8156, 2.2241, 48.9021, 2.4699);
);
out center tags;
"""

response = requests.post(url, data={"data": query})

if response.status_code == 200:
    data = response.json()
    print(data)  # Contains all POI data
    with open("tags.json", "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

else:
    print(f"Error: {response.status_code}")