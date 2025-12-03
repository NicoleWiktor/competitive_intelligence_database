import json
import requests
from bs4 import BeautifulSoup
import re

# Honeywell pressure transmitter sources
sources = [
    {
        "product": "ST 3000 Pressure Transmitter",
        "url": "https://sps.honeywell.com/us/en/products/pressure-transmitters",
        "category": "Specs"
    },
    {
        "product": "ST 7000 Series Pressure Transmitter",
        "url": "https://www.omega.com",
        "category": "Specs & Reviews"
    },
    {
        "product": "ST 3000 Pressure Transmitter",
        "url": "https://www.controleng.com",
        "category": "Reviews"
    },
    {
        "product": "ST 7000 Series Pressure Transmitter",
        "url": "https://www.datasheet.com",
        "category": "Specs"
    },
    {
        "product": "ST 3000 Pressure Transmitter",
        "url": "https://www.reddit.com/r/Instrumentation",
        "category": "Reviews"
    }
]

def get_text_snippets(url, max_snippets=5):
    """Scrape first few paragraphs and tables as snippets"""
    snippets = []
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return ["Could not fetch content"]

        soup = BeautifulSoup(response.text, "html.parser")

        # Extract first paragraphs
        for p_tag in soup.find_all("p", limit=max_snippets):
            text = p_tag.get_text(strip=True)
            if text:
                snippets.append(text)

        # Extract tables
        for table_tag in soup.find_all("table", limit=max_snippets):
            text = table_tag.get_text(separator=" ", strip=True)
            if text:
                snippets.append(text)

        return snippets if snippets else ["No text snippet found"]

    except Exception as e:
        return [f"Error fetching snippet: {e}"]

def extract_features_from_snippet(snippet):
    """Try to extract structured features like pressure range, accuracy, output"""
    features = []
    # Pressure range (psi, bar, Pa)
    range_match = re.search(r'(\d+[\-–]\d+)\s*(psi|bar|Pa)', snippet)
    if range_match:
        features.append(f"Pressure range: {range_match.group(1)} {range_match.group(2)}")
    # Accuracy
    accuracy_match = re.search(r'±\s*\d+\.?\d*%?\s*(FS)?', snippet)
    if accuracy_match:
        features.append(f"Accuracy: {accuracy_match.group(0)}")
    # Output signal
    output_match = re.search(r'(4-20 mA|HART|Modbus|0-10 V)', snippet, re.IGNORECASE)
    if output_match:
        features.append(f"Output: {output_match.group(0)}")
    return features

# Build schema-compliant entries
entries = []
for source in sources:
    snippets = get_text_snippets(source["url"])
    structured_flag = source["category"] in ["Specs", "Specs & Reviews"]

    relationships = []
    for snippet in snippets:
        features = extract_features_from_snippet(snippet)
        for feature in features:
            relationships.append({
                "source_type": "Product",
                "source": source["product"],
                "relationship": "HAS_FEATURE",
                "target_type": "Feature",
                "target": feature
            })

    # If no features detected, just store snippet as one feature
    if not relationships:
        relationships.append({
            "source_type": "Product",
            "source": source["product"],
            "relationship": "HAS_FEATURE",
            "target_type": "Feature",
            "target": snippets[0][:200]  # truncate for quick reference
        })

    entry = {
        "Industry": "Industrial Automation",
        "CustomerSegment": "",
        "CustomerNeed": [],
        "HoneywellProduct": source["product"],
        "Competitor": "",
        "CompetitiveProduct": "",
        "Relationships": relationships,
        "Doc": {
            "source_url": source["url"],
            "snippets": snippets
        },
        "Flags": {
            "scraped": True,
            "structured": structured_flag
        }
    }

    entries.append(entry)

# Save to JSON
with open("honeywell_schema_advanced.json", "w") as f:
    json.dump(entries, f, indent=4)

print("Honeywell schema entries saved to honeywell_schema_advanced.json")