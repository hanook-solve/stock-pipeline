from dotenv import load_dotenv
from groq import Groq
import os
import json

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY)

# Add your topics here — same list as generate.py
topics = [
     "Abandoned_City_Streets",
    "Abandoned_Urban_Cityscape_Background",
    "Dramatic_Sunset_Desert_Horizon",
    "Dystopian_Industrial_Zones",
    "Galactic_Cosmic_Night_Sky",
    "Misty_Mountain_Range",
    "Moody_Foggy_Forest_Landscape",
    "Neon_Cyberpunk_Cityscape_Night",
    "Planetary_Ring_System_Shine",
    "rabbit_garden_flower_scene",
    "Retro_Futuristic_Landscapes",
    "Vintage_Science_Fiction",
    "Surreal sunset landscape on a remote exoplanet"
]

def generate_keywords(topic):
    print(f"  Generating keywords for: {topic}...")

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a senior Adobe Stock SEO strategist with deep knowledge of "
"how buyers search for stock images on Adobe Stock, Shutterstock, and Freepik. "
"Your goal is to maximize search visibility and get images ranked on page 1. "

"TITLE RULES: "
"Maximum 70 characters. Title Case. No punctuation at the end. "
"Put the most searched keyword FIRST in the title — this is critical for ranking. "
"Be descriptive and specific — avoid vague titles like 'Beautiful Background'. "
"Include the main subject, style, and use case in the title. "
"Example: 'Dark Marble Texture With Gold Veining Luxury Background' "
"Example: 'Ramadan Kareem Night Sky With Glowing Lanterns Festive Background' "

"DESCRIPTION RULES: "
"Maximum 200 characters. One to two sentences. "
"Start with the primary use case (e.g. Perfect for holiday greeting cards...). "
"Include 3 to 5 natural keywords embedded in the sentence — not keyword stuffing. "
"Describe what the buyer will USE the image for, not just what it looks like. "

"KEYWORD RULES — this is the most important part for ranking: "
"Return exactly 45 keywords — Adobe Stock allows up to 50, use 45 for maximum coverage. "
"ORDER matters — put highest search volume keywords FIRST. "
"Follow this keyword structure in order: "
"1. PRIMARY KEYWORDS (1-5): Exact topic match — what someone types first. "
"2. STYLE KEYWORDS (6-12): Visual style, mood, atmosphere, lighting. "
"3. COLOR KEYWORDS (13-18): Dominant colors in the image. "
"4. USE CASE KEYWORDS (19-28): What buyers use it for (website banner, social media, greeting card). "
"5. SEASONAL/OCCASION KEYWORDS (29-35): Season, holiday, event, time of year. "
"6. TECHNICAL KEYWORDS (36-40): Resolution quality, orientation, style type. "
"7. BROAD MATCH KEYWORDS (41-45): Broader category terms to catch more searches. "
"All keywords lowercase. Single words or 2-word phrases only. "
"No duplicate meanings. No irrelevant keywords — Adobe penalizes keyword stuffing. "

"RETURN FORMAT: "
"Raw JSON object only. No markdown. No explanation. No extra text. "
"Exactly this structure: "
#"{\"title\": \"...\", \"description\": \"...\", \"keywords\": [\"kw1\", \"kw2\", ...]}"
                )
            },
            {
                "role": "user",
                "content": f"Generate title, description and keywords for: {topic} background stock image"
            }
        ]
    )

    raw = response.choices[0].message.content.strip()

    try:
        data = json.loads(raw)
        return data
    except json.JSONDecodeError:
        print(f"  Warning: Could not parse JSON for {topic}, saving raw text")
        return {"title": topic, "description": "", "keywords": raw}


def save_results(results):
    # Save as JSON for reference
    with open("keywords.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Save as readable TXT for easy copy-paste during upload
    with open("keywords.txt", "w", encoding="utf-8") as f:
        for topic, data in results.items():
            f.write(f"{'='*60}\n")
            f.write(f"TOPIC:       {topic}\n")
            f.write(f"TITLE:       {data.get('title', '')}\n")
            f.write(f"DESCRIPTION: {data.get('description', '')}\n")
            f.write(f"KEYWORDS:    {data.get('keywords', '')}\n")
            f.write(f"\n")

    print("\nSaved to keywords.json and keywords.txt")


# Run
results = {}

for topic in topics:
    data = generate_keywords(topic)
    results[topic] = data

    print(f"  Title:    {data.get('title', '')}")
    print(f"  Keywords: {str(data.get('keywords', ''))[:60]}...")
    print()

save_results(results)
print("Done! Open keywords.txt to copy-paste during Adobe Stock upload.")