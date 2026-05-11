from dotenv import load_dotenv
from groq import Groq
import requests
import os
import time
import json
import random
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2 import service_account

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

load_dotenv()

# ================================
# API KEYS
# ================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

HF_TOKENS = [
    os.getenv(f"HF_TOKEN{i}") for i in [""] + list(range(1, 25))
]
HF_TOKENS = [t for t in HF_TOKENS if t]

if not HF_TOKENS:
    print("WARNING: No HuggingFace tokens found in .env")

if not GROQ_API_KEY:
    print("ERROR: GROQ_API_KEY missing from .env file")
    exit()

groq_client = Groq(api_key=GROQ_API_KEY)
current_hf_index = 0

# ================================
# GOOGLE DRIVE SETUP
# ================================
DRIVE_FOLDER_ID = os.getenv("DRIVE_FOLDER_ID")
CREDENTIALS_FILE = "credentials.json"

def get_drive_service():
    try:
        creds = service_account.Credentials.from_service_account_file(
            CREDENTIALS_FILE,
            scopes=["https://www.googleapis.com/auth/drive"]
        )
        return build("drive", "v3", credentials=creds)
    except Exception as e:
        print(f"  ✗ Drive connection failed: {e}")
        return None

def upload_to_drive(filepath, filename):
    try:
        if not DRIVE_FOLDER_ID:
            print("  ⚠ No Drive folder ID set — skipping upload")
            return False

        service = get_drive_service()
        if not service:
            return False

        file_metadata = {
            "name": filename,
            "parents": [DRIVE_FOLDER_ID]
        }

        media = MediaFileUpload(
            filepath,
            mimetype="image/jpeg",
            resumable=True
        )

        file = service.files().create(
            body=file_metadata,
            media_body=media,
            fields="id"
        ).execute()

        print(f"  ✓ Uploaded to Google Drive: {filename}")
        return True

    except Exception as e:
        print(f"  ✗ Drive upload failed: {e}")
        return False

# ================================
# SETTINGS
# ================================
IMAGE_COUNT = 10

ASPECT_RATIOS = {
    "landscape": (2560, 1600),
    "portrait":  (1600, 2560),
    "square":    (2048, 2048),
}
RATIO = "landscape"

IMAGE_MODELS = [
    "gptimage-large",
    "flux-realism",
    "flux",
]

os.makedirs("images", exist_ok=True)

# ================================
# STEP 0 — AI AGENT
# Thinks of category and topics
# No manual input needed
# ================================
def agent_decide_category():
    print(f"\nAI Agent deciding category and topics...")

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an autonomous Adobe Stock content strategist agent. "
                        "Your job is to independently decide what stock image category "
                        "and topics will generate the most downloads and revenue today. "

                        "Think about: "
                        "1. What season or month is it? (consider current global seasons) "
                        "2. What holidays or events are coming up in the next 30-60 days? "
                        "3. What evergreen categories always sell well? "
                        "4. What niches are undersupplied on Adobe Stock right now? "
                        "5. What topics have high commercial buyer demand? "

                        "Categories to rotate between: "
                        "holiday backgrounds, nature and landscape, islamic and cultural, "
                        "textures and surfaces, business and technology, food and drink, "
                        "abstract backgrounds, sky and atmosphere, travel and architecture, "
                        "health and wellness, seasonal themes, commercial backgrounds. "

                        "STRICT RULES: "
                        "No people, no faces, no body parts. "
                        "Backgrounds, textures, atmospheric scenes only. "
                        "Each topic must be visually distinct from others. "
                        "Topics must be 3 to 6 words each. "
                        "Think commercially — what do buyers actually purchase? "

                        "Return a JSON object with exactly 2 fields: "
                        "category: the chosen category name as a string. "
                        f"topics: array of exactly {IMAGE_COUNT} specific topic strings. "
                        "Return raw JSON only. No markdown. No explanation. "
                        "Format: {\"category\": \"...\", \"topics\": [\"topic1\", \"topic2\", ...]}"
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Today's date context: {time.strftime('%B %Y')}. "
                        f"Autonomously decide the best category and {IMAGE_COUNT} specific "
                        "high demand topics to generate stock images for right now. "
                        "Think strategically about what will sell."
                    )
                }
            ]
        )

        raw = response.choices[0].message.content.strip()
        clean = raw.replace("```json", "").replace("```", "").strip()
        result = json.loads(clean)

        if "category" not in result or "topics" not in result:
            raise ValueError("Missing category or topics")

        if not isinstance(result["topics"], list) or len(result["topics"]) == 0:
            raise ValueError("Invalid topics format")

        result["topics"] = result["topics"][:IMAGE_COUNT]

        print(f"  ✓ Agent chose category: {result['category']}")
        print(f"  ✓ Agent generated {len(result['topics'])} topics:")
        for i, t in enumerate(result["topics"], 1):
            print(f"    {i}. {t}")

        return result["category"], result["topics"]

    except Exception as e:
        print(f"  ✗ Agent failed: {e}")
        # Fallback category and topics
        fallback_category = "holiday backgrounds"
        fallback_topics = [f"holiday background scene {i}" for i in range(1, IMAGE_COUNT + 1)]
        return fallback_category, fallback_topics


# ================================
# STEP 2 — GENERATE PROMPT + METADATA
# Uses upgraded formula:
# [Subject] + [Action/Pose] +
# [Environment/Background] +
# [Lighting & Atmosphere] +
# [Camera & Style Specs]
# ================================
def generate_prompt(topic, category):
    print(f"\n  Writing prompt and metadata for: {topic}...")

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a world class Adobe Stock image prompt engineer and SEO expert. "
                        "For every topic return a JSON object with exactly 4 fields. "

                        "PROMPT FIELD — use this exact formula: "
                        "[Subject]: clearly describe the main visual element. "
                        "[Action/State]: describe what the subject is doing or its condition "
                        "(glowing, falling, blooming, melting, flowing, scattered, arranged). "
                        "[Environment/Background]: describe the setting and surroundings in detail. "
                        "[Lighting & Atmosphere]: describe light source, direction, quality, and mood "
                        "(soft golden backlight, dramatic rim lighting, diffused studio light, "
                        "warm candlelight, cool moonlight, misty atmospheric haze). "
                        "[Camera & Style Specs]: describe the shot technically "
                        "(macro lens extreme close up, wide angle panorama, overhead flat lay, "
                        "shallow depth of field, 85mm portrait lens, fisheye). "

                        "Apply correct style per topic type: "
                        "HOLIDAY/EVENT: warm cinematic atmosphere, bokeh lights, rich festive colors. "
                        "NATURE/LANDSCAPE: photorealistic, golden or blue hour lighting, vast depth. "
                        "ABSTRACT/BACKGROUND: clean geometry, smooth gradients, elegant minimal. "
                        "BUSINESS/TECHNOLOGY: clean modern, cool blue tones, sharp professional. "
                        "TEXTURE/SURFACE: macro mode, describe material finish color detail lighting. "
                        "FOOD/DRINK: appetizing, studio lighting, overhead or 45 degree angle. "
                        "CULTURAL/RELIGIOUS: respectful, atmospheric, symbolic, warm dignified. "

                        "UNIQUENESS RULE: avoid the most obvious interpretation. "
                        "Find unexpected angle, unusual composition, or specific detail. "
                        "Ask: has this exact image been made a thousand times? If yes — find different angle. "

                        "NEVER include: people, faces, hands, body parts, text, watermarks, "
                        "logos, cartoons, anime, random unrelated objects. "

                        "End prompt with: sharp focus, highly detailed, 4K resolution, "
                        "Adobe Stock style, award winning photography. "
                        "Keep prompt 70 to 90 words. "

                        "TITLE FIELD: "
                        "Professional Adobe Stock title. Minimum 5 words. Title Case. "
                        "No punctuation at end. Most searched keyword FIRST. Specific not vague. "

                        "DESCRIPTION FIELD: "
                        "One to two sentences. Maximum 200 characters. "
                        "Start with use case (Perfect for... or Ideal for...). "
                        "Include 3 to 5 natural keywords. "

                        "TAGS FIELD: "
                        "Exactly 45 keywords for maximum Adobe Stock SEO coverage. "
                        "All lowercase. Single words or 2 word phrases only. "
                        "Order by search volume highest first. "
                        "Cover: primary subject, style, mood, colors, use case, "
                        "season, occasion, technical specs, broad category. "
                        "No duplicates. No irrelevant tags. "

                        "STRICT: Return raw JSON only. No markdown. No explanation. "
                        "Exactly this format: "
                        "{\"prompt\": \"...\", \"title\": \"...\", "
                        "\"description\": \"...\", \"tags\": [\"tag1\", \"tag2\"]}"
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Category: {category}. "
                        f"Generate prompt and metadata for: {topic} stock image."
                    )
                }
            ]
        )

        raw = response.choices[0].message.content.strip()
        clean = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean)

        for field in ["prompt", "title", "description", "tags"]:
            if field not in data:
                raise ValueError(f"Missing field: {field}")

        if len(data["prompt"]) < 20:
            raise ValueError("Prompt too short")

        print(f"  ✓ Prompt: {data['prompt'][:80]}...")
        print(f"  ✓ Title: {data['title']}")
        print(f"  ✓ Tags: {len(data['tags'])} keywords generated")
        return data

    except json.JSONDecodeError as e:
        print(f"  ✗ JSON parse failed: {e}")
        return _fallback_data(topic)

    except Exception as e:
        print(f"  ✗ Generation failed: {e}")
        return _fallback_data(topic)


def _fallback_data(topic):
    return {
        "prompt": f"{topic}, professional stock photo, sharp focus, highly detailed, 4K, Adobe Stock style",
        "title": topic.title(),
        "description": f"Professional stock image of {topic} suitable for commercial use.",
        "tags": [topic.lower(), "background", "stock", "professional", "commercial"]
    }


# ================================
# STEP 3A — GENERATE VIA HUGGINGFACE
# ================================
def generate_huggingface(topic, prompt, data, attempt=1):
    global current_hf_index

    if not HF_TOKENS:
        return False

    while current_hf_index < len(HF_TOKENS):
        token = HF_TOKENS[current_hf_index]
        width, height = ASPECT_RATIOS[RATIO]

        API_URL = "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell"
        headers = {"Authorization": f"Bearer {token}"}
        payload = {
            "inputs": prompt,
            "parameters": {"width": width, "height": height}
        }

        try:
            print(f"  Generating via HuggingFace account {current_hf_index + 1}/{len(HF_TOKENS)}...")
            response = requests.post(API_URL, headers=headers, json=payload, timeout=120)

            if response.status_code == 200:
                print(f"  ✓ Success with account {current_hf_index + 1}")
                return save_image(topic, response.content, data, "HuggingFace FLUX")

            elif response.status_code == 402:
                print(f"  ✗ Account {current_hf_index + 1} exhausted — trying next...")
                current_hf_index += 1
                continue

            elif response.status_code == 503:
                print(f"  Model loading, waiting 20 seconds...")
                time.sleep(20)
                continue

            elif response.status_code == 400:
                print(f"  ✗ Bad request — falling back to Pollinations...")
                return False

            else:
                print(f"  ✗ HuggingFace failed: {response.status_code}")
                return False

        except requests.exceptions.Timeout:
            if attempt < 3:
                print(f"  Timed out, retrying in 10 seconds...")
                time.sleep(10)
                attempt += 1
                continue
            print(f"  ✗ HuggingFace timed out after 3 attempts")
            return False

        except Exception as e:
            print(f"  ✗ HuggingFace error: {e}")
            return False

    print(f"  ✗ All HuggingFace accounts exhausted — falling back to Pollinations")
    return False


# ================================
# STEP 3B — GENERATE VIA POLLINATIONS
# ================================
def generate_pollinations(topic, prompt, data, attempt=1):
    width, height = ASPECT_RATIOS[RATIO]
    seed = random.randint(1, 99999)

    for model in IMAGE_MODELS:
        try:
            url = (
                f"https://image.pollinations.ai/prompt/{requests.utils.quote(prompt)}"
                f"?width={width}&height={height}&model={model}&nologo=true&seed={seed}"
            )

            print(f"  Trying Pollinations model: {model}...")
            response = requests.get(url, timeout=150)

            if response.status_code == 200:
                print(f"  ✓ Success with model: {model}")
                return save_image(topic, response.content, data, f"Pollinations {model}")
            else:
                print(f"  ✗ {model} failed: {response.status_code}, trying next...")

        except requests.exceptions.Timeout:
            print(f"  ✗ {model} timed out, trying next...")
            continue

        except Exception as e:
            print(f"  ✗ {model} error: {e}, trying next...")
            continue

    print(f"  ✗ All Pollinations models failed for {topic}")
    return False


# ================================
# SAVE PROMPT TO NOTES FILE
# ================================
def save_prompt_log(topic, data, model_used, category):
    log_file = "prompt_notes.txt"
    tags_str = ", ".join(data.get("tags", []))

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"{'='*60}\n")
        f.write(f"TOPIC:       {topic}\n")
        f.write(f"CATEGORY:    {category}\n")
        f.write(f"MODEL:       {model_used}\n")
        f.write(f"DATE:        {time.strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"\n")
        f.write(f"PROMPT:      {data.get('prompt', '')}\n")
        f.write(f"\n")
        f.write(f"TITLE:       {data.get('title', '')}\n")
        f.write(f"DESCRIPTION: {data.get('description', '')}\n")
        f.write(f"TAGS:        {tags_str}\n")
        f.write(f"\n")


# ================================
# SAVE IMAGE + UPLOAD TO DRIVE
# ================================
def save_image(topic, content, data=None, model_used="unknown", category=""):
    try:
        filename = topic.replace(" ", "_").replace("/", "_") + ".jpg"
        filepath = os.path.join("images", filename)

        with open(filepath, "wb") as f:
            f.write(content)
        print(f"  ✓ Saved locally: {filepath}")

        # Upload to Google Drive
        upload_to_drive(filepath, filename)

        if data:
            save_prompt_log(topic, data, model_used, category)

        return True

    except Exception as e:
        print(f"  ✗ Failed to save: {e}")
        return False


# ================================
# MAIN RUNNER
# ================================
def run():
    print(f"\n{'='*50}")
    print(f"AI Stock Image Generator")
    print(f"Date: {time.strftime('%Y-%m-%d %H:%M')}")
    print(f"Ratio: {RATIO} {ASPECT_RATIOS[RATIO]}")
    print(f"HuggingFace accounts: {len(HF_TOKENS)}")
    print(f"{'='*50}")

    # Step 0 — AI Agent decides category and topics
    category, topics = agent_decide_category()

    if not topics:
        print("\nNo topics generated. Exiting.")
        return

    print(f"\n{'='*50}")
    print(f"Generating {len(topics)} images for: {category}")
    print(f"{'='*50}")

    success_count = 0
    fail_count = 0

    for i, topic in enumerate(topics, 1):
        print(f"\n[{i}/{len(topics)}] {topic}")

        # Generate prompt and metadata
        data = generate_prompt(topic, category)
        prompt = data["prompt"]

        # Try HuggingFace first then Pollinations
        success = generate_huggingface(topic, prompt, data)
        if not success:
            print(f"  Falling back to Pollinations...")
            success = generate_pollinations(topic, prompt, data)

        if success:
            success_count += 1
        else:
            fail_count += 1
            print(f"  ✗ Completely failed for: {topic}")

        time.sleep(5)

    print(f"\n{'='*50}")
    print(f"Done! Results:")
    print(f"  ✓ Success: {success_count}")
    print(f"  ✗ Failed:  {fail_count}")
    print(f"  Images saved locally and uploaded to Google Drive")
    print(f"{'='*50}")

run()