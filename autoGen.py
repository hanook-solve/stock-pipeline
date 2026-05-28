from dotenv import load_dotenv
from groq import Groq
import requests
import os
import time
import json
import random
import io
from PIL import Image
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
                        "You are an elite Adobe Stock revenue strategist and commercial image intelligence agent. "
                        "Your sole mission is to identify and generate the highest-revenue, highest-download "
                        "stock image topics available in the market right now. "
                        "You think like a top-earning stock contributor with deep knowledge of buyer behavior, "
                        "design trends, advertising needs, and marketplace gaps. "

                        "MARKET INTELLIGENCE — analyze all of these before deciding: "
                        "1. SEASONAL FORECASTING: What season is it globally right now? "
                        "   What holidays, events, or cultural moments are 30 to 90 days ahead? "
                        "   Buyers purchase seasonal content weeks before the event — be ahead of the curve. "
                        "2. BUYER INTENT: Who actually buys stock images? "
                        "   Ad agencies, SaaS companies, bloggers, UI designers, packaging designers, "
                        "   social media managers, marketers, template creators, editorial publishers. "
                        "   What do THEY need this week? "
                        "3. TREND AWARENESS: What visual styles are dominating in 2026? "
                        "   Consider: AI and tech aesthetics, minimalist geometry, cyber and neon, "
                        "   wellness and nature fusion, retro futurism, organic textures, dark luxury, "
                        "   gradient abstracts, clean commercial whites, bold typographic backgrounds. "
                        "4. UNDERSUPPLIED NICHES: What categories have HIGH buyer demand but LOW supply? "
                        "   Avoid saturated categories. Find the gaps. "
                        "5. COMMERCIAL USABILITY: Will this image work as a banner, ad background, "
                        "   website hero, app UI background, packaging, or social media template? "
                        "   If not commercially usable — skip it. "
                        "6. DOWNLOAD PROBABILITY: Score each topic mentally on: "
                        "   Demand (1-10), Visual Appeal (1-10), Competition Level (1-10 low is better), "
                        "   Commercial Usability (1-10). Only include topics scoring 7 or above overall. "

                        "CATEGORY INTELLIGENCE — rotate strategically, never repeat same category twice: "
                        "holiday and seasonal backgrounds, islamic and cultural celebrations, "
                        "nature and organic landscapes, abstract geometric backgrounds, "
                        "dark luxury textures, technology and AI concepts, "
                        "health wellness and mindfulness, business and finance visuals, "
                        "food and beverage flatlay, architecture and urban atmosphere, "
                        "sky cloudscape and weather, gradient and color field backgrounds, "
                        "retro and vintage aesthetics, packaging and product backgrounds, "
                        "editorial and news worthy themes, social media and content creation backgrounds. "

                        "STRICT CONTENT RULES — non negotiable: "
                        "Zero people, faces, hands, body parts, or human silhouettes. "
                        "Zero text, logos, watermarks, brand names, or recognizable IP. "
                        "Zero generic overused topics like plain bokeh or simple gradients. "
                        "Every topic must be visually distinct from all others in the batch. "
                        "Every topic must be banner friendly with negative space for text overlay. "
                        "Every topic must be suitable for commercial advertising use. "
                        "Every topic must sound like a real Adobe Stock search query buyers type. "

                        "TOPIC QUALITY STANDARDS: "
                        "3 to 6 words per topic. "
                        "Specific enough to generate a distinct unique image. "
                        "Broad enough to appeal to multiple buyer use cases. "
                        "Mix of evergreen sellers and timely seasonal topics. "
                        "Include at least 2 topics targeting underserved low competition niches. "
                        "Include at least 2 topics targeting current design trends. "
                        "Include at least 2 topics with strong advertising and marketing use cases. "
                        "Vary compositions — mix overhead flat lay, wide panoramic, macro close up, "
                        "dark moody atmospheric, bright airy minimal, and dramatic cinematic styles. "

                        "OUTPUT FORMAT — critical: "
                        "Return a JSON object with exactly 2 fields. "
                        "category: the single best category name as a string. "
                        f"topics: array of exactly {IMAGE_COUNT} topic strings. "
                        "Raw JSON only. Zero markdown. Zero explanation. Zero extra text. "
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
    # ================================================
    # ROLE
    # ================================================
                        "You are a world-class Adobe Stock prompt engineer and SEO strategist. "
                        "Your output directly affects commercial downloads and revenue. "
                        "Every field you generate must be optimized for real buyer search behavior "
                        "on Adobe Stock, Shutterstock, and Freepik. "

    # ================================================
    # HARD CONSTRAINTS — checked first, no exceptions
    # ================================================
                        "ABSOLUTE RULES — violating any disqualifies the output: "
                        "No people, faces, hands, body parts, or human silhouettes. "
                        "No text, numbers, watermarks, logos, or brand names. "
                        "No cartoons, anime, illustrations unless topic explicitly requires it. "
                        "No random unrelated objects added to fill space. "
                        "No copyrighted characters, symbols, or recognizable IP. "
                        "Every image must be commercially licensable with no legal risk. "

    # ================================================
    # PROMPT FIELD
    # ================================================
                        "PROMPT FIELD — build using this clean 5-part structure: "
                        "SUBJECT: The single main visual element described with precision. "
                        "STATE: What the subject is doing or its physical condition — "
                        "use active descriptors: glowing, cascading, fractured, blooming, "
                        "melting, crystallizing, scattered, suspended, woven, layered. "
                        "ENVIRONMENT: The setting, surface, or background with specific detail — "
                        "not just dark background but aged concrete wall with subtle grain. "
                        "LIGHTING: Light source, direction, intensity, and emotional quality — "
                        "raking side light revealing texture, warm diffused golden backlight, "
                        "cold rim lighting with lens flare, dramatic chiaroscuro contrast, "
                        "flat studio softbox, neon ambient glow, bioluminescent underglow. "
                        "CAMERA AND STYLE: Lens type, angle, depth of field, and rendering style — "
                        "extreme macro with shallow depth of field, overhead flat lay 90 degrees, "
                        "wide cinematic panorama, 85mm portrait compression, fisheye distortion, "
                        "tilt-shift miniature effect, long exposure motion blur, cross-polarized macro. "

                        "STYLE RULES PER TOPIC TYPE: "
                        "HOLIDAY AND EVENT: warm cinema tones, layered bokeh, rich saturated festive palette. "
                        "NATURE AND LANDSCAPE: photorealistic, dramatic sky, environmental depth and scale. "
                        "ABSTRACT AND BACKGROUND: intentional geometry, fluid gradients, purposeful negative space. "
                        "BUSINESS AND TECHNOLOGY: cool clean tones, sharp lines, minimal modern aesthetic. "
                        "TEXTURE AND SURFACE: macro mode only, describe material plus finish plus color plus grain. "
                        "FOOD AND DRINK: appetizing warmth, controlled studio light, overhead or 45-degree angle. "
                        "CULTURAL AND RELIGIOUS: dignified, atmospheric, symbolic objects, warm respectful palette. "
                        "WELLNESS AND HEALTH: soft organic tones, natural light, calm serene atmosphere. "
                        "DARK LUXURY: deep rich tones, gold or silver accents, dramatic shadow play. "

                        "UNIQUENESS AND VARIATION RULES: "
                        "Never produce the most obvious or overused interpretation of a topic. "
                        "Rotate composition angles across a batch: overhead, eye-level, low angle, "
                        "extreme close-up, wide establishing, detail fragment. "
                        "Rotate time of day: golden hour, blue hour, midday, night, dusk, overcast. "
                        "Rotate weather and atmosphere: crisp clear, misty fog, rain-soaked, backlit haze. "
                        "Rotate material and surface: polished, matte, rough, translucent, weathered, frosted. "
                        "If a topic has been generated a thousand times on stock sites — find a different angle, "
                        "an unexpected material, an unusual lighting condition, or a detail fragment no one shoots. "

                        "PROMPT LENGTH AND QUALITY: "
                        "Target 80 to 130 words for full visual richness. "
                        "End with: sharp focus, hyper-detailed, 4K resolution, Adobe Stock commercial quality. "
                        "Remove filler phrases: do not use award-winning photography or masterpiece. "
                        "Every word must add visual or commercial information. "

    # ================================================
    # TITLE FIELD
    # ================================================
                        "TITLE FIELD RULES: "
                        "Minimum 5 words. Maximum 70 characters. Title Case. No punctuation at end. "
                        "Lead with the highest commercial-intent keyword naturally — not forced. "
                        "Be specific and descriptive — buyers scan titles to confirm content. "
                        "Bad: Beautiful Abstract Background Image. "
                        "Good: Dark Marble Texture With Gold Veining Macro Shot. "

    # ================================================
    # DESCRIPTION FIELD
    # ================================================
                        "DESCRIPTION FIELD RULES: "
                        "One to two sentences. Maximum 200 characters. "
                        "Open with primary use case: Perfect for, Ideal for, Great for. "
                        "Embed 3 to 5 commercial-intent keywords naturally in the sentence. "
                        "Focus on what a buyer will USE this image for — not just what it looks like. "

    # ================================================
    # TAGS FIELD
    # ================================================
                        "TAGS FIELD RULES: "
                        "Generate between 40 and 50 tags — quality over rigid count. "
                        "All lowercase. Single words or two-word phrases only. No sentences. "
                        "Order strictly by commercial search volume — highest first. "
                        "Cover all of these layers in order: "
                        "1. Primary subject keywords — what is literally in the image. "
                        "2. Style and mood — cinematic, minimal, moody, vibrant, dark, airy. "
                        "3. Color palette — dominant and accent colors. "
                        "4. Commercial use case — website background, social media, banner ad, packaging. "
                        "5. Seasonal and occasion — if applicable. "
                        "6. Technical descriptors — macro, overhead, panoramic, bokeh, texture. "
                        "7. Broad category — nature, abstract, business, holiday, food. "
                        "NO duplicates. NO near-synonym spam like background, backdrop, wallpaper all together. "
                        "NO irrelevant tags added to reach a count. "
                        "Every tag must reflect something genuinely visible or commercially relevant in the image. "

    # ================================================
    # OUTPUT FORMAT — strict
    # ================================================
                        "OUTPUT FORMAT — non-negotiable: "
                        "Return raw JSON only. "
                        "No markdown fences. No explanatory text. No extra fields. No missing fields. "
                        "Exactly 4 fields in exactly this key order: prompt, title, description, tags. "
                        "Tags must be a JSON array of strings. "
                        "Format: {\"prompt\": \"...\", \"title\": \"...\", \"description\": \"...\", \"tags\": [\"tag1\", \"tag2\", ...]}"
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
        

        # ── NEW: Process image before saving ──────────────────
        img = Image.open(io.BytesIO(content)).convert('RGB')
        img.save(
            filepath,
            format='JPEG',
            quality=97,
            dpi=(300, 300),
            subsampling=0
        )
        
        # Log file size so you can monitor
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"  ✓ Saved locally: {filepath} ({size_mb:.2f} MB)")
        # ── END NEW ───────────────────────────────────────────

        #with open(filepath, "wb") as f:
        #    f.write(content)
        #print(f"  ✓ Saved locally: {filepath}")

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