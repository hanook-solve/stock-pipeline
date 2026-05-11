from dotenv import load_dotenv
from groq import Groq
import requests
import os
import time
import json
import random
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# ================================
# LOAD ALL HUGGINGFACE TOKENS
# Add as many as you have in .env
# ================================
HF_TOKENS = [
    os.getenv("HF_TOKEN"),
    os.getenv("HF_TOKEN1"),
    os.getenv("HF_TOKEN2"),
    os.getenv("HF_TOKEN3"),
    os.getenv("HF_TOKEN4"),
    os.getenv("HF_TOKEN5"),
    os.getenv("HF_TOKEN6"),
    os.getenv("HF_TOKEN7"),
    os.getenv("HF_TOKEN8"),
    os.getenv("HF_TOKEN9"),
    os.getenv("HF_TOKEN10"),
    os.getenv("HF_TOKEN11"),
    os.getenv("HF_TOKEN12"),
    os.getenv("HF_TOKEN13"),
    os.getenv("HF_TOKEN14"),
    os.getenv("HF_TOKEN15"),
    os.getenv("HF_TOKEN16"),
    os.getenv("HF_TOKEN17"),
    os.getenv("HF_TOKEN18"),
    os.getenv("HF_TOKEN19"),
    os.getenv("HF_TOKEN21"),
    os.getenv("HF_TOKEN22"),
    os.getenv("HF_TOKEN23"),
    os.getenv("HF_TOKEN24"),
    os.getenv("HF_TOKEN20"),
]
# Remove any None values (missing keys)
HF_TOKENS = [t for t in HF_TOKENS if t]

if not HF_TOKENS:
    print("WARNING: No HuggingFace tokens found in .env")

# Active token tracker
current_hf_index = 0

if not GROQ_API_KEY:
    print("ERROR: GROQ_API_KEY missing from .env file")
    exit()

groq_client = Groq(api_key=GROQ_API_KEY)

# ================================
# CHANGE THIS — type your category
# Examples:
# "holiday backgrounds"
# "nature textures"
# "islamic and ramadan"
# "business and technology"
# "marble and stone textures"
# "sky and atmosphere"
# "food"
# "Retor Computer"
# "travel"
#=================================
# "Business & Remote Work Scenes"
# "AI & Technology Concepts"
# "Finance & Money Growth"
# "Healthcare & Wellness"
# "Social Media & Content Creation"
# "E-commerce & Online Shopping"
# "Education & Learning"
# "Lifestyle + Everyday Moments"

# ================================
CATEGORY = "Single shot of different vegetable with white background with studio level lighting."

#image Count

IMAGE_COUNT = 10  #Need to upload 7 image a day for 100 image

# ================================
# ASPECT RATIO — pick one
# ================================
ASPECT_RATIOS = {
    "landscape": (2560, 1600),
    "portrait":  (1600, 2560),
    "square":    (2048, 2048),
}
RATIO = "landscape"

# ================================
# MODEL PRIORITY LIST
# ================================
IMAGE_MODELS = [
    "gptimage-large",
    "flux-realism",
    "flux",
]

os.makedirs("images", exist_ok=True)

# ================================
# STEP 1 — GENERATE TOPICS
# ================================
def generate_topics(category):
    print(f"\nGenerating topics for category: {category}...")

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an Adobe Stock content strategist. "
                        "Your job is to generate high demand stock image topic ideas. "
                        "Rules: "
                        "Topics must be specific and commercially searchable on Adobe Stock. "
                        "Each topic should produce a visually distinct image. "
                        "No people, no faces — backgrounds, textures, and atmospheric scenes only. "
                        f"Return a JSON array of exactly {IMAGE_COUNT} topic strings. "
                        "Each topic should be 3 to 6 words. "
                        "Return raw JSON array only — no explanation, no markdown, nothing else."
                    )
                },
                {
                    "role": "user",
                    "content": f"Generate {IMAGE_COUNT} high demand Adobe Stock image topics for this category: {category}"
                }
            ]
        )

        raw = response.choices[0].message.content.strip()
        clean = raw.replace("```json", "").replace("```", "").strip()
        topics = json.loads(clean)

        if not isinstance(topics, list) or len(topics) == 0:
            raise ValueError("Invalid topics format returned")
        # Trim or pad to exact count
        topics = topics[:IMAGE_COUNT]

        print(f"  ✓ Generated {len(topics)} topics:")
        for i, t in enumerate(topics, 1):
            print(f"    {i}. {t}")

        return topics

    except json.JSONDecodeError:
        print("  ✗ Failed to parse topics JSON — using fallback topics")
        return [f"{category} background {i}" for i in range(1, 6)]

    except Exception as e:
        print(f"  ✗ Topic generation failed: {e}")
        return []


# ================================
# STEP 2 — GENERATE PROMPT + METADATA
# ================================
def generate_prompt(topic):
    print(f"\n  Writing prompt and metadata for: {topic}...")

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a world class Adobe Stock image prompt engineer and SEO expert. "
                        "For every topic return a JSON object with exactly 4 fields: "

                        "prompt: image generation prompt for FLUX model. "
                        "Identify topic type and apply correct style — "
                        "HOLIDAY/EVENT: warm cinematic atmosphere, bokeh lights, rich colors. "
                        "NATURE/LANDSCAPE: photorealistic, golden hour, dramatic sky. "
                        "ABSTRACT/BACKGROUND: clean geometry, smooth gradients, minimal. "
                        "BUSINESS/TECHNOLOGY: clean modern, cool blue tones, sharp lines. "
                        "TEXTURE/SURFACE: macro photography mode, describe material, finish, color, detail, lighting. "
                        "CULTURAL/RELIGIOUS: respectful, atmospheric, symbolic objects, warm lighting. "
                        "Apply variety — rotate between: extreme close up macro, wide cinematic panorama, "
                        "overhead flat lay, dramatic side lighting, soft dreamy light, high contrast moody, "
                        "bright airy minimal, golden hour glow, blue hour cool, misty atmospheric. "
                        "Find unique unexpected angle — avoid obvious interpretations. "
                        "Never include people, faces, text, watermarks, logos, cartoons. "
                        "End prompt with: sharp focus, highly detailed, 4K, Adobe Stock style, award winning photography. "
                        "Keep prompt 60 to 80 words. "

                        "title: professional Adobe Stock title. "
                        "Minimum 5 words. Title Case. No punctuation at end. "
                        "Put most searched keyword FIRST. Be specific not vague. "

                        "description: one to two sentences maximum 200 characters. "
                        "Start with use case (Perfect for...). "
                        "Include 3 to 5 natural keywords. "

                        "tags: exactly 45 keywords for maximum Adobe Stock coverage. "
                        "All lowercase. Single words or 2 word phrases only. "
                        "Order by search volume highest first. "
                        "Cover primary keywords, style, colors, use case, season, occasion, technical, broad category. "
                        "No duplicates. No irrelevant tags. "

                        "STRICT: Return raw JSON only. No markdown. No explanation. Nothing else. "
                        "Exactly this format: "
                        "{\"prompt\": \"...\", \"title\": \"...\", \"description\": \"...\", \"tags\": [\"tag1\", \"tag2\"]}"
                    )
                },
                {
                    "role": "user",
                    "content": f"Generate prompt and metadata for: {topic} stock image"
                }
            ]
        )

        raw = response.choices[0].message.content.strip()
        clean = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean)

        # Validate all fields exist
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
        print(f"  Raw response: {raw[:100]}...")
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

    # Try each token starting from current index
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
                print(f"  ✗ Account {current_hf_index + 1} credits exhausted — trying next account...")
                current_hf_index += 1
                continue

            elif response.status_code == 503:
                print(f"  Model loading, waiting 20 seconds...")
                time.sleep(20)
                continue

            elif response.status_code == 400:
                print(f"  ✗ Bad request — resolution too high, falling back to Pollinations...")
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

    # All tokens exhausted
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
def save_prompt_log(topic, data, model_used):
    log_file = "prompt_notes.txt"
    tags_str = ", ".join(data.get("tags", []))

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"{'='*60}\n")
        f.write(f"TOPIC:       {topic}\n")
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
# SAVE IMAGE HELPER
# ================================
def save_image(topic, content, data=None, model_used="unknown"):
    try:
        filename = topic.replace(" ", "_").replace("/", "_") + ".jpg"
        filepath = os.path.join("images", filename)
        with open(filepath, "wb") as f:
            f.write(content)
        print(f"  ✓ Saved: {filepath}")
        if data:
            save_prompt_log(topic, data, model_used)
        return True
    except Exception as e:
        print(f"  ✗ Failed to save image: {e}")
        return False

# ================================
# MAIN RUNNER
# ================================
def run():
    print(f"\n{'='*50}")
    print(f"Category: {CATEGORY}")
    print(f"Ratio: {RATIO} {ASPECT_RATIOS[RATIO]}")
    print(f"{'='*50}")

    # Step 1 — get topics
    topics = generate_topics(CATEGORY)

    if not topics:
        print("\nNo topics generated. Exiting.")
        return

    print(f"\n{'='*50}")
    print(f"Starting image generation for {len(topics)} topics")
    print(f"{'='*50}")

    success_count = 0
    fail_count = 0

    for i, topic in enumerate(topics, 1):
        print(f"\n[{i}/{len(topics)}] {topic}")

        # Step 2 — generate prompt
        data = generate_prompt(topic)
        prompt = data["prompt"]

        # Step 3 — try HuggingFace first, fallback to Pollinations
        success = generate_huggingface(topic, prompt,data)
        if not success:
            print(f"  Falling back to Pollinations...")
            success = generate_pollinations(topic, prompt,data)

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
    print(f"  Check your images folder!")
    print(f"{'='*50}")

run()