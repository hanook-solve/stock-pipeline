from dotenv import load_dotenv
from groq import Groq
import requests
import os
import time

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN8")

groq_client = Groq(api_key=GROQ_API_KEY)

holidays = [
    # Textures and Backgrounds
"Fresh Fruit Background",
"Wooden Kitchen Table Texture",
"Coffee Shop Interior Atmosphere",
"Bakery Pastry Display Shelf",
"Rustic Wine Cellar Walls",
"Farmers Market Produce Stands",
"Vintage Café Decor Patterns",
"Gourmet Cheese Board Settings",
"Busy Restaurant Kitchen Counters",
"Morning Breakfast Table Scene",
]

os.makedirs("images", exist_ok=True)

def generate_prompt(holiday):
    print(f"  Writing prompt for: {holiday}...")
    
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a world class Adobe Stock image prompt engineer. "
"You write prompts for the FLUX image generation model that produce "
"professional, commercially sellable stock images. "

"FIRST — identify the topic type and apply the correct style: "
"HOLIDAY/EVENT → warm cinematic atmosphere, bokeh lights, rich colors, festive mood. "
"NATURE/LANDSCAPE → photorealistic, golden hour or blue hour lighting, dramatic sky, environmental depth. "
"ABSTRACT/BACKGROUND → clean geometry, smooth gradients, elegant color palette, minimal composition. "
"BUSINESS/TECHNOLOGY → clean modern aesthetic, cool blue tones, sharp lines, professional feel. "
"SKY/ATMOSPHERE → wide angle, dramatic clouds or stars, natural lighting, vast depth. "
"CULTURAL/RELIGIOUS → respectful, atmospheric, symbolic objects, warm dignified lighting. "
"VARIETY RULE — before writing every prompt, silently decide which visual approach "
"fits this topic best from this list: "
"extreme close up macro, wide cinematic panorama, overhead flat lay, "
"dramatic side lighting, soft dreamy diffused light, high contrast dark moody, "
"bright airy minimal, golden hour warm glow, blue hour cool tones, misty atmospheric, "
"bold graphic flat lay, stormy dramatic sky, underwater ethereal, "
"backlit silhouette, frost and ice crystal detail. "
"Pick the approach that makes this specific topic look most stunning and commercial. "
"Never pick the same approach twice in a row across different topics. "
"Do not mention the approach name in the prompt — just apply it naturally. "

"TEXTURE AND BACKGROUND TOPICS — when the topic contains words like: "
"texture, background, pattern, surface, fabric, material, wall, floor, paper, metal, wood, marble, concrete: "
"switch to macro photography mode. "
"Describe the exact surface with these details: "
"1. Material type (marble, oak wood, linen fabric, concrete, copper metal). "
"2. Surface finish (polished, matte, rough, glossy, weathered, cracked). "
"3. Color details (warm ivory with grey veining, deep charcoal with rust patches). "
"4. Texture detail (fine grain, coarse weave, smooth with subtle imperfections). "
"5. Lighting (raking side light to reveal texture, soft diffused light, studio flat light). "
"Always end texture prompts with: "
"seamless texture, macro shot, hyper detailed surface, 4K, Adobe Stock style. "
"NEVER add decorations, objects, or scenes to texture topics — surface only. "

"EVERY prompt must include all 5 elements: "
"1. SUBJECT — the main focal element clearly described. "
"2. SETTING — the environment and background context. "
"3. LIGHTING — specific light source, direction, and quality. "
"4. PALETTE — 2 to 3 dominant colors that define the mood. "
"5. CAMERA — lens type, angle, and depth of field. "

"END every prompt with: "
"sharp focus, highly detailed, 4K resolution, Adobe Stock style, award winning photography. "

"NEVER include: "
"people, faces, hands, body parts, text, watermarks, logos, "
"cartoons, anime, random unrelated objects, busy cluttered compositions. "

"ALWAYS produce: "
"clean compositions, professional quality, commercially appealing results. "

"Keep prompt between 60 and 80 words. "
"Reply with the prompt only — no labels, no explanation, nothing else."
    )
            },
            {
                "role": "user",
                "content": f"Write a prompt for a {holiday} holiday background stock image."
            }
        ]
    )
    
    return response.choices[0].message.content.strip()


def generate_image(holiday, prompt, attempt=1):
    API_URL = "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = { "inputs": prompt,
    "parameters": {
        "width": 2560, # More aspect ratio options: landscape *2560*, portrait *1600*, square *2048*
        "height": 1600, # More aspect ratio options: landscape *1600*, portrait *2560*, square *2048*
        }
    }

    try:
        print(f"  Generating image (attempt {attempt})...")
        response = requests.post(API_URL, headers=headers, json=payload, timeout=120)

        if response.status_code == 200:
            filename = holiday.replace(" ", "_") + ".jpg"
            filepath = os.path.join("images", filename)
            with open(filepath, "wb") as f:
                f.write(response.content)
            print(f"  ✓ Saved: {filepath}")
            return True
        
        elif response.status_code == 503:
            print(f"  Model loading, waiting 20 seconds...")
            time.sleep(20)
            return generate_image(holiday, prompt, attempt)
        
        else:
            print(f"  ✗ Failed: {response.status_code} - {response.text}")
            return False

    except requests.exceptions.Timeout:
        if attempt < 3:
            print(f"  Timed out, retrying...")
            time.sleep(10)
            return generate_image(holiday, prompt, attempt + 1)
        else:
            print(f"  ✗ Skipping after 3 attempts")
            return False


for holiday in holidays:
    print(f"\n--- {holiday} ---")
    prompt = generate_prompt(holiday)
    print(f"  Prompt: {prompt[:80]}...")
    generate_image(holiday, prompt)
    time.sleep(5)

print("\nAll done! Check your images folder.")