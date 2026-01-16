#!/usr/bin/env python3
"""
BrandDiffusion V39.0 - "EVENT BOOSTER"
- FIX 1 (Event Priority): drastically boosts event keywords (Diwali, Christmas) in the prompt to force the theme.
- FIX 2 (BG Freedom): Lowers ControlNet scale to 0.05 for events so the AI can draw lights/fireworks freely.
- FIX 3 (Subject Context): Auto-injects "Traditional/Festive" clothing for cultural events.
"""
# --- NEW IMPORTS FOR REFINEMENT ---
import mediapipe as mp
from gfpgan import GFPGANer

from diffusers import AutoPipelineForInpainting
import os
import sys
import argparse
import re
import cv2
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from collections import Counter
import gc
from sklearn.cluster import KMeans
import json
import requests
import io
import tempfile
from pathlib import Path


# --- IMPORTS FOR POSTERLLAMA ---
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# ============================================================================
# 🎯 V9.4: CONTEXT-AWARE PRODUCT GENERATION
# ============================================================================

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def extract_products_from_prompt(intent):
    """Extract product mentions from user's prompt - HIGHEST PRIORITY"""
    print("   🔍 Extracting products from user prompt...")

    product_keywords = {
        "shoes": ["shoes", "sneakers", "footwear", "boots", "sandals", "slippers", "heels"],
        "athletic footwear": ["running shoes", "sports shoes", "athletic shoes", "trainers"],
        "clothing": ["clothes", "apparel", "wear", "garments", "outfit"],
        "t-shirts": ["t-shirt", "tshirt", "tee", "tops"],
        "shirts": ["shirt", "blouse"],
        "pants": ["pants", "jeans", "trousers", "denim"],
        "dresses": ["dress", "gown", "frock"],
        "ethnic wear": ["ethnic", "traditional", "kurta", "saree", "lehenga", "salwar"],
        "electronics": ["electronics", "gadgets", "devices"],
        "smartphones": ["phone", "smartphone", "mobile", "iphone", "android"],
        "laptops": ["laptop", "notebook", "computer", "macbook"],
        "tablets": ["tablet", "ipad"],
        "watches": ["watch", "smartwatch", "timepiece"],
        "headphones": ["headphone", "earphone", "airpods", "earbuds"],
        "furniture": ["furniture", "sofa", "chair", "table", "bed"],
        "home decor": ["decor", "decoration", "furnishing"],
        "kitchen": ["kitchen", "cookware", "utensils", "appliances"],
        "cosmetics": ["makeup", "cosmetics", "beauty", "lipstick", "foundation"],
        "skincare": ["skincare", "cream", "serum", "moisturizer"],
        "jewelry": ["jewelry", "jewellery", "necklace", "ring", "earring", "bracelet"],
        "bags": ["bag", "handbag", "backpack", "purse", "wallet"],
        "food": ["food", "snacks", "meal", "dish"],
        "beverages": ["drink", "beverage", "juice", "soda", "coffee", "tea"],
        "sports equipment": ["sports", "fitness", "gym", "equipment", "gear"],
        "books": ["book", "novel", "magazine", "reading"],
        "toys": ["toy", "game", "play"],
        "kids wear": ["kids", "children", "baby"]
    }

    intent_lower = intent.lower()
    detected_products = []

    for category, keywords in product_keywords.items():
        for keyword in keywords:
            if keyword in intent_lower:
                detected_products.append(category)
                break

    detected_products = list(dict.fromkeys(detected_products))

    if detected_products:
        print(f"      ✅ Found products in prompt: {', '.join(detected_products)}")
    else:
        print(f"      ℹ️  No specific products mentioned in prompt")

    print()
    return detected_products

def detect_products_clip(image):
    """Detect specific product types from REFERENCE using CLIP"""
    print("   🔍 Detecting products from reference with CLIP...")

    try:
        from transformers import CLIPProcessor, CLIPModel
        import os
        os.environ['HF_HUB_DISABLE_TORCH_LOAD_CHECK'] = '1'

        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        product_categories = [
            "shoes and sneakers", "athletic footwear", "running shoes",
            "clothing and apparel", "t-shirts and shirts", "pants and jeans",
            "electronics and gadgets", "phones and tablets", "laptops and computers",
            "home goods and furniture", "kitchen appliances", "home decor",
            "toys and games", "sports equipment", "outdoor gear",
            "beauty and cosmetics", "skincare products", "makeup",
            "food and beverages", "snacks and candy", "drinks and sodas",
            "books and media", "office supplies", "stationery",
            "watches and jewelry", "bags and backpacks", "accessories"
        ]

        inputs = processor(text=product_categories, images=image, return_tensors="pt", padding=True)

        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)

        top_probs, top_indices = torch.topk(probs[0], k=5)

        detected_products = []
        for prob, idx in zip(top_probs, top_indices):
            if prob > 0.1:
                product = product_categories[idx]
                confidence = prob.item()
                detected_products.append((product, confidence))

        if detected_products:
            print(f"      ✅ Detected products from reference:")
            for product, conf in detected_products:
                print(f"         - {product} ({conf*100:.1f}%)")
        else:
            print(f"      ⚠️  No specific products detected")
            detected_products = [("retail products and merchandise", 0.5)]

        print()
        return detected_products

    except Exception as e:
        print(f"      ⚠️  CLIP detection failed: {e}\n")
        return [("retail merchandise and products", 0.5)]

def removebg_cutout(logo_path: str, cache_dir: str = "./outputs/logo_cache") -> Image.Image:
    """
    Returns RGBA logo with transparent background using remove.bg API.
    Caches output to avoid repeated API calls.
    """
    api_key = os.getenv("REMOVEBG_API_KEY")
    if not api_key:
        raise RuntimeError("REMOVEBG_API_KEY is not set")

    os.makedirs(cache_dir, exist_ok=True)

    src = Path(logo_path)
    cached = Path(cache_dir) / f"{src.stem}_removebg.png"
    if cached.exists():
        return Image.open(cached).convert("RGBA")

    url = "https://api.remove.bg/v1.0/removebg"
    with open(logo_path, "rb") as f:
        files = {"image_file": f}
        data = {"size": "auto"}  # best available (may consume credits)
        headers = {"X-Api-Key": api_key}

        resp = requests.post(url, files=files, data=data, headers=headers, timeout=60)

    if resp.status_code != 200:
        # helpful debug print
        raise RuntimeError(f"remove.bg failed: {resp.status_code} -> {resp.text[:300]}")

    img = Image.open(io.BytesIO(resp.content)).convert("RGBA")
    img.save(cached)
    return img
#!/usr/bin/env python3
def detect_use_case_with_ai(user_prompt, parsed):
    """Use LLM to intelligently detect use case and suggest appropriate products"""
    print("   🤖 Analyzing intent with AI...")
    
    intent = user_prompt 
    try:
        from groq import Groq
        api_key = os.getenv("GROQ_API_KEY")
        
        if not api_key:
            raise Exception("GROQ_API_KEY not set")
        
        client = Groq(api_key=api_key)
        
        context = f"""
Analyze this marketing intent and determine:

INTENT: "{intent}"

Provide a JSON response with:
1. use_case: One of ["product_launch", "sale", "festival", "general"]
2. festival_name: If festival, which one? (diwali/christmas/holi/eid/new_year/valentine/halloween/thanksgiving) or null
3. suggested_products: Array of 3 contextually relevant products to generate
4. reasoning: Brief explanation

CRITICAL DISTINCTION:
- "Happy Diwali wishes" or "Merry Christmas greetings" → use_case = "festival", suggest decorative festival items (diyas, gifts)
- "Christmas sale 70% off on shoes" → use_case = "sale", suggest shoes (the product being sold)
- "Launching new iPhone" → use_case = "product_launch", suggest iPhone

RULES:
- If intent mentions launching/introducing/unveiling a SPECIFIC product → use_case = "product_launch", suggested_products = that product
- If intent is about festival/celebration → use_case = "festival", suggested_products = festival-appropriate items
- If intent mentions sale/discount/offer → use_case = "sale", suggested_products = relevant retail products
- For festivals: NEVER suggest products like shoes/electronics unless explicitly mentioned. Use diyas for Diwali, gifts for Christmas, etc.

EXAMPLES:

Input: "Launching new iPhone 15 Pro"
Output: {{"use_case": "product_launch", "festival_name": null, "suggested_products": ["iPhone 15 Pro smartphone", "premium phone", "mobile device"], "reasoning": "Specific product launch"}}

Input: "Happy Diwali wishes for customers"
Output: {{"use_case": "festival", "festival_name": "diwali", "suggested_products": ["decorated golden diya lamp", "colorful rangoli design", "traditional gift box"], "reasoning": "Diwali greeting requires festival-appropriate items, not regular products"}}

Input: "50% off on summer collection"
Output: {{"use_case": "sale", "festival_name": null, "suggested_products": ["summer clothing", "light fabric apparel", "seasonal wear"], "reasoning": "Sale with seasonal context"}}

Input: "Check out our new sneakers"
Output: {{"use_case": "general", "festival_name": null, "suggested_products": ["athletic sneakers", "sports shoes", "casual footwear"], "reasoning": "General product showcase"}}

Return ONLY valid JSON.
"""
        
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an expert marketing analyst. Analyze intent and suggest contextually appropriate products."},
                {"role": "user", "content": context}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.3,  # Lower temp for consistent analysis
            max_tokens=400
        )
        
        import json
        result = response.choices[0].message.content.strip()
        
        # Extract JSON
        match = re.search(r'\{.*\}', result, re.DOTALL)
        if match:
            parsed = json.loads(match.group())
            
            use_case = parsed.get("use_case", "general")
            festival_name = parsed.get("festival_name")
            suggested_products = parsed.get("suggested_products", [])
            reasoning = parsed.get("reasoning", "")
            
            print(f"      ✅ USE CASE: {use_case.upper()}")
            if festival_name:
                print(f"      🎭 FESTIVAL: {festival_name.upper()}")
            print(f"      💡 AI REASONING: {reasoning}")
            print(f"      📦 SUGGESTED PRODUCTS: {', '.join(suggested_products[:2])}")
            print()
            
            return use_case, festival_name, suggested_products
        else:
            raise Exception("No valid JSON in response")
    
    except Exception as e:
        print(f"      ⚠️  AI detection failed: {e}")
        print(f"      ℹ️  Falling back to keyword-based detection\n")
        
        # Fallback to keyword matching
        intent_lower = intent.lower()
        
        # Festival Detection
        festival_keywords = {
            "diwali": ["diwali", "deepavali"],
            "christmas": ["christmas", "xmas"],
            "holi": ["holi"],
            "eid": ["eid"],
            "new_year": ["new year"],
            "valentine": ["valentine"],
            "halloween": ["halloween"],
            "thanksgiving": ["thanksgiving"]
        }
        
        for festival, keywords in festival_keywords.items():
            if any(kw in intent_lower for kw in keywords):
                festival_products = get_festival_products(festival)
                return "festival", festival, festival_products
        
        # Sale
        if any(kw in intent_lower for kw in ["sale", "discount", "off", "%"]):
            return "sale", None, []
        
        # Launch
        if any(kw in intent_lower for kw in ["launch", "new product", "introducing"]):
            return "product_launch", None, []
        
        return "general", None, []

def get_festival_products(festival_name):
    """Return appropriate products for each festival (fallback)"""
    festival_products = {
        "diwali": [
            "decorated golden diya lamp", "ornate oil lamp with flame",
            "colorful rangoli design", "traditional Indian gift box"
        ],
        "christmas": [
            "wrapped gift box with red ribbon", "christmas ornament ball",
            "festive present with bow", "holiday gift hamper"
        ],
        "holi": [
            "colorful powder gulal", "vibrant color packets",
            "festive sweets box", "traditional holi treats"
        ],
        "eid": [
            "decorative gift box with crescent", "traditional sweet box",
            "ornate lantern", "festive eid decoration"
        ],
        "new_year": [
            "champagne bottle with gold foil", "celebration gift box",
            "party popper", "festive hamper"
        ],
        "valentine": [
            "heart-shaped gift box", "red roses bouquet",
            "chocolate box with ribbon", "romantic present"
        ],
        "halloween": [
            "decorative orange pumpkin", "halloween candy bowl",
            "spooky decoration", "trick-or-treat basket"
        ],
        "thanksgiving": [
            "decorative harvest pumpkin", "autumn wreath",
            "festive centerpiece", "thanksgiving basket"
        ]
    }
    
    return festival_products.get(festival_name, ["decorative gift box"])

# ============================================================================
# SMART PRODUCT PRIORITIZATION (WITH AI SUGGESTIONS)
# ============================================================================

def prioritize_products_with_ai(use_case, festival_name, ai_suggested_products, prompt_products, reference_products):
    """AI-POWERED product prioritization"""
    print("   🎲 Prioritizing products with AI suggestions...")

    if use_case == "festival":
        # FESTIVAL: Use AI-suggested festival products
        if ai_suggested_products:
            print(f"      ✅ FESTIVAL MODE: Using AI-suggested {festival_name.upper()} products")
            print(f"      📌 Products: {', '.join(ai_suggested_products[:2])}")
            print(f"      ⚠️  IGNORING reference products (not contextually relevant)\n")
            return ai_suggested_products
        else:
            # Fallback
            festival_products = get_festival_products(festival_name)
            print(f"      ✅ FESTIVAL MODE: Using fallback {festival_name.upper()} products\n")
            return festival_products
    
    elif use_case == "product_launch":
        # PRODUCT LAUNCH: AI suggestions > prompt > references
        if ai_suggested_products:
            print(f"      ✅ PRODUCT LAUNCH: Using AI-suggested products")
            print(f"      📌 Products: {', '.join(ai_suggested_products[:2])}\n")
            return ai_suggested_products
        elif prompt_products:
            print(f"      ✅ PRODUCT LAUNCH: Using PROMPT products")
            print(f"      📌 Products: {', '.join(prompt_products[:2])}\n")
            return prompt_products[:3]
        else:
            reference_product_names = [p[0] if isinstance(p, tuple) else p for p in reference_products]
            print(f"      ⚠️  No AI/prompt products, using references\n")
            return reference_product_names[:3]
    
    elif use_case == "sale":
        # SALE: AI > prompt > references
        if ai_suggested_products:
            print(f"      ✅ SALE: Using AI-suggested products")
            print(f"      📌 Products: {', '.join(ai_suggested_products[:2])}\n")
            return ai_suggested_products
        elif prompt_products:
            print(f"      ✅ SALE: Using PROMPT products\n")
            return prompt_products[:3]
        else:
            reference_product_names = [p[0] if isinstance(p, tuple) else p for p in reference_products]
            print(f"      ✅ SALE: Using REFERENCE products\n")
            return reference_product_names[:3]
    
    else:  # general
        # GENERAL: Mix AI > prompt > references
        if ai_suggested_products:
            print(f"      ✅ GENERAL: Using AI-suggested products")
            print(f"      📌 Products: {', '.join(ai_suggested_products[:2])}\n")
            return ai_suggested_products
        elif prompt_products:
            primary_products = prompt_products[:3]
            reference_product_names = [p[0] if isinstance(p, tuple) else p for p in reference_products]
            for ref_prod in reference_product_names[:2]:
                if ref_prod not in primary_products:
                    primary_products.append(ref_prod)
                    if len(primary_products) >= 3:
                        break
            print(f"      ✅ GENERAL: Mixing PROMPT + REFERENCE\n")
            return primary_products
        else:
            reference_product_names = [p[0] if isinstance(p, tuple) else p for p in reference_products]
            print(f"      ✅ GENERAL: Using REFERENCE products\n")
            return reference_product_names[:3]

def extract_colors_advanced(image, n=12):
    """Advanced color extraction with KMeans"""
    print("   🎨 Extracting brand colors...")

    arr = np.array(image)
    pixels = arr.reshape(-1, 3)

    if len(pixels) > 10000:
        idx = np.random.choice(len(pixels), 10000, replace=False)
        pixels = pixels[idx]

    kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
    kmeans.fit(pixels)

    colors = kmeans.cluster_centers_
    labels = kmeans.labels_
    counts = np.bincount(labels)

    idx = np.argsort(-counts)
    sorted_colors = colors[idx]
    sorted_counts = counts[idx]

    results = []
    for color, count in zip(sorted_colors, sorted_counts):
        hex_color = "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))
        percentage = (count / len(labels)) * 100
        results.append((hex_color, percentage))

    print(f"      Top colors:")
    for i, (color, pct) in enumerate(results[:5], 1):
        print(f"         {i}. {color} ({pct:.1f}%)")

    print()
    return results

# ============================================================================
# COMPREHENSIVE REFERENCE ANALYSIS
# ============================================================================

def analyze_reference_comprehensive(image):
    """Complete reference analysis"""
    reference_products = detect_products_clip(image)
    colors = extract_colors_advanced(image, 12)

    arr = np.array(image)
    brightness = arr.mean()

    return {
        "reference_products": reference_products,
        "cleaned_image": image,
        "colors": colors,
        "brightness": brightness,
    }

def analyze_all_references(ref_images, user_prompt, parsed):
    """Analyze ALL references with AI-POWERED USE CASE DETECTION"""
    print("\n" + "="*70)
    print("STEP 1: AI-POWERED CONTEXT ANALYSIS")
    print("="*70 + "\n")

    # ✅ YOU REMOVED THIS BY ACCIDENT
    use_case, festival_name, ai_suggested_products = detect_use_case_with_ai(user_prompt, parsed)
    prompt_products = extract_products_from_prompt(user_prompt)

    print(f"📸 Analyzing {len(ref_images)} reference image(s)...\n")

    all_reference_products = []
    all_colors = []
    brightness_vals = []

    cleaned_images = []
    for img in ref_images:
        analysis = analyze_reference_comprehensive(img)
        cleaned_images.append(analysis["cleaned_image"])
        all_reference_products.extend([p[0] for p in analysis["reference_products"]])
        all_colors.extend([c[0] for c in analysis["colors"][:6]])
        brightness_vals.append(analysis["brightness"])

    # de-dupe while preserving order
    all_reference_products = list(dict.fromkeys(all_reference_products))
    unified_colors = list(dict.fromkeys(all_colors))
    avg_brightness = sum(brightness_vals) / max(1, len(brightness_vals))

    top_reference_products = all_reference_products[:5]

    # ✅ YOU REMOVED THIS TOO
    final_products = prioritize_products_with_ai(
        use_case, festival_name, ai_suggested_products, prompt_products, top_reference_products
    )

    print("\n" + "="*70)
    print("🤖 AI-POWERED INTELLIGENCE:")
    print("="*70)
    print(f"🎭 USE CASE: {use_case.upper()}" + (f" ({festival_name})" if festival_name else ""))
    print(f"📦 FINAL PRODUCTS: {', '.join(final_products[:3])}")
    print(f"🎨 BRAND COLORS: {', '.join(unified_colors[:5])}")
    print(f"💡 BRIGHTNESS: {int(avg_brightness)}/255")
    print("="*70 + "\n")

    return {
        "use_case": use_case,
        "festival_name": festival_name,
        "products": final_products,
        "prompt_products": prompt_products,
        "reference_products": all_reference_products,
        "colors": unified_colors,
        "brightness": avg_brightness,
        "images": cleaned_images,
        "cleaned_image": cleaned_images[0] if cleaned_images else ref_images[0]
    }


def create_canny_control(image):
    """Create Canny edge map"""
    print("   📐 Creating edge map...")

    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 200)
    kernel = np.ones((2,2), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    canny_image = Image.fromarray(edges_rgb)

    print("      ✅ Edge map created\n")
    return canny_image

# ============================================================================
# V9.4 CONTEXT-AWARE PROMPT BUILDING
# ============================================================================

def build_v9_hero_prompt_for_v39(parsed, ref_colors, use_case, final_products):
    """Build V9.4 context-aware prompt for V39 hero product"""
    
    # UNIVERSAL NEGATIVE - NO HUMANS EVER
    base_negative = (
        "human, person, people, face, eyes, nose, mouth, lips, teeth, tongue, skin, "
        "hands, fingers, arms, legs, feet, body, portrait, selfie, model, "
        "holding, grasping, touching, applying, wearing, using, "
        "body parts, human anatomy, facial features, human skin, "
        "text, label, logo, words, letters, watermark, digits, writing, typography, "
        "blurry, distorted, deformed, cluttered, messy, ugly, amateur, "
        "multiple objects, scattered, collage, grid, bad quality, low resolution"
    )
    
    product_type = parsed["product_type"]
    detected_color = parsed["detected_color"]
    
    # Use final prioritized products
    if final_products:
        product_type = final_products[0]
    
    # Color emphasis
    color_weight = f"({detected_color}:1.5)" if detected_color != "neutral" else ""
    
    if use_case == "festival":
        # Festival: Generate festival items, ignore regular products
        prompt = (
            f"professional studio photograph of single decorative festive item, "
            f"celebratory and elegant style, {color_weight}, "
            f"centered composition on clean white surface, "
            f"isolated on pure white background, "
            f"warm professional lighting, commercial quality, "
            f"inspired by color palette: {', '.join(ref_colors[:3])}, "
            f"8k ultra resolution, sharp focus, macro detail, "
            f"product only, no humans, no hands, no body parts"
        )
        controlnet_scale = 0.25
        blur_radius = 15
    
    elif use_case == "product_launch":
        # Product launch: Focus on the specific product
        prompt = (
            f"professional studio photograph of single {product_type}, "
            f"the entire product is prominently colored in {color_weight}, "
            f"centered on pure white background, "
            f"commercial product photography, isolated and floating, "
            f"no hands, no person holding it, just the product object itself, "
            f"inspired by brand colors: {', '.join(ref_colors[:3])}, "
            f"8k ultra high resolution, sharp macro focus, highly detailed, "
            f"professional studio lighting, commercial quality"
        )
        controlnet_scale = 0.40
        blur_radius = 10
    
    else:  # sale or general
        prompt = (
            f"professional studio photograph of single {product_type}, "
            f"the entire product is prominently colored in {color_weight}, "
            f"centered composition, isolated on white background, "
            f"commercial studio lighting, product only without any human, "
            f"brand color palette: {', '.join(ref_colors[:3])}, "
            f"8k resolution, sharp focus, macro detail, "
            f"no hands, no person, no holding, just the product"
        )
        controlnet_scale = 0.60
        blur_radius = 8
    
    return prompt, base_negative, controlnet_scale, blur_radius

# ============================================================================
# V9.4 COMPLETE HERO GENERATION
# ============================================================================

def generate_hero_v9_for_v39(parsed, ref_images, user_prompt):
    primary_ref = ref_images[0]
    """V9.4 COMPLETE Context-Aware Hero Generation"""
    print("\n" + "="*70)
    print("🎯 V9.4: COMPLETE CONTEXT-AWARE HERO GENERATION")
    print("="*70)
    
    clear_memory()
    
    # Step 1: COMPREHENSIVE REFERENCE ANALYSIS
    ref_analysis = analyze_all_references(ref_images, user_prompt, parsed)
    
    prompt_products = ref_analysis['prompt_products']
    clip_products = ref_analysis['reference_products']
    ref_colors = ref_analysis['colors']
    use_case = ref_analysis['use_case']
    festival_name = ref_analysis.get('festival_name')
    final_products = ref_analysis['products']
    
    # Step 2: Build Context-Aware Prompt
    prompt, negative, cn_scale, blur_rad = build_v9_hero_prompt_for_v39(
        parsed, ref_colors, use_case, final_products
    )
    
    print(f"\n✅ V9.4 FINAL CONFIGURATION:")
    print(f"   Use Case: {use_case.upper()}")
    print(f"   Festival: {festival_name.upper() if festival_name else 'N/A'}")
    print(f"   Final Products: {', '.join(final_products[:2])}")
    print(f"   Brand Colors: {', '.join(ref_colors[:3])}")
    print(f"   ControlNet Scale: {cn_scale}")
    print(f"   Blur Radius: {blur_rad}")
    
    # Step 3: Prepare Reference
    w, h = primary_ref.size
    crop_size = int(min(w, h) * 0.6)
    left = int((w - crop_size) / 2)
    top = int((h - crop_size) / 2)
    hero_ref_cropped = primary_ref.crop((left, top, left + crop_size, top + crop_size)).resize((1024, 1024))
    
    # Variable blur based on use case
    hero_ref_blurred = hero_ref_cropped.filter(ImageFilter.GaussianBlur(radius=blur_rad))
    
    # Create Canny edges using V9.4 function
    hero_canny = create_canny_control(hero_ref_blurred)
    
    # Step 4: Generate with V9.4 Logic
    try:
        from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, DPMSolverMultistepScheduler
        
        print(f"\n🔄 Loading V9.4 ControlNet Pipeline...")
        
        controlnet = ControlNetModel.from_pretrained(
            Config.CONTROLNET_CANNY,
            torch_dtype=Config.DTYPE
        ).to(Config.DEVICE)
        
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            Config.SDXL_MODEL_ID,
            controlnet=controlnet,
            torch_dtype=Config.DTYPE,
            variant="fp16"
        ).to(Config.DEVICE)
        
        if Config.DEVICE == "cuda":
            pipe.enable_model_cpu_offload()
            pipe.enable_attention_slicing()
        
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            use_karras_sigmas=True
        )
        
        print(f"🎨 Generating V9.4 Hero Product...")
        
        result = pipe(
    prompt=prompt,
    negative_prompt=negative,
    image=hero_canny,
    num_inference_steps=45,
    guidance_scale=10.0 if use_case == "festival" else 9.0,
    controlnet_conditioning_scale=cn_scale,
    width=1024,
    height=1024
).images[0]

        
        print(f"✅ V9.4 Hero Generated ({use_case.upper()} mode)")
        print(f"   Products: {', '.join(final_products[:2])}")
        print("="*70 + "\n")
        
        del pipe, controlnet
        clear_memory()
        
        return result
    
    except Exception as e:
        print(f"❌ V9.4 generation failed: {e}")
        print(f"⚠️ Falling back to V39 original\n")
        
        # Fallback to V39 original hero generation
        pipe_cn = get_pipeline(Config.SDXL_MODEL_ID, Config.CONTROLNET_CANNY)
        
        result = pipe_cn(
            prompt=parsed["hero_prompt"],
            negative_prompt="text, label, logo, words, hands, holding, blurry",
            image=hero_canny,
            num_inference_steps=40,
            controlnet_conditioning_scale=0.60,
            width=1024,
            height=1024
        ).images[0]
        
        return result

# ============================================================================
# 0. CONFIGURATION
# ============================================================================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

class Config:
    SDXL_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
    CONTROLNET_CANNY = "diffusers/controlnet-canny-sdxl-1.0"
    LAYOUT_BASE_MODEL = "codellama/CodeLlama-7b-hf"
    LAYOUT_ADAPTER_PATH = "./PosterLlama-PKU-Adapter" 
    OUTPUT_DIR = "./outputs"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# ============================================================================
# 🧠 1. POSTERLLAMA LAYOUT ENGINE
# ============================================================================

class LayoutGenerator:
    def __init__(self):
        print(f"   🧠 Loading Layout Model: {Config.LAYOUT_ADAPTER_PATH}...")
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            self.base_model = AutoModelForCausalLM.from_pretrained(
                Config.LAYOUT_BASE_MODEL,
                quantization_config=bnb_config,
                device_map="auto"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(Config.LAYOUT_BASE_MODEL)
            self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = PeftModel.from_pretrained(self.base_model, Config.LAYOUT_ADAPTER_PATH)
            self.model.eval()
            print("   ✅ Layout Model Loaded Successfully.")
            self.active = True
        except Exception as e:
            print(f"   ⚠️ FAILED to load Layout Model: {e}")
            print("   -> Running in FALLBACK mode (Manual Layout).")
            self.active = False

    def generate_layout(self):
        if not self.active: return None
        prompt = "<s>[INST] Design an advertising poster layout. [/INST]"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(Config.DEVICE)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=512, temperature=0.7, do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "[/INST]" in generated_text:
            generated_text = generated_text.split("[/INST]")[1]
        return self.parse_layout(generated_text)

    def parse_layout(self, text):
        pattern = r"<elem type='(.*?)' x='(.*?)' y='(.*?)' w='(.*?)' h='(.*?)' />"
        matches = re.findall(pattern, text)
        elements = []
        for m in matches:
            try:
                bbox = [float(m[1]), float(m[2]), float(m[3]), float(m[4])]
                elements.append({"type": m[0], "bbox": bbox})
            except: continue
        print(f"   📐 Generated {len(elements)} layout elements.")
        return elements

# ============================================================================
# 🧠 2. DYNAMIC PARSER (EVENT BOOST)
# ============================================================================
# ============================================================================
# 💅 HUMAN REFINEMENT PIPELINE (Face + Hands)
# ============================================================================

class HumanRefiner:
    def __init__(self):
        self.face_enhancer = None
        self.mp_hands = None
        self.inpaint_pipe = None
        
        # Initialize Face Restorer (CodeFormer/GFPGAN)
        try:
            # model_path auto-downloads in standard gfpgan usage
            self.face_enhancer = GFPGANer(model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
                                          upscale=1, arch='clean', channel_multiplier=2, bg_upsampler=None)
            print("✅ Face Restoration (CodeFormer/GFPGAN) Loaded")
        except Exception as e:
            print(f"⚠️ Face Restorer Failed to Load: {e}")

        # Initialize Hand Detector (MediaPipe)
        try:
            self.mp_hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
            print("✅ Hand Detector (MediaPipe) Loaded")
        except Exception:
            pass

    def restore_face(self, pil_image):
        """Step 1: Face Restoration"""
        if not self.face_enhancer: return pil_image
        print("      💅 Restoring Face...")
        
        # Convert PIL to CV2
        img_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        # Run Inference
        _, _, output = self.face_enhancer.enhance(img_cv, has_aligned=False, only_center_face=False, paste_back=True)
        
        # Convert back to PIL
        return Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))

    def create_hand_mask(self, pil_image):
        """Generates a mask specifically for hands using MediaPipe"""
        if not self.mp_hands: return None
        
        img_cv = np.array(pil_image)
        results = self.mp_hands.process(img_cv)
        
        if not results.multi_hand_landmarks:
            return None
            
        h, w, _ = img_cv.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        
        padding = 40 # Dilate mask slightly
        
        for hand_landmarks in results.multi_hand_landmarks:
            x_min, y_min = w, h
            x_max, y_max = 0, 0
            
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)
            
            # Draw rectangle on mask
            cv2.rectangle(mask, 
                         (max(0, x_min - padding), max(0, y_min - padding)), 
                         (min(w, x_max + padding), min(h, y_max + padding)), 
                         255, -1)
                         
        return Image.fromarray(mask)

    def inpaint_hands(self, pil_image, prompt):
        """Step 2: Hand Inpainting"""
        mask = self.create_hand_mask(pil_image)
        if mask is None:
            print("      ℹ️ No hands detected, skipping inpainting.")
            return pil_image
            
        print("      💅 Inpainting Hands...")
        
        # Load Inpaint Pipeline on demand to save VRAM
        if self.inpaint_pipe is None:
            self.inpaint_pipe = AutoPipelineForInpainting.from_pretrained(
                "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
                torch_dtype=Config.DTYPE,
                variant="fp16"
            ).to(Config.DEVICE)
            
        # Inpaint
        result = self.inpaint_pipe(
            prompt=prompt,
            negative_prompt="malformed hands, extra fingers, missing fingers, fused fingers, bad anatomy, mutation",
            image=pil_image,
            mask_image=mask,
            strength=0.55, # Keep it subtle
            guidance_scale=7.5,
            num_inference_steps=20
        ).images[0]
        
        return result

    def process(self, image, prompt):
        # 1. Restore Face
        restored = self.restore_face(image)
        # 2. Fix Hands
        final = self.inpaint_hands(restored, prompt)
        return final
        #!/usr/bin/env python3

# ============================================================================
# 🧠 2. DYNAMIC PARSER (EVENT BOOST + REGEX FIX)
# ============================================================================

class DynamicParser:
    @staticmethod
    def parse(user_prompt):
        print(f"🧠 AI AGENT: Analyzing -> '{user_prompt}'")
        text = user_prompt.lower()
        parsed = {
            "subject_type": None, 
            "product_type": None,
            "detected_color": "neutral", 
            "heading": "SPECIAL OFFER",
            "environment_type": None,
            "color_name": "neutral",
            "is_event": False,
            "event_name": ""
        }

        # Helper for Regex Matching
        def has_word(word_list):
            if isinstance(word_list, str): word_list = [word_list]
            for w in word_list:
                # Matches standalone word, ignores "woman", "human", "manual"
                if re.search(r'\b' + re.escape(w) + r'\b', text):
                    return True
            return False

        # --- A. EVENT DETECTION ---
        events = {
            "diwali": {"env": "(Happy Diwali Festival background:1.5), (glowing diyas and rangoli:1.3), golden lights", "outfit": "traditional indian festive clothing, saree or kurta"},
            "christmas": {"env": "(Christmas winter wonderland:1.5), (christmas tree:1.3), red and gold bokeh", "outfit": "winter coat, scarf, festive red clothing"},
            "holi": {"env": "(Holi festival background:1.5), colorful powder explosions", "outfit": "white clothes stained with colorful powder"},
            "eid": {"env": "(Eid Mubarak night background:1.5), crescent moon, lanterns", "outfit": "traditional elegant clothing"},
            "new year": {"env": "(New Year Fireworks background:1.5), party lights", "outfit": "glamorous party dress or suit"},
            "halloween": {"env": "(Halloween spooky background:1.5), pumpkins", "outfit": "halloween costume"}
        }
        
        greeting_keywords = ["happy", "merry", "wishes", "greetings", "celebrate", "mubarak"]
        is_greeting = any(kw in text for kw in greeting_keywords)
        
        for evt, details in events.items():
            if has_word(evt) and is_greeting:
                parsed["is_event"] = True
                parsed["event_name"] = evt
                parsed["environment_type"] = details["env"]
                parsed["event_outfit"] = details["outfit"]
                print(f"   🎉 Event GREETING Detected: {evt.upper()}")
                break
        
        # --- B. PRODUCT DETECTION ---
        common_items = {
            "lipstick": "lipstick", "balm": "lip balm tube",
            "perfume": "glass perfume bottle", "shoe": "sneaker",
            "watch": "luxury wrist watch", "bag": "designer handbag",
            "car": "luxury car", "phone": "smartphone"
        }
        for key, val in common_items.items():
            if has_word(key): 
                parsed["product_type"] = val
                break
        
        if not parsed["product_type"]: parsed["product_type"] = "luxury gift box" 

        # --- C. COLOR ---
        color_adjectives = ["neon", "pastel", "metallic", "matte", "glossy", "dark", "light", "bright", "red", "green", "blue", "yellow", "pink", "purple", "orange", "black", "white", "gold", "silver"]
        for w in text.split():
            clean = w.strip(".,")
            if clean in color_adjectives: 
                parsed["detected_color"] = clean
                parsed["color_name"] = clean
                break
        
        # --- D. SUBJECT DETECTION (REGEX FIX) ---
        user_specified_subject = False
        
        # Strict checking using Word Boundaries (\b)
        if has_word(["woman", "girl", "lady", "female"]):
            parsed["subject_type"] = "woman"
            user_specified_subject = True
        elif has_word(["man", "guy", "boy", "male", "gentleman"]):
            parsed["subject_type"] = "man"
            user_specified_subject = True
        elif has_word("model"):
            parsed["subject_type"] = "woman" # Default for 'model'
            user_specified_subject = True
        else:
            parsed["subject_type"] = "woman" # Fallback Default

        # --- E. HEADING ---
        quoted_match = re.search(r"['\"]([\\w\\s!]+)['\"]", user_prompt)
        if quoted_match:
            parsed["heading"] = quoted_match.group(1).strip().upper()[:25]
        elif parsed["is_event"]:
            parsed["heading"] = f"HAPPY {parsed['event_name'].upper()}"
        else:
            match = re.search(r'(?:called|named|title)\s+([\w\s]+?)(?:\s+(?:of|for|with|in)|$)', user_prompt, re.IGNORECASE)
            if match:
                 words = match.group(1).strip().split()[:4] 
                 parsed["heading"] = " ".join(words).upper()

        # --- PROMPT CONSTRUCTION ---
# --- PROMPT CONSTRUCTION ---
        cw = f"({parsed['detected_color']}:1.5)"
        
        # 1. Check for Custom Background Keyword (e.g. "beach background", "city view")
        # Captures words before "background", "view", or "setting"
        bg_match = re.search(r'([\w\s]+?)\s+(?:background|view|setting|scene)', text)
        custom_bg = bg_match.group(1).strip() if bg_match else None

        if parsed["is_event"]:
            env_prompt = parsed["environment_type"]
        
        # 2. Specific Landmarks (Hardcoded for high quality)
        elif has_word(["tower", "paris", "eiffel"]):
            env_prompt = "iconic view of Eiffel Tower, single large tower in background, parisian sky"
        elif has_word(["bridge", "golden gate", "suspension"]): # <--- ADDED THIS
            env_prompt = "cinematic view of Golden Gate Bridge, massive suspension bridge structure in background, golden hour sky"
        elif has_word(["beach", "ocean", "sea"]):
            env_prompt = "luxury tropical beach background, blurred ocean waves, sunlight"
        elif has_word(["city", "urban", "street"]):
            env_prompt = "blurred modern city street background, bokeh city lights, urban vibe"
            
        # 3. Generic "Smart" Fallback
        # If user said "forest background", we use "forest"
        elif custom_bg and len(custom_bg) > 3:
            env_prompt = f"cinematic background of {custom_bg}, professional photography backdrop"
            
        # 4. Final Default
        else:
            env_prompt = "luxury studio background, clean minimal aesthetic"

        clothing = parsed["event_outfit"] if parsed.get("is_event") else f"elegant {cw} clothing"
        parsed["subject_prompt"] = f"professional photo of ({parsed['subject_type']}:1.5), {env_prompt}, wearing {clothing}, holding {parsed['product_type']}, 8k, highly detailed"

        parsed["hero_prompt"] = f"professional studio photograph of one luxury {parsed['product_type']}, the entire object is prominently colored in {cw}, isolated on white background, 8k, highly detailed, sharp focus"
            
        return parsed
# ============================================================================
# 🛠️ 3. GENERATION ENGINE
# ============================================================================

def get_pipeline(model_id, controlnet=None):
    from diffusers import StableDiffusionXLPipeline, StableDiffusionXLControlNetPipeline, ControlNetModel, DPMSolverMultistepScheduler
    if controlnet:
        cn = ControlNetModel.from_pretrained(controlnet, torch_dtype=Config.DTYPE).to(Config.DEVICE)
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(model_id, controlnet=cn, torch_dtype=Config.DTYPE, variant="fp16")
    else:
        pipe = StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=Config.DTYPE, variant="fp16", use_safetensors=True)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
    pipe.to(Config.DEVICE)
    if Config.DEVICE == "cuda": pipe.enable_model_cpu_offload()
    return pipe

def prepare_uploaded_hero(product_path: str, out_size: int = 1024) -> Image.Image:
    """
    Loads the uploaded product image and returns a square hero image.
    - If PNG with alpha: keeps transparency
    - If JPG/opaque: keeps as-is (you can optionally remove bg later)
    """
    img = Image.open(product_path).convert("RGBA")

    # center-crop to square
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))

    # resize to standard hero size
    img = img.resize((out_size, out_size), Image.LANCZOS)
    return img


def generate_assets(parsed, ref_images, size):
    primary_ref = ref_images[0]
    print(f"   🎨 Generating Visuals...")
    pipe_cn = get_pipeline(Config.SDXL_MODEL_ID, Config.CONTROLNET_CANNY)
    pipe_base = get_pipeline(Config.SDXL_MODEL_ID)

    # 1. Background
    bg_ref_blurred = primary_ref.filter(ImageFilter.GaussianBlur(radius=100))
    bg_edges = cv2.Canny(cv2.cvtColor(np.array(bg_ref_blurred), cv2.COLOR_RGB2BGR), 30, 100)
    bg_canny = Image.fromarray(np.stack([bg_edges]*3, axis=2))
    
    cn_scale_bg = 0.05 if parsed["is_event"] else 0.4
    prompt_bg = f"blurred background view of {parsed['environment_type']}, {parsed['detected_color']} theme"
    
    bg = pipe_cn(
        prompt=prompt_bg, 
        negative_prompt="text, people, objects, man, woman, face, distorted, messy, busy, plain wall", 
        image=bg_canny, 
        num_inference_steps=35, 
        controlnet_conditioning_scale=cn_scale_bg, 
        width=size[0], height=size[1]
    ).images[0]
    
    # 2. Hero Product
    print(f"   📦 Hero selection...")
    use_case, festival_name, ai_products = detect_use_case_with_ai(parsed.get("original_prompt", ""), parsed)

    if use_case == "product_launch" and parsed.get("product_image_path"):
        print("   ✅ Using uploaded product image as HERO (product_launch mode)")
        hero = prepare_uploaded_hero(parsed["product_image_path"], out_size=1024)
    else:
        print("   🎨 Generating Hero with V9.4 Intelligence...")
        hero = generate_hero_v9_for_v39(parsed, ref_images, parsed.get("original_prompt", ""))

    # 3. Subject
    print(f"   👤 Generating {parsed['subject_type']}...")
    base_neg = "text, watermark, ugly, deformed, bad anatomy, blur, extra fingers, fewer digits, extra limbs, extra arms, extra legs, malformed limbs, fused fingers, too many fingers, (text:3.0), (words:3.0), (letters:3.0), (typography:3.0), (watermark:3.0), (logo:2.5), (label:3.0), (caption:3.0), (brand name:3.0), (signature:3.0)"
    
    target = parsed["color_name"]
    if target not in ["red", "pink", "rose", "coral"]:
        base_neg += ", red lipstick, pink lipstick, magenta lipstick, red lips"

    if parsed["subject_type"] == "woman":
         subject_neg = f"{base_neg}, (man:2.0), male, beard, stubble, masculine"
    else:
         subject_neg = f"{base_neg}, (woman:2.0), female, feminine features"

    raw_subject = pipe_base(
        prompt=parsed["subject_prompt"],
        negative_prompt=subject_neg,
        width=896, height=1152
    ).images[0]

    # --- REFINEMENT STEP ---
    # Apply Face Restoration and Hand Inpainting
    print("   ✨ Refining Subject (Face & Hands)...")
    refiner = HumanRefiner()
    subject = refiner.process(raw_subject, parsed["subject_prompt"])
    
    return bg, subject, hero

    
# def textiness_score(pil_img: Image.Image) -> float:
#     """
#     Heuristic: detects small thin connected components similar to text strokes.
#     Lower = better (less text-like clutter).
#     """
#     img = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2GRAY)
#     # emphasize edges
#     edges = cv2.Canny(img, 80, 200)

#     # close small gaps to form components
#     kernel = np.ones((2, 2), np.uint8)
#     closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

#     num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
#     score = 0.0

#     # stats: [label, x, y, w, h, area]
#     for i in range(1, num_labels):
#         x, y, w, h, area = stats[i]
#         if area < 10:
#             continue
#         # text strokes often: small-ish, thin-ish, not huge blocks
#         if 8 <= h <= 80 and 8 <= w <= 200 and area <= 2000:
#             aspect = w / max(1, h)
#             if 1.2 <= aspect <= 12.0:
#                 score += 1.0

#     return score


# def pick_best_hero(candidates):
#     scored = [(textiness_score(img), img) for img in candidates]
#     scored.sort(key=lambda t: t[0])
#     best_score, best_img = scored[0]
#     print(f"   🔎 Hero textiness scores: {[s for s,_ in scored]}  -> picked {best_score}")
#     return best_img

def extract_offer_and_name(user_prompt: str):
    """
    Extracts:
    - collection_name (e.g. SKY JOY)
    - discount (e.g. 50% OFF)
    - launch verb intent
    """
    text = user_prompt.strip()

    # Collection name: called/named "XYZ" or called XYZ
    name = None
    m = re.search(r"(?:called|named|title)\s+'([^']+)'", text, re.IGNORECASE)
    if not m:
        m = re.search(r'(?:called|named|title)\s+"([^"]+)"', text, re.IGNORECASE)
    if not m:
        m = re.search(r"(?:called|named|title)\s+([A-Z0-9][A-Z0-9\s]{2,30})", text)
    if m:
        name = m.group(1).strip()

    # Discount: "50% off", "flat 30% OFF", "upto 70% off"
    disc = None
    m = re.search(r"(\d{1,2}\s?%)(\s*(?:off|OFF))", text)
    if m:
        disc = (m.group(1).replace(" ", "") + " OFF").upper()
    else:
        m = re.search(r"(flat\s+\d{1,2}\s?%)(?:\s*off)?", text, re.IGNORECASE)
        if m:
            disc = (m.group(1).upper().replace(" ", "") + " OFF")

    # Sale trigger
    is_sale = bool(re.search(r"\b(sale|discount|offer|off|deal|limited time)\b", text, re.IGNORECASE))

    return {
        "collection_name": name,
        "discount": disc,
        "is_sale": is_sale
    }

def generate_marketing_copy(parsed, user_prompt):
    """
    Returns dict: {"headline": ..., "subheadline": ..., "cta": ...}
    Forces grounding on product/collection name + discount/sale details.
    """
    meta = extract_offer_and_name(user_prompt)
    collection_name = meta["collection_name"]
    discount = meta["discount"]
    is_sale = meta["is_sale"]

    product = parsed.get("product_type", "collection")
    color = parsed.get("detected_color", "signature")
    use_case = "festival" if parsed.get("is_event") else "product_launch/sale/general"
    if parsed.get("is_event"):
        evt = parsed.get("event_name", "").lower().strip()

        # fallback (if Groq fails)
        festival_fallback = {
            "headline": f"HAPPY {evt.upper()}"[:28],
            "subheadline": "Wishing you joy, light, and warm moments."[:70],
            "cta": "CELEBRATE"[:18],
        }

        try:
            from groq import Groq

            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                return festival_fallback

            client = Groq(api_key=api_key)

            festival_prompt = f"""
Write short festival greeting copy for a poster.

Festival: {evt}
User prompt: "{user_prompt}"

Return ONLY valid JSON:
{{
  "headline": "...",
  "subheadline": "...",
  "cta": "..."
}}

RULES:
- This is a FESTIVAL GREETING, NOT a sale.
- Headline must include the festival name (e.g., "HAPPY DIWALI").
- Subheadline: 6–12 words max, warm and festive, no product push.
- CTA: 1–2 words max (e.g., "CELEBRATE", "SPREAD JOY", "SEND LOVE").
- Do NOT mention discounts, "shop now", prices, or "luxury gift box".
- Keep it simple and not cringe.
"""

            resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You write short, tasteful festival greetings for posters. Return strict JSON only."},
                    {"role": "user", "content": festival_prompt},
                ],
                temperature=0.25,
                max_tokens=140,
            )

            txt = resp.choices[0].message.content.strip()
            m = re.search(r"\{.*\}", txt, re.DOTALL)
            if not m:
                return festival_fallback

            data = json.loads(m.group())

            headline = str(data.get("headline", festival_fallback["headline"])).strip().upper()[:28]
            sub = str(data.get("subheadline", festival_fallback["subheadline"])).strip()[:70]
            cta = str(data.get("cta", festival_fallback["cta"])).strip().upper()[:18]

            # hard safety checks
            banned = ["%", "OFF", "SALE", "DISCOUNT", "SHOP", "BUY", "LUXURY", "GIFT BOX"]
            if any(b in headline for b in banned) or any(b in sub.upper() for b in banned):
                return festival_fallback

            # enforce festival keyword
            if evt and evt.upper() not in headline:
                headline = f"HAPPY {evt.upper()}"[:28]

            return {"headline": headline, "subheadline": sub, "cta": cta}

        except Exception:
            return festival_fallback


    # Strong fallback (grounded, not abstract)
    if discount:
        fallback_headline = f"{discount}"
        fallback_sub = f"Limited-time deal on {collection_name or product}."
        fallback_cta = "SHOP NOW"
    elif collection_name:
        fallback_headline = f"{collection_name}".upper()
        fallback_sub = f"New {product}. {color.title()} vibes, bold finish."
        fallback_cta = "GET IT NOW"
    else:
        fallback_headline = parsed.get("heading", "SPECIAL OFFER").upper()
        fallback_sub = f"New {product} in {color} tones."
        fallback_cta = "BUY NOW"

    fallback = {
        "headline": fallback_headline[:28],
        "subheadline": fallback_sub[:70],
        "cta": fallback_cta[:18],
    }

    try:
        from groq import Groq
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return fallback

        client = Groq(api_key=api_key)

        # 🚨 Hard constraints: must include name/discount when provided
        hard_rules = []
        if collection_name:
            hard_rules.append(f'You MUST include "{collection_name}" in either HEADLINE or SUBHEADLINE.')
        if discount:
            hard_rules.append(f'You MUST include "{discount}" in the HEADLINE.')
        if is_sale:
            hard_rules.append("This is a SALE/OFFER. Copy must clearly communicate the offer (not abstract vibes).")

        hard_rules_text = "\n".join(hard_rules) if hard_rules else "Stay grounded to the product."

        prompt = f"""
You are a senior performance ad copywriter.
Write copy for ONE poster.

Context:
- Use case: {use_case}
- Product type: {product}
- Color theme: {color}
- Collection name (if any): {collection_name}
- Discount/offer (if any): {discount}
- User intent: {user_prompt}

HARD RULES:
{hard_rules_text}

Output format (RETURN ONLY JSON):
{{
  "headline": "...",
  "subheadline": "...",
  "cta": "..."
}}

STYLE RULES:
- Headline: 2–5 words, ALL CAPS, not generic, must be product/offer-driven
- Subheadline: 6–14 words, benefit-led, must mention product if headline is offer-only
- CTA: 1–3 words, ALL CAPS (SHOP NOW / BUY NOW / GET IT NOW / CLAIM OFFER)
- Never invent brand-new names (no “LOVE IN JOY” unless user wrote it)
"""

        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You write grounded, high-converting ad copy. No abstract fluff."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,   # lower randomness = less cringe
            max_tokens=180,
        )

        txt = resp.choices[0].message.content.strip()
        m = re.search(r"\{.*\}", txt, re.DOTALL)
        if not m:
            return fallback

        data = json.loads(m.group())

        headline = str(data.get("headline", fallback["headline"])).strip().upper()
        sub = str(data.get("subheadline", fallback["subheadline"])).strip()
        cta = str(data.get("cta", fallback["cta"])).strip().upper()

        # ✅ enforce constraints even if model ignores them
        if discount and discount not in headline:
            headline = discount

        if collection_name and (collection_name.lower() not in (headline + " " + sub).lower()):
            # force inject into subheadline
            sub = f"{collection_name}: {sub}"[:70]

        return {
            "headline": headline[:28],
            "subheadline": sub[:70],
            "cta": cta[:18]
        }

    except Exception:
        return fallback


def make_logo_background_transparent(logo_rgba: Image.Image, thresh: int = 240) -> Image.Image:
    """
    Converts near-white pixels to transparent.
    Works even if logo has a white box background.
    """
    logo = logo_rgba.convert("RGBA")
    arr = np.array(logo)
    r, g, b, a = arr[..., 0], arr[..., 1], arr[..., 2], arr[..., 3]

    # If alpha already exists and isn't fully opaque everywhere, keep it (still remove near-white).
    near_white = (r >= thresh) & (g >= thresh) & (b >= thresh)
    arr[..., 3] = np.where(near_white, 0, a)

    return Image.fromarray(arr, "RGBA")


def paste_logo_fixed_top_right(
    bg: Image.Image,
    logo_path: str,
    fixed_w: int = 200,     # ✅ fixed width
    fixed_h: int = 150,     # ✅ fixed height box
    margin: int = 0       # ✅ corner margin
) -> bool:
    try:
        # ✅ remove.bg cutout (transparent background)
        logo = removebg_cutout(logo_path)

        # ✅ resize to fit inside a fixed box, preserve aspect ratio
        scale = min(fixed_w / logo.width, fixed_h / logo.height)
        new_w = max(1, int(logo.width * scale))
        new_h = max(1, int(logo.height * scale))
        logo = logo.resize((new_w, new_h), Image.LANCZOS)

        # ✅ top-right corner
        x = bg.width - new_w - margin
        y = margin
        bg.paste(logo, (x, y), logo)
        return True

    except Exception as e:
        print(f"⚠️ Logo paste failed: {e}")
        return False

def bbox_to_px(bbox, w, h):
    bx, by, bw, bh = bbox
    # PosterLlama sometimes outputs slightly > 1.0 even though normalized
    if max(bx, by, bw, bh) <= 2.0:
        bx, by, bw, bh = bx * w, by * h, bw * w, bh * h
    return bx, by, bw, bh

# ============================================================================
# 📐 4. COMPOSITOR
# ============================================================================

class LayoutManager:
    @staticmethod
    def get_font(size):
        options = ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "arialbd.ttf"]
        for path in options:
            if os.path.exists(path): return ImageFont.truetype(path, size)
        return ImageFont.load_default()


    @staticmethod
    def draw_locked_copy(draw, parsed, w, h, text_color):
        copy = parsed.get("copy") or {}
        headline = copy.get("headline", parsed.get("heading", "SPECIAL OFFER")).upper()
        sub = copy.get("subheadline", "")
        cta = copy.get("cta", "SHOP NOW").upper()

        # This is the SAME hero circle geometry you already use:
        p_size = int(w * 0.38)
        circle_x = int(w * 0.08)
        circle_y = h - p_size - 40

        # Define a "safe text box" ABOVE the circle
        box_left = circle_x
        box_right = circle_x + p_size
        box_top = int(h * 0.06)
        box_bottom = circle_y - 20

        box_w = box_right - box_left
        box_h = box_bottom - box_top

        # Fonts
        headline_font = LayoutManager.get_font(int(h * 0.11))  # big
        sub_font = LayoutManager.get_font(int(h * 0.040))
        cta_font = LayoutManager.get_font(int(h * 0.045))

        # Shrink headline to fit width
        hs = int(h * 0.11)
        while hs > 30:
            headline_font = LayoutManager.get_font(hs)
            hb = draw.textbbox((0, 0), headline, font=headline_font)
            if (hb[2] - hb[0]) <= box_w:
                break
            hs -= 4

        shadow = "black" if text_color == "white" else "white"

        x = box_left
        y = box_top

        # HEADLINE (with shadow)
        draw.text((x+4, y+4), headline, font=headline_font, fill=shadow)
        draw.text((x, y), headline, font=headline_font, fill=text_color)

        # SUBHEADLINE
        y += (draw.textbbox((0,0), headline, font=headline_font)[3] + 12)
        if sub:
            # shrink sub to fit
            ss = int(h * 0.040)
            while ss > 18:
                sub_font = LayoutManager.get_font(ss)
                sb = draw.textbbox((0,0), sub, font=sub_font)
                if (sb[2]-sb[0]) <= box_w:
                    break
                ss -= 2

            draw.text((x+2, y+2), sub, font=sub_font, fill=shadow)
            draw.text((x, y), sub, font=sub_font, fill=text_color)
            y += (draw.textbbox((0,0), sub, font=sub_font)[3] + 16)

        # CTA "pill" (optional but catchy)
        # Draw a rounded-ish rectangle using a normal rectangle (simple)
        cta_bbox = draw.textbbox((0,0), cta, font=cta_font)
        pad_x, pad_y = 18, 10
        pill_w = (cta_bbox[2]-cta_bbox[0]) + pad_x*2
        pill_h = (cta_bbox[3]-cta_bbox[1]) + pad_y*2

        pill_x = x
        pill_y = min(y, box_bottom - pill_h)

        fill_rgba = (0,0,0,110) if text_color == "white" else (255,255,255,140)
        draw.rounded_rectangle(
            [pill_x, pill_y, pill_x + pill_w, pill_y + pill_h],
            radius=18,
            fill=fill_rgba
        )
        draw.text((pill_x+pad_x, pill_y+pad_y), cta, font=cta_font, fill=text_color)

    @staticmethod
    def composite(bg, subject, hero, parsed, layout_elements, logo_path=None):
        w, h = bg.size
        draw = ImageDraw.Draw(bg, "RGBA")

        # 1) Subject (Right)
        if subject:
            s_h = int(h * 0.95)
            s_w = int(subject.width * (s_h / subject.height))
            subject = subject.resize((s_w, s_h), Image.LANCZOS)

            mask = Image.new("L", subject.size, 0)
            draw_m = ImageDraw.Draw(mask)
            for i in range(s_w):
                alpha = int(255 * (i / 150)) if i < 150 else 255
                draw_m.line([(i, 0), (i, s_h)], fill=alpha)

            bg.paste(subject, (w - s_w + 50, h - s_h), mask)

        # 2) Hero (Bottom Left)
        if hero:
            p_size = int(w * 0.38)

            try:
                hero_rgba = hero.convert("RGBA")
                hero = hero_rgba.resize((p_size, p_size), Image.LANCZOS)
            except Exception:
                hero = hero.resize((p_size, p_size), Image.LANCZOS)

            mask = Image.new("L", (p_size, p_size), 0)
            ImageDraw.Draw(mask).ellipse((5, 5, p_size - 5, p_size - 5), fill=255)
            
            # Store hero position for text positioning
            hero_x = int(w * 0.08)
            hero_y = h - p_size - 40
            bg.paste(hero, (hero_x, hero_y), mask)

        # 3) Text color
        text_color = "white" if "dark" in parsed.get("detected_color", "") else "black"
        if parsed.get("is_event"):
            text_color = "white"
        shadow = "black" if text_color == "white" else "white"

        # ✅ FIXED TEXT POSITIONING - Above hero, left-aligned
        copy = parsed.get("copy") or {}
        headline = str(copy.get("headline", parsed.get("heading", "SPECIAL OFFER"))).strip().upper()
        sub = str(copy.get("subheadline", "")).strip()

        # Define text area: left-aligned above hero
        text_x = int(w * 0.08)  # Same left margin as hero
        text_y = int(h * 0.02)  # Start from top with padding
        max_text_width = int(w * 0.42)  # Don't go past center

        # ✅ HEADLINE: Each word on new line, left-aligned
        headline_words = headline.split()
        current_y = text_y
        text_bottom_limit = hero_y - 30 
        
        for word in headline_words:
            # Size to fit width
            font_size = 70  # Start large
            font = LayoutManager.get_font(font_size)
            
            while font_size > 40:
                bb = draw.textbbox((0, 0), word, font=font)
                if (bb[2] - bb[0]) < max_text_width:
                    break
                font_size -= 5
                font = LayoutManager.get_font(font_size)
            
            # Draw with shadow (NO white highlight)
            draw.text((text_x, current_y), word, font=font, fill=text_color)
            
            current_y += font_size + 8 # Move down for next word

        # ✅ SUBHEADLINE: Below headline, BIGGER, no white highlight
        if sub:
            current_y += 15  # Gap before subheadline
            
            # BIGGER subheadline - 70% of headline size
            font_size = 20  # Bigger than before
            font = LayoutManager.get_font(font_size)
            
            # Shrink to fit
            while font_size > 28:
                bb = draw.textbbox((0, 0), sub, font=font)
                if (bb[2] - bb[0]) < max_text_width:
                    break
                font_size -= 2
                font = LayoutManager.get_font(font_size)
            
            # Draw WITHOUT shadow (no white highlight)
            draw.text((text_x, current_y), sub, font=font, fill=text_color)

        # Logo: always fixed top-right
        if logo_path: 
            paste_logo_fixed_top_right( bg, logo_path, fixed_w=150, fixed_h=150, margin=0 )

        return bg



# ============================================================================
# 🚀 API-CALLABLE FUNCTION
# ============================================================================

def generate_poster(
    prompt: str,
    ref_images: list,  # List of PIL Images
    logo_image=None,  # Optional PIL Image
    product_image=None,  # Optional PIL Image
    output_dir=None
):
    """
    Generate a marketing poster (API-callable version)
    
    Args:
        prompt: User's text prompt
        ref_images: List of PIL Images (reference images)
        logo_image: Optional PIL Image (logo)
        product_image: Optional PIL Image (product image for hero)
        output_dir: Optional output directory (for temporary files)
    
    Returns:
        Dictionary with:
        - composite: PIL Image of final composite
        - layers: dict of layer_name -> PIL Image
        - metadata: dict with copy, use_case, etc.
    """
    from typing import Optional, List, Dict
    
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="branddiffusion_")
    Config.OUTPUT_DIR = output_dir
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    print("\n🚀 BRANDDIFFUSION V39.0 (API MODE) STARTING")
    
    layout_gen = LayoutGenerator()
    parsed = DynamicParser.parse(prompt)
    parsed["original_prompt"] = prompt
    
    # Store product image path if provided (save to temp file)
    if product_image:
        product_temp = os.path.join(output_dir, "temp_product.png")
        product_image.save(product_temp)
        parsed["product_image_path"] = product_temp
    else:
        parsed["product_image_path"] = None
    
    parsed["copy"] = generate_marketing_copy(parsed, prompt)
    
    print("\n📝 COPY GOING ON IMAGE:")
    print(f"   HEADLINE: {parsed['copy'].get('headline')}")
    print(f"   SUBHEAD:  {parsed['copy'].get('subheadline')}")
    print(f"   CTA:      {parsed['copy'].get('cta')}")
    print()
    
    # Canvas dimensions
    CANVAS_W, CANVAS_H = 1280, 800
    
    # Resize reference images to canvas size
    ref_images_resized = [img.convert("RGB").resize((CANVAS_W, CANVAS_H), Image.LANCZOS) for img in ref_images]
    primary_ref = ref_images_resized[0]
    
    # 1. GENERATE RAW ASSETS
    bg, subject, hero = generate_assets(parsed, ref_images_resized, (CANVAS_W, CANVAS_H))
    
    # ============================================================================
    # 💾 PROCESS LAYERS (store as PIL Images)
    # ============================================================================
    print("\n💾 PROCESSING SEPARATE LAYERS...")
    layer_images = {}
    layer_meta = {}
    
    # --- A. Background ---
    layer_images["background"] = bg
    
    # --- B. Subject with Gradient Mask ---
    if subject:
        s_h = int(CANVAS_H * 0.95)
        s_w = int(subject.width * (s_h / subject.height))
        subject_resized = subject.resize((s_w, s_h), Image.LANCZOS)
        
        # Create Gradient Alpha Mask
        mask = Image.new("L", subject_resized.size, 0)
        draw_m = ImageDraw.Draw(mask)
        for i in range(s_w):
            alpha = int(255 * (i / 150)) if i < 150 else 255
            draw_m.line([(i, 0), (i, s_h)], fill=alpha)
        
        subject_layer = subject_resized.copy()
        subject_layer.putalpha(mask)
        layer_images["subject"] = subject_layer
        
        layer_meta["subject"] = {
            "x": CANVAS_W - s_w + 50,
            "y": CANVAS_H - s_h,
            "width": s_w,
            "height": s_h
        }
    
    # --- C. Hero with Circle Mask ---
    if hero:
        p_size = int(CANVAS_W * 0.38)
        hero_resized = hero.resize((p_size, p_size), Image.LANCZOS)
        
        mask = Image.new("L", (p_size, p_size), 0)
        ImageDraw.Draw(mask).ellipse((5, 5, p_size - 5, p_size - 5), fill=255)
        
        hero_layer = hero_resized.copy()
        hero_layer.putalpha(mask)
        layer_images["hero"] = hero_layer
        
        layer_meta["hero"] = {
            "x": int(CANVAS_W * 0.08),
            "y": CANVAS_H - p_size - 40,
            "width": p_size,
            "height": p_size
        }
    
    # --- D. Logo ---
    if logo_image:
        try:
            # Save logo to temp file for removebg_cutout
            logo_temp = os.path.join(output_dir, "temp_logo.png")
            logo_image.save(logo_temp)
            logo_clean = removebg_cutout(logo_temp)
            layer_images["logo"] = logo_clean
        except Exception as e:
            print(f"⚠️ Could not process logo layer: {e}")
    
    # ============================================================================
    # 🖼️ COMPOSITE FINAL IMAGE
    # ============================================================================
    logo_path = os.path.join(output_dir, "temp_logo.png") if logo_image else None
    if logo_image and not os.path.exists(logo_path):
        logo_image.save(logo_path)
    
    layout_data = layout_gen.generate_layout()
    final = LayoutManager.composite(bg, subject, hero, parsed, layout_data, logo_path)
    
    # ============================================================================
    # 📄 RETURN DICTIONARY
    # ============================================================================
    return {
        "composite": final,
        "layers": layer_images,
        "metadata": {
            "generated_copy": parsed["copy"],  # Frontend expects "generated_copy"
            "use_case": parsed.get("use_case", "general"),
            "is_event": parsed.get("is_event", False),
            "event_name": parsed.get("event_name"),
            "text_color": "white" if parsed.get("is_event") or "dark" in parsed.get("detected_color", "") else "black",
            "layout_meta": layer_meta
        }
    }

# ============================================================================
# 🚀 CLI MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--ref", required=True)
    parser.add_argument("--logo", default=None)
    parser.add_argument("--product_image", default=None, help="Optional: image of the product/collection to use as hero in product_launch")
    parser.add_argument("--outputdir", default="./outputs")
    args = parser.parse_args()
    
    Config.OUTPUT_DIR = args.outputdir
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    print("\n🚀 BRANDDIFFUSION V39.0 (EVENT BOOSTER) STARTING")
    
    layout_gen = LayoutGenerator()
    parsed = DynamicParser.parse(args.prompt)
    parsed["original_prompt"] = args.prompt
    parsed["product_image_path"] = args.product_image
    parsed["copy"] = generate_marketing_copy(parsed, args.prompt)
    
    print("\n📝 COPY GOING ON IMAGE:")
    print(f"   HEADLINE: {parsed['copy'].get('headline')}")
    print(f"   SUBHEAD:  {parsed['copy'].get('subheadline')}")
    print(f"   CTA:      {parsed['copy'].get('cta')}")
    print()

    # Canvas dimensions
    CANVAS_W, CANVAS_H = 1280, 800

    ref_images = [Image.open(args.ref).convert("RGB").resize((CANVAS_W, CANVAS_H))]
    primary_ref = ref_images[0]
    
    # 1. GENERATE RAW ASSETS
    bg, subject, hero = generate_assets(parsed, ref_images, (CANVAS_W, CANVAS_H))

    # ============================================================================
    # 💾 NEW: PROCESS AND SAVE SEPARATE LAYERS
    # ============================================================================
    print("\n💾 SAVING SEPARATE LAYERS...")
    layer_paths = {}
    layer_meta = {}

    # --- A. Save Background ---
    bg_path = os.path.join(Config.OUTPUT_DIR, "layer_1_background.png")
    bg.save(bg_path)
    layer_paths["background"] = bg_path

    # --- B. Process & Save Subject (Woman) with Gradient Mask ---
    # We replicate the LayoutManager logic here to get the isolated PNG
    if subject:
        # Calculate size based on logic in LayoutManager
        s_h = int(CANVAS_H * 0.95)
        s_w = int(subject.width * (s_h / subject.height))
        subject_resized = subject.resize((s_w, s_h), Image.LANCZOS)

        # Create the Gradient Alpha Mask
        mask = Image.new("L", subject_resized.size, 0)
        draw_m = ImageDraw.Draw(mask)
        for i in range(s_w):
            alpha = int(255 * (i / 150)) if i < 150 else 255
            draw_m.line([(i, 0), (i, s_h)], fill=alpha)
        
        # Apply mask to image
        subject_layer = subject_resized.copy()
        subject_layer.putalpha(mask)

        # Save
        subj_path = os.path.join(Config.OUTPUT_DIR, "layer_2_subject.png")
        subject_layer.save(subj_path)
        layer_paths["subject"] = subj_path
        
        # Store coordinates for JSON
        layer_meta["subject"] = {
            "x": CANVAS_W - s_w + 50,
            "y": CANVAS_H - s_h,
            "width": s_w,
            "height": s_h
        }

    # --- C. Process & Save Hero (Product) with Circle Mask ---
    if hero:
        p_size = int(CANVAS_W * 0.38)
        hero_resized = hero.resize((p_size, p_size), Image.LANCZOS)

        # Create Circle Mask
        mask = Image.new("L", (p_size, p_size), 0)
        ImageDraw.Draw(mask).ellipse((5, 5, p_size - 5, p_size - 5), fill=255)

        # Apply mask
        hero_layer = hero_resized.copy()
        hero_layer.putalpha(mask)

        # Save
        hero_path = os.path.join(Config.OUTPUT_DIR, "layer_3_hero.png")
        hero_layer.save(hero_path)
        layer_paths["hero"] = hero_path

        # Store coordinates for JSON
        layer_meta["hero"] = {
            "x": int(CANVAS_W * 0.08),
            "y": CANVAS_H - p_size - 40,
            "width": p_size,
            "height": p_size
        }

    # --- D. Process & Save Logo ---
    if args.logo:
        try:
            # Use your existing helper to get transparent logo
            logo_clean = removebg_cutout(args.logo)
            logo_path = os.path.join(Config.OUTPUT_DIR, "layer_4_logo.png")
            logo_clean.save(logo_path)
            layer_paths["logo"] = logo_path
        except Exception as e:
            print(f"⚠️ Could not save logo layer: {e}")

    # ============================================================================
    # 🖼️ COMPOSITE FINAL IMAGE
    # ============================================================================
    
    layout_data = layout_gen.generate_layout()
    final = LayoutManager.composite(bg, subject, hero, parsed, layout_data, args.logo)
    
    final_path = os.path.join(Config.OUTPUT_DIR, "final_composite.png")
    final.save(final_path)
    print(f"✅ Saved composite to: {final_path}")

    # ============================================================================
    # 📄 GENERATE JSON RESPONSE
    # ============================================================================
    
    json_response = {
        "status": "success",
        "project_name": "BrandDiffusion_V39_Hackathon",
        "dimensions": {"width": CANVAS_W, "height": CANVAS_H},
        "assets": {
            "composite": final_path,
            "layers": layer_paths
        },
        "layout_meta": layer_meta,
        "generated_copy": {
            "headline": parsed['copy'].get('headline'),
            "subheadline": parsed['copy'].get('subheadline'),
            "cta": parsed['copy'].get('cta'),
            "text_color": "white" if parsed.get("is_event") or "dark" in parsed.get("detected_color", "") else "black"
        },
        "marketing_logic": {
            "detected_intent": parsed.get("use_case", "general"),
            "event_detected": parsed.get("is_event", False),
            "event_name": parsed.get("event_name", None)
        }
    }

    # Save JSON to file
    json_path = os.path.join(Config.OUTPUT_DIR, "response.json")
    with open(json_path, "w") as f:
        json.dump(json_response, f, indent=4)

    # Print JSON to console for the Hackathon Demo
    print("\n" + "="*60)
    print("🔮 JSON API RESPONSE (For Frontend/Editor)")
    print("="*60)
    print(json.dumps(json_response, indent=4))
    print("="*60 + "\n")
    #!/usr/bin/env pytho
if __name__ =="__main__":
    main()