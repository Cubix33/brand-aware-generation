#!/usr/bin/env python3
"""
BrandDiffusion V39.0 - "GROQ AGENT EDITION"
Replaced PosterLlama and regex parsing with LLM Agents.
"""
import os
import sys
import argparse
import re
import cv2
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import gc
from sklearn.cluster import KMeans
import json
import requests
import io
from pathlib import Path
import textwrap
import mediapipe as mp
from gfpgan import GFPGANer
from diffusers import AutoPipelineForInpainting
from groq import Groq

# ============================================================================
# 0. CONFIGURATION
# ============================================================================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ['HF_HUB_DISABLE_TORCH_LOAD_CHECK'] = '1'

class Config:
    SDXL_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
    CONTROLNET_CANNY = "diffusers/controlnet-canny-sdxl-1.0"
    OUTPUT_DIR = "./outputs"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# ============================================================================
# 🧠 1. STRATEGIST AGENT
# ============================================================================
class StrategistAgent:
    @staticmethod
    def parse(user_prompt):
        print(f"🧠 STRATEGIST AGENT: Analyzing -> '{user_prompt}'")
        
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key: raise ValueError("GROQ_API_KEY is missing in environment variables.")
        client = Groq(api_key=api_key)

        system_instruction = """
        You are an adaptable, world-class Creative Director and Copywriter. 
        Analyze the user's request, identify the core use case (e.g., product launch, festival greeting, clearance sale), and determine the appropriate brand vibe (e.g., luxury, sporty, playful, tech, everyday retail).
        
        Write HIGH-CONVERTING, CONTEXT-AWARE marketing copy and design prompts that perfectly match the inferred vibe.
        
        Output JSON format strictly:
        {
            "is_event": boolean,
            "event_name": "string (e.g., 'Diwali', 'Black Friday', or empty)",
            "detected_color": "string (e.g., red, neon green, matte black, pastel pink)",
            "environment_prompt": "string (highly detailed SDXL photography background tailored to the vibe. MUST BE COMPLETELY EMPTY OF PEOPLE OR HUMANS. e.g., 'empty neon cyberpunk street', 'clean minimal white studio')",
            "product_type": "string (e.g., lipstick, running sneakers, coffee mug)",
            "subject_type": "string (man, woman, or empty if no human needed)",
            "marketing_copy": {
                "headline": "string (MUST INCLUDE any exact discounts/offers mentioned. Adapt tone to the product. ALL CAPS. 5 words only.)",
                "subheadline": "string (Write exactly 2 complete sentences. Each sentence should be 8-12 words. Total length 16-18. Every sentence must be concrete, meaningful, and directly tied to the user's prompt.",
                "cta": "string (Action-oriented, ALL CAPS, relevant to the context prompt, don't create information on your own)."
            }
        }
        
        CRITICAL RULES:
1. PROMPT FIDELITY FIRST: The subheadline must directly reflect the user's prompt, product, occasion, and any stated offer or collection.
2. NO ABSTRACT FLUFF: Avoid vague phrases like "shine bright", "embrace elegance", "radiant beauty", "timeless glamour" unless the user explicitly asked for that style.
3. BE SPECIFIC: Mention the product use, collection, sale, finish, wear, occasion, or offer when relevant.
4. TONE ADAPTATION: Sneakers = energetic and sporty. Lipstick/beauty = polished and beauty-focused. Jewelry = elegant. Festival = festive but still concrete.
5. ACCURACY: Do not hallucinate discounts, benefits, ingredients, or performance claims.
6. SUBHEADLINE STYLE: Write exactly 2 or 3 complete sentences.
7. Each sentence should be 5 to 7 words.
8. Every sentence must make sense on its own.
9. Avoid keyword stuffing, vague emotional filler, and poetic nonsense.
10. Write like a strong professional ad copywriter, not a generic inspirational quote generator.

GOOD EXAMPLES:
- "Discover festive shades for Diwali nights. Smooth color glides on effortlessly. Shop the limited lipstick collection."
- "Bold red shades for evening wear. Lightweight formulas feel comfortable longer. Explore the Diwali collection today."
- "Celebrate Diwali with statement shades. Creamy color delivers a polished finish. Shop the exclusive lipstick range."

BAD EXAMPLES:
- "Radiant hues celebrate festive elegance. Glamour glides across every smile."
- "Shine bright with timeless beauty. Indulge in luxury tonight."
        """

        try:
            completion = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            return json.loads(completion.choices[0].message.content)
        except Exception as e:
            print(f"⚠️ Strategist failed: {e}")
            return {
                "is_event": False, "event_name": "", "detected_color": "neutral",
                "environment_prompt": "luxury studio background, clean minimal aesthetic",
                "product_type": "product", "subject_type": "woman",
                "marketing_copy": {"headline": "SPECIAL OFFER", "subheadline": "Limited time only.", "cta": "SHOP NOW"}
            }

# ============================================================================
# 📐 2. LAYOUT AGENT (Collision-Aware Engine)
# ============================================================================
class LayoutAgent:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def generate_layout(self, width, height, copy_data, layer_meta):
        print(f"   📐 LAYOUT AGENT: Calculating collision-free layout for {width}x{height}...")
        
        # Extract the forbidden zones (where the images were pasted)
        subj = layer_meta.get('subject', {"x": 700, "y": 0, "w": 500, "h": 800})
        hero = layer_meta.get('hero', {"x": 100, "y": 400, "width": 400, "height": 400})
        
        head_text = copy_data.get("headline", "SPECIAL OFFER")
        sub_text = copy_data.get("subheadline", "")
        cta_text = copy_data.get("cta", "SHOP NOW")
        # Safely get the exact logo dimensions calculated in main()
        logo_w = layer_meta.get("logo", {}).get("w", 200)
        logo_h = layer_meta.get("logo", {}).get("h", 100)
        
        # We pre-calculate the target math to guide the LLM's JSON example
        logo_target_x = width - logo_w - 25

        prompt = f"""
        You are a highly precise Graphic Design Layout Engine. 
        Canvas Size: {width}px wide x {height}px high.
        
        YOUR GOAL: Place 4 text elements without overlapping the FORBIDDEN ZONES.
        
        🚨 FORBIDDEN ZONES (Image Layers - DO NOT OVERLAP THESE):
        1. Subject Person: Starts at X:{subj['x']}, Y:{subj['y']}. Spans to X:{subj['x']+subj['w']}, Y:{subj['y']+subj['h']}. (Occupies right side).
        2. Hero Product: Starts at X:{hero['x']}, Y:{hero['y']}. Spans to X:{hero['x']+hero['width']}, Y:{hero['y']+hero['height']}. (Occupies bottom-left).
        
        ELEMENTS TO PLACE:
        1. "headline": "{head_text}" (40px of padding from the left canvas edge. Requires at least 500px width. Put in Top-Left empty space).
        2. "subheadline": "{sub_text}" (Indent to create a staggered look. MUST NOT overlap the headline or forbidden zones).
        3. "cta": "{cta_text}" (Must be near the bottom, but leave at least 40px of padding from the canvas edges).
        4. "logo": The logo image is EXACTLY {logo_w}px wide and {logo_h}px high. You MUST calculate the exact X coordinate for the Top-Right corner so there is exactly 25px of padding from the right edge. (Hint: X should be Canvas Width - Logo Width - 25). Y should be 25.
        
        Output ONLY valid JSON in exact absolute pixels:
        {{
            "headline": {{"x": 80, "y": 80, "w": 500, "h": 150}},
            "subheadline": {{"x": 600, "y": 100, "w": 400, "h": 100}},
            "cta": {{"x": 540, "y": 670, "w": 200, "h": 60}},
            "logo": {{"x": {logo_target_x}, "y": 25, "w": {logo_w}, "h": {logo_h}}}
        }}
        """
        try:
            completion = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "system", "content": prompt}],
                temperature=0.1, 
                response_format={"type": "json_object"}
            )
            layout_dict = json.loads(completion.choices[0].message.content)
            print(f"   ✅ Smart Layout generated avoiding Hero & Subject.")
            return layout_dict
        except Exception as e:
            print(f"⚠️ Layout Agent failed: {e}")
            return {}

# ============================================================================
# 💅 3. HUMAN REFINEMENT PIPELINE
# ============================================================================
class HumanRefiner:
    def __init__(self):
        self.face_enhancer = None
        self.mp_hands = None
        self.inpaint_pipe = None
        try:
            self.face_enhancer = GFPGANer(model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
                                          upscale=1, arch='clean', channel_multiplier=2, bg_upsampler=None)
            print("✅ Face Restoration (GFPGAN) Loaded")
        except Exception as e:
            print(f"⚠️ Face Restorer Failed to Load: {e}")
        try:
            self.mp_hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
            print("✅ Hand Detector (MediaPipe) Loaded")
        except Exception:
            pass

    def restore_face(self, pil_image):
        if not self.face_enhancer: return pil_image
        print("      💅 Restoring Face...")
        img_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        _, _, output = self.face_enhancer.enhance(img_cv, has_aligned=False, only_center_face=False, paste_back=True)
        return Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))

    def create_hand_mask(self, pil_image):
        if not self.mp_hands: return None
        img_cv = np.array(pil_image)
        results = self.mp_hands.process(img_cv)
        if not results.multi_hand_landmarks: return None
        h, w, _ = img_cv.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        padding = 40
        for hand_landmarks in results.multi_hand_landmarks:
            x_min, y_min = w, h
            x_max, y_max = 0, 0
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min, y_min = min(x_min, x), min(y_min, y)
                x_max, y_max = max(x_max, x), max(y_max, y)
            cv2.rectangle(mask, (max(0, x_min - padding), max(0, y_min - padding)), 
                          (min(w, x_max + padding), min(h, y_max + padding)), 255, -1)
        return Image.fromarray(mask)

    def inpaint_hands(self, pil_image, prompt):
        mask = self.create_hand_mask(pil_image)
        if mask is None:
            print("      ℹ️ No hands detected, skipping inpainting.")
            return pil_image
        print("      💅 Inpainting Hands...")
        if self.inpaint_pipe is None:
            self.inpaint_pipe = AutoPipelineForInpainting.from_pretrained(
                "diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=Config.DTYPE, variant="fp16"
            ).to(Config.DEVICE)
        result = self.inpaint_pipe(prompt=prompt, negative_prompt="malformed hands, extra fingers", 
                                   image=pil_image, mask_image=mask, strength=0.55, guidance_scale=7.5, num_inference_steps=20).images[0]
        return result

    def process(self, image, prompt):
        restored = self.restore_face(image)
        return self.inpaint_hands(restored, prompt)

# ============================================================================
# 🛠️ 4. GENERATION UTILS
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

def removebg_cutout(logo_path: str, cache_dir: str = "./outputs/logo_cache") -> Image.Image:
    api_key = os.getenv("REMOVEBG_API_KEY")
    if not api_key: raise RuntimeError("REMOVEBG_API_KEY is not set")
    os.makedirs(cache_dir, exist_ok=True)
    src = Path(logo_path)
    cached = Path(cache_dir) / f"{src.stem}_removebg.png"
    if cached.exists(): return Image.open(cached).convert("RGBA")
    url = "https://api.remove.bg/v1.0/removebg"
    with open(logo_path, "rb") as f:
        resp = requests.post(url, files={"image_file": f}, data={"size": "auto"}, headers={"X-Api-Key": api_key}, timeout=60)
    if resp.status_code != 200: raise RuntimeError(f"remove.bg failed: {resp.status_code}")
    img = Image.open(io.BytesIO(resp.content)).convert("RGBA")
    img.save(cached)
    return img

def prepare_uploaded_hero(product_path: str, out_size: int = 1024) -> Image.Image:
    img = Image.open(product_path).convert("RGBA")
    w, h = img.size
    side = min(w, h)
    left, top = (w - side) // 2, (h - side) // 2
    return img.crop((left, top, left + side, top + side)).resize((out_size, out_size), Image.LANCZOS)

def create_canny_control(image):
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 200)
    edges = cv2.dilate(edges, np.ones((2,2), np.uint8), iterations=1)
    return Image.fromarray(cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB))

def get_subject_content_bbox(pil_image):
    """Uses Edge Detection to find the actual person inside the image, ignoring empty space."""
    # Convert to grayscale OpenCV image
    gray = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2GRAY)
    
    # Slight blur to ignore noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Detect edges (the outline of the person)
    edges = cv2.Canny(blurred, 50, 150)
    
    # Find all non-empty pixels (the person)
    pts = cv2.findNonZero(edges)
    
    # If it fails to find anything, return the full image dimensions as a fallback
    if pts is None:
        return 0, 0, pil_image.width, pil_image.height
        
    # Get the tight bounding box around the edges
    x, y, w, h = cv2.boundingRect(pts)
    
    # Add a small 20px padding around the person so text doesn't touch them directly
    x = max(0, x - 20)
    y = max(0, y - 20)
    w = min(pil_image.width - x, w + 40)
    h = min(pil_image.height - y, h + 40)
    
    return x, y, w, h

def rects_overlap(a, b, padding=20):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ax1 -= padding
    ay1 -= padding
    ax2 += padding
    ay2 += padding
    bx1 -= padding
    by1 -= padding
    bx2 += padding
    by2 += padding

    return not (ax2 < bx1 or ax1 > bx2 or ay2 < by1 or ay1 > by2)


def get_text_block_bbox(draw, lines, font, x, y, line_gap=8):
    max_w = 0
    total_h = 0
    line_heights = []

    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font, stroke_width=2)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        max_w = max(max_w, w)
        line_heights.append(h)

    if line_heights:
        total_h = sum(line_heights) + line_gap * (len(line_heights) - 1)

    return (x, y, x + max_w, y + total_h)

def normalize_subheadline(text):
    text = re.sub(r"\s+", " ", str(text or "")).strip()

    parts = [p.strip() for p in re.split(r"[.!?]+", text) if p.strip()]
    cleaned = []

    for part in parts:
        words = part.split()

        # enforce roughly 6 words per sentence
        if len(words) < 4:
            continue
        if len(words) > 6:
            words = words[:6]

        sentence = " ".join(words).strip()
        if sentence:
            cleaned.append(sentence)

    # keep 2 or 3 sentences only
    cleaned = cleaned[:3]

    # fallback if model gives garbage
    if len(cleaned) < 2:
        words = text.split()
        if len(words) >= 12:
            cleaned = [
                " ".join(words[0:6]),
                " ".join(words[6:12])
            ]
            if len(words) >= 18:
                cleaned.append(" ".join(words[12:18]))
        else:
            cleaned = [
                "Radiant colors celebrate festive elegance",
                "Silky formulas glide through celebrations"
            ]

    return ". ".join(s.strip().capitalize() for s in cleaned if s.strip()) + "."
    
def extract_colors_advanced(image, n=12):
    arr = np.array(image)
    pixels = arr.reshape(-1, 3)
    if len(pixels) > 10000:
        pixels = pixels[np.random.choice(len(pixels), 10000, replace=False)]
    kmeans = KMeans(n_clusters=n, random_state=42, n_init=10).fit(pixels)
    colors, counts = kmeans.cluster_centers_, np.bincount(kmeans.labels_)
    idx = np.argsort(-counts)
    return ["#{:02x}{:02x}{:02x}".format(int(c[0]), int(c[1]), int(c[2])) for c in colors[idx]]

def generate_hero_v9_for_v39(parsed, ref_images, user_prompt):
    clear_memory()
    primary_ref = ref_images[0]
    ref_colors = extract_colors_advanced(primary_ref)
    
    color_weight = f"({parsed['detected_color']}:1.5)" if parsed['detected_color'] != "neutral" else ""
    prompt = f"professional studio photograph of single {parsed['product_type']}, {color_weight}, isolated on pure white background, 8k, sharp focus"
    if parsed.get('is_event') and parsed.get('event_name'):
        prompt = f"professional studio photograph of single {parsed['product_type']}, {color_weight}, themed for {parsed['event_name']}, festive atmospheric background, 8k, sharp focus"
    else:
        prompt = f"professional studio photograph of 2-3 {parsed['product_type']}, {color_weight}, isolated on pure white background, 8k, sharp focus"
        
    base_negative = "human, hands, person, text, logo, watermark, ugly, blurry, multiple objects"
    
    hero_ref_blurred = primary_ref.filter(ImageFilter.GaussianBlur(radius=10)).resize((1024, 1024))
    hero_canny = create_canny_control(hero_ref_blurred)
    
    pipe_cn = get_pipeline(Config.SDXL_MODEL_ID, Config.CONTROLNET_CANNY)
    return pipe_cn(prompt=prompt, negative_prompt=base_negative, image=hero_canny, 
                   num_inference_steps=40, controlnet_conditioning_scale=0.4, width=1024, height=1024).images[0]

def generate_assets(parsed, ref_images, size):
    primary_ref = ref_images[0]
    print(f"   🎨 Generating Visuals...")

    pipe_cn = get_pipeline(Config.SDXL_MODEL_ID, Config.CONTROLNET_CANNY)
    from diffusers import StableDiffusionXLPipeline
    pipe_base = StableDiffusionXLPipeline(
        vae=pipe_cn.vae, text_encoder=pipe_cn.text_encoder, text_encoder_2=pipe_cn.text_encoder_2,
        tokenizer=pipe_cn.tokenizer, tokenizer_2=pipe_cn.tokenizer_2, unet=pipe_cn.unet, scheduler=pipe_cn.scheduler
    )
    if Config.DEVICE == "cuda":
        pipe_base.enable_model_cpu_offload()
    else:
        pipe_base.to(Config.DEVICE)

    # 1. Background
    bg_ref_blurred = primary_ref.filter(ImageFilter.GaussianBlur(radius=100))
    bg_canny = create_canny_control(bg_ref_blurred).resize(size)
    cn_scale_bg = 0.05 if parsed.get("is_event") else 0.4
    
    prompt_bg = f"empty scenery, completely devoid of people, blurred background view of {parsed['environment_type']}, {parsed['detected_color']} theme"
    bg_negative = "(human:2.0), (person:2.0), (people:2.0), (woman:2.0), (man:2.0), (face:2.0), (body:2.0), crowd, silhouettes, pedestrians, text, objects"
    
    bg = pipe_cn(prompt=prompt_bg, negative_prompt=bg_negative, image=bg_canny, 
                 num_inference_steps=35, controlnet_conditioning_scale=cn_scale_bg, width=size[0], height=size[1]).images[0]

    # 2. Hero
    print(f"   📦 Hero selection...")
    if parsed.get("product_image_path"):
        hero = prepare_uploaded_hero(parsed["product_image_path"])
    else:
        hero = generate_hero_v9_for_v39(parsed, ref_images, parsed.get("original_prompt", ""))

    # 3. Subject
    print(f"   👤 Generating {parsed['subject_type']}...")
    base_neg = "(giant object:2.0), (oversized product:2.0), (holding near mouth:2.0), (eating:2.0), (floating object:1.5), text, watermark, ugly, deformed, bad anatomy, bad hands"
    subject_neg = f"{base_neg}, (man:2.0)" if parsed["subject_type"] == "woman" else f"{base_neg}, (woman:2.0)"
    
    raw_subject = pipe_base(prompt=parsed["subject_prompt"], negative_prompt=subject_neg, width=896, height=1152).images[0]

    print("   ✨ Refining Subject (Face & Hands)...")
    refiner = HumanRefiner()
    subject = refiner.process(raw_subject, parsed["subject_prompt"])
    
    return bg, subject, hero

# ============================================================================
# 📐 5. COMPOSITOR
# ============================================================================
class LayoutManager:
    @staticmethod
    def get_font(size):
        options = ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "arialbd.ttf"]
        for path in options:
            if os.path.exists(path): return ImageFont.truetype(path, size)
        return ImageFont.load_default()

    @staticmethod
    def composite(bg, subject, hero, parsed, layout_elements, logo_path=None):
        base_canvas = bg.convert("RGBA")
        w, h = base_canvas.size
        draw = ImageDraw.Draw(base_canvas, "RGBA")

        # 1) Subject (Right) - RESTORED EXACT ORIGINAL PERFECT BLEND
        if subject:
            s_h = int(h * 0.95)
            s_w = int(subject.width * (s_h / subject.height))
            subject = subject.resize((s_w, s_h), Image.LANCZOS).convert("RGBA")
            mask = Image.new("L", subject.size, 0)
            draw_m = ImageDraw.Draw(mask)
            # The exact left-edge gradient logic you had originally
            for i in range(s_w): 
                draw_m.line([(i, 0), (i, s_h)], fill=int(255 * (i / 150)) if i < 150 else 255)
            base_canvas.paste(subject, (w - s_w + 50, h - s_h), mask)

        # 2) Hero (Bottom Left)
        if hero:
            p_size = int(w * 0.38)
            hero = hero.resize((p_size, p_size), Image.LANCZOS).convert("RGBA")
            mask = Image.new("L", (p_size, p_size), 0)
            ImageDraw.Draw(mask).ellipse((5, 5, p_size - 5, p_size - 5), fill=255)
            base_canvas.paste(hero, (int(w * 0.08), h - p_size - 40), mask)

                # 3) Text & Effects

        if not isinstance(layout_elements, dict): layout_elements = {}
        copy = parsed.get("copy") or {}

        # 🚨 FORCED WHITE TEXT
        pure_white = (255, 255, 255, 255)

        # --- A. HEADLINE: Neon Glow + Pure White Text ---
        headline = str(copy.get("headline", "SPECIAL OFFER")).upper()
        head_box = layout_elements.get("headline", {"x": 100, "y": 80, "w": 600})
        fs_head = 75
        h_font = LayoutManager.get_font(fs_head)
        wrapped_h = textwrap.wrap(headline, width=22) 
        glow_layer = Image.new("RGBA", base_canvas.size, (0, 0, 0, 0))
        g_draw = ImageDraw.Draw(glow_layer)
        curr_y = head_box.get('y', 80)
        for line in wrapped_h:
            for off in range(1, 12):
                g_draw.text((head_box.get('x', 100), curr_y), line, font=h_font, fill=(255, 215, 0, 35))

            # Headline forced to pure white
            g_draw.text((head_box.get('x', 100), curr_y), line, font=h_font, fill=pure_white)
            curr_y += fs_head + 10
        base_canvas = Image.alpha_composite(base_canvas, glow_layer)
        draw = ImageDraw.Draw(base_canvas, "RGBA") 

        # --- B. SUBHEADLINE: Sentence-Aware Placement ---
        sub = str(copy.get("subheadline", ""))
        if sub:
            sub_box = layout_elements.get("subheadline", {"x": 300, "y": 300, "w": 360})
            fs_sub = 24
            s_font = LayoutManager.get_font(fs_sub)

            sub_x = sub_box.get("x", 300)
            sub_y = sub_box.get("y", 300)

            # one sentence per visual line
            wrapped_s = [s.strip() + "." for s in re.split(r"[.!?]+", sub) if s.strip()]
            wrapped_s = wrapped_s[:3]

            line_gap = 10
            head_x = head_box.get("x", 100)
            head_y = head_box.get("y", 80)

            actual_head_w = max(
                [(draw.textbbox((0, 0), line, font=h_font)[2] - draw.textbbox((0, 0), line, font=h_font)[0]) for line in wrapped_h] + [0]
            )
            actual_head_h = len(wrapped_h) * (fs_head + 10)

            headline_bbox = (
                head_x,
                head_y,
                head_x + actual_head_w,
                head_y + actual_head_h
            )

            # avoid headline
            sub_bbox = get_text_block_bbox(draw, wrapped_s, s_font, sub_x, sub_y, line_gap=line_gap)

            if rects_overlap(sub_bbox, headline_bbox, padding=24):
                # first try shifting right
                candidate_x = headline_bbox[2] + 40
                candidate_bbox = get_text_block_bbox(draw, wrapped_s, s_font, candidate_x, sub_y, line_gap=line_gap)

                if candidate_bbox[2] < w - 40:
                    sub_x = candidate_x
                    sub_bbox = candidate_bbox
                else:
                    # otherwise move below headline
                    candidate_y = headline_bbox[3] + 28
                    sub_y = candidate_y
                    sub_bbox = get_text_block_bbox(draw, wrapped_s, s_font, sub_x, sub_y, line_gap=line_gap)

            # avoid hero
            hero_meta = layout_elements.get("_hero_meta")
            if not hero_meta:
                hero_meta = {"x": int(w * 0.08), "y": h - int(w * 0.38) - 40, "w": int(w * 0.38), "h": int(w * 0.38)}

            hero_bbox = (
                hero_meta["x"],
                hero_meta["y"],
                hero_meta["x"] + hero_meta["w"],
                hero_meta["y"] + hero_meta["h"]
            )

            if rects_overlap(sub_bbox, hero_bbox, padding=25):
                candidate_x = hero_bbox[2] + 36
                candidate_bbox = get_text_block_bbox(draw, wrapped_s, s_font, candidate_x, sub_y, line_gap=line_gap)

                if candidate_bbox[2] < w - 40:
                    sub_x = candidate_x
                    sub_bbox = candidate_bbox
                else:
                    candidate_y = hero_bbox[1] - (sub_bbox[3] - sub_bbox[1]) - 24
                    if candidate_y > 40:
                        sub_y = candidate_y
                        sub_bbox = get_text_block_bbox(draw, wrapped_s, s_font, sub_x, sub_y, line_gap=line_gap)

                        # avoid subject
            subject_meta = layout_elements.get("_subject_meta")
            if subject_meta:
                subject_bbox = (
                    subject_meta["x"],
                    subject_meta["y"],
                    subject_meta["x"] + subject_meta["w"],
                    subject_meta["y"] + subject_meta["h"]
                )

                if rects_overlap(sub_bbox, subject_bbox, padding=30):
                    # first try moving left of subject
                    candidate_x = subject_bbox[0] - (sub_bbox[2] - sub_bbox[0]) - 40
                    candidate_bbox = get_text_block_bbox(draw, wrapped_s, s_font, candidate_x, sub_y, line_gap=line_gap)

                    if candidate_x > 40 and candidate_bbox[0] > 40:
                        sub_x = candidate_x
                        sub_bbox = candidate_bbox
                    else:
                        # otherwise move above subject
                        candidate_y = subject_bbox[1] - (sub_bbox[3] - sub_bbox[1]) - 30
                        if candidate_y > 40:
                            sub_y = candidate_y
                            sub_bbox = get_text_block_bbox(draw, wrapped_s, s_font, sub_x, sub_y, line_gap=line_gap)
                            
            final_sub_y = sub_y
            sub_bbox = get_text_block_bbox(draw, wrapped_s, s_font, sub_x, sub_y, line_gap=line_gap)
            if rects_overlap(sub_bbox, headline_bbox, padding=24):
                sub_y = headline_bbox[3] + 30
                sub_bbox = get_text_block_bbox(draw, wrapped_s, s_font, sub_x, sub_y, line_gap=line_gap)
            for line in wrapped_s:
                draw.text((sub_x + 2, final_sub_y + 2), line, font=s_font, fill=(0, 0, 0, 150))
                draw.text((sub_x, final_sub_y), line, font=s_font, fill=pure_white, stroke_width=2, stroke_fill=(0, 0, 0, 255))
                final_sub_y += fs_sub + line_gap
                
        # --- C. CTA: Glassmorphism Pill + Centered Text ---
        cta = str(copy.get("cta", "SHOP NOW")).upper()
        if cta:
            c_font = LayoutManager.get_font(28)
            
            # 1. Get precise pixel dimensions of the actual text
            c_bbox = draw.textbbox((0, 0), cta, font=c_font)
            text_w = c_bbox[2] - c_bbox[0]
            text_h = c_bbox[3] - c_bbox[1]
            
            # 2. Size the pill dynamically based on the text size
            pill_w = text_w + 80  # 40px padding on left/right
            pill_h = text_h + 50  # 25px padding on top/bottom
            
            # 3. Get AI coordinates (with safety bounds so it doesn't fall off canvas)
            cta_box = layout_elements.get("cta", {"x": (w // 2) - (pill_w // 2), "y": h - 130})
            cta_x = cta_box.get('x', (w // 2) - (pill_w // 2))
            cta_y = min(cta_box.get('y', h - 130), h - pill_h - 30) 
            
            # 4. Draw the Glass Pill
            glass = Image.new("RGBA", base_canvas.size, (0, 0, 0, 0))
            ov_draw = ImageDraw.Draw(glass)
            ov_draw.rounded_rectangle([cta_x, cta_y, cta_x + pill_w, cta_y + pill_h], 
                                      radius=pill_h//2, fill=(255, 255, 255, 60), outline=(255, 255, 255, 150), width=2)
            
            base_canvas = Image.alpha_composite(base_canvas, glass)
            draw = ImageDraw.Draw(base_canvas, "RGBA") 
            
            # 5. Calculate perfect mathematical center for the text
            text_x = cta_x + (pill_w - text_w) / 2 - c_bbox[0]
            text_y = cta_y + (pill_h - text_h) / 2 - c_bbox[1]
            
            draw.text((text_x, text_y), cta, font=c_font, fill=pure_white)

            
# 4) Logo (Driven completely by AI Layout Engine)
        if logo_path:
            try:
                # Default fallback just in case the AI fails
                l_box = layout_elements.get("logo", {"x": w-200-25, "y": 25, "w": 200, "h": 100})
                
                logo = removebg_cutout(logo_path).convert("RGBA")
                logo.thumbnail((l_box.get('w', 200), l_box.get('h', 100)), Image.LANCZOS)
                
                # Trust the X and Y coordinates generated by the LLM
                base_canvas.paste(logo, (l_box.get('x'), l_box.get('y')), logo)
            except Exception as e: 
                print(f"⚠️ Logo placement failed: {e}")

        return base_canvas.convert("RGB")
        #!/usr/bin/env python3
def save_layout_debug_image(bg_image, layout_elements, output_path):
    if not layout_elements or not isinstance(layout_elements, dict): return None
    debug_img = bg_image.copy().convert("RGB")
    draw = ImageDraw.Draw(debug_img)
    
    for label, coords in layout_elements.items():
        if isinstance(coords, dict):
            x1 = coords.get("x", 0)
            y1 = coords.get("y", 0)
            x2 = x1 + coords.get("w", 0)
            y2 = y1 + coords.get("h", 0)
            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=5)
            draw.text((x1, y1 - 15), label, fill=(255, 0, 0))
            
    debug_img.save(output_path)
    print(f"    ✅ Layout debug image saved: {output_path}")
    return output_path

# ============================================================================
# 🚀 MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--ref", required=True)
    parser.add_argument("--logo", default=None)
    parser.add_argument("--product_image", default=None)
    parser.add_argument("--outputdir", default="./outputs")
    args = parser.parse_args()
    
    Config.OUTPUT_DIR = args.outputdir
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    print("\n🚀 BRANDDIFFUSION V39.0 (GROQ AGENT EDITION) STARTING")
    
    # 1. STRATEGIST AGENT
    parsed_data = StrategistAgent.parse(args.prompt)
    if "marketing_copy" in parsed_data:
        parsed_data["marketing_copy"]["subheadline"] = normalize_subheadline(
            parsed_data.get("marketing_copy", {}).get("subheadline", "")
        )
    parsed = {
        "heading": parsed_data.get('marketing_copy', {}).get('headline', 'SPECIAL OFFER'),
        "copy": parsed_data.get('marketing_copy', {}),
        "environment_type": parsed_data.get('environment_prompt', 'studio background'),
        "product_type": parsed_data.get('product_type', 'product'),
        "subject_type": parsed_data.get('subject_type', 'woman'),
        "detected_color": parsed_data.get('detected_color', 'neutral'),
        "color_name": parsed_data.get('detected_color', 'neutral'),
        "is_event": parsed_data.get('is_event', False),
        "event_name": parsed_data.get('event_name', ''),
        "subject_prompt": f"professional elegant photo of a {parsed_data.get('subject_type', 'woman')}, {parsed_data.get('environment_prompt', 'studio')}. They are posing naturally. A small, correctly-proportioned {parsed_data.get('product_type', 'product')} is featured subtly. 8k, photorealistic, natural scaling.",
        "original_prompt": args.prompt,
        "product_image_path": args.product_image
    }

    print("\n📝 STRATEGY GENERATED:")
    print(f"   Intent: {parsed['event_name'] if parsed['is_event'] else 'General'}")
    print(f"   Copy: {parsed['heading']}")

    CANVAS_W, CANVAS_H = 1280, 800
    ref_images = [Image.open(args.ref).convert("RGB").resize((CANVAS_W, CANVAS_H))]
    
    # 2. GENERATE RAW ASSETS
    bg, subject, hero = generate_assets(parsed, ref_images, (CANVAS_W, CANVAS_H))

    print("\n💾 SAVING SEPARATE LAYERS...")
    layer_paths = {}
    layer_meta = {}

    bg_path = os.path.join(Config.OUTPUT_DIR, "layer_1_background.png")
    bg.save(bg_path)
    layer_paths["background"] = bg_path

    if subject:
        s_h = int(CANVAS_H * 0.95)
        s_w = int(subject.width * (s_h / subject.height))
        subj_x = CANVAS_W - s_w + 50
        subj_y = CANVAS_H - s_h
        
        # Resize subject first
        resized_subject = subject.resize((s_w, s_h), Image.LANCZOS)
        
        # 🚨 USE COMPUTER VISION TO FIND THE ACTUAL PERSON
        cx, cy, cw, ch = get_subject_content_bbox(resized_subject)
        
        # Calculate the absolute canvas coordinates of just the person
        true_x = subj_x + cx
        true_y = subj_y + cy
        
        # Pass ONLY the true bounding box to the AI as the "Forbidden Zone"
        layer_meta["subject"] = {"x": true_x, "y": true_y, "w": cw, "h": ch}
        
        subj_path = os.path.join(Config.OUTPUT_DIR, "layer_2_subject.png")
        resized_subject.save(subj_path)
        layer_paths["subject"] = subj_path
        print(f"    ✅ Subject layer saved. True content bounding box: {layer_meta['subject']}")

    if hero:
        p_size = int(CANVAS_W * 0.38)
        hero_path = os.path.join(Config.OUTPUT_DIR, "layer_3_hero.png")
        hero.resize((p_size, p_size)).save(hero_path)
        layer_paths["hero"] = hero_path
        layer_meta["hero"] = {"x": int(CANVAS_W * 0.08), "y": CANVAS_H - p_size - 40, "width": p_size, "height": p_size}
        
    if args.logo:
        try:
            logo_path = os.path.join(Config.OUTPUT_DIR, "layer_4_logo.png")
            logo_img = removebg_cutout(args.logo)
            # Resize the logo to find its final bounding box dimensions early
            logo_img.thumbnail((200, 100), Image.LANCZOS)
            logo_w, logo_h = logo_img.size
            logo_img.save(logo_path)
            
            layer_paths["logo"] = logo_path
            # Save the exact dimensions to pass to the LayoutAgent
            layer_meta["logo"] = {"w": logo_w, "h": logo_h} 
        except Exception as e:
            print(f"⚠️ Logo processing failed: {e}")

    # 3. GENERATE LAYOUT & COMPOSITE
    print("   📐 Generating Context-Aware Layout with Groq...")
    layout_agent = LayoutAgent()
    # Pass the ENTIRE layer_meta dictionary so the AI knows where both images are
    layout_data = layout_agent.generate_layout(CANVAS_W, CANVAS_H, parsed['copy'], layer_meta)
    if isinstance(layout_data, dict) and "hero" in layer_meta:
        layout_data["_hero_meta"] = {
            "x": layer_meta["hero"]["x"],
            "y": layer_meta["hero"]["y"],
            "w": layer_meta["hero"]["width"],
            "h": layer_meta["hero"]["height"]
        }
    if isinstance(layout_data, dict) and "subject" in layer_meta:
        layout_data["_subject_meta"] = {
            "x": layer_meta["subject"]["x"],
            "y": layer_meta["subject"]["y"],
            "w": layer_meta["subject"]["w"],
            "h": layer_meta["subject"]["h"]
        }
    debug_path = os.path.join(Config.OUTPUT_DIR, "debug_layout_bboxes.png")
    save_layout_debug_image(bg, layout_data, debug_path)
    
    final = LayoutManager.composite(bg, subject, hero, parsed, layout_data, args.logo)
    final_path = os.path.join(Config.OUTPUT_DIR, "final_compositee.png")
    final.save(final_path)
    print(f"✅ Saved composite to: {final_path}")

    # 4. JSON RESPONSE
    json_response = {
        "status": "success",
        "dimensions": {"width": CANVAS_W, "height": CANVAS_H},
        "assets": {"composite": final_path, "layers": layer_paths, "debug_layout": debug_path},
        "layout_meta": layer_meta,
        "generated_copy": parsed['copy']
    }

    json_path = os.path.join(Config.OUTPUT_DIR, "response.json")
    with open(json_path, "w") as f:
        json.dump(json_response, f, indent=4)

    print("\n" + "="*60)
    print("🔮 JSON API RESPONSE")
    print("="*60)
    print(json.dumps(json_response, indent=4))
    print("="*60 + "\n")

if __name__ =="__main__":
    main()
