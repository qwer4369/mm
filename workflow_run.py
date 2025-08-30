import requests
import json
import io
from PIL import Image

import os

# --- إعداداتك الشخصية (يفضل وضع التوكنات في متغيرات بيئة) ---
# Environment variables: HF_TOKEN, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
YOUR_HF_TOKEN = os.environ.get('HF_TOKEN') or ''
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN') or ''
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID') or ''

# --- عناوين API للنماذج المختارة ---
# Candidate hosted inference endpoints to try (first one that responds will be used)
DEEPSEEK_CANDIDATES = [
    "https://api-inference.huggingface.co/models/deepseek-ai/DeepSeek-R1",
    "https://api-inference.huggingface.co/models/meta-llama/Llama-3.1-8B-Instruct",
    "https://api-inference.huggingface.co/models/google/flan-t5-large",
    "https://api-inference.huggingface.co/models/gpt2",
]
DEEPSEEK_API_URL = None
FLUX_API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-dev"
FLUX_FILL_API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-Fill-dev"

# Fallback image generation endpoints to try if FLUX fails
IMAGE_CANDIDATES = [
    "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2",
    "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5",
    "https://api-inference.huggingface.co/models/CompVis/stable-diffusion-v1-4",
]

headers = {"Authorization": f"Bearer {YOUR_HF_TOKEN}"}

def generate_prompt_with_deepseek(task_description):
    print("1. محاولة استخدام نموذج مستضاف لتوليد الأمر (probe ثم استدعاء)...")
    payload = {
        "inputs": f"""
        Act as a world-class visual artist and a descriptive writer. 
        Generate a highly detailed, photorealistic prompt for an image generation model. 
        The scene should be: {task_description}.
        Describe the lighting, atmosphere, composition, colors, and specific details to make the image stunning and hyper-realistic.
        """,
        "parameters": {
            "max_new_tokens": 256, 
            "return_full_text": False,
        }
    }

    # probe candidates to find a working endpoint
    global DEEPSEEK_API_URL
    if DEEPSEEK_API_URL is None:
        for cand in DEEPSEEK_CANDIDATES:
            try:
                probe = {"inputs": "Hello", "parameters": {"max_new_tokens": 2}}
                r = requests.post(cand, headers=headers, json=probe, timeout=8)
                if r.status_code == 200:
                    DEEPSEEK_API_URL = cand
                    print(f"استخدام endpoint: {cand}")
                    break
            except Exception:
                continue

    if DEEPSEEK_API_URL:
        try:
            r = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=30)
            r.raise_for_status()
            data = r.json()
            # try common response shapes
            if isinstance(data, list) and len(data) and isinstance(data[0], dict):
                if 'generated_text' in data[0]:
                    generated_text = data[0]['generated_text']
                elif 'text' in data[0]:
                    generated_text = data[0]['text']
                else:
                    generated_text = json.dumps(data[0])
            elif isinstance(data, dict):
                generated_text = data.get('generated_text') or data.get('text') or json.dumps(data)
            else:
                generated_text = str(data)
            print("✅ تم استلام الأمر بنجاح من الـ endpoint")
            return str(generated_text).strip()
        except requests.exceptions.RequestException as e:
            print(f"❌ فشل استدعاء الـ endpoint المختار: {e}")

    # local fallback template
    print("⚠️ لم يتم العثور على endpoint صالح، استخدام مولد محلي للـ prompt (fallback)")
    # Stronger, structured fallback prompt engineered for image diffusion models.
    template = (
        "You are a world-class visual concept artist and prompt engineer. "
        "Return a single, high-quality English prompt suitable for image diffusion models (do NOT add commentary). "
        f"Subject: {task_description}. "
        "Include: lighting (type, angle, color), mood, camera model and lens, aperture/focal length, composition, color palette, textures, and 3-4 vivid specific details. "
        "Also include a short NEGATIVE_PROMPT line listing common artifacts to avoid (comma-separated): e.g. lowres, deformed, watermark, extra limbs, bad anatomy. "
        "Format strictly as two lines: the first line is the prompt; the second line begins with 'NEGATIVE_PROMPT:' followed by comma-separated terms."
    )
    return template

def generate_image_with_flux(prompt, try_alternatives=True, max_retries=3):
    """Generate image via FLUX. Retries and optional alternative endpoints.

    Returns image bytes or None.
    """
    print("\n2. جاري إرسال الأمر إلى FLUX.1-dev لتوليد الصورة (قد يستغرق هذا بعض الوقت)...")
    payload = {"inputs": prompt, "options": {"wait_for_model": True}}

    # helper to attempt a single endpoint with retries
    def _attempt_endpoint(url):
        last_err = None
        for attempt in range(1, max_retries+1):
            try:
                print(f"Attempt {attempt} -> {url} ...")
                response = requests.post(url, headers=headers, json=payload, timeout=120)
                response.raise_for_status()
                # if response is raw image bytes
                ctype = response.headers.get('content-type', '')
                if ctype.startswith('image'):
                    print("✅ تم توليد الصورة بنجاح!")
                    return response.content
                # try parse json for base64
                data = response.json()
                if isinstance(data, dict) and data.get('image_base64'):
                    import base64
                    print("✅ تم توليد الصورة بنجاح (base64)!")
                    return base64.b64decode(data['image_base64'])
                # some models return list/dict with 'generated_image' or similar
                if isinstance(data, dict) and any(k in data for k in ('generated_image','image')):
                    v = data.get('generated_image') or data.get('image')
                    if isinstance(v, str):
                        import base64
                        return base64.b64decode(v)
                # otherwise log and treat as failure
                print('Unknown response shape from image endpoint:', type(data), str(data)[:400])
                return None
            except requests.exceptions.RequestException as e:
                last_err = e
                print(f"Request failed (attempt {attempt}) -> {e}")
        if last_err:
            print(f"All {max_retries} attempts failed for {url}: {last_err}")
        return None

    # try main FLUX endpoint first
    res = _attempt_endpoint(FLUX_API_URL)
    if res:
        return res

    # try alternatives if enabled
    if try_alternatives:
        print('FLUX failed — محاولة استخدام بدائل لتوليد الصورة...')
        for cand in IMAGE_CANDIDATES:
            try:
                res = _attempt_endpoint(cand)
                if res:
                    return res
            except Exception as e:
                print('Alternative endpoint error:', e)

    print('❌ فشل توليد الصورة عبر جميع المسارات المتاحة.')
    return None


def inpaint_image_with_flux_fill(image_bytes, inpaint_prompt, mask_bytes=None):
    """Attempt to call FLUX Fill endpoint to edit an image.
    If the endpoint requires different payload shape, this function may need
    to be adapted to the model's API. Currently it will try a simple JSON
    payload with base64-encoded image and prompt. Returns edited image bytes
    or None on failure.
    """
    print("\n3. محاولة تعديل الصورة: أولاً محاولة تعديل محلي عالي الجودة (diffusers) إن أمكن، وإلا سنحاول FLUX Fill ثم الوقوع إلى البدائل المحلية.")
    # Try a local diffusers inpainting pipeline if available (best quality if model & resources exist)
    try:
        from io import BytesIO
        try:
            import torch
            from diffusers import StableDiffusionInpaintPipeline
        except Exception as _err:
            raise RuntimeError('diffusers/torch not available')

        print('Local diffusers available — attempting inpainting locally (may require model weights to be present or to download them).')
        # model id to try (user can change to any inpainting-compatible model)
        model_id = 'runwayml/stable-diffusion-inpainting'  # typical HF model id
        pipe = None
        try:
            pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32, revision='fp16' if torch.cuda.is_available() else None)
        except Exception as e:
            print('Could not load local diffusers model from hub or cache:', repr(e))
            pipe = None

        if pipe is not None:
            try:
                pipe = pipe.to('cuda') if torch.cuda.is_available() else pipe
                # prepare inputs
                init_image = Image.open(BytesIO(image_bytes)).convert('RGB')
                if mask_bytes:
                    mask_image = Image.open(BytesIO(mask_bytes)).convert('RGB')
                else:
                    # if no mask provided, create a full-image fill mask (replace background) — user prompt should direct
                    mask_image = Image.new('RGB', init_image.size, (255,255,255))

                result = pipe(prompt=inpaint_prompt, image=init_image, mask_image=mask_image, guidance_scale=7.5)
                out_img = result.images[0]
                buf = BytesIO()
                out_img.save(buf, format='PNG')
                print('Local diffusers inpainting succeeded')
                return buf.getvalue()
            except Exception as e:
                print('Local diffusers inpainting failed:', repr(e))
    except Exception as e:
        # not available or failed, continue to remote attempts
        print('Local diffusers not usable:', repr(e))

    print("\n3. جاري إرسال الصورة والقناع إلى FLUX.1-Fill-dev للتعديل...")
    import base64
    # First attempt: JSON base64 payload (existing)
    try:
        payload = {
            "inputs": inpaint_prompt,
            "image": base64.b64encode(image_bytes).decode('utf-8')
        }
        if mask_bytes:
            payload['mask_image'] = base64.b64encode(mask_bytes).decode('utf-8')
        response = requests.post(FLUX_FILL_API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        # expected binary image content or base64 in json
        if response.headers.get('content-type', '').startswith('image'):
            return response.content
        data = response.json()
        if isinstance(data, dict) and 'image_base64' in data:
            return base64.b64decode(data['image_base64'])
        # unexpected shape: log and fallthrough to next attempt
        print('Unknown response shape from inpainting endpoint (json):', type(data), data if isinstance(data, dict) else str(data)[:400])
    except Exception as e:
        print('Inpainting JSON request failed:', repr(e))

    # Second attempt: multipart/form-data with files (some endpoints expect files)
    try:
        files = {'image': ('image.png', image_bytes, 'image/png')}
        data = {'inputs': inpaint_prompt}
        if mask_bytes:
            files['mask_image'] = ('mask.png', mask_bytes, 'image/png')
        # no Authorization header for files: pass token as header still
        response = requests.post(FLUX_FILL_API_URL, headers=headers, files=files, data=data, timeout=60)
        response.raise_for_status()
        if response.headers.get('content-type', '').startswith('image'):
            return response.content
        # try parse json for base64
        data = response.json()
        if isinstance(data, dict) and 'image_base64' in data:
            return base64.b64decode(data['image_base64'])
        print('Unknown response shape from inpainting endpoint (multipart):', type(data), str(data)[:400])
    except Exception as e:
        print('Inpainting multipart request failed:', repr(e))

    # Final fallback: local simple edit using Pillow to guarantee a result for common edits
    try:
        from PIL import Image, ImageDraw
        im = Image.open(io.BytesIO(image_bytes)).convert('RGBA')
        w, h = im.size

        def draw_iraq_flag(size):
            fw, fh = size
            flag = Image.new('RGBA', (fw, fh), (255, 255, 255, 255))
            draw = ImageDraw.Draw(flag)
            # three horizontal stripes: red, white, black
            draw.rectangle([0, 0, fw, fh // 3], fill=(206, 17, 38))
            draw.rectangle([0, fh // 3, fw, 2 * fh // 3], fill=(255, 255, 255))
            draw.rectangle([0, 2 * fh // 3, fw, fh], fill=(0, 0, 0))
            # simple green rectangle in center of white stripe to suggest emblem
            gx0 = fw // 2 - fw // 8
            gy0 = fh // 2 - fh // 24
            gx1 = fw // 2 + fw // 8
            gy1 = fh // 2 + fh // 24
            draw.rectangle([gx0, gy0, gx1, gy1], fill=(0, 128, 0))
            return flag

        instr_lower = (inpaint_prompt or '').lower()
        # if user explicitly asked for Iraq flag, use flag background
        if 'عراق' in instr_lower or 'iraq' in instr_lower or 'علم العراق' in instr_lower:
            print('Using improved local fallback: placing Iraqi flag background with feathered composite')
            flag_bg = draw_iraq_flag((w, h))

            # Preferred mask: use rembg (U2Net) if available for high-quality subject alpha
            alpha = None
            try:
                from rembg import remove
                print('rembg available — extracting alpha matte using U2Net')
                cleaned = remove(image_bytes)
                try:
                    im_clean = Image.open(io.BytesIO(cleaned)).convert('RGBA')
                    alpha = im_clean.split()[-1]
                    print('rembg produced alpha mask')
                except Exception as _open_err:
                    print('rembg returned bytes but failed to open as image:', repr(_open_err))
                    alpha = None
            except Exception:
                # rembg not available or failed -> fall back to OpenCV GrabCut
                try:
                    import cv2
                    import numpy as np
                    print('OpenCV available — using GrabCut for better segmentation')
                    # convert PIL RGBA to BGR for OpenCV
                    bgr = cv2.cvtColor(np.array(im.convert('RGB'))[:, :, ::-1], cv2.COLOR_BGR2RGB)
                    mask_cv = np.zeros(bgr.shape[:2], np.uint8)
                    # initial rectangle roughly center of image (avoid edges)
                    rect = (max(10, w//10), max(10, h//10), max(10, w - w//10), max(10, h - h//10))
                    bgdModel = np.zeros((1,65), np.float64)
                    fgdModel = np.zeros((1,65), np.float64)
                    cv2.grabCut(bgr, mask_cv, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
                    # mask_cv: probable foreground pixels marked as 2 or 1
                    mask2 = np.where((mask_cv==2)|(mask_cv==1), 255, 0).astype('uint8')
                    # smooth mask
                    mask2 = cv2.GaussianBlur(mask2, (15,15), 0)
                    alpha = Image.fromarray(mask2).convert('L')
                except Exception:
                # fallback to color-difference heuristic
                    corners = [im.getpixel((0,0))[:3], im.getpixel((w-1,0))[:3], im.getpixel((0,h-1))[:3], im.getpixel((w-1,h-1))[:3]]
                    avg_corner = tuple(int(sum([c[i] for c in corners]) / len(corners)) for i in range(3))
                    mask = Image.new('L', (w, h), 0)
                    mpx = mask.load()
                    px = im.load()
                    for y in range(h):
                        for x in range(w):
                            r,g,b,a = px[x,y]
                            diff = abs(r-avg_corner[0]) + abs(g-avg_corner[1]) + abs(b-avg_corner[2])
                            mpx[x,y] = 255 if diff > 50 else 0
                    mask = mask.filter(ImageFilter.GaussianBlur(radius=8))
                    alpha = mask.point(lambda p: int(max(0, min(255, (p)))))
                    alpha = alpha.filter(ImageFilter.GaussianBlur(radius=4))

            # compute bbox of likely subject and expand slightly
            bbox = alpha.getbbox()
            if bbox:
                x0,y0,x1,y1 = bbox
                margin_x = int((x1-x0) * 0.12) + 10
                margin_y = int((y1-y0) * 0.12) + 10
                x0 = max(0, x0 - margin_x)
                y0 = max(0, y0 - margin_y)
                x1 = min(w, x1 + margin_x)
                y1 = min(h, y1 + margin_y)
            else:
                x0,y0,x1,y1 = 0,0,w,h

            # crop and resize subject slightly smaller to create natural border
            subject = im.crop((x0,y0,x1,y1))
            subj_w, subj_h = subject.size
            scale = 0.95
            new_subj = subject.resize((int(subj_w*scale), int(subj_h*scale)), Image.LANCZOS)

            # place flag and composite subject using alpha mask (feathered)
            composite = Image.new('RGBA', (w, h))
            composite.paste(flag_bg, (0,0))

            # create a full-size alpha with subject centered at bbox center
            full_alpha = Image.new('L', (w,h), 0)
            paste_x = x0 + ( (x1-x0) - new_subj.size[0] )//2
            paste_y = y0 + ( (y1-y0) - new_subj.size[1] )//2
            # place blurred alpha for smooth blending
            subj_alpha = alpha.crop((x0,y0,x1,y1)).resize(new_subj.size, Image.LANCZOS)
            subj_alpha = subj_alpha.filter(ImageFilter.GaussianBlur(radius=3))
            full_alpha.paste(subj_alpha, (paste_x, paste_y))

            # optional subtle shadow under subject
            shadow = Image.new('RGBA', (w,h), (0,0,0,0))
            shadow_draw = ImageDraw.Draw(shadow)
            sh_x = paste_x + 8
            sh_y = paste_y + int(new_subj.size[1]*0.85)
            shadow_draw.ellipse([sh_x, sh_y, sh_x + new_subj.size[0]*0.6, sh_y + new_subj.size[1]*0.2], fill=(0,0,0,100))
            shadow = shadow.filter(ImageFilter.GaussianBlur(radius=8))
            composite = Image.alpha_composite(composite, shadow)

            # paste subject onto composite using the alpha mask
            temp = Image.new('RGBA', (w,h), (0,0,0,0))
            temp.paste(new_subj, (paste_x, paste_y), new_subj)
            composite = Image.composite(temp, composite, full_alpha)

            out_buf = io.BytesIO()
            composite.convert('RGB').save(out_buf, format='PNG')
            return out_buf.getvalue()

        # generic fallback: slightly blend original with a light vignette to indicate edit attempt
        print('Using generic local fallback: blending image to indicate edit (no specialized instruction detected)')
        overlay = Image.new('RGBA', (w, h), (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)
        draw.rectangle([0, 0, w, h], fill=(255, 255, 255, 40))
        composite = Image.alpha_composite(im, overlay)
        out_buf = io.BytesIO()
        composite.convert('RGB').save(out_buf, format='PNG')
        return out_buf.getvalue()
    except Exception as e:
        print('Local inpainting fallback failed:', repr(e))
        return None

def send_telegram_photo(chat_id, photo_bytes, caption=""):
    print("\n4. جاري إرسال الصورة إلى تليجرام...")
    telegram_api_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    files = {'photo': ('image.png', photo_bytes, 'image/png')}
    # Telegram caption limit: 1024 chars -> truncate to be safe
    max_caption = 1000
    if caption and len(caption) > max_caption:
        caption = caption[:max_caption] + '...'
    data = {'chat_id': chat_id, 'caption': caption}
    try:
        response = requests.post(telegram_api_url, files=files, data=data)
        response.raise_for_status()
        print("✅ تم إرسال الصورة إلى تليجرام بنجاح!")
    except requests.exceptions.RequestException as e:
        print(f"❌ فشل إرسال الصورة إلى تليجرام: {e}")
        if response and response.text:
            print(f"رسالة الخطأ من تليجرام: {response.text}")

def send_telegram_message(chat_id, text):
    print("\n4. جاري إرسال رسالة نصية إلى تليجرام...")
    telegram_api_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {'chat_id': chat_id, 'text': text}
    try:
        response = requests.post(telegram_api_url, data=data)
        response.raise_for_status()
        print("✅ تم إرسال الرسالة إلى تليجرام بنجاح!")
    except requests.exceptions.RequestException as e:
        print(f"❌ فشل إرسال الرسالة إلى تليجرام: {e}")
        if response and response.text:
            print(f"رسالة الخطأ من تليجرام: {response.text}")

if __name__ == "__main__":
    user_idea = "A futuristic marketplace in old Cairo, blending ancient architecture with holographic technology and flying vehicles. The market is bustling with people and small flying drones delivering goods. The lighting is warm and golden from the setting sun, casting long shadows. Hyper-realistic, cinematic, 8K."
    detailed_prompt = generate_prompt_with_deepseek(user_idea)

    if detailed_prompt:
        send_telegram_message(TELEGRAM_CHAT_ID, f"تم توليد الأمر بنجاح:\n\n{detailed_prompt}")
        image_bytes = generate_image_with_flux(detailed_prompt)
        if image_bytes:
            with open("generated_image.png", "wb") as f:
                f.write(image_bytes)
            print("🖼️ تم حفظ الصورة المولدة باسم 'generated_image.png'")
            send_telegram_photo(TELEGRAM_CHAT_ID, image_bytes, caption="صورة تم توليدها بواسطة AI بناءً على وصفك.")
            send_telegram_message(TELEGRAM_CHAT_ID, "تم الانتهاء من سير العمل. يرجى مراجعة الصورة في تليجرام.")
        else:
            send_telegram_message(TELEGRAM_CHAT_ID, "❌ فشل توليد الصورة. يرجى مراجعة السجلات.")
    else:
        send_telegram_message(TELEGRAM_CHAT_ID, "❌ فشل توليد الأمر. يرجى مراجعة السجلات.")
