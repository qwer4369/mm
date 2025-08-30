import time
import base64
import json
import io
import requests
from workflow_run import (
    YOUR_HF_TOKEN,
    FLUX_API_URL,
    FLUX_FILL_API_URL,
    generate_prompt_with_deepseek,
    generate_image_with_flux,
    send_telegram_message,
    send_telegram_photo,
    inpaint_image_with_flux_fill,
)
import os

# Read TELEGRAM_BOT_TOKEN from environment; fail fast if not provided
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN') or ''
if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError('TELEGRAM_BOT_TOKEN environment variable is required to run the bot')

GET_UPDATES_URL = lambda: f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates"
GET_FILE_URL = lambda: f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getFile"

print('Bot service starting (polling getUpdates)...')

# in-memory chat state: {chat_id: {state: str, photo_file_id: str, ack_msg_id: int}}
chat_states = {}
offset = None

ACK_MESSAGES = [
    'تم استلام طلبك — جاري التحضير. سنتواصل معك عند الانتهاء.',
    'وصلنا طلبك! الآن نقوم بصياغة الوصف وتشغيل المحرك. الرجاء الصبر لحظات.',
    'شكراً، بدأنا العمل على طلبك الآن. سيتم إرسال النتيجة فور الانتهاء.',
]

def send_message(chat_id, text, reply_markup=None):
    """Send a Telegram message and return the sent message object (dict)"""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {'chat_id': chat_id, 'text': text, 'parse_mode': 'HTML'}
    if reply_markup:
        payload['reply_markup'] = json.dumps(reply_markup)
    r = requests.post(url, data=payload)
    try:
        r.raise_for_status()
        return r.json().get('result')
    except Exception:
        print('send_message error', r.status_code, r.text)
        return None


def send_photo(chat_id, photo_bytes, caption=''):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    files = {'photo': ('image.png', photo_bytes, 'image/png')}
    data = {'chat_id': chat_id, 'caption': caption, 'parse_mode': 'HTML'}
    try:
        r = requests.post(url, files=files, data=data)
        r.raise_for_status()
        return r.json().get('result')
    except Exception:
        print('send_photo error', getattr(r, 'status_code', None), getattr(r, 'text', None))
        return None
ACK_MESSAGES = [
    'تم استلام طلبك — جاري التحضير. سنتواصل معك عند الانتهاء.',
    'وصلنا طلبك! الآن نقوم بصياغة الوصف وتشغيل المحرك. الرجاء الصبر لحظات.',
    'شكراً، بدأنا العمل على طلبك الآن. سيتم إرسال النتيجة فور الانتهاء.',
]


def delete_message(chat_id, message_id):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/deleteMessage"
    try:
        r = requests.post(url, data={'chat_id': chat_id, 'message_id': message_id})
        return r.ok
    except Exception as e:
        print('delete_message error', e)
        return False


def answer_callback(callback_id, text=None):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/answerCallbackQuery"
    data = {'callback_query_id': callback_id}
    if text:
        data['text'] = text
    try:
        requests.post(url, data=data)
    except Exception:
        pass


def get_file_path(file_id, attempts=3):
    """Call getFile and return file_path or None (with attempts)."""
    url = GET_FILE_URL() + f"?file_id={file_id}"
    for i in range(attempts):
        try:
            r = requests.get(url, timeout=10)
            if not r.ok:
                print('getFile error', r.status_code, r.text)
                time.sleep(1 + i)
                continue
            data = r.json()
            fp = data.get('result', {}).get('file_path')
            if fp:
                return fp
            print('getFile returned unexpected payload:', data)
        except Exception as e:
            print('getFile exception:', e)
        time.sleep(1 + i)
    return None


def download_file_bytes(file_path):
    try:
        file_url = f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{file_path}"
        r = requests.get(file_url, timeout=20)
        if r.ok:
            return r.content
        print('download_file_bytes failed', r.status_code, r.text)
    except Exception as e:
        print('download_file_bytes exception', e)
    return None


def make_welcome_keyboard():
    return {
        'inline_keyboard': [
            [
                {'text': 'إنشاء صورة', 'callback_data': 'create'},
                {'text': 'تعديل صورة', 'callback_data': 'edit'},
            ]
        ]
    }


def backoff_sleep(attempts):
    # exponential backoff with cap and jitter
    import random
    base = min(30, 1 + attempts * 3)
    t = base + random.random()
    time.sleep(t)


attempts = 0
while True:
    try:
        params = {'timeout': 30}
        if offset:
            params['offset'] = offset
        r = requests.get(GET_UPDATES_URL(), params=params, timeout=40)
        if not r.ok:
            print('getUpdates error', r.status_code, r.text)
            # if webhook conflict, try once to delete webhook then back off more aggressively
            if r.status_code == 409:
                if attempts < 3:
                    try:
                        delr = requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/deleteWebhook")
                        print('deleteWebhook', delr.status_code, delr.text)
                    except Exception as e:
                        print('deleteWebhook error', e)
                attempts += 1
                backoff_sleep(attempts)
            else:
                time.sleep(2)
            continue
        attempts = 0
        data = r.json()
        for upd in data.get('result', []):
            offset = upd['update_id'] + 1

            # callback_query (button press)
            if 'callback_query' in upd:
                cq = upd['callback_query']
                cid = cq['message']['chat']['id']
                cbid = cq['id']
                data_cb = cq.get('data')
                # acknowledge callback to remove loading state
                answer_callback(cbid)
                if data_cb == 'create':
                    chat_states[cid] = {'state': 'awaiting_prompt'}
                    send_message(cid, 'حسناً — أرسل وصف الصورة الذي تريد إنشاءه باللغة العربية أو الإنجليزية. سأقوم بصياغة وصف احترافي ثم توليد الصورة.')
                elif data_cb == 'edit':
                    chat_states[cid] = {'state': 'awaiting_edit_photo'}
                    send_message(cid, 'جيد — أرسل الصورة التي تريد تعديلها مع وضع وصف مختصر في الـ caption أو أرسل الصورة أولاً ثم أرسل الوصف.')
                continue

            # handle message types
            if 'message' not in upd:
                continue
            msg = upd['message']
            chat = msg['chat']
            chat_id = chat['id']

            # handle /start
            if msg.get('text', '').strip() == '/start':
                send_message(chat_id, '<b>أهلاً وسهلاً!</b> اضغط أحد الأزرار للاختيار:', reply_markup=make_welcome_keyboard())
                continue

            # determine current state
            state = chat_states.get(chat_id, {}).get('state')

            # if awaiting a prompt for creation
            if state == 'awaiting_prompt' and msg.get('text'):
                user_desc = msg['text'].strip()
                # send ack (varied)
                import random
                ack_text = random.choice(ACK_MESSAGES)
                ack = send_message(chat_id, ack_text)
                if ack and 'message_id' in ack:
                    chat_states[chat_id]['ack_msg_id'] = ack['message_id']

                prompt = generate_prompt_with_deepseek(user_desc)
                if not prompt:
                    send_message(chat_id, 'عذراً، لم أتمكن من تكوين وصف احترافي الآن. يرجى المحاولة لاحقاً.')
                    chat_states.pop(chat_id, None)
                    continue
                img = generate_image_with_flux(prompt)
                # delete ack before sending result
                if chat_states.get(chat_id, {}).get('ack_msg_id'):
                    delete_message(chat_id, chat_states[chat_id]['ack_msg_id'])
                if img:
                    caption = '<b>تم إنشاء الصورة بنجاح</b>\nهذا الوصف الذي استخدمناه لإنشاء الصورة:\n' + (prompt[:1000])
                    send_photo(chat_id, img, caption=caption)
                else:
                    send_message(chat_id, 'عذراً، فشلت عملية توليد الصورة. حاول مرة أخرى لاحقاً.')
                chat_states.pop(chat_id, None)
                continue

            # if awaiting an edit photo
            if state == 'awaiting_edit_photo' and 'photo' in msg:
                # if caption contains edit: use it immediately
                caption = msg.get('caption', '') or ''
                photo_info = msg['photo'][-1]
                file_id = photo_info['file_id']
                # treat any caption as edit instruction; if empty, store photo and ask for instruction
                instr = caption.strip()
                if instr:
                    # get file path
                    file_path = get_file_path(file_id)
                    if not file_path:
                        send_message(chat_id, 'عذراً، لم أتمكن من تنزيل الصورة من تليجرام. حاول إرسالها مرة أخرى.')
                        chat_states.pop(chat_id, None)
                        continue
                    img_bytes = download_file_bytes(file_path)
                    if not img_bytes:
                        send_message(chat_id, 'عذراً، حدث خطأ عند تنزيل الصورة. حاول مرة أخرى.')
                        chat_states.pop(chat_id, None)
                        continue
                    ack = send_message(chat_id, 'تم استلام الصورة والتعليمات، جاري تطبيق التعديل. سترى النتيجة هنا قريباً.')
                    out = None
                    try:
                        out = inpaint_image_with_flux_fill(img_bytes, instr, None)
                    except Exception as e:
                        print('inpaint raised', e)
                    if ack and 'message_id' in ack:
                        delete_message(chat_id, ack['message_id'])
                    if out:
                        send_photo(chat_id, out, caption='<b>تم تعديل الصورة بنجاح</b>')
                    else:
                        send_message(chat_id, 'عذراً، فشل التعديل أو غير مدعوم من قبل الخدمة. تم تجربة حل محلي بديل.')
                    chat_states.pop(chat_id, None)
                else:
                    # store file_id and ask for instruction
                    chat_states[chat_id] = {'state': 'awaiting_edit_instruction', 'photo_file_id': file_id}
                    send_message(chat_id, 'حسناً، استلمت الصورة. الآن أرسل وصف التعديل (مثال: ازل الخلفية واستبدلها بسماء ليلية أو: اضف علم العراق خلفه).')
                continue

            # awaiting instruction after photo upload
            if state == 'awaiting_edit_instruction' and msg.get('text'):
                instr = msg['text'].strip()
                file_id = chat_states[chat_id].get('photo_file_id')
                file_path = get_file_path(file_id)
                if not file_path:
                    send_message(chat_id, 'عذراً، لم أتمكن من تنزيل الصورة. حاول إرسالها مرة أخرى.')
                    chat_states.pop(chat_id, None)
                    continue
                img_bytes = download_file_bytes(file_path)
                if not img_bytes:
                    send_message(chat_id, 'عذراً، حدث خطأ عند تنزيل الصورة. حاول مرة أخرى.')
                    chat_states.pop(chat_id, None)
                    continue
                ack = send_message(chat_id, 'تم استلام التعليمات، جاري تطبيق التعديل...')
                out = None
                try:
                    out = inpaint_image_with_flux_fill(img_bytes, instr, None)
                except Exception as e:
                    print('inpaint raised', e)
                if ack and 'message_id' in ack:
                    delete_message(chat_id, ack['message_id'])
                if out:
                    send_photo(chat_id, out, caption='<b>تم تعديل الصورة بنجاح</b>')
                else:
                    send_message(chat_id, 'عذراً، فشل التعديل أو غير مدعوم من قبل الخدمة. حاول وصفاً أبسط أو أرسل صورة أوضح.')
                chat_states.pop(chat_id, None)
                continue

            # default: if plain text and no state, suggest buttons
            if msg.get('text'):
                send_message(chat_id, 'للبدء اختر: /start ثم اضغط أحد الأزرار، أو ارسل /start الآن.', reply_markup=make_welcome_keyboard())
        time.sleep(0.3)
    except KeyboardInterrupt:
        print('Bot stopped by user')
        break
    except Exception as e:
        print('Exception in polling loop:', e)
        time.sleep(2)
