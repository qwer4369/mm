import requests
import time
import json

import os

TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN') or ''
if not TOKEN:
    raise RuntimeError('Please set TELEGRAM_BOT_TOKEN in your environment to use wait_for_updates.py')

URL = f"https://api.telegram.org/bot{TOKEN}/getUpdates"

print("انتظر رسالة من البوت الآن. أرسل رسالة إلى البوت من تطبيق Telegram (أو أعد توجيه رسالة).")
print("سيتم طباعة أي تحديث يصل، ثم الخروج بعد اكتشاف chat.id.")

offset = None
try:
    while True:
        params = {'timeout': 30}
        if offset is not None:
            params['offset'] = offset
        try:
            r = requests.get(URL, params=params, timeout=40)
        except Exception as e:
            print('خطأ في الاتصال:', e)
            time.sleep(2)
            continue
        if not r.ok:
            print('HTTP', r.status_code, r.text)
            # If webhook was re-enabled, advise deletion
            if r.status_code == 409:
                print('Conflict: يبدو أن webhook مفعل — سيتم محاولة حذف webhook تلقائياً.')
                del_url = f"https://api.telegram.org/bot{TOKEN}/deleteWebhook"
                try:
                    dr = requests.post(del_url)
                    print('deleteWebhook result:', dr.status_code, dr.text)
                except Exception as e:
                    print('deleteWebhook error:', e)
            time.sleep(2)
            continue
        data = r.json()
        results = data.get('result', [])
        if not results:
            # poll again
            continue
        for upd in results:
            print(json.dumps(upd, indent=2, ensure_ascii=False))
            offset = upd['update_id'] + 1
            chat = None
            # common locations for chat
            if 'message' in upd and 'chat' in upd['message']:
                chat = upd['message']['chat']
            elif 'edited_message' in upd and 'chat' in upd['edited_message']:
                chat = upd['edited_message']['chat']
            elif 'callback_query' in upd and 'message' in upd['callback_query'] and 'chat' in upd['callback_query']['message']:
                chat = upd['callback_query']['message']['chat']
            if chat:
                print('\n=== Found chat ===')
                print(json.dumps(chat, indent=2, ensure_ascii=False))
                print('\nUse this chat id in workflow_run.py as TELEGRAM_CHAT_ID (or I can update it).')
                raise SystemExit(0)
        time.sleep(0.5)
except KeyboardInterrupt:
    print('\nAborted by user')
