"""Flask app exposing a unified API for the MultiModelAgent."""
from flask import Flask, request, jsonify
from multi_model_agent import MultiModelAgent
import logging

logging.basicConfig(level=logging.INFO)
app = Flask(__name__)
agent = MultiModelAgent()


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


@app.route('/generate', methods=['POST'])
def generate():
    """POST JSON {"model":"deepseek"|"gptoss", "prompt":"...", "max_tokens": 128}

    Returns: {"result": "..."}
    """
    data = request.get_json(force=True)
    model = data.get('model', 'deepseek')
    prompt = data.get('prompt', '')
    max_tokens = int(data.get('max_tokens', 256))
    if not prompt:
        return jsonify({'error': 'prompt is required'}), 400
    try:
        out = agent.generate_text(model, prompt, max_tokens=max_tokens)
        return jsonify({'result': out})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/image_edit', methods=['POST'])
def image_edit():
    """POST JSON {"image_base64": "...", "instruction": "..."}

    Returns {"image_base64": "..."}
    """
    data = request.get_json(force=True)
    b64 = data.get('image_base64')
    instruction = data.get('instruction', '')
    if not b64 or not instruction:
        return jsonify({'error': 'image_base64 and instruction are required'}), 400
    try:
        out_b64 = agent.edit_image_base64(b64, instruction)
        return jsonify({'image_base64': out_b64})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
