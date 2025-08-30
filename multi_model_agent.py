# multi_model_agent.py
"""
Unified Multi-Model Agent

This module provides lightweight wrappers that attempt to load the real models
from local folders (if transformers/diffusers are installed) and fall back to
mock implementations when those libraries or model files are not available.

Exports:
- MultiModelAgent: main entrypoint used by the Flask app in `app.py`.
"""
from typing import Optional
import base64
import io
import os
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class BaseTextModel:
    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        raise NotImplementedError()


class HFTextModel(BaseTextModel):
    """Attempt to load a HuggingFace-style causal LM from a local folder."""
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.tokenizer = None
        self.model = None
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch

            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            self.model = AutoModelForCausalLM.from_pretrained(model_dir)
            # move to cpu by default; user can modify later
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if hasattr(self.model, 'to'):
                self.model.to(self.device)
            logger.info(f"Loaded HF model from {model_dir} onto {self.device}")
        except Exception as e:
            logger.warning(f"Could not load HF model from {model_dir}: {e}")
            self.model = None

    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        # If tokenizer/model not available, try a robust pipeline fallback
        if not self.model or not self.tokenizer:
            try:
                from transformers import pipeline
                # explicitly use a small hub model for fallback to avoid loading
                # a partially-present local model directory
                pipe = pipeline('text-generation', model='gpt2', device=-1)
                out = pipe(prompt, max_new_tokens=max_tokens, do_sample=False)
                if isinstance(out, list) and len(out) > 0 and 'generated_text' in out[0]:
                    return out[0]['generated_text']
                # unexpected pipeline shape
                return str(out)
            except Exception as e:
                logger.warning(f"Pipeline fallback failed: {e}")
                return f"[fallback-HF-text] {prompt}"

        # Preferred path: use model + tokenizer directly (fast path)
        try:
            from transformers import GenerationConfig
            import torch

            inputs = self.tokenizer(prompt, return_tensors='pt')
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model.generate(**inputs, max_new_tokens=max_tokens)
            text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return text
        except Exception as e:
            logger.warning(f"Direct generate failed, falling back to pipeline: {e}")
            try:
                from transformers import pipeline
                pipe = pipeline('text-generation', model='gpt2', device=-1)
                out = pipe(prompt, max_new_tokens=max_tokens, do_sample=False)
                if isinstance(out, list) and len(out) > 0 and 'generated_text' in out[0]:
                    return out[0]['generated_text']
                return str(out)
            except Exception as e2:
                logger.warning(f"Pipeline fallback also failed: {e2}")
                return f"[fallback-HF-text] {prompt}"


class MockTextModel(BaseTextModel):
    def __init__(self, name: str):
        self.name = name

    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        return f"[{self.name} - mock] {prompt}"


class ImageEditor:
    """Wrapper for image-editing model (attempt diffusers or fallback).

    The edit method accepts an image as bytes or PIL Image and an instruction string.
    """
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.ready = False
        try:
            # don't import heavy libs at module import time unless available
            from PIL import Image
            # optional: import diffusers or custom runtime here
            self.ready = True
            logger.info(f"Image editor initialized for {model_dir} (no runtime ops yet)")
        except Exception as e:
            logger.warning(f"Image editor will run in mock mode: {e}")
            self.ready = False

    def edit(self, image_bytes: bytes, instruction: str) -> bytes:
        # simple mock: return the same bytes, prefixed with a tiny note if possible
        if not image_bytes:
            raise ValueError("No image bytes provided")
        # real implementation would run a pipeline; here we just return the input
        return image_bytes


class MultiModelAgent:
    """Facade that holds the three models and exposes a simple API.

    Loading strategy:
    - Try to use HF loader for text models if transformers installed and local folder exists.
    - Otherwise use mock text model that returns predictable strings.
    - Image editor attempts basic initialization; otherwise works as passthrough.
    """

    def __init__(self, base_path: Optional[str] = None):
        base = base_path or os.getcwd()
        # map model names to their local folder (from repo attachments)
        self.paths = {
            'deepseek': os.path.join(base, 'DeepSeek-V3.1'),
            'gptoss': os.path.join(base, 'gpt-oss-20b'),
            'qwen_image': os.path.join(base, 'Qwen-Image-Edit'),
        }

        # Text models
        self.deepseek = self._make_text_model('deepseek')
        self.gptoss = self._make_text_model('gptoss')

        # Image edit model
        self.qwen = ImageEditor(self.paths['qwen_image'])

    def _make_text_model(self, key: str) -> BaseTextModel:
        path = self.paths.get(key)
        # If transformers exists and folder looks like a model, try HF loader
        try:
            from transformers import AutoTokenizer  # type: ignore
            # try local path first
            if path and os.path.isdir(path):
                hf = HFTextModel(path)
                # if hf loaded a model or tokenizer, use it
                if hf.model or hf.tokenizer:
                    return hf
            # fallback: for gptoss, try a small hub model to provide a working generator
            if key == 'gptoss':
                try:
                    return HFTextModel('gpt2')
                except Exception:
                    pass
        except Exception:
            pass
        return MockTextModel(key)

    def generate_text(self, model: str, prompt: str, max_tokens: int = 256) -> str:
        if model == 'deepseek':
            return self.deepseek.generate(prompt, max_tokens=max_tokens)
        elif model == 'gptoss':
            return self.gptoss.generate(prompt, max_tokens=max_tokens)
        else:
            raise ValueError(f'Unknown text model: {model}')

    def edit_image_base64(self, b64: str, instruction: str) -> str:
        # accept base64-encoded image, run edit, and return base64 of result
        raw = base64.b64decode(b64)
        out = self.qwen.edit(raw, instruction)
        return base64.b64encode(out).decode('utf-8')


if __name__ == '__main__':
    agent = MultiModelAgent()
    print(agent.generate_text('deepseek', 'اكتب جملة اختبار'))
    print(agent.generate_text('gptoss', 'اشرح البرمجة'))
    # mock image roundtrip
    sample = base64.b64encode(b'fake-image-bytes').decode('utf-8')
    print(agent.edit_image_base64(sample, 'تحسين الوضوح'))
