"""CPU-only, quantized multilingual CLIP encoders; no PyTorch dependency."""
import gc
import io
import os
import threading
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

TEXT_REPO = 'sentence-transformers/clip-ViT-B-32-multilingual-v1'
TEXT_REV = '58edf8cada9e398793dca955574a48cbb7f18be2'
VISION_REPO = 'Xenova/clip-vit-base-patch32'
VISION_REV = 'd15189d7028b43f1d3e65039190477f6af591c2a'
CACHE_SIZE = 512


def normal(vector):
    vector = np.asarray(vector, dtype=np.float32)
    return vector / max(float(np.linalg.norm(vector)), 1e-9)


def model_file(repo, name, revision):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, name, revision=revision, cache_dir=os.getenv('MODEL_CACHE'))


class Embeddings:
    def __init__(self):
        self.lock = threading.Lock()
        self.text_cache = OrderedDict()
        self.image_cache = OrderedDict()

    def remember(self, cache, key, value):
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > CACHE_SIZE:
            cache.popitem(last=False)

    @contextmanager
    def session(self, path):
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False
        session = ort.InferenceSession(path, sess_options=options, providers=['CPUExecutionProvider'])
        try:
            yield session
        finally:
            del session
            gc.collect()

    def texts(self, texts):
        missing = list(dict.fromkeys(t for t in texts if t not in self.text_cache))
        if missing:
            from tokenizers import Tokenizer
            from safetensors.numpy import load_file
            tokenizer = Tokenizer.from_file(model_file(TEXT_REPO, 'tokenizer.json', TEXT_REV))
            tokenizer.enable_truncation(max_length=128)
            dense = load_file(model_file(TEXT_REPO, '2_Dense/model.safetensors', TEXT_REV))['linear.weight']
            with self.session(model_file(TEXT_REPO, 'onnx/model_quint8_avx2.onnx', TEXT_REV)) as session:
                names = {i.name for i in session.get_inputs()}
                for text in missing:
                    encoded = tokenizer.encode(text)
                    inputs = {'input_ids': np.array([encoded.ids], dtype=np.int64), 'attention_mask': np.array([encoded.attention_mask], dtype=np.int64)}
                    if 'token_type_ids' in names:
                        inputs['token_type_ids'] = np.array([encoded.type_ids], dtype=np.int64)
                    result = session.run(None, {k: v for k, v in inputs.items() if k in names})[0]
                    mask = inputs['attention_mask'][..., None]
                    pooled = (result * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1)
                    self.remember(self.text_cache, text, normal(pooled[0] @ dense.T))
        return [self.text_cache[t] for t in texts]

    def images(self, images):
        """images: list of (cache key, JPEG/PNG bytes)."""
        if not images:
            return []
        missing = {key: value for key, value in images if key not in self.image_cache}
        if missing:
            with self.session(model_file(VISION_REPO, 'onnx/vision_model_quantized.onnx', VISION_REV)) as session:
                output = next((v.name for v in session.get_outputs() if v.name == 'image_embeds'), session.get_outputs()[0].name)
                for key, data in missing.items():
                    with Image.open(io.BytesIO(data)) as opened:
                        image = ImageOps.exif_transpose(opened).convert('RGB')
                        width, height = image.size
                        # Match CLIP's resize-shortest-edge then center-crop processor.
                        size = (224, int(height * 224 / width)) if width <= height else (int(width * 224 / height), 224)
                        image = image.resize(size, Image.Resampling.BICUBIC)
                        left, top = (image.width - 224) // 2, (image.height - 224) // 2
                        array = np.asarray(image.crop((left, top, left + 224, top + 224)), dtype=np.float32) / 255.0
                    array = (array - np.array([.48145466, .4578275, .40821073], dtype=np.float32)) / np.array([.26862954, .26130258, .27577711], dtype=np.float32)
                    pixels = np.transpose(array, (2, 0, 1))[None, ...]
                    vector = session.run([output], {'pixel_values': pixels})[0][0]
                    self.remember(self.image_cache, key, normal(vector))
        return [self.image_cache[key] for key, _ in images]


engine = Embeddings()


def download_models():
    for repo, revision, files in [
        (TEXT_REPO, TEXT_REV, ['tokenizer.json', '2_Dense/model.safetensors', 'onnx/model_quint8_avx2.onnx']),
        (VISION_REPO, VISION_REV, ['onnx/vision_model_quantized.onnx']),
    ]:
        for name in files:
            model_file(repo, name, revision)


if __name__ == '__main__':
    download_models()
    print('Quantized search models downloaded.')
