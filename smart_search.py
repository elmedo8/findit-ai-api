"""Search reads Firestore using the caller's token, so Firestore rules still apply."""
import hashlib
import io
from PIL import Image, ImageOps
import os
import time
from functools import lru_cache
from urllib.parse import urlparse

import numpy as np
import requests
from fastapi import HTTPException
from embeddings import engine, normal

MAX_REPORTS = 150
ALLOWED_CATEGORIES = {'الكل', 'إلكترونيات', 'مفاتيح', 'شنط', 'أخرى'}


def read_reports(token):
    project = os.environ['FIREBASE_PROJECT_ID']
    response = requests.post(
        f'https://firestore.googleapis.com/v1/projects/{project}/databases/(default)/documents:runQuery',
        headers={'Authorization': f'Bearer {token}'},
        json={'structuredQuery': {'from': [{'collectionId': 'reports'}], 'orderBy': [{'field': {'fieldPath': 'createdAt'}, 'direction': 'DESCENDING'}], 'limit': MAX_REPORTS + 1}}, timeout=25,
    )
    if response.status_code != 200:
        raise HTTPException(502, 'Could not read reports')
    records = []
    for entry in response.json():
        doc = entry.get('document')
        if not doc:
            continue
        fields = doc['fields']
        def value(name, kind='stringValue', default=''):
            return fields.get(name, {}).get(kind, default)
        records.append({'id': doc['name'].rsplit('/', 1)[-1], 'title': value('title'), 'description': value('description'), 'place': value('place'), 'category': value('category'), 'image': value('imagePath'), 'found': value('found', 'booleanValue', False), 'returned': value('returned', 'booleanValue', False)})
    return records[:MAX_REPORTS], len(records) > MAX_REPORTS


@lru_cache(maxsize=64)
def fetch_image(url):
    parsed = urlparse(url)
    cloud = os.getenv('CLOUDINARY_CLOUD_NAME', '')
    # Never fetch client-supplied URLs or follow redirects to internal networks.
    if parsed.scheme != 'https' or parsed.hostname != 'res.cloudinary.com' or not cloud or not parsed.path.startswith(f'/{cloud}/image/upload/'):
        return None
    try:
        with requests.get(url, stream=True, allow_redirects=False, timeout=(5, 12)) as response:
            if response.status_code != 200 or not response.headers.get('Content-Type', '').startswith('image/'):
                return None
            chunks, total = [], 0
            for chunk in response.iter_content(65536):
                total += len(chunk)
                if total > 5 * 1024 * 1024:
                    return None
                chunks.append(chunk)
            from app import validate_image
            data = b''.join(chunks)
            validate_image(data)
            with Image.open(io.BytesIO(data)) as opened:
                thumbnail = ImageOps.fit(ImageOps.exif_transpose(opened).convert('RGB'), (224,224), method=Image.Resampling.BICUBIC)
                output = io.BytesIO()
                thumbnail.save(output, 'JPEG', quality=95)
                return output.getvalue()
    except (requests.RequestException, HTTPException, OSError, ValueError):
        return None


def rank(query_text, image, mode, category, token, top_k=10):
    if mode not in {'text', 'image', 'hybrid'} or category not in ALLOWED_CATEGORIES:
        raise HTTPException(422, 'Invalid search options')
    if mode == 'text' and not query_text or mode == 'image' and not image or not query_text and not image:
        raise HTTPException(422, 'Provide text or a photo for the selected mode')
    if not engine.lock.acquire(blocking=False):
        raise HTTPException(429, 'Search is busy; try again shortly')
    try:
        records, truncated = read_reports(token)
        records = [r for r in records if r['found'] and not r['returned'] and (category == 'الكل' or r['category'] == category)]
        if not records:
            return {'results': [], 'mode': mode, 'searched_count': 0, 'truncated': truncated, 'skipped_images': 0}
        text_vectors = {}
        query_t = None
        if mode != 'image':
            contents = [' '.join([r['title'], r['description'], r['category']])[:2000] for r in records]
            vectors = engine.texts(([query_text] if query_text else []) + contents)
            if query_text:
                query_t, vectors = vectors[0], vectors[1:]
            text_vectors = {r['id']: v for r, v in zip(records, vectors)}
        query_i, image_vectors, skipped = None, {}, 0
        if mode != 'text':
            jobs = []
            if image:
                jobs.append(('query:' + hashlib.sha256(image).hexdigest(), image))
            valid = []
            for r in records:
                if r['image']:
                    data = fetch_image(r['image'])
                    if data:
                        valid.append(r)
                        jobs.append((r['image'], data))
                    else:
                        skipped += 1
            # Treat malformed stored photos as unavailable, not as server crashes.
            clean_jobs, clean_valid = [], []
            from app import validate_image
            for key, data in jobs:
                try:
                    validate_image(data)
                    clean_jobs.append((key, data))
                except HTTPException:
                    if not key.startswith('query:'):
                        skipped += 1
            valid_keys = {key for key, _ in clean_jobs}
            valid = [r for r in valid if r['image'] in valid_keys]
            vectors = engine.images(clean_jobs)
            lookup = dict(zip([key for key, _ in clean_jobs], vectors))
            if image:
                query_i = lookup.get('query:' + hashlib.sha256(image).hexdigest())
            image_vectors = {r['id']: lookup[r['image']] for r in valid}
        query_parts = [v for v in (query_t, query_i) if v is not None]
        query_vector = normal(np.mean(query_parts, axis=0))
        results = []
        for r in records:
            parts = [v for v in (text_vectors.get(r['id']), image_vectors.get(r['id'])) if v is not None]
            if not parts:
                continue
            score = float(np.dot(query_vector, normal(np.mean(parts, axis=0))))
            results.append({'id': r['id'], 'title': r['title'], 'place': r['place'], 'imagePath': r['image'] or None, 'score': round(max(0.0, min(1.0, score)), 4)})
        results.sort(key=lambda r: (-r['score'], r['id']))
        return {'results': results[:top_k], 'mode': mode, 'searched_count': len(records), 'truncated': truncated, 'skipped_images': skipped}
    finally:
        engine.lock.release()
