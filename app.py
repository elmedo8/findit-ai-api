"""Authenticated image proxy. Cloudinary secrets remain on the server."""
import io
import os
from functools import lru_cache

import cloudinary
import cloudinary.uploader
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, Form
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from google.auth.transport.requests import Request
from google.oauth2.id_token import verify_firebase_token
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

MAX_BYTES = 5 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 20_000_000
app = FastAPI(title="FindIt AI image service", docs_url=None, redoc_url=None)
bearer = HTTPBearer(auto_error=False)

class ImageResult(BaseModel):
    image_url: str

@lru_cache
def configure_cloud():
    names = ('CLOUDINARY_CLOUD_NAME', 'CLOUDINARY_API_KEY', 'CLOUDINARY_API_SECRET')
    values = [os.getenv(name) for name in names]
    if not all(values):
        raise HTTPException(503, 'Image storage is not configured')
    cloudinary.config(cloud_name=values[0], api_key=values[1], api_secret=values[2], secure=True)
    return values[0]

async def authenticated_user(credentials: HTTPAuthorizationCredentials | None = Depends(bearer)):
    if not credentials:
        raise HTTPException(401, 'Sign in first')
    project = os.getenv('FIREBASE_PROJECT_ID')
    if not project:
        raise HTTPException(503, 'Authentication is not configured')
    try:
        claims = await run_in_threadpool(verify_firebase_token, credentials.credentials, Request(), project)
        uid = claims['sub']
        if not uid or len(uid) > 128 or '/' in uid:
            raise ValueError('Invalid user')
        return uid
    except Exception:
        raise HTTPException(401, 'Invalid or expired sign-in token') from None

def validate_image(data: bytes):
    if not data or len(data) > MAX_BYTES:
        raise HTTPException(413, 'Image must be at most 5 MB')
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.format not in ('JPEG', 'PNG'):
                raise HTTPException(415, 'Use a JPEG or PNG image')
            if image.width * image.height > Image.MAX_IMAGE_PIXELS:
                raise HTTPException(413, 'Image dimensions are too large')
            image.verify()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise HTTPException(415, 'Invalid image') from None

@app.get('/health')
def health():
    return {'status': 'ok', 'images_configured': all(os.getenv(n) for n in ('CLOUDINARY_CLOUD_NAME','CLOUDINARY_API_KEY','CLOUDINARY_API_SECRET','FIREBASE_PROJECT_ID'))}

@app.post('/reports/{report_id}/image', response_model=ImageResult)
async def upload_image(report_id: str, file: UploadFile = File(...), uid: str = Depends(authenticated_user)):
    if not report_id.isascii() or not report_id.isalnum() or len(report_id) != 20:
        raise HTTPException(400, 'Invalid report ID')
    configure_cloud()
    try:
        data = await file.read(MAX_BYTES + 1)
    finally:
        await file.close()
    await run_in_threadpool(validate_image, data)
    try:
        result = await run_in_threadpool(cloudinary.uploader.upload, io.BytesIO(data), resource_type='image', public_id=f'findit/{uid}/{report_id}', overwrite=False, allowed_formats=['jpg','png'], transformation={'width': 1600, 'height': 1600, 'crop': 'limit'})
        return ImageResult(image_url=result['secure_url'])
    except Exception:
        raise HTTPException(502, 'Image upload failed; retry later') from None


@app.post('/search')
async def search_reports(
    query: str = Form('', max_length=1000),
    mode: str = Form('hybrid'),
    category: str = Form('الكل'),
    top_k: int = Form(10, ge=1, le=30),
    file: UploadFile | None = File(None),
    uid: str = Depends(authenticated_user),
    credentials: HTTPAuthorizationCredentials = Depends(bearer),
):
    image = None
    if file is not None:
        try:
            image = await file.read(MAX_BYTES + 1)
        finally:
            await file.close()
        await run_in_threadpool(validate_image, image)
    from smart_search import rank
    try:
        return await run_in_threadpool(rank, query.strip(), image, mode, category, credentials.credentials, top_k)
    except HTTPException:
        raise
    except Exception:
        # Do not leak tokens, API secrets, response bodies or server paths.
        raise HTTPException(503, 'Search could not finish; retry shortly') from None
