import io
import os
from unittest.mock import patch
from fastapi.testclient import TestClient
from PIL import Image
from app import app, authenticated_user, configure_cloud

client = TestClient(app)
REPORT = 'A' * 20

def png():
    output = io.BytesIO()
    Image.new('RGB', (10, 10)).save(output, format='PNG')
    return output.getvalue()

def test_sign_in_required():
    assert client.post(f'/reports/{REPORT}/image', files={'file': ('x.png', png())}).status_code == 401

def test_reject_forged_token():
    with patch.dict(os.environ, {'FIREBASE_PROJECT_ID': 'findit-ai-d5212'}):
        assert client.post(f'/reports/{REPORT}/image', headers={'Authorization': 'Bearer invalid'}, files={'file': ('x.png', png())}).status_code == 401

def test_image_validation_and_upload():
    app.dependency_overrides[authenticated_user] = lambda: 'test-user'
    configure_cloud.cache_clear()
    try:
        with patch.dict(os.environ, {'CLOUDINARY_CLOUD_NAME': 'test', 'CLOUDINARY_API_KEY': 'test', 'CLOUDINARY_API_SECRET': 'test'}):
            assert client.post(f'/reports/{REPORT}/image', files={'file': ('fake.jpg', b'not an image')}).status_code == 415
            assert client.post(f'/reports/{REPORT}/image', files={'file': ('big.png', b'x' * (5 * 1024 * 1024 + 1))}).status_code == 413
            assert client.post('/reports/bad/image', files={'file': ('x.png', png())}).status_code == 400
            with patch('app.cloudinary.uploader.upload', return_value={'secure_url': f'https://res.cloudinary.com/test/image/upload/v1/findit/test-user/{REPORT}.png'}) as upload:
                response = client.post(f'/reports/{REPORT}/image', files={'file': ('x.png', png())})
                assert response.status_code == 200
                assert upload.call_args.kwargs['public_id'] == f'findit/test-user/{REPORT}'
                assert 'API_SECRET' not in response.text
    finally:
        app.dependency_overrides.clear()
        configure_cloud.cache_clear()
