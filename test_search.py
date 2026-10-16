import io
import threading
from unittest.mock import patch
import numpy as np
from fastapi import HTTPException
from fastapi.testclient import TestClient
from PIL import Image
import pytest
from app import app, authenticated_user
import smart_search as search

client = TestClient(app)

def record(id, title='headphones', found=True, returned=False, category='إلكترونيات', image=''):
    return {'id': id, 'title': title, 'description': '', 'place': 'library', 'category': category, 'image': image, 'found': found, 'returned': returned}

class FakeEngine:
    def __init__(self): self.lock = threading.Lock()
    def texts(self, texts):
        return [np.array([1.,0.],dtype=np.float32) if 'headphone' in t else np.array([0.,1.],dtype=np.float32) for t in texts]
    def images(self, jobs):
        return [np.array([1.,0.],dtype=np.float32) for _ in jobs]

def test_search_requires_authentication():
    assert client.post('/search', data={'query':'headphones'}).status_code == 401

def test_text_ranking_filters_and_limit():
    records = [record('a'),record('b','keys'),record('c',found=False),record('d',returned=True),record('e',category='مفاتيح')]
    with patch.object(search, 'engine', FakeEngine()), patch.object(search,'read_reports',return_value=(records,True)):
        data = search.rank('headphones',None,'text','إلكترونيات','token',1)
        assert data['results'][0]['id']=='a'
        assert len(data['results'])==1
        assert data['searched_count']==2
        assert data['truncated'] is True

def test_image_mode_skips_records_without_images():
    photo=io.BytesIO();Image.new('RGB',(10,10)).save(photo,'PNG');image=photo.getvalue()
    with patch.object(search,'engine',FakeEngine()), patch.object(search,'read_reports',return_value=([record('a',image='url'),record('b')],False)),patch.object(search,'fetch_image',return_value=image):
        data=search.rank('',image,'image','الكل','token')
        assert [r['id'] for r in data['results']]==['a']

def test_modes_validate_input_and_busy_response():
    with patch.object(search,'engine',FakeEngine()):
        with pytest.raises(HTTPException) as error: search.rank('',None,'hybrid','الكل','token')
        assert error.value.status_code==422
        search.engine.lock.acquire()
        with pytest.raises(HTTPException) as error: search.rank('headphones',None,'text','الكل','token')
        assert error.value.status_code==429
        search.engine.lock.release()

def test_candidate_url_restriction():
    search.fetch_image.cache_clear()
    with patch.dict('os.environ',{'CLOUDINARY_CLOUD_NAME':'mycloud'}),patch('smart_search.requests.get') as get:
        assert search.fetch_image('http://127.0.0.1/secret') is None
        assert search.fetch_image('https://res.cloudinary.com/other/image/upload/x.jpg') is None
        get.assert_not_called()
