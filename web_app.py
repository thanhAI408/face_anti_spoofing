"""Render web service. Images are processed in memory and never stored."""
import io
import os
import secrets
import threading
import time
from flask import Flask,request,jsonify,send_from_directory
from PIL import Image,UnidentifiedImageError
import cv2
import numpy as np
from pipeline import BASE,Pipeline,TemporalDecision

cv2.setNumThreads(1)
app=Flask(__name__,static_folder=str(BASE/'static'))
app.config['MAX_CONTENT_LENGTH']=750_000
engine=Pipeline(BASE/'web_models/liveness.tflite',BASE/'web_models/labels.json')
lock=threading.Lock()
sessions={}

@app.after_request
def headers(response):
    response.headers['Cache-Control']='no-store'
    response.headers['X-Content-Type-Options']='nosniff'
    response.headers['Referrer-Policy']='same-origin'
    response.headers['Permissions-Policy']='camera=(self), microphone=()'
    response.headers['Content-Security-Policy']="default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' blob: data:; media-src 'self' blob:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'"
    return response

@app.get('/')
def index():return send_from_directory(app.static_folder,'index.html')

@app.get('/healthz')
def health():return jsonify(status='ok',model='liveness-tflite',detector='yunet',version='web-1')

@app.post('/api/session')
def new_session():
    with lock:
        now=time.monotonic()
        for token in list(sessions):
            if now-sessions[token]['seen']>30:sessions.pop(token)
        if len(sessions)>=32:return jsonify(error='Máy chủ đang bận. Vui lòng thử lại sau.'),503
        token=secrets.token_urlsafe(24)
        sessions[token]={'seen':now,'last':0,'temporal':TemporalDecision(min_samples=4,max_gap=2.0,window_seconds=4.0)}
    return jsonify(token=token)

@app.delete('/api/session')
def end_session():
    with lock:sessions.pop(request.headers.get('X-Session-Token'),None)
    return '',204

@app.post('/api/frame')
def frame():
    if request.mimetype!='image/jpeg':return jsonify(error='Cần ảnh JPEG.'),415
    data=request.get_data()
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.format!='JPEG' or max(image.size)>960 or min(image.size)<32:
                return jsonify(error='Ảnh cần có kích thước từ 32 đến 960 pixel.'),422
            image.verify()
        pixels=cv2.imdecode(np.frombuffer(data,dtype=np.uint8),cv2.IMREAD_COLOR)
        if pixels is None:raise ValueError('invalid image')
    except (UnidentifiedImageError,OSError,ValueError,Image.DecompressionBombError):
        return jsonify(error='Không đọc được ảnh.'),422
    if not lock.acquire(blocking=False):return jsonify(error='Máy chủ đang xử lý. Đang thử lại.'),429
    try:
        now=time.monotonic()
        session=sessions.get(request.headers.get('X-Session-Token'))
        if session is None or now-session['seen']>30:return jsonify(error='Phiên đã hết hạn.'),401
        if now-session['last']<.12:return jsonify(error='Gửi ảnh quá nhanh.'),429
        session['seen']=session['last']=now
        faces=engine.process(pixels,now,session['temporal'])
        clean=[{k:v for k,v in f.items() if k!='landmarks'} for f in faces]
        return jsonify(faces=clean,width=pixels.shape[1],height=pixels.shape[0],processing_ms=round((time.monotonic()-now)*1000))
    finally:lock.release()

@app.errorhandler(413)
def too_large(error):return jsonify(error='Ảnh quá lớn.'),413

if __name__=='__main__':app.run(host='127.0.0.1',port=int(os.environ.get('PORT',8080)),threaded=True)
