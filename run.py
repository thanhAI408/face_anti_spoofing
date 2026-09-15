"""Run the camera or a reproducible headless video check."""
import argparse
import json
import os
from pathlib import Path
import time
import cv2
from pipeline import BASE, Pipeline


def open_camera(index=None):
    for idx in ([index] if index is not None else [1,0,2,3]):
        for backend in ([cv2.CAP_DSHOW,cv2.CAP_MSMF] if os.name=='nt' else [cv2.CAP_ANY]):
            cap = cv2.VideoCapture(idx,backend)
            for _ in range(15 if cap.isOpened() else 0):
                ok,frame = cap.read()
                if ok and frame is not None and frame.std(axis=(0,1)).max()>2:
                    print(f'[INFO] camera {idx}, {cap.getBackendName()}',flush=True)
                    return cap
            cap.release()
    raise RuntimeError('No usable camera. Check privacy settings or select --camera INDEX.')


def render(frame, faces, fps):
    canvas = frame.copy()
    for face in faces:
        x,y,r,b = face['box']
        status = face['status']
        color = (60,210,70) if status=='real' else (50,50,240) if status=='fake' else (0,200,255)
        label = face['quality'] or {'checking':'Checking...', 'uncertain':'Uncertain - retry'}.get(status,status.upper())
        if status in ('fake','real'):
            label += f" | score {face['scores'][status]:.3f}"
        cv2.rectangle(canvas,(x,y),(r,b),color,2)
        tx,ty = max(0,min(x,canvas.shape[1]-min(350,canvas.shape[1]))), max(20,y-8)
        (tw,th),_ = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,.5,1)
        cv2.rectangle(canvas,(tx,ty-th-5),(min(canvas.shape[1],tx+tw+5),ty+4),(20,20,20),-1)
        cv2.putText(canvas,label,(tx+2,ty),cv2.FONT_HERSHEY_SIMPLEX,.5,color,1,cv2.LINE_AA)
    # Separate status bars: never paint over the image sent to either model.
    canvas = cv2.copyMakeBorder(canvas,55,32,0,0,cv2.BORDER_CONSTANT,value=(24,27,32))
    cv2.putText(canvas,f'FACE ANTI-SPOOFING    {fps:.1f} FPS',(12,23),cv2.FONT_HERSHEY_SIMPLEX,.55,(235,235,235),1,cv2.LINE_AA)
    cv2.putText(canvas,'No face detected' if not faces else 'Green: real | Red: fake | Yellow: checking / adjust',(12,44),cv2.FONT_HERSHEY_SIMPLEX,.45,(180,200,220),1,cv2.LINE_AA)
    cv2.putText(canvas,'Q / ESC: quit | Scores are model outputs, not guarantees',(10,canvas.shape[0]-11),cv2.FONT_HERSHEY_SIMPLEX,.43,(190,190,190),1,cv2.LINE_AA)
    return canvas


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('-i','--input')
    ap.add_argument('--camera',type=int)
    ap.add_argument('-m','--model',default=str(BASE/'liveness.h5'))
    ap.add_argument('-l','--le',default=str(BASE/'le.pickle'))
    ap.add_argument('-d','--detector',default=str(BASE/'face_detector/face_detection_yunet_2023mar.onnx'))
    ap.add_argument('-c','--confidence',type=float,default=.85)
    ap.add_argument('--threshold',type=float,default=.8)
    ap.add_argument('--snapshot')
    ap.add_argument('--headless',action='store_true')
    ap.add_argument('--max-frames',type=int,default=0)
    ap.add_argument('--report')
    args=ap.parse_args()
    if not 0<args.confidence<1 or not .5<args.threshold<=1 or args.max_frames<0:
        ap.error('Invalid confidence, threshold or max-frames')
    print('[INFO] loading models...',flush=True)
    pipe=Pipeline(args.model,args.le,args.detector,args.confidence,args.threshold)
    cap=None
    window='Face Anti-Spoofing | Q / ESC to quit'
    records=[]
    try:
        cap=cv2.VideoCapture(args.input) if args.input else open_camera(args.camera)
        if not cap.isOpened():
            raise RuntimeError(f'Cannot open {args.input}')
        source_fps=cap.get(cv2.CAP_PROP_FPS)
        source_fps=source_fps if 0<source_fps<240 else 30
        if not args.headless:
            cv2.namedWindow(window,cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window,960,760)
        count=0
        fps=0.
        while True:
            start=time.perf_counter()
            ok,frame=cap.read()
            if not ok:
                if args.input:
                    break
                raise RuntimeError('Camera stopped delivering frames')
            if not args.input:
                frame=cv2.flip(frame,1)
            if frame.shape[1]>960:
                frame=cv2.resize(frame,(960,round(frame.shape[0]*960/frame.shape[1])))
            now=count/source_fps if args.input else time.monotonic()
            faces=pipe.process(frame,now)
            elapsed=time.perf_counter()-start
            fps=.9*fps+.1/max(elapsed,1e-6) if count else 1/max(elapsed,1e-6)
            canvas=render(frame,faces,fps)
            count+=1
            if args.report:
                records.append({'frame':count,'faces':[{k:v for k,v in face.items() if k!='landmarks'} for face in faces]})
            if args.snapshot and count==31:
                Path(args.snapshot).parent.mkdir(parents=True,exist_ok=True)
                if not cv2.imwrite(args.snapshot,canvas):
                    raise IOError('Failed to save snapshot')
            if count==1 or count%100==0:
                print(f'[INFO] frames={count} faces={len(faces)} statuses={[f["status"] for f in faces]} fps={fps:.1f}',flush=True)
            if not args.headless:
                cv2.imshow(window,canvas)
                if cv2.waitKey(1)&255 in (ord('q'),27) or cv2.getWindowProperty(window,cv2.WND_PROP_VISIBLE)<1:
                    break
            if args.max_frames and count>=args.max_frames:
                break
        if count==0:
            raise RuntimeError('Input contained no readable frames')
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        if args.report:
            Path(args.report).parent.mkdir(parents=True,exist_ok=True)
            Path(args.report).write_text(json.dumps(records,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
