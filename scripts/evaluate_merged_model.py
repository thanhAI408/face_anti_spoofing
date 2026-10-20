"""Check merged candidate on frozen Sep30 anchors and actual API routes.

This is a development benchmark and in-process integration check. It does not
measure network latency, camera FPS, or independent production accuracy.
"""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.retraining_metrics import evaluate_probabilities, paired_video_bootstrap


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def predict(model, x):
    return np.concatenate([np.asarray(model(x[start:start+128], training=False))
                           for start in range(0, len(x), 128)])


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='outputs/merged-20261001')
    parser.add_argument('--dataset', default='dataset/merged-20261001')
    args = parser.parse_args()
    run, data = ROOT/args.run, ROOT/args.dataset
    evaluation = json.loads((run/'evaluation.json').read_text(encoding='utf-8'))
    manifest = json.loads((data/'manifest.json').read_text(encoding='utf-8'))
    old_data = ROOT/'dataset/retrain-20260930'
    old_run = ROOT/'outputs/retrain-20260930'
    old_manifest = json.loads((old_data/'manifest.json').read_text(encoding='utf-8'))
    prior_evaluation = json.loads((old_run/'evaluation.json').read_text(encoding='utf-8'))
    if (sha(old_run/'liveness.h5') != prior_evaluation['new_model_sha256']
            or evaluation['configuration']['baseline_model_sha256'] != prior_evaluation['new_model_sha256']
            or sha(run/'liveness.h5') != evaluation['new_model_sha256']):
        raise ValueError('Before/after model hash does not match the intended runs')
    if sha(data/'manifest.json') != evaluation['configuration']['dataset_manifest_sha256']:
        raise ValueError('Merged manifest differs from the trained run')
    if sha(old_data/'frames.npz') != old_manifest['array']['sha256']:
        raise ValueError('Frozen anchor archive changed')
    train_records = [r for r in manifest['frames'] if r['split']=='train']
    train_videos = {r['video_id'].split(':', 1)[-1] for r in train_records}
    anchor_videos = {r['video_id'].split(':', 1)[-1] for r in old_manifest['frames']
                     if r['split'] in ('test', 'regression_old')}
    if train_videos & anchor_videos:
        raise ValueError('An anchor video leaked into merged training')
    train_pixels = {r.get('frame_pixel_sha256') or r.get('crop_pixel_sha256') for r in train_records}
    arrays = np.load(old_data/'frames.npz')['x']
    for i, r in enumerate(old_manifest['frames']):
        if r['split'] in ('test','regression_old') and hashlib.sha256(arrays[i].tobytes()).hexdigest() in train_pixels:
            raise ValueError('Exact anchor pixels leaked into merged training')
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    models = {phase: tf.keras.models.load_model(path/'liveness.h5', compile=False)
              for phase,path in [('before',old_run),('after',run)]}
    labels = {phase: tuple(pickle.loads((path/'le.pickle').read_bytes()).classes_)
              for phase,path in [('before',old_run),('after',run)]}
    anchor = {'method':'Both frozen models on the same Sep30 original uint8 crops; argmax, no augmentation',
              'baseline_model_sha256':sha(old_run/'liveness.h5'),
              'candidate_model_sha256':sha(run/'liveness.h5'),
              'manifest_sha256':sha(old_data/'manifest.json'),
              'input_hashes':{'prior_manifest':sha(old_data/'manifest.json'),
                              'prior_archive':sha(old_data/'frames.npz'),
                              'merged_manifest':sha(data/'manifest.json')}, 'groups':{},
              'limitations':['Development benchmark previously inspected, not a new independent final test.',
                             'Old anchor people can overlap training people; videos are held out.']}
    for split,title in [('test','new'),('regression_old','old')]:
        ids = [i for i,r in enumerate(old_manifest['frames']) if r['split']==split]
        records = [old_manifest['frames'][i] for i in ids]
        x = arrays[ids].astype(np.float32)/255.0
        group = {'prior_split':split, 'probability_files':{}}
        for phase in models:
            probs = predict(models[phase],x)
            np.save(run/f'anchor_{phase}_{title}_probabilities.npy',probs)
            group['probability_files'][phase] = f'anchor_{phase}_{title}_probabilities.npy'
            group[phase] = evaluate_probabilities(probs,records,labels=labels[phase])
            if phase=='before':
                saved = np.load(old_run/f'after_{split}_probabilities.npy')
                if not np.allclose(saved,probs,rtol=0,atol=1e-6):
                    raise ValueError('Frozen baseline anchor predictions differ from the prior run')
        group['paired_video_bootstrap'] = paired_video_bootstrap(group['before'],group['after'])
        anchor['groups'][title] = group
        print(json.dumps({'phase':'anchor','source':title,'frames':len(ids),
                          **{p:group[p]['classification_report']['macro avg']['f1-score'] for p in models}}),flush=True)
    save(run/'anchor_comparison.json',anchor)

    from api_app import create_app
    from clients.antispoof_client import encode_jpeg
    class Clock:
        value = 0.0
        def __call__(self):
            return self.value
    clock = Clock()
    app = create_app(run/'liveness.h5',run/'le.pickle',api_key='',clock=clock)
    app.testing = True
    client = app.test_client()
    health = client.get('/v1/health')
    if health.status_code != 200 or health.json['model']['sha256'] != sha(run/'liveness.h5'):
        raise ValueError('Candidate health/model binding failed')
    prior_smoke = json.loads((old_run/'api_smoke.json').read_text(encoding='utf-8'))
    videos = {v['video_id']:v for v in old_manifest['videos']}
    examples = []
    for item in prior_smoke['examples']:
        video = videos[item['video_id']]
        image_path = old_data/video['preview']['frame']
        frame = cv2.imdecode(np.frombuffer(image_path.read_bytes(),dtype=np.uint8),cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError('Missing API sample pixels')
        payload = encode_jpeg(frame)
        response = client.post('/v1/analyze',data=payload,content_type='image/jpeg')
        session = client.post('/v1/sessions')
        if session.status_code != 201:
            raise ValueError('Candidate session creation failed')
        token = session.json['session_id']
        responses = [response]
        statuses = []
        for _ in range(7):
            clock.value += .1
            result = client.post(f'/v1/sessions/{token}/frames',data=payload,content_type='image/jpeg')
            responses.append(result)
            statuses.append(result.json.get('status'))
        deleted = client.delete(f'/v1/sessions/{token}')
        payload_sha = hashlib.sha256(payload).hexdigest()
        bound = all(r.status_code==200 and r.json['image']['sha256']==payload_sha
                    and r.json['model']['sha256']==sha(run/'liveness.h5')
                    for r in responses)
        if not bound or deleted.status_code not in (200,204):
            raise ValueError('Candidate image binding or route response failed')
        examples.append({'video_id':item['video_id'],'true_label':item['true_label'],
                         'input_path':str(image_path),'input_file_sha256':sha(image_path),
                         'jpeg_payload_sha256':payload_sha,
                         'single_status':response.json['status'],'session_statuses':statuses,
                         'http_statuses':[r.status_code for r in responses],
                         'response_image_hashes':[r.json['image']['sha256'] for r in responses],
                         'response_model_hashes':[r.json['model']['sha256'] for r in responses],
                         'session_create_status':session.status_code,
                         'session_delete_status':deleted.status_code,
                         'all_http_statuses_ok':True,'exact_image_hash_binding_ok':True})
    save(run/'api_smoke.json',{'method':'Actual candidate H5 with Flask in-process test client and SDK JPEG',
                             'model_sha256':sha(run/'liveness.h5'),'health_http_status':200,'examples':examples,
                             'model_activated_for_current_api':False,
                             'limitations':['Integration only, not accuracy or real liveness proof.',
                                            'Repeated JPEGs test state, not independent frames.',
                                            'Network latency and live camera FPS not measured.']})
    print(json.dumps({'phase':'api_smoke','passed':True,'examples':len(examples)}),flush=True)


if __name__=='__main__':
    main()
