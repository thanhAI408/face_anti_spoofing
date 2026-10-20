"""Validation-only controlled check of BatchNorm statistics on the merged model.

Only training pixels update moving statistics. Convolution, dense, BN gamma/beta
weights remain unchanged. Test predictions are never computed in this script.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.retraining_metrics import evaluate_probabilities


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def balanced_loss(probs, records):
    losses = -np.log(np.clip(probs[np.arange(len(records)),
                       [0 if r['label']=='fake' else 1 for r in records]], 1e-7, 1-1e-7))
    groups = {}
    for source in sorted({r['source'] for r in records}):
        for label in ('fake', 'real'):
            videos = sorted({r['video_id'] for r in records if r['source']==source and r['label']==label})
            groups[f'{source}:{label}'] = float(np.mean([
                losses[[i for i,r in enumerate(records) if r['video_id']==video]].mean() for video in videos]))
    return {'mean':float(np.mean(list(groups.values()))),'groups':groups}


def main():
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.keras.utils.set_random_seed(42)
    tf.config.experimental.enable_op_determinism()
    data=ROOT/'dataset/merged-20261001'
    run=ROOT/'outputs/merged-20261001'
    out=ROOT/'outputs/merged-investigation-20261001/bn'
    out.mkdir(parents=True, exist_ok=False)
    m=json.loads((data/'manifest.json').read_text(encoding='utf-8'))
    assert sha(data/'frames.npz')==m['array']['sha256']
    original_sha=sha(run/'liveness.h5')
    x=np.load(data/'frames.npz')['x'].astype(np.float32)/255
    train=np.asarray([i for i,r in enumerate(m['frames']) if r['split']=='train'])
    val=np.asarray([i for i,r in enumerate(m['frames']) if r['split']=='validation'])
    records=[m['frames'][i] for i in val]
    original=tf.keras.models.load_model(run/'liveness.h5',compile=False)
    def prediction(model):
        return np.concatenate([np.asarray(model(x[val[i:i+128]],training=False)) for i in range(0,len(val),128)])
    before=prediction(original)
    saved=np.load(run/'after_validation_probabilities.npy')
    assert np.allclose(before,saved,rtol=0,atol=1e-6)
    report={'method':'Validation-only BatchNorm ablation; training raw pixels only; Dropout disabled during calibration',
            'model_sha256':original_sha,'dataset_manifest_sha256':sha(data/'manifest.json'),
            'test_evaluated':False,'steps_per_calibration':638,'batch_size':32,'seed':42,
            'original':{'validation':evaluate_probabilities(before,records),
                        'balanced_validation_loss':balanced_loss(before,records)},'trials':{}}
    for sampling in ('uniform','source_class_video_balanced'):
        def clone(layer):
            config=layer.get_config()
            if isinstance(layer,tf.keras.layers.Dropout):
                config['rate']=0.0
            return layer.__class__.from_config(config)
        model=tf.keras.models.clone_model(original,clone_function=clone)
        model.set_weights(original.get_weights())
        trainable=[v.numpy().copy() for v in model.trainable_weights]
        rng=np.random.default_rng(42)
        groups=[]
        for source in sorted({m['frames'][i]['source'] for i in train}):
            for label in ('fake','real'):
                ids=[i for i in train if m['frames'][i]['source']==source and m['frames'][i]['label']==label]
                videos=sorted({m['frames'][i]['video_id'] for i in ids})
                groups.append([np.asarray([i for i in ids if m['frames'][i]['video_id']==v]) for v in videos])
        seen=set()
        for _ in range(638):
            if sampling=='uniform':
                ids=rng.choice(train,size=32,replace=True)
            else:
                ids=np.asarray([rng.choice(group[rng.integers(len(group))]) for group in groups for _ in range(8)])
                rng.shuffle(ids)
            seen.update(ids.tolist())
            model(x[ids],training=True)
        assert all(np.array_equal(v.numpy(),saved_v) for v,saved_v in zip(model.trainable_weights,trainable))
        probs=prediction(model)
        np.save(out/f'{sampling}_validation_probabilities.npy',probs)
        # Restoring dropout rates does not affect inference, but preserves the architecture for future use.
        original.set_weights(model.get_weights())
        original.save(out/f'{sampling}.h5')
        # Reload untouched original for a matched next ablation.
        original=tf.keras.models.load_model(run/'liveness.h5',compile=False)
        result={'validation':evaluate_probabilities(probs,records),
                'balanced_validation_loss':balanced_loss(probs,records),
                'train_rows_seen':len(seen),'trainable_weights_unchanged':True,
                'model_sha256':sha(out/f'{sampling}.h5')}
        report['trials'][sampling]=result
        print(json.dumps({'sampling':sampling,'val_loss':result['validation']['loss'],
                          'val_f1':result['validation']['classification_report']['macro avg']['f1-score'],
                          'balanced_val_loss':result['balanced_validation_loss']['mean']}),flush=True)
    assert sha(run/'liveness.h5')==original_sha
    (out/'diagnosis.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')


if __name__=='__main__':
    main()
