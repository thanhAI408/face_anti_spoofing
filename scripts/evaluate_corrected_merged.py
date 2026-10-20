"""Evaluate a validation-selected corrected candidate against two frozen models.

Selection must be written before this script starts. All three models are scored
on exactly the same merged rows; the test cannot influence the saved selection.
"""
import argparse
import json
from pathlib import Path
import pickle
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.train_retraining_comparison import (
    sha, check_split, source_summary, evaluate_splits, baseline_training_metadata,
)
from scripts.retraining_metrics import paired_video_bootstrap


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',default='outputs/merged-corrected-20261001')
    args=parser.parse_args()
    run=ROOT/args.run
    selection=read(run/'selection.json')
    summary=read(run/'training_summary.json')
    config=read(run/'configuration.json')
    data=ROOT/'dataset/merged-20261001'
    manifest=read(data/'manifest.json')
    if (selection['candidate_model_sha256']!=sha(run/'liveness.h5')
            or selection['dataset_manifest_sha256']!=sha(data/'manifest.json')
            or selection['test_used_for_selection'] is not False):
        raise ValueError('Candidate was not frozen using the intended validation policy')
    if sha(data/'frames.npz')!=manifest['array']['sha256']:
        raise ValueError('Merged test pixels changed')
    records=manifest['frames']
    check_split(records,manifest['split_policy'])
    arrays=np.load(data/'frames.npz')['x']
    assert arrays.shape==(len(records),32,32,3) and arrays.dtype==np.uint8
    assert all(r['array_index']==i for i,r in enumerate(records))
    x=arrays.astype(np.float32)/255
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    phases={'before':ROOT/'outputs/retrain-20260930',
            'failed':ROOT/'outputs/merged-20261001','after':run}
    evaluation={}
    for phase,path in phases.items():
        labels=tuple(pickle.loads((path/'le.pickle').read_bytes()).classes_)
        if labels!=('fake','real'):
            raise ValueError('Class order mismatch')
        model=tf.keras.models.load_model(path/'liveness.h5',compile=False)
        if model.input_shape[1:]!=(32,32,3) or model.output_shape[-1]!=2:
            raise ValueError('Candidate is not compatible with current API preprocessing')
        evaluation[phase]=evaluate_splits(model,labels,x,records,run,phase)
        if phase in ('before','failed'):
            previous=ROOT/'outputs/merged-20261001'
            for split in ('train','validation','test'):
                original=np.load(previous/f'{"before" if phase=="before" else "after"}_{split}_probabilities.npy')
                fresh=np.load(run/f'{phase}_{split}_probabilities.npy')
                if not np.allclose(original,fresh,rtol=0,atol=1e-6):
                    raise ValueError(f'Frozen {phase}/{split} predictions changed')
        del model
    config.update({'dataset_manifest_sha256':sha(data/'manifest.json'),
                   'baseline_model_path':'outputs/retrain-20260930/liveness.h5',
                   'baseline_model_sha256':sha(phases['before']/'liveness.h5'),
                   'baseline_training':baseline_training_metadata(phases['before']/'liveness.h5'),
                   'failed_merge_model_sha256':sha(phases['failed']/'liveness.h5'),
                   'candidate_role':'after: corrected training on unchanged merged dataset',
                   'evaluation_loss':'unweighted original frames, inference mode',
                   'selection_sha256':sha(run/'selection.json')})
    result={'configuration':config,**evaluation,'selection':selection,
            'training_summary':summary,'new_model_sha256':sha(run/'liveness.h5'),
            'source_split_summary':source_summary(records),'dataset_summary':manifest['summary'],
            'test_video_paired_bootstrap':paired_video_bootstrap(evaluation['before']['test'],evaluation['after']['test']),
            'test_video_paired_bootstrap_by_source':{
                s:paired_video_bootstrap(evaluation['before']['test']['per_source'][s],
                                         evaluation['after']['test']['per_source'][s])
                for s in ('legacy-cache','drive-20260930')},
            'limitations':['Previously inspected development test, not a new independent final test.',
                           'Legacy videos are held out but identities can overlap train; new identities are supplied folder identifiers.',
                           'Raw argmax frame error rates are not actual API temporal acceptance rates.']}
    with (run/'evaluation.json').open('x',encoding='utf-8') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps({'complete':True,'test_f1':{
        phase:evaluation[phase]['test']['classification_report']['macro avg']['f1-score'] for phase in phases}}))


if __name__=='__main__':
    main()
