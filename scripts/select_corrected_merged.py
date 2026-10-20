"""Freeze the best of two completed validation-only trials before any test run."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.train_balanced_merged import source_class_video_loss
from scripts.train_retraining_comparison import sha


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    manifest=ROOT/'dataset/merged-20261001/manifest.json'
    records=[r for r in read(manifest)['frames'] if r['split']=='validation']
    trials=[]
    for directory in ('outputs/merged-balanced-20261001','outputs/merged-groupnorm-20261001'):
        path=ROOT/directory
        summary=read(path/'training_summary.json')
        config=read(path/'configuration.json')
        if (summary['test_predictions_or_metrics_created'] is not False
                or not summary['validation_only_trial']
                or config['dataset_manifest_sha256']!=sha(manifest)
                or summary['model_sha256']!=sha(path/'liveness.h5')
                or summary['configuration_sha256']!=sha(path/'configuration.json')):
            raise ValueError('Selection trial provenance is invalid')
        loss=source_class_video_loss(np.load(path/'selected_validation_probabilities.npy'),records)
        if not np.isclose(loss,summary['minimum_weighted_validation_ce'],rtol=0,atol=1e-6):
            raise ValueError('Trial validation score does not match saved probabilities')
        trials.append({'run_directory':directory,'candidate_model_sha256':sha(path/'liveness.h5'),
                       'validation_source_class_video_loss':loss,
                       'configuration_sha256':sha(path/'configuration.json'),
                       'training_summary_sha256':sha(path/'training_summary.json'),
                       'epochs_run':summary['epochs_run'],'best_epoch':summary['best_epoch']})
    chosen=min(trials,key=lambda trial:trial['validation_source_class_video_loss'])
    out=ROOT/'outputs/merged-corrected-20261001'
    out.mkdir(parents=True,exist_ok=False)
    source=ROOT/chosen['run_directory']
    for item in source.iterdir():
        if item.is_file():
            shutil.copy2(item,out/item.name)
    selection={'created_at':datetime.now(timezone.utc).isoformat(),
               'metric':'val_source_class_video_loss','rule':'minimum validation CE; earliest trial breaks exact ties',
               'candidates':trials,'selected_run_directory':chosen['run_directory'],
               'candidate_model_sha256':chosen['candidate_model_sha256'],
               'dataset_manifest_sha256':sha(manifest),'test_used_for_selection':False,
               'test_previously_inspected':True,
               'caveat':'Previously inspected development test; not an independent final benchmark'}
    (out/'selection.json').write_text(json.dumps(selection,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'selected':chosen['run_directory'],'validation_loss':chosen['validation_source_class_video_loss'],
                      'test_evaluated':False,'output':str(out)}))


if __name__=='__main__':
    main()
