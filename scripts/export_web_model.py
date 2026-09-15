import tensorflow as tf
from pathlib import Path
import pickle,json
m=tf.keras.models.load_model('liveness.h5',compile=False)
# Export inference only with a fixed single-face signature.
@tf.function(input_signature=[tf.TensorSpec([1,32,32,3],tf.float32)])
def serving(x):return m(x,training=False)
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
frozen=convert_variables_to_constants_v2(serving.get_concrete_function())
converter=tf.lite.TFLiteConverter.from_concrete_functions([frozen])
converter.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS]
Path('web_models/liveness.tflite').write_bytes(converter.convert())
with open('le.pickle','rb') as f: labels=pickle.load(f).classes_.tolist()
Path('web_models/labels.json').write_text(json.dumps(labels))
print('EXPORTED',Path('web_models/liveness.tflite').stat().st_size,labels)
