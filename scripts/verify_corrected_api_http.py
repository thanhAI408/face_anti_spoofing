"""Check the frozen candidate over real loopback HTTP, then stop the test server."""
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from clients.antispoof_client import AntiSpoofClient, encode_jpeg


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    run = ROOT / 'outputs/merged-corrected-20261001'
    selection = read(run / 'selection.json')
    model = run / 'liveness.h5'
    labels = run / 'le.pickle'
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    if digest != selection['candidate_model_sha256'] or selection['test_used_for_selection'] is not False:
        raise ValueError('Candidate was not frozen by validation')
    output = run / 'http_smoke.json'
    benchmark = run / 'http_benchmark.json'
    if output.exists() or benchmark.exists():
        raise FileExistsError('Keep the earlier HTTP evidence intact')
    host, port = '127.0.0.1', 8001
    with socket.socket() as probe:
        probe.bind((host, port))  # Fail without disturbing an occupied server.
    url = f'http://{host}:{port}'
    import cv2
    import numpy as np
    smoke = read(run / 'api_smoke.json')
    examples = []
    flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
    with (run / 'http_server.log').open('x', encoding='utf-8') as log:
        process = subprocess.Popen([
            sys.executable, '-X', 'utf8', 'api_app.py', '--host', host,
            '--port', str(port), '--cpu-threads', '2', '--model', str(model), '--labels', str(labels),
        ], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
        try:
            with AntiSpoofClient(url, timeout=(1, 10)) as client:
                deadline = time.monotonic() + 45
                while True:
                    if process.poll() is not None:
                        raise RuntimeError('Test server failed to start; read http_server.log')
                    try:
                        health = client.health()
                        break
                    except Exception:
                        if time.monotonic() >= deadline:
                            raise TimeoutError('Test server did not become ready')
                        time.sleep(.25)
                if health['model']['sha256'] != digest:
                    raise ValueError('HTTP served a different model')
                for item in smoke['examples']:
                    path = Path(item['input_path'])
                    frame = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
                    payload = encode_jpeg(frame)
                    payload_sha = hashlib.sha256(payload).hexdigest()
                    single = client.analyze(payload)
                    session = client.create_session()
                    responses = [single]
                    try:
                        for _ in range(7):
                            time.sleep(.1)
                            responses.append(client.frame(session, payload))
                    finally:
                        client.close_session(session)
                    if not all(r['model']['sha256'] == digest and r['image']['sha256'] == payload_sha
                               for r in responses):
                        raise ValueError('HTTP image/model binding failed')
                    examples.append({'video_id': item['video_id'], 'true_label': item['true_label'],
                                     'jpeg_sha256': payload_sha,
                                     'statuses': [r['status'] for r in responses],
                                     'response_model_hashes': [r['model']['sha256'] for r in responses],
                                     'response_image_hashes': [r['image']['sha256'] for r in responses],
                                     'analyze_and_session_routes_passed': True})
            subprocess.run([
                sys.executable, '-X', 'utf8', 'scripts/benchmark_api.py', '--url', url,
                '--image', smoke['examples'][0]['input_path'], '--model', str(model),
                '--labels', str(labels), '--cpu-threads', '2', '--iterations', '10', '--output', str(benchmark),
            ], cwd=ROOT, check=True, creationflags=flags)
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
    result = {'passed': True, 'url': url, 'model_sha256': digest, 'health': health,
              'examples': examples, 'server_stopped': process.poll() is not None,
              'model_activated_for_default_api': False,
              'limitations': ['Loopback HTTP integration; external network and live camera were not tested.',
                              'Repeated JPEG session checks verify transport/state, not independent liveness.',
                              'Latency is measured using one repeated image and a serial client.']}
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({'http_passed': True, 'examples': len(examples), 'server_stopped': result['server_stopped']}))


if __name__ == '__main__':
    main()
