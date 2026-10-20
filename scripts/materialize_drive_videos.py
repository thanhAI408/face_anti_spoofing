"""Materialize authenticated Drive file references received through stdin.

Download references never enter the persistent inventory or logs.
"""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import sys
import threading
import time
import requests

ROOT = Path(__file__).resolve().parents[1]
LOCK = threading.Lock()
STATUS = ROOT / 'outputs/drive-20260930/download_status.jsonl'


def download(item):
    destination = (ROOT / item['relative_path']).resolve()
    if not destination.is_relative_to(ROOT / 'videos/drive-20260930'):
        raise ValueError('Destination outside authorized dataset')
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected = int(item['size'])
    if destination.exists():
        if destination.stat().st_size != expected:
            raise ValueError('Existing destination has wrong size')
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        status = 'already_materialized'
    else:
        partial = destination.with_suffix(destination.suffix + '.partial')
        digest = hashlib.sha256()
        size = partial.stat().st_size if partial.exists() else 0
        if size > expected:
            raise ValueError('Existing partial exceeds expected size')
        if size:
            with partial.open('rb') as previous:
                for block in iter(lambda: previous.read(1024 * 1024), b''):
                    digest.update(block)
        try:
            deadline = time.monotonic() + 900
            headers = {'Range': f'bytes={size}-'} if size else {}
            with requests.get(item['download_url'], headers=headers,
                              stream=True, timeout=(20, 20)) as response:
                if size and response.status_code == 206:
                    content_range = response.headers.get('Content-Range', '')
                    if content_range != f'bytes {size}-{expected - 1}/{expected}':
                        raise ValueError('Unexpected Content-Range')
                    mode = 'ab'
                elif response.status_code == 200:
                    # A provider can ignore Range; restart rather than append.
                    size, digest, mode = 0, hashlib.sha256(), 'wb'
                else:
                    raise RuntimeError(f'Provider HTTP {response.status_code}')
                with partial.open(mode) as output:
                    for block in response.iter_content(64 * 1024):
                        if time.monotonic() > deadline:
                            raise TimeoutError('Whole-download deadline exceeded')
                        output.write(block)
                        digest.update(block)
                        size += len(block)
            if size != expected:
                raise ValueError(f'Download size mismatch {size}/{expected}')
            partial.replace(destination)
            digest = digest.hexdigest()
            status = 'downloaded'
        except Exception as exc:
            print(json.dumps({'id': item['id'], 'error_type': type(exc).__name__}), flush=True)
            raise RuntimeError('Materialization failed; reference withheld') from None
    result = {key: item[key] for key in ('id', 'relative_path', 'size')}
    result.update(sha256=digest, status=status)
    with LOCK:
        STATUS.parent.mkdir(parents=True, exist_ok=True)
        with STATUS.open('a', encoding='utf-8') as output:
            output.write(json.dumps(result, ensure_ascii=False) + '\n')
        print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


if __name__ == '__main__':
    items = json.load(sys.stdin)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(download, items))
