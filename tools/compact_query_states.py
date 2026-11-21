"""Recompress exact worker-state chunks without changing snapshot bytes."""

import argparse
import gzip
import lzma
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from queryreplay import decode_chunk, load_profile


def compact(data):
    if data['version'] != 2:
        raise ValueError('chunked worker snapshots required')
    chunks = {}
    for key in data['chunks']:
        raw = decode_chunk(data, key)
        chunks[key] = lzma.compress(raw, preset=6)
        if lzma.decompress(chunks[key]) != raw:
            raise ValueError('worker-state bytes changed')
    return dict(data, codec='xz', chunks=chunks)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = compact(load_profile(args.input))
    with gzip.GzipFile(filename=str(args.out), mode='wb', mtime=0) as stream:
        pickle.dump(result, stream, protocol=5)
    print(f'{len(result["chunks"])} chunks; {args.out.stat().st_size} bytes', flush=True)
