"""Cache validated bundled evidence during image construction."""

from __future__ import annotations

import hashlib
import pickle
import sys
from pathlib import Path


def fingerprint(path, loader):
    from comparisoncheck import REVISION
    root = Path(__file__).resolve().parent
    return (1, REVISION, tuple(sys.version_info[:2]), loader,
            hashlib.sha256(Path(path).read_bytes()).digest(),
            tuple(hashlib.sha256((root / name).read_bytes()).digest()
                  for name in ('profilecache.py', 'behaviorcheck.py',
                               'comparisoncheck.py', 'relationcheck.py')))


def load(path, loader):
    """A stale or unavailable cache falls back to the original validator."""
    path = Path(path)
    cache = path.with_name(path.name + '.validated.pkl')
    if not cache.is_file():
        return None
    try:
        expected = fingerprint(path, loader)
        with cache.open('rb') as stream:
            recorded, digest = pickle.load(stream)
            if recorded != expected:
                return None
            payload = stream.read()
            if hashlib.sha256(payload).digest() != digest:
                return None
            return pickle.loads(payload)
    except (OSError, ValueError, TypeError, EOFError, pickle.UnpicklingError):
        return None


def prepare(path, loader_name, loader):
    # Run the complete existing validation and expansion before caching. No
    # evaluation inputs, patches, or outcome labels participate in this step.
    path = Path(path)
    data = loader(path, use_cache=False)
    if data is None:
        raise ValueError('invalid bundled profile: ' + path.name)
    cache = path.with_name(path.name + '.validated.pkl')
    payload = pickle.dumps(data, protocol=5)
    with cache.open('wb') as stream:
        pickle.dump((fingerprint(path, loader_name), hashlib.sha256(payload).digest()),
                    stream, protocol=5)
        stream.write(payload)


if __name__ == '__main__':
    from behaviorcheck import load_profile
    from comparisoncheck import load_comparison_profile
    root = Path(__file__).resolve().parent / 'maps'
    prepare(root / 'behavior.json.gz', 'behavior', load_profile)
    prepare(root / 'comparisons.json.gz', 'comparisons', load_comparison_profile)
