# syntax=docker/dockerfile:1.7
FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90 AS baseline
ADD --checksum=sha256:93b2445f15f395b81c20556777ece19a6c7507df950f13fa9fac4a3be739af0d \
    https://codeload.github.com/tobymao/sqlglot/tar.gz/eb11aec4be8cb2122afe713a4816662c3f0fb2fa /tmp/sqlglot.tar.gz
RUN mkdir /source && tar -xzf /tmp/sqlglot.tar.gz --strip-components=1 -C /source \
    && chmod -R a+rX /source
RUN PYTHONHASHSEED=0 python -m compileall -q --invalidation-mode checked-hash /source/sqlglot /source/tests

FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHON_CPU_COUNT=2 PYTHONHASHSEED=0 \
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1 HOME=/tmp
COPY requirements.lock /app/requirements.lock
RUN python -m pip install --no-cache-dir -r /app/requirements.lock
COPY --from=baseline /source /app/source
COPY selector.py runtime.py querycheck.py comparisoncheck.py valuecheck.py queryreplay.py behaviorcheck.py compiledcheck.py fieldcheck.py statementcheck.py fixturecheck.py passcheck.py relationcheck.py statecheck.py stagecheck.py dispatchcheck.py warmcheck.py /app/
COPY maps/baseline.json.gz maps/runtime.json.gz maps/settings.json.gz maps/comparisons.json.gz maps/values.json.gz maps/query-values.pkl.gz maps/queries.pkl.gz maps/behavior.json.gz maps/timing.json.gz maps/compiled.json.gz maps/fields.json.gz maps/fixtures.json.gz maps/passes.pkl.xz maps/dispatch.json.gz /app/maps/
RUN python -c 'import gzip,lzma; from pathlib import Path; p=Path("/app/maps/query-values.pkl.gz"); raw=gzip.decompress(p.read_bytes()); raw=lzma.decompress(raw) if raw.startswith(b"\xfd7zXZ\x00") else raw; p.write_bytes(gzip.compress(raw,mtime=0))' \
    && python /app/statecheck.py --source /app/source --out /app/maps/import-state.json.gz
COPY LICENSE tools/SQLGLOT_LICENSE /app/licenses/
WORKDIR /app
USER 65534:65534
CMD ["python", "/app/selector.py", "--aggressive", "--aggressive-min-cpu", "1.0"]
