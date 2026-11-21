FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90
ENV PYTHONDONTWRITEBYTECODE=1
COPY selector.py /app/selector.py
COPY maps/baseline.json.gz /app/maps/baseline.json.gz
COPY LICENSE tools/SQLGLOT_LICENSE /app/licenses/
CMD ["python", "/app/selector.py"]
