FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml .
COPY src/ src/

RUN pip install --no-cache-dir -e ".[all]" || pip install --no-cache-dir -e .

EXPOSE 8080

CMD ["python", "-m", "m2m"]
