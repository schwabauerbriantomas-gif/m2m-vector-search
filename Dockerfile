FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml .
COPY src/ src/

RUN pip install --no-cache-dir -e ".[all]" || pip install --no-cache-dir -e .

EXPOSE 8080

# audit r3a-18: `python -m m2m` escucha en 127.0.0.1 por defecto (lo que promete
# SECURITY.md). Bind explícito para que el contenedor siga accesible; definí
# M2M_API_KEY antes de publicar el puerto.
ENV M2M_HOST=0.0.0.0

CMD ["python", "-m", "m2m"]
