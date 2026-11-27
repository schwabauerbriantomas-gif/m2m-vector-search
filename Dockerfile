FROM python:3.12-slim

WORKDIR /app

# pyproject.toml lee README.md (readme) y license-files apunta a LICENSE; MANIFEST.in hace
# que setuptools empaquete los shaders .comp y py.typed dentro del paquete instalado.
COPY pyproject.toml README.md MANIFEST.in LICENSE ./
COPY src/ ./src/

# audit r3b-04 / r3b-11: antes era `pip install -e ".[all]" || pip install -e .`. El extra
# `all` trae torch+sentence-transformers (>1 GB) que `python -m m2m` no usa, y el `||`
# convertia un fallo de ese extra en una imagen sin uvicorn que moria al arrancar el CMD.
# Instalacion no editable con el extra `api` (lo que importa m2m.api.edge_api + uvicorn).
RUN pip install --no-cache-dir ".[api]"

# Usuario non-root (r3b-04).
RUN useradd --create-home --uid 1000 m2m
USER m2m

EXPOSE 8080

# audit r3a-18: `python -m m2m` escucha en 127.0.0.1 por defecto (lo que promete
# SECURITY.md). Bind explícito para que el contenedor siga accesible; definí
# M2M_API_KEY antes de publicar el puerto.
ENV M2M_HOST=0.0.0.0

CMD ["python", "-m", "m2m"]
