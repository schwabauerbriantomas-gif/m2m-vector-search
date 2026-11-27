FROM python:3.12-slim

WORKDIR /app

# Lo mínimo que necesita `pip install .`: pyproject.toml lee README.md (readme) y
# license-files apunta a LICENSE; y MANIFEST.in, sin el cual setuptools NO empaqueta los
# shaders .comp ni py.typed (el paquete instalado en la imagen quedaba sin shaders).
COPY pyproject.toml README.md MANIFEST.in LICENSE ./
COPY src/ ./src/

# audit r3b-04: antes era `pip install .` -> solo numpy+scikit-learn, y el CMD era
# `uvicorn m2m.api.coordinator_api:app`: los 4 contenedores del compose morian al arrancar
# con ModuleNotFoundError: fastapi / httpx / uvicorn. Instalamos el extra `api`
# (fastapi, uvicorn, httpx, pydantic, requests). Sin gcc: todas las deps tienen wheel.
RUN pip install --no-cache-dir ".[api]"

# Usuario non-root (r3b-04): la imagen corria como root.
RUN useradd --create-home --uid 1000 m2m
USER m2m

# Expose port
EXPOSE 8000

# audit r3a-18: `python -m m2m` (override documentado del CMD) escucha en
# 127.0.0.1 por defecto; bind explicito para que el contenedor siga accesible.
ENV M2M_HOST=0.0.0.0

# HEALTHCHECK a nivel TCP porque sirve para los dos roles de esta imagen: el edge tiene
# GET /health real (src/m2m/api/edge_api.py) pero el coordinator NO tiene ninguna ruta
# GET, asi que un probe HTTP fallaria ahi. El compose usa el probe HTTP por servicio.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import socket; socket.create_connection(('127.0.0.1', 8000), timeout=3).close()"

# Default command (can be overridden to run edge_api)
CMD ["uvicorn", "m2m.api.coordinator_api:app", "--host", "0.0.0.0", "--port", "8000"]
