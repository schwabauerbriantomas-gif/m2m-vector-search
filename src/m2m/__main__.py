"""python -m m2m — serve the M2M edge API (FastAPI + uvicorn).

The Dockerfile ships CMD ["python", "-m", "m2m"] with EXPOSE 8080; without this
module the container exited immediately at startup. This serves the same app
documented in the README (m2m.api.edge_api).

Bind address (audit r3a-18): the default is 127.0.0.1:8080 — what SECURITY.md
promises — so `python -m m2m` is not reachable from the network unless you opt
in with M2M_HOST. The Dockerfile sets M2M_HOST=0.0.0.0 to keep the container
reachable; set M2M_API_KEY there before publishing the port.
"""

import ipaddress
import logging
import os

logger = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080


def resolve_bind_addr() -> tuple[str, int]:
    """(host, port) que usará uvicorn: M2M_HOST / M2M_PORT, default 127.0.0.1:8080."""
    host = os.environ.get("M2M_HOST") or DEFAULT_HOST
    port = int(os.environ.get("M2M_PORT") or DEFAULT_PORT)
    return host, port


def _is_loopback(host: str) -> bool:
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host == "localhost"


def main() -> None:
    try:
        import uvicorn
    except ImportError:
        raise SystemExit(
            "uvicorn is required to serve the M2M API. "
            "Install with: pip install 'm2m-vector-search[all]'"
        )

    from m2m.api.edge_api import app

    host, port = resolve_bind_addr()
    # audit r3a-18: la API sólo exige clave si M2M_API_KEY está definida; atar a un
    # host no-loopback sin clave deja la base abierta a quien alcance el puerto.
    if not _is_loopback(host) and not os.environ.get("M2M_API_KEY"):
        logger.warning(
            "M2M API escuchando en %s:%d (host NO loopback) sin M2M_API_KEY: cualquiera "
            "que alcance el puerto puede leer y escribir la base. Definí M2M_API_KEY "
            "o volvé a M2M_HOST=127.0.0.1.",
            host,
            port,
        )
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
