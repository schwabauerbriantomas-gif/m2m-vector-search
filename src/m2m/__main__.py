"""python -m m2m — serve the M2M edge API (FastAPI + uvicorn).

The Dockerfile ships CMD ["python", "-m", "m2m"] with EXPOSE 8080; without this
module the container exited immediately at startup. This serves the same app
documented in the README (m2m.api.edge_api).
"""

import os


def main() -> None:
    try:
        import uvicorn
    except ImportError:
        raise SystemExit(
            "uvicorn is required to serve the M2M API. "
            "Install with: pip install 'm2m-vector-search[all]'"
        )

    from m2m.api.edge_api import app

    host = os.environ.get("M2M_HOST", "0.0.0.0")
    port = int(os.environ.get("M2M_PORT", "8080"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
