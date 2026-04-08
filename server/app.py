"""FastAPI app for the vulnerability triage environment."""

from __future__ import annotations

try:
    from openenv.core.env_server.http_server import create_app
except Exception as exc:  # pragma: no cover
    raise ImportError("openenv-core is required to run this server") from exc

try:
    from ..models import VulnTriageAction, VulnTriageObservation
    from .vuln_triage_env_environment import VulnTriageEnvironment
except (ModuleNotFoundError, ImportError):
    from models import VulnTriageAction, VulnTriageObservation
    from server.vuln_triage_env_environment import VulnTriageEnvironment


app = create_app(
    VulnTriageEnvironment,
    VulnTriageAction,
    VulnTriageObservation,
    env_name="vulnops",
    max_concurrent_envs=4,
)


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
