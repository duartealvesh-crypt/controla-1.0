"""L'application web : une API JSON mince au-dessus de ce qui existe déjà.

Discipline de cette couche, la même qu'ailleurs dans le projet : **elle ne
recalcule rien**. Le classement vient du manifeste, les métriques du manifeste,
la carte de l'instantané, les coups légaux du masque de l'environnement. Si une
valeur affichée n'existe pas en amont, c'est en amont qu'il faut l'ajouter —
sinon l'écran et le moteur finissent par raconter deux histoires.

Le serveur est **local et sans authentification**. Il lit un dossier de runs et
lance des parties ; il n'est pas fait pour être exposé. La seule protection est
celle qui compte à ce niveau : aucun chemin servi ne peut sortir de la racine
(`library.resolve_within`).
"""

from __future__ import annotations

from pathlib import Path

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from controla.config.loader import load_config
from controla.registry.catalog import build_catalog, describe_formulas
from controla.replay.render import building_glyph
from controla.ui import library
from controla.ui.session import HUMAN_SEAT, Session, SessionSpec, SessionStore

STATIC_DIR = Path(__file__).resolve().parent / "static"


def create_app(root: Path | str = library.DEFAULT_ROOT, config_path: str | None = None):
    root = Path(root)
    cfg = load_config(config_path)
    catalog = build_catalog(cfg)
    sessions = SessionStore()
    # Partagé entre toutes les sessions : un seul client par modèle, comme dans
    # un tournoi, et des statistiques d'appel qui s'accumulent.
    backends: dict = {}

    app = FastAPI(title="Controla", docs_url=None, redoc_url=None)

    # ── contexte ─────────────────────────────────────────────────────────────

    @app.get("/api/context")
    def context() -> dict:
        """Tout ce qui ne change pas d'une requête à l'autre.

        Chargé une fois au démarrage de la page : libellés, sièges disponibles,
        coûts d'action, formules. Le navigateur n'a alors plus besoin de
        connaître le catalogue pour rendre quoi que ce soit.
        """
        from controla.policies.factory import available_seats

        seats = available_seats()
        return {
            "seats": {**seats, "human": HUMAN_SEAT},
            "max_turns": int(cfg.world.max_turns),
            "grid_size": int(cfg.world.grid_size),
            "num_factions": int(cfg.world.num_factions),
            "terrain_keys": list(catalog.terrain_keys),
            "building_keys": list(catalog.building_keys),
            "trait_keys": list(catalog.trait_keys),
            "glyphs": {key: building_glyph(key) for key in catalog.building_keys},
            "labels": {
                key: catalog.label(key)
                for key in (
                    *catalog.terrain_keys, *catalog.building_keys,
                    *catalog.trait_keys,
                    *(spec.key for spec in catalog.actions),
                    *(spec.key for spec in catalog.resources),
                )
            },
            "actions": [
                {
                    "key": spec.key,
                    "label": spec.label,
                    "kind": spec.kind.value,
                    "cost": spec.cost,
                    "note": spec.note,
                    # Dérivée du descripteur, comme le règlement du prompt : ce
                    # que l'écran explique est ce que le moteur applique.
                    "describe": spec.describe(cfg, catalog),
                }
                for spec in catalog.actions
            ],
            "formulas": describe_formulas(cfg),
            "score_weights": dict(cfg.score.weights),
        }

    @app.get("/api/models")
    def models() -> dict:
        """Audit du registre de modèles — sert la page « fournisseur »."""
        from controla.policies.llm.catalogue import (
            audit,
            campaign_cost,
            load_provider_catalogues,
        )
        from controla.policies.llm.registry import load_dotenv_if_present, load_models

        load_dotenv_if_present()
        declared = load_models()
        catalogues, errors = load_provider_catalogues(declared)
        reports = audit(declared, catalogues)
        for report in reports:
            report["cost_48_games"] = campaign_cost(report, games=48)
        return {"models": reports, "errors": errors}

    # ── bibliothèque ─────────────────────────────────────────────────────────

    @app.get("/api/runs")
    def runs() -> dict:
        return {
            "root": str(root),
            "runs": library.list_runs(root),
            "loose": library.list_loose_games(root),
        }

    @app.get("/api/run")
    def run(path: str = Query(...)) -> dict:
        try:
            return {
                "manifest": library.load_manifest(root, path),
                "games": library.list_games(root, path),
            }
        except library.OutsideRoot as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except (OSError, ValueError) as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.get("/api/game")
    def game(path: str = Query(...)) -> dict:
        try:
            return library.load_game(root, path)
        except library.OutsideRoot as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except (OSError, ValueError) as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    # ── partie jouable ───────────────────────────────────────────────────────

    @app.post("/api/session")
    def create_session(payload: dict = Body(...)) -> dict:
        seats = payload.get("seats") or []
        try:
            human_seat = seats.index(HUMAN_SEAT)
        except ValueError as error:
            raise HTTPException(
                status_code=400,
                detail=f"aucun siège '{HUMAN_SEAT}' dans {seats}",
            ) from error
        try:
            spec = SessionSpec(
                seed=int(payload.get("seed", 42)),
                seats=[str(name) for name in seats],
                human_seat=human_seat,
            )
            session = sessions.add(Session.create(spec, cfg, catalog, backends))
        except (KeyError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        session.advance()
        return session.snapshot()

    @app.get("/api/session/{session_id}")
    def read_session(session_id: str) -> dict:
        return _session(session_id).snapshot()

    @app.post("/api/session/{session_id}/act")
    def act(session_id: str, payload: dict = Body(...)) -> dict:
        session = _session(session_id)
        try:
            session.submit(int(payload["index"]), str(payload.get("argument", "")))
        except KeyError as error:
            raise HTTPException(status_code=400, detail="champ 'index' absent") from error
        except (RuntimeError, ValueError, TypeError) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        return session.snapshot()

    @app.post("/api/session/{session_id}/save")
    def save(session_id: str, payload: dict = Body(default={})) -> dict:
        """Écrit la partie au format de replay, dans la racine servie."""
        from controla.arena.storage import write_game

        session = _session(session_id)
        name = str(payload.get("name") or f"human-{session.id}.jsonl")
        if not name.endswith(".jsonl"):
            name += ".jsonl"
        try:
            target = library.resolve_within(root, f"human/{Path(name).name}")
        except library.OutsideRoot as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        write_game(target, session.result())
        return {"path": str(target.relative_to(Path(root).resolve())).replace("\\", "/")}

    @app.delete("/api/session/{session_id}")
    def drop(session_id: str) -> dict:
        sessions.drop(session_id)
        return {"dropped": session_id}

    def _session(session_id: str) -> Session:
        session = sessions.get(session_id)
        if session is None:
            raise HTTPException(
                status_code=404,
                detail="session inconnue ou expirée — le serveur ne les conserve "
                       "qu'en mémoire",
            )
        return session

    # ── pages ────────────────────────────────────────────────────────────────

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    return app
