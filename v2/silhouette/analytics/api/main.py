from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from analytics.api.routers.animation import router as animation_router
from analytics.api.routers.files import router as files_router
from analytics.api.routers.brand import router as brand_router
from analytics.api.routers.category import router as category_router
from analytics.api.routers.meta import router as meta_router
from analytics.api.routers.momentum import router as momentum_router
from analytics.api.routers.overview import router as overview_router
from analytics.api.routers.performance import router as performance_router
from analytics.api.routers.price import router as price_router
from analytics.api.routers.semantic import router as semantic_router
from analytics.api.routers.text import router as text_router
from analytics.api.routers.thumbnails import router as thumbnails_router
from analytics.api.routers.trends import router as trends_router
from analytics.api.settings import get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_title, version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.allow_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(meta_router)
    app.include_router(overview_router)
    app.include_router(price_router)
    app.include_router(momentum_router)
    app.include_router(semantic_router)
    app.include_router(text_router)
    app.include_router(performance_router)
    app.include_router(trends_router)
    app.include_router(brand_router)
    app.include_router(category_router)
    app.include_router(thumbnails_router)
    app.include_router(animation_router)
    app.include_router(files_router)
    return app


app = create_app()
