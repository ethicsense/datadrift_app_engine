from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .database import Base, engine
from .routers import datasets, eda, drift, embeddings, files, report, runs, status

# 테이블 생성
Base.metadata.create_all(bind=engine)

app = FastAPI(title="drift_studio API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(datasets.router)
app.include_router(eda.router)
app.include_router(drift.router)
app.include_router(status.router)
app.include_router(embeddings.router)
app.include_router(files.router)
app.include_router(report.router)
app.include_router(runs.router)

@app.get("/")
def root():
    return {"status": "ok"}
