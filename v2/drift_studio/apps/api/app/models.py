from sqlalchemy import Column, String, Integer, DateTime, JSON
from sqlalchemy.sql import func
from .database import Base

class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(String, primary_key=True, index=True)
    name = Column(String, index=True)
    type = Column(String)              # csv / text / image / video ...
    size = Column(Integer)             # 파일 크기(옵션)
    rows = Column(Integer)
    cols = Column(Integer)
    missing_rate = Column(JSON)        # {col: rate}
    preview = Column(JSON)             # 예: head(5)
    dvc_path = Column(String)          # data/.. 경로
    version = Column(String)           # DVC 버전 태그
    created_at = Column(DateTime, server_default=func.now())

