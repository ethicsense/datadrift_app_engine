from typing import Any, Optional

from pydantic import BaseModel, ConfigDict

class DatasetSchema(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,
        json_encoders={
            float: lambda x: None
            if (x != x or x in [float("inf"), float("-inf")])
            else x
        },
    )

    id: str
    name: str
    type: str
    rows: int
    cols: int
    missing_rate: Optional[Any]
    preview: Optional[Any]
    dvc_path: str
    version: str