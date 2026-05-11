from __future__ import annotations

from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _BaseMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(1, description="메타 스키마 버전(정수)")
    # 공통(선택): 표시용 이름
    name: Optional[str] = None


class VisionImageData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data_dir: str = Field(".", description="데이터 루트(압축 해제 루트 기준 상대경로)")


class VisionImageMeta(_BaseMeta):
    modality: Literal["vision_image"] = "vision_image"
    data: VisionImageData


class VisionVideoData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data_dir: str = Field(".", description="비디오 파일 루트(압축 해제 루트 기준 상대경로)")


class VisionVideoMeta(_BaseMeta):
    modality: Literal["vision_video"] = "vision_video"
    data: VisionVideoData


class TextData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    csv: str = Field(..., description="CSV 파일 경로(압축 해제 루트 기준 상대경로)")
    columns: list[str] = Field(..., description="분석 대상 컬럼(텍스트 컬럼) 목록")
    # 선택(하위호환/다국어 처리용)
    id_column: Optional[str] = Field(None, description="ID 컬럼명(선택)")
    language: Optional[str] = Field(None, description="언어 힌트(예: english, korean) - 선택")

    @field_validator("csv")
    @classmethod
    def _csv_must_be_csv(cls, v: str) -> str:
        if not v.lower().endswith(".csv"):
            raise ValueError("text modality는 .csv 포맷만 지원합니다 (data.csv)")
        return v

    @field_validator("columns")
    @classmethod
    def _columns_nonempty_unique(cls, v: list[str]) -> list[str]:
        cols = [c.strip() for c in v if c and c.strip()]
        if not cols:
            raise ValueError("text modality는 columns가 1개 이상 필요합니다")
        if len(set(cols)) != len(cols):
            raise ValueError("text modality columns에 중복이 있습니다")
        return cols


class TextMeta(_BaseMeta):
    modality: Literal["text"] = "text"
    data: TextData


class TimeSeriesData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    csv: str = Field(..., description="CSV 파일 경로(압축 해제 루트 기준 상대경로)")
    timestamp_column: str = Field(..., description="timestamp 컬럼명")
    numeric_columns: list[str] = Field(default_factory=list, description="숫자형 컬럼 목록")
    categorical_columns: list[str] = Field(default_factory=list, description="범주형 컬럼 목록")

    @field_validator("csv")
    @classmethod
    def _csv_must_be_csv(cls, v: str) -> str:
        if not v.lower().endswith(".csv"):
            raise ValueError("timeseries modality는 .csv 포맷만 지원합니다 (data.csv)")
        return v

    @field_validator("timestamp_column")
    @classmethod
    def _ts_nonempty(cls, v: str) -> str:
        v2 = v.strip()
        if not v2:
            raise ValueError("timeseries timestamp_column은 비어있을 수 없습니다")
        return v2

    @field_validator("numeric_columns", "categorical_columns")
    @classmethod
    def _list_strip_unique(cls, v: list[str]) -> list[str]:
        cols = [c.strip() for c in v if c and c.strip()]
        if len(set(cols)) != len(cols):
            raise ValueError("timeseries columns 목록에 중복이 있습니다")
        return cols

    @model_validator(mode="after")
    def _validate_groups(self):
        if not self.numeric_columns and not self.categorical_columns:
            raise ValueError("timeseries는 numeric_columns 또는 categorical_columns 중 최소 1개는 지정해야 합니다")
        if self.timestamp_column in set(self.numeric_columns) or self.timestamp_column in set(
            self.categorical_columns
        ):
            raise ValueError("timestamp_column은 numeric/categorical 컬럼 목록에 포함되면 안 됩니다")
        if set(self.numeric_columns) & set(self.categorical_columns):
            raise ValueError("numeric_columns와 categorical_columns는 서로 겹치면 안 됩니다")
        return self


class TimeSeriesMeta(_BaseMeta):
    modality: Literal["timeseries"] = "timeseries"
    data: TimeSeriesData


class AudioWaveData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data_dir: str = Field(".", description="파형 오디오 파일 루트(압축 해제 루트 기준 상대경로)")


class AudioWaveMeta(_BaseMeta):
    modality: Literal["audio_wave"] = "audio_wave"
    data: AudioWaveData


class AudioMidiData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data_dir: str = Field(".", description="MIDI(.mid/.midi) 파일 루트(압축 해제 루트 기준 상대경로)")


class AudioMidiMeta(_BaseMeta):
    modality: Literal["audio_midi"] = "audio_midi"
    data: AudioMidiData


class MlflowLogData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tracking_dir: str = Field(
        ..., description="MLflow tracking 디렉토리(압축 해제 루트 기준 상대경로, auto 허용)"
    )
    default_experiment_id: Optional[str] = Field(None, description="기본 experiment id(선택)")
    default_run_id: Optional[str] = Field(None, description="기본 run id(선택)")

    @field_validator("tracking_dir")
    @classmethod
    def _tracking_dir_nonempty(cls, v: str) -> str:
        v2 = v.strip()
        if not v2:
            raise ValueError("mlflow_log tracking_dir는 비어있을 수 없습니다")
        return v2


class MlflowLogMeta(_BaseMeta):
    modality: Literal["mlflow_log"] = "mlflow_log"
    data: MlflowLogData


DatasetMeta = Annotated[
    Union[
        VisionImageMeta,
        VisionVideoMeta,
        TextMeta,
        TimeSeriesMeta,
        AudioWaveMeta,
        AudioMidiMeta,
        MlflowLogMeta,
    ],
    Field(discriminator="modality"),
]


