import CardSelector from "../CardSelector";
import DriftStatusCard from "./drift/DriftStatusCard";
import OverallScoreCard from "./drift/OverallScoreCard";
import AttributeDriftCard from "./drift/AttributeDriftCard";
import EmbeddingDriftCard from "./drift/EmbeddingDriftCard";
import FileChangeCard from "./drift/FileChangeCard";
import EmbeddingOverlayProjectionCard from "./drift/EmbeddingOverlayProjectionCard";
import AttributeDistributionCompareCard from "./drift/AttributeDistributionCompareCard";
import EDASummaryCard from "./eda/EDASummaryCard";
import DistributionCard from "./eda/DistributionCard";
import StatisticsCard from "./eda/StatisticsCard";
import EmbeddingProjectionCard from "./common/EmbeddingProjectionCard";
import TrainlogSummaryCard from "./trainlog/TrainlogSummaryCard";
import TrainlogRunListCard from "./trainlog/TrainlogRunListCard";
import TrainlogMetricsOverlayCard from "./trainlog/TrainlogMetricsOverlayCard";
import TrainlogMlflowGuideCard from "./trainlog/TrainlogMlflowGuideCard";
import TrainlogDriftAggregateCard from "./trainlog/TrainlogDriftAggregateCard";
import TrainlogDriftPairsCard from "./trainlog/TrainlogDriftPairsCard";
import TrainlogPreviewImageCard from "./trainlog/TrainlogPreviewImageCard";

const getPayload = (artifact, payloads) =>
  payloads?.[artifact.id] ??
  (artifact.payload?.mode === "inline" ? artifact.payload.data : null);

const findPayloadByType = (artifactIndex, payloads, type) => {
  const target = artifactIndex?.artifacts?.find((item) => item.type === type);
  if (!target) return null;
  return getPayload(target, payloads);
};

const registry = [
  {
    id: "drift-status",
    name: "Drift 상태",
    priority: 1,
    supportedArtifactTypes: ["drift.status.v1"],
    extractData: ({ payload }) => ({
      status: payload?.status,
      overallScore: payload?.overall_score,
      modality: payload?.modality,
    }),
    component: DriftStatusCard,
  },
  {
    id: "drift-overall-score",
    name: "Overall Score",
    priority: 2,
    supportedArtifactTypes: ["drift.overall_score.v1"],
    // drift.status.v1에 overall_score가 포함되므로 동시에 두 카드를 띄우지 않음
    match: ({ artifactIndex }) => {
      const hasStatus = artifactIndex?.artifacts?.some((a) => a.type === "drift.status.v1");
      return !hasStatus;
    },
    extractData: ({ payload }) => ({ overallScore: payload?.overall_score }),
    component: OverallScoreCard,
  },
  {
    id: "drift-attribute",
    name: "Attribute Drifts",
    priority: 3,
    supportedArtifactTypes: ["drift.attribute_drifts.v1"],
    extractData: ({ payload }) => ({
      attributeDrifts: payload?.attribute_drifts,
    }),
    component: AttributeDriftCard,
  },
  {
    id: "drift-attribute-distributions",
    name: "Attribute Distributions",
    priority: 3.5,
    supportedArtifactTypes: ["drift.attribute_distributions.v1"],
    extractData: ({ payload }) => payload,
    component: AttributeDistributionCompareCard,
  },
  {
    id: "drift-embedding",
    name: "Embedding Drift",
    priority: 4,
    supportedArtifactTypes: ["drift.embedding.summary.v1"],
    extractData: ({ payload }) => ({
      embeddingDrift: payload?.embedding_drift,
      embeddingDriftDetailed: payload?.embedding_drift_detailed,
    }),
    component: EmbeddingDriftCard,
  },
  {
    id: "drift-embedding-projection",
    name: "Embedding Projection (Drift)",
    priority: 4.5,
    supportedArtifactTypes: ["drift.embedding.projection.2d.v1"],
    extractData: ({ payload }) => payload,
    component: EmbeddingOverlayProjectionCard,
  },
  {
    id: "drift-file-change",
    name: "파일 변화",
    priority: 5,
    supportedArtifactTypes: ["drift.file_changes.v1"],
    extractData: ({ payload }) => ({
      filesAdded: payload?.files_added,
      filesRemoved: payload?.files_removed,
      filesCommon: payload?.files_common,
    }),
    component: FileChangeCard,
  },
  {
    id: "eda-summary",
    name: "EDA 요약",
    priority: 10,
    supportedArtifactTypes: ["eda.summary.v1"],
    extractData: ({ payload }) => ({ summary: payload }),
    component: EDASummaryCard,
  },
  {
    id: "eda-distributions",
    name: "Distributions",
    priority: 11,
    supportedArtifactTypes: ["eda.distributions.v1"],
    extractData: ({ payload }) => ({
      summary: { label_distributions: payload?.label_distributions },
      distributions: payload?.distributions,
    }),
    component: DistributionCard,
  },
  {
    id: "eda-distributions-basic",
    name: "기본 정보 분포",
    priority: 11.1,
    supportedArtifactTypes: ["eda.distributions.basic.v1"],
    extractData: ({ payload }) => ({
      distributions: payload?.distributions,
    }),
    component: DistributionCard,
  },
  {
    id: "eda-distributions-attributes",
    name: "컬럼 분포",
    priority: 11.2,
    supportedArtifactTypes: ["eda.distributions.attributes.v1"],
    extractData: ({ payload }) => ({
      distributions: payload?.distributions,
    }),
    component: DistributionCard,
  },
  {
    id: "eda-statistics",
    name: "통계 지표",
    priority: 12,
    supportedArtifactTypes: ["eda.metrics.numeric.v1"],
    extractData: ({ payload, artifactIndex, payloads }) => ({
      stats: payload || null,
      summary: findPayloadByType(artifactIndex, payloads, "eda.summary.v1") || null,
    }),
    component: StatisticsCard,
  },
  {
    id: "embedding-projection",
    name: "Embedding Projection",
    priority: 15,
    supportedArtifactTypes: ["embedding.projection.2d.v1"],
    extractData: ({ payload, artifactIndex, payloads }) => ({
      projection: payload,
      clustering: findPayloadByType(artifactIndex, payloads, "embedding.clustering.v1"),
    }),
    component: EmbeddingProjectionCard,
  },
  {
    id: "trainlog-summary",
    name: "MLflow Summary",
    priority: 20,
    supportedArtifactTypes: ["trainlog.mlflow.summary.v1"],
    extractData: ({ payload }) => ({ summary: payload }),
    component: TrainlogSummaryCard,
  },
  {
    id: "trainlog-runs",
    name: "MLflow Runs",
    priority: 21,
    supportedArtifactTypes: ["trainlog.mlflow.runs.index.v1"],
    extractData: ({ payload }) => ({ runs: payload }),
    component: TrainlogRunListCard,
  },
  {
    id: "trainlog-metrics-overlay",
    name: "MLflow Metrics Overlay",
    priority: 22,
    supportedArtifactTypes: ["trainlog.mlflow.metrics.index.v1"],
    extractData: ({ payload, artifactIndex, payloads }) => ({
      metrics: payload,
      runs: findPayloadByType(artifactIndex, payloads, "trainlog.mlflow.runs.index.v1") || [],
      summary: findPayloadByType(artifactIndex, payloads, "trainlog.mlflow.summary.v1") || {},
    }),
    component: TrainlogMetricsOverlayCard,
  },
  {
    id: "trainlog-mlflow-guide",
    name: "MLflow Guide",
    priority: 23,
    supportedArtifactTypes: ["trainlog.mlflow.mlflow_ui.guide.v1"],
    extractData: ({ payload }) => payload,
    component: TrainlogMlflowGuideCard,
  },
  {
    id: "trainlog-preview-image",
    name: "MLflow Preview",
    priority: 23.5,
    supportedArtifactTypes: ["trainlog.mlflow.preview.image.v1"],
    extractData: ({ payload }) => payload,
    component: TrainlogPreviewImageCard,
  },
  {
    id: "trainlog-drift-aggregate",
    name: "MLflow Drift Aggregate",
    priority: 24,
    supportedArtifactTypes: ["trainlog.mlflow.drift.aggregate.v1"],
    extractData: ({ payload }) => payload,
    component: TrainlogDriftAggregateCard,
  },
  {
    id: "trainlog-drift-matched-pairs",
    name: "MLflow Drift Matched Pairs",
    priority: 25,
    supportedArtifactTypes: ["trainlog.mlflow.drift.matched_pairs.v1"],
    extractData: ({ payload }) => ({ pairs: payload }),
    component: TrainlogDriftPairsCard,
  },
];

export function getMatchingCards(artifactIndex, payloads) {
  const selector = new CardSelector(registry);
  return selector.selectCards({ artifactIndex, payloads });
}
