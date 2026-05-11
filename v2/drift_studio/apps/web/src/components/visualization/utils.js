export const MODALITY_ALIASES = {
  image: "vision_image",
  video: "vision_video",
  audio: "audio_wave",
};

export function normalizeModality(raw) {
  if (!raw) return null;
  const key = String(raw).trim().toLowerCase();
  return MODALITY_ALIASES[key] || key;
}

export function formatNumber(value, digits = 4) {
  if (value === null || value === undefined) return "-";
  if (typeof value !== "number" || Number.isNaN(value)) return String(value);
  const abs = Math.abs(value);
  if (abs === 0) return "0";
  if (abs < 0.001) return value.toExponential(2);
  return value.toFixed(digits);
}

export function formatMetricValue(key, value) {
  if (value === null || value === undefined) return "-";
  if (typeof value !== "number" || Number.isNaN(value)) return String(value);
  const k = String(key || "").toLowerCase();
  if (k.includes("sharpness")) {
    if (value > 0) {
      return `${formatNumber(Math.log10(value), 4)} (log10)`;
    }
    return formatNumber(value, 4);
  }
  if (k.includes("size_mb") || k.endsWith("_mb")) {
    if (Math.abs(value) < 1) {
      return `${formatNumber(value * 1024, 3)} KB`;
    }
    return `${formatNumber(value, 3)} MB`;
  }
  if (k.includes("size")) {
    return formatNumber(value, 4);
  }
  if (Math.abs(value) < 1e-6) {
    return value.toExponential(2);
  }
  return formatNumber(value, 4);
}

export function formatMetricLabel(key) {
  if (!key) return "";
  let label = String(key);
  if (label.endsWith("_mb")) {
    label = label.slice(0, -3);
  }
  if (label.endsWith("_kb")) {
    label = label.slice(0, -3);
  }
  return label;
}

export function toEntries(obj) {
  if (!obj || typeof obj !== "object") return [];
  return Object.entries(obj);
}

export function toChartData(obj, valueKey = "value") {
  return toEntries(obj).map(([name, value]) => ({
    name,
    [valueKey]: value,
  }));
}

export function pickNumeric(obj) {
  if (!obj || typeof obj !== "object") return {};
  return Object.fromEntries(
    Object.entries(obj).filter(
      ([, v]) => typeof v === "number" && Number.isFinite(v)
    )
  );
}

export function omitKeys(obj, keys) {
  if (!obj || typeof obj !== "object") return {};
  const set = new Set(keys);
  return Object.fromEntries(Object.entries(obj).filter(([k]) => !set.has(k)));
}
