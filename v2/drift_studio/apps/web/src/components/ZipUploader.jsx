import React, { useState, useRef, useCallback } from "react";
import JSZip from "jszip";

/**
 * ddoc.yaml 기반 멀티모달 ZIP 업로더
 *
 * - ZIP 내부에서 ddoc.yaml을 먼저 확인해 modality를 결정하고
 * - modality 규약에 맞는 최소 구조 검증을 수행합니다.
 */
export default function ZipUploader({ backend, onUploadComplete }) {
  const [isDragging, setIsDragging] = useState(false);
  const [file, setFile] = useState(null);
  const [validationResult, setValidationResult] = useState(null);
  const [isValidating, setIsValidating] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  // 드래그 이벤트 핸들러
  const handleDragEnter = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  }, []);

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const droppedFiles = e.dataTransfer.files;
    if (droppedFiles.length > 0) {
      handleFileSelect(droppedFiles[0]);
    }
  }, []);

  // 파일 선택 핸들러
  const handleFileSelect = async (selectedFile) => {
    setError(null);
    setValidationResult(null);

    // ZIP 파일 확인
    if (!selectedFile.name.toLowerCase().endsWith(".zip")) {
      setError("ZIP 파일만 업로드할 수 있습니다.");
      return;
    }

    setFile(selectedFile);
    setIsValidating(true);

    try {
      const result = await validateZipByDdoc(selectedFile);
      setValidationResult(result);
      
      if (!result.isValid) {
        setError(result.error);
      }
    } catch (err) {
      setError(`파일 검증 중 오류 발생: ${err.message}`);
    } finally {
      setIsValidating(false);
    }
  };

  // 불필요한 파일/폴더 필터링 헬퍼
  const isJunkPath = (path) => {
    const lowerPath = path.toLowerCase();
    // __MACOSX 폴더
    if (lowerPath.includes("__macosx")) return true;
    // .DS_Store 파일
    if (lowerPath.includes(".ds_store")) return true;
    // ._ 로 시작하는 macOS 리소스 포크 파일
    const fileName = path.split("/").pop();
    if (fileName.startsWith("._")) return true;
    // Thumbs.db (Windows)
    if (lowerPath.includes("thumbs.db")) return true;
    return false;
  };

  const normalizeYamlValue = (raw) => {
    if (!raw) return "";
    const s = String(raw).trim();
    return s.replace(/^['"]|['"]$/g, "").trim();
  };

  // 의존성 추가 없이 ddoc.yaml에서 핵심 키만 읽는 초경량 파서
  const parseSimpleYaml = (yamlText) => {
    const root = {};
    const data = {};

    const lines = String(yamlText || "")
      .split(/\r?\n/)
      .map((l) => l.replace(/\t/g, "  "));

    let inData = false;
    for (const line of lines) {
      const noComment = line.replace(/\s+#.*$/, "");
      if (!noComment.trim()) continue;

      const indent = noComment.match(/^\s*/)?.[0]?.length ?? 0;
      const m = noComment.match(/^\s*([A-Za-z0-9_]+)\s*:\s*(.*)\s*$/);
      if (!m) continue;

      const key = m[1];
      const val = normalizeYamlValue(m[2]);

      if (indent === 0 && key === "data") {
        inData = true;
        continue;
      }
      if (indent === 0) {
        inData = false;
        root[key] = val;
        continue;
      }
      if (inData && indent >= 2) {
        data[key] = val;
      }
    }

    return { root, data };
  };

  const findDdocPath = (nonDirFiles, zipStem) => {
    const lower = nonDirFiles.map((p) => p.toLowerCase());
    const idxRoot = lower.indexOf("ddoc.yaml");
    if (idxRoot >= 0) return nonDirFiles[idxRoot];
    if (zipStem) {
      const stemPath = `${zipStem.toLowerCase()}/ddoc.yaml`;
      const idxStem = lower.indexOf(stemPath);
      if (idxStem >= 0) return nonDirFiles[idxStem];
    }
    return null;
  };

  // ddoc.yaml 기반 최소 검증
  const validateZipByDdoc = async (zipFile) => {
    const zip = new JSZip();
    const contents = await zip.loadAsync(zipFile);
    
    // 불필요한 파일 제외
    const files = Object.keys(contents.files).filter(path => !isJunkPath(path));
    
    // ddoc.yaml 존재 여부 확인(업로드 전 빠른 피드백)
    // - v2 규약: 압축 해제 루트에 ddoc.yaml 필수
    // - 실무적으로 zip 내부에 "단일 루트 폴더(zip 이름)"가 있는 경우도 많아서 그 케이스도 허용
    const zipStem = (zipFile?.name || "").replace(/\.zip$/i, "");
    const nonDirFiles = files.filter((p) => !p.endsWith("/"));
    const topFolders = new Set(
      nonDirFiles
        .map((p) => p.split("/")[0])
        .filter((x) => x && x !== "." && !isJunkPath(x))
    );

    const ddocPath = findDdocPath(nonDirFiles, zipStem);
    if (!ddocPath) {
      return {
        isValid: false,
        error:
          "ddoc.yaml 파일이 없습니다. ZIP 압축 해제 루트(또는 zip 이름과 동일한 단일 루트 폴더) 아래에 ddoc.yaml이 반드시 필요합니다.",
        stats: {
          modality: null,
          ddocPath: null,
          imageCount: 0,
          labelCount: 0,
          hasDataYaml: false,
          csvCount: 0,
          folders: Array.from(topFolders).slice(0, 10),
        },
      };
    }

    // read modality from ddoc.yaml
    let ddocText = "";
    try {
      ddocText = await contents.file(ddocPath).async("string");
    } catch (e) {
      return {
        isValid: false,
        error: `ddoc.yaml을 읽을 수 없습니다: ${e?.message || String(e)}`,
        stats: {
          modality: null,
          ddocPath,
          imageCount: 0,
          labelCount: 0,
          hasDataYaml: false,
          csvCount: 0,
          folders: Array.from(topFolders).slice(0, 10),
        },
      };
    }

    const parsed = parseSimpleYaml(ddocText);
    const modality = String(parsed.root.modality || "").trim().toLowerCase();
    if (!modality) {
      return {
        isValid: false,
        error: "ddoc.yaml에 modality가 없습니다. 예) modality: vision",
        stats: {
          modality: null,
          ddocPath,
          imageCount: 0,
          labelCount: 0,
          hasDataYaml: false,
          csvCount: 0,
          folders: Array.from(topFolders).slice(0, 10),
        },
      };
    }

    // 폴더 구조 분석(표시용)
    const folders = new Set();
    files.forEach((path) => {
      const parts = path.split("/");
      if (parts.length > 1) {
        const subFolder = parts.length > 2 ? parts[1] : parts[0];
        folders.add(subFolder.toLowerCase());
        if (parts.length > 2) {
          folders.add(`${parts[1].toLowerCase()}/${parts[2].toLowerCase()}`);
        }
      }
    });

    // stats
    const imageExtensions = [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"];
    const audioMidiExtensions = [".mid", ".midi"];
    const audioWaveExtensions = [".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"];
    const videoExtensions = [".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"];

    const countByExt = (exts) =>
      nonDirFiles.reduce((acc, p) => {
        const lower = p.toLowerCase();
        return acc + (exts.some((e) => lower.endsWith(e)) ? 1 : 0);
      }, 0);

    const csvCount = nonDirFiles.filter((p) => p.toLowerCase().endsWith(".csv")).length;
    const imageCount = countByExt(imageExtensions);
    const audioMidiCount = countByExt(audioMidiExtensions);
    const audioWaveCount = countByExt(audioWaveExtensions);
    const videoCount = countByExt(videoExtensions);
    const labelCount = nonDirFiles.filter((p) => p.toLowerCase().endsWith(".txt")).length;
    const jsonCount = nonDirFiles.filter((p) => p.toLowerCase().endsWith(".json")).length;
    const hasDataYaml = nonDirFiles.some((p) => p.toLowerCase().endsWith("data.yaml"));

    const dataDir = parsed.data.data_dir || ".";
    // driftstudio_spec 스키마는 timeseries/text에서 data.csv를 사용합니다.
    // 일부 레거시/플러그인 케이스를 위해 data.path도 허용합니다.
    const dataCsv = parsed.data.csv || "";
    const dataPath = parsed.data.path || "";

    const resolveCandidatePaths = (p) => {
      const candidates = [];
      const clean = String(p || "").replace(/^\.?\//, "");
      if (!clean) return candidates;
      candidates.push(clean);
      if (dataDir && dataDir !== "." && dataDir !== "./") {
        const dd = String(dataDir).replace(/\/+$/, "").replace(/^\.?\//, "");
        candidates.push(`${dd}/${clean}`);
      }
      return candidates;
    };

    const existsInZip = (candidates) => {
      const set = new Set(nonDirFiles.map((p) => p.replace(/^\.?\//, "")));
      return candidates.some((c) => set.has(String(c).replace(/^\.?\//, "")));
    };

    // modality-specific validation
    const issues = [];
    if (modality === "vision_image") {
      if (imageCount === 0) issues.push("이미지 파일이 없습니다.");
    } else if (modality === "timeseries") {
      const declared = dataCsv || dataPath;
      if (declared) {
        if (!existsInZip(resolveCandidatePaths(declared))) {
          issues.push(`ddoc.yaml의 data.csv/path='${declared}' 파일이 ZIP에 없습니다.`);
        }
      } else if (csvCount === 0) {
        issues.push("시계열(timeseries) 데이터는 최소 1개 이상의 .csv 파일이 필요합니다.");
      }
    } else if (modality === "text") {
      const declared = dataCsv || dataPath;
      if (declared) {
        if (!existsInZip(resolveCandidatePaths(declared))) {
          issues.push(`ddoc.yaml의 data.csv/path='${declared}' 파일이 ZIP에 없습니다.`);
        }
      } else if (csvCount === 0) {
        issues.push("텍스트(text) 데이터는 최소 1개 이상의 .csv 파일이 필요합니다.");
      }
    } else if (modality === "audio_midi") {
      if (dataPath) {
        if (!existsInZip(resolveCandidatePaths(dataPath))) {
          issues.push(`ddoc.yaml의 data.path='${dataPath}' 파일이 ZIP에 없습니다.`);
        }
      } else if (audioMidiCount === 0) {
        issues.push("오디오(audio_midi) MIDI 파일이 없습니다. (.mid/.midi)");
      }
      if (jsonCount === 0) {
        issues.push("오디오(audio_midi) 라벨(.json) 파일이 없습니다.");
      }

      // MIDI ↔ JSON 페어링(동일 경로+파일명 stem 기준) 최소 검증
      if (audioMidiCount > 0 && jsonCount > 0) {
        const toStem = (p) => {
          const lower = String(p || "").toLowerCase();
          if (lower.endsWith(".mid")) return lower.slice(0, -4);
          if (lower.endsWith(".midi")) return lower.slice(0, -5);
          if (lower.endsWith(".json")) return lower.slice(0, -5);
          return lower;
        };

        const midiStems = new Set(
          nonDirFiles
            .filter((p) => {
              const lower = p.toLowerCase();
              return lower.endsWith(".mid") || lower.endsWith(".midi");
            })
            .map(toStem)
        );
        const jsonStems = new Set(nonDirFiles.filter((p) => p.toLowerCase().endsWith(".json")).map(toStem));

        const missingJson = [];
        midiStems.forEach((s) => {
          if (!jsonStems.has(s)) missingJson.push(s);
        });
        if (missingJson.length > 0) {
          const sample = missingJson.slice(0, 3).join(", ");
          issues.push(
            `MIDI 파일과 매칭되는 라벨(.json)이 없습니다: ${sample}${missingJson.length > 3 ? " ..." : ""}`
          );
        }
      }
    } else if (modality === "audio_wave") {
      if (dataPath) {
        if (!existsInZip(resolveCandidatePaths(dataPath))) {
          issues.push(`ddoc.yaml의 data.path='${dataPath}' 파일이 ZIP에 없습니다.`);
        }
      } else if (audioWaveCount === 0) {
        issues.push("오디오(audio_wave) 파형 파일이 없습니다. (.wav/.mp3 등)");
      }
    } else if (modality === "vision_video") {
      if (dataPath) {
        if (!existsInZip(resolveCandidatePaths(dataPath))) {
          issues.push(`ddoc.yaml의 data.path='${dataPath}' 파일이 ZIP에 없습니다.`);
        }
      } else if (videoCount === 0) {
        issues.push("비디오(vision_video) 파일이 없습니다. (.mp4/.mov 등)");
      }
    } else if (modality === "mlflow_log") {
      const trackingDir = parsed.data.tracking_dir || "auto";
      const cleanTracking = String(trackingDir).replace(/^\.?\//, "").replace(/\/+$/, "");
      const hasExplicitTracking = cleanTracking && cleanTracking.toLowerCase() !== "auto";

      const metaCandidates = nonDirFiles.filter((p) => p.toLowerCase().endsWith("/meta.yaml"));
      const experimentMeta = metaCandidates.filter((p) => p.split("/").length >= 2);
      const runMetaCandidates = metaCandidates.filter((p) => p.split("/").length >= 3);

      const expMeta = hasExplicitTracking
        ? experimentMeta.some((p) => p.toLowerCase().startsWith(`${cleanTracking.toLowerCase()}/`))
        : experimentMeta.length > 0;

      const runMeta = hasExplicitTracking
        ? runMetaCandidates.some((p) => p.toLowerCase().startsWith(`${cleanTracking.toLowerCase()}/`))
        : runMetaCandidates.length > 0;

      if (!expMeta) {
        issues.push("mlflow_log: <tracking_dir>/<experiment>/meta.yaml 파일이 필요합니다.");
      }
      if (!runMeta) {
        issues.push("mlflow_log: <tracking_dir>/<experiment>/<run>/meta.yaml 파일이 필요합니다.");
      }
    } else {
      issues.push(`지원하지 않는 modality 입니다: ${modality}`);
    }

    const isValid = issues.length === 0;

    return {
      isValid,
      error: issues.length > 0 ? issues.join(" ") : null,
      stats: {
        modality,
        ddocPath,
        imageCount,
        labelCount,
        hasDataYaml,
        csvCount,
        audioMidiCount,
        audioWaveCount,
        jsonCount,
        videoCount,
        folders: Array.from(folders).slice(0, 10),
      },
    };
  };

  // 업로드 실행
  const handleUpload = async () => {
    if (!file || !validationResult?.isValid) return;

    setIsUploading(true);
    setError(null);

    try {
      const form = new FormData();
      form.append("file", file);

      const response = await fetch(`${backend}/datasets/upload`, {
        method: "POST",
        body: form,
      });

      if (!response.ok) {
        // FastAPI는 보통 {detail: "..."} 형태로 실패 사유를 내려줍니다.
        let detail = null;
        try {
          const data = await response.json();
          detail = data?.detail;
        } catch (_) {
          // ignore
        }
        if (!detail) {
          try {
            const text = await response.text();
            detail = text?.slice(0, 400);
          } catch (_) {
            // ignore
          }
        }
        throw new Error(detail || `업로드 실패 (HTTP ${response.status})`);
      }

      // 성공 시 초기화
      setFile(null);
      setValidationResult(null);
      onUploadComplete?.();
    } catch (err) {
      setError(`업로드 불가: ${err.message}`);
    } finally {
      setIsUploading(false);
    }
  };

  // 취소
  const handleCancel = () => {
    setFile(null);
    setValidationResult(null);
    setError(null);
  };

  return (
    <div className="w-full">
      {/* 드래그 앤 드롭 영역 */}
      {!file && (
        <div
          className={`
            relative border-2 border-dashed rounded-lg p-8 text-center cursor-pointer
            transition-all duration-200 ease-in-out
            ${isDragging
              ? "border-blue-500 bg-blue-50"
              : "border-gray-300 hover:border-blue-400 hover:bg-gray-50"
            }
          `}
          onDragEnter={handleDragEnter}
          onDragLeave={handleDragLeave}
          onDragOver={handleDragOver}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".zip"
            className="hidden"
            onChange={(e) => e.target.files[0] && handleFileSelect(e.target.files[0])}
          />
          
          <div className="flex flex-col items-center gap-3">
            <div className="text-4xl">📦</div>
            <div className="text-sm font-medium text-gray-700">
              {isDragging ? "파일을 놓으세요" : "ZIP 파일을 드래그하거나 클릭하여 선택"}
            </div>
            <div className="text-xs text-gray-500">
              ddoc.yaml 기반 멀티모달 ZIP 업로드를 지원합니다
            </div>
            <div className="text-xs text-gray-400 mt-2">
              필수: ddoc.yaml (modality 규약에 따라 최소 구조 검증)
            </div>
          </div>
        </div>
      )}

      {/* 파일 선택됨 - 검증 중 */}
      {file && isValidating && (
        <div className="border rounded-lg p-6 bg-gray-50">
          <div className="flex items-center gap-3">
            <div className="animate-spin h-5 w-5 border-2 border-blue-500 border-t-transparent rounded-full"></div>
            <div className="text-sm text-gray-600">파일 검증 중...</div>
          </div>
        </div>
      )}

      {/* 파일 선택됨 - 검증 완료 */}
      {file && !isValidating && validationResult && (
        <div className={`border rounded-lg p-4 ${validationResult.isValid ? "bg-green-50 border-green-200" : "bg-red-50 border-red-200"}`}>
          <div className="flex items-start justify-between mb-3">
            <div className="flex items-center gap-2">
              <span className="text-2xl">{validationResult.isValid ? "✅" : "❌"}</span>
              <div>
                <div className="font-medium text-sm">{file.name}</div>
                <div className="text-xs text-gray-500">
                  {(file.size / 1024 / 1024).toFixed(2)} MB
                </div>
              </div>
            </div>
            <button
              onClick={handleCancel}
              className="text-gray-400 hover:text-gray-600 text-lg"
            >
              ✕
            </button>
          </div>

          {/* 검증 결과 상세 */}
          {validationResult.isValid ? (
            <div className="space-y-2">
              <div className="text-xs text-green-700 font-medium">
                ✓ ddoc.yaml 포맷 확인됨 (modality: {validationResult?.stats?.modality || "unknown"})
              </div>
              <div className="grid grid-cols-3 gap-2 text-xs">
                <div className="bg-white p-2 rounded">
                  <div className="text-gray-500">이미지</div>
                  <div className="font-semibold">{validationResult.stats.imageCount}개</div>
                </div>
                <div className="bg-white p-2 rounded">
                  <div className="text-gray-500">CSV</div>
                  <div className="font-semibold">{validationResult.stats.csvCount}개</div>
                </div>
                <div className="bg-white p-2 rounded">
                  <div className="text-gray-500">ddoc.yaml</div>
                  <div className="font-semibold">{validationResult.stats.ddocPath ? "있음" : "없음"}</div>
                </div>
              </div>
              
              {/* 업로드 버튼 */}
              <button
                onClick={handleUpload}
                disabled={isUploading}
                className="w-full mt-3 px-4 py-2 bg-blue-600 text-white rounded-md text-sm font-medium hover:bg-blue-700 disabled:bg-gray-400 transition"
              >
                {isUploading ? (
                  <span className="flex items-center justify-center gap-2">
                    <span className="animate-spin h-4 w-4 border-2 border-white border-t-transparent rounded-full"></span>
                    업로드 중...
                  </span>
                ) : (
                  "업로드"
                )}
              </button>
            </div>
          ) : (
            <div className="text-xs text-red-700">
              <div className="font-medium mb-1">포맷 검증 실패:</div>
              <div>{validationResult.error}</div>
              <div className="mt-2 text-gray-500">
                modality: {validationResult?.stats?.modality || "unknown"} · 감지된 폴더:{" "}
                {validationResult.stats.folders.join(", ") || "없음"}
              </div>
            </div>
          )}
        </div>
      )}

      {/* 에러 메시지 (검증 단계/업로드 단계 모두 표시) */}
      {error && (
        <div className="mt-2 p-3 bg-red-50 border border-red-200 rounded-lg">
          <div className="text-sm text-red-700">{error}</div>
        </div>
      )}
    </div>
  );
}


