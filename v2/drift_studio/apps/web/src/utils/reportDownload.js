function parseFilenameFromContentDisposition(contentDisposition) {
  if (!contentDisposition) return null;
  const utf8Match = contentDisposition.match(/filename\*=UTF-8''([^;]+)/i);
  if (utf8Match?.[1]) return decodeURIComponent(utf8Match[1]);
  const quotedMatch = contentDisposition.match(/filename="([^"]+)"/i);
  if (quotedMatch?.[1]) return quotedMatch[1];
  const simpleMatch = contentDisposition.match(/filename=([^;]+)/i);
  return simpleMatch?.[1]?.trim() || null;
}

export async function downloadReportFile({ backend, path, filename }) {
  if (!path) throw new Error("다운로드할 리포트 경로가 없습니다.");

  const response = await fetch(
    `${backend}/report/download?path=${encodeURIComponent(path)}`,
    { credentials: "same-origin" }
  );
  const contentType = response.headers.get("content-type") || "";

  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new Error(
      data?.detail || `리포트 다운로드 실패 (HTTP ${response.status})`
    );
  }

  if (contentType.includes("application/json")) {
    const data = await response.json().catch(() => ({}));
    throw new Error(data?.detail || "다운로드 응답이 JSON입니다. 프록시/경로를 확인하세요.");
  }

  const blob = await response.blob();
  if (!blob || blob.size === 0) {
    throw new Error("다운로드한 파일이 비어 있습니다.");
  }

  const contentDisposition = response.headers.get("content-disposition");
  const serverFilename = parseFilenameFromContentDisposition(contentDisposition);
  const fallbackFilename = path.split("/").pop() || "report.bin";
  const resolvedFilename = filename || serverFilename || fallbackFilename;

  const blobUrl = window.URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = blobUrl;
  anchor.download = resolvedFilename;
  anchor.rel = "noopener";
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  window.setTimeout(() => window.URL.revokeObjectURL(blobUrl), 30_000);
}
