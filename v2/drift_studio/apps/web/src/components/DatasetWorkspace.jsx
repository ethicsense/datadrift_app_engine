import React, { useState } from "react";

export default function DatasetWorkspace({ datasets, backend = "/api", onUploaded, onSelect, onCompare }) {
  const [file, setFile] = useState(null);
  const [error, setError] = useState(null);

  const upload = () => {
    const form = new FormData();
    form.append("file", file);

    fetch(`${backend}/datasets/upload`, {
      method: "POST",
      body: form,
    })
      .then(async (r) => {
        const data = await r.json().catch(() => ({}));
        if (!r.ok) throw new Error(data?.detail || `업로드 실패 (HTTP ${r.status})`);
        return data;
      })
      .then(() => {
        setError(null);
      onUploaded();
      setFile(null);
      })
      .catch((e) => setError(e?.message || String(e)));
  };

  return (
    <div className="max-w-4xl mx-auto">
      <h2 className="text-2xl font-bold mb-4">📁 Dataset Workspace</h2>

      {/* Upload */}
      <div className="p-4 bg-white shadow rounded-xl mb-6">
        <div className="flex gap-4">
          <input
            type="file"
            onChange={(e) => setFile(e.target.files[0])}
            className="text-sm"
          />
          <button
            onClick={upload}
            disabled={!file}
            className="bg-blue-600 text-white px-4 py-2 rounded-lg disabled:bg-gray-300"
          >
            Upload
          </button>
        </div>
        {error && (
          <div className="mt-3 p-3 bg-red-50 border border-red-200 rounded text-sm text-red-700">
            업로드 불가: {error}
          </div>
        )}
      </div>

      {/* List */}
      <h3 className="text-xl font-semibold mb-3">📄 Uploaded Datasets</h3>

      <div className="space-y-3">
        {datasets.map((ds) => (
          <div
            key={ds.id}
            className="p-4 bg-white shadow rounded-lg border hover:border-blue-400 transition"
          >
            <div className="flex justify-between items-center">
              <div>
                <div className="font-bold text-lg">{ds.name}</div>
                <div className="text-sm text-gray-600">
                  {ds.rows} rows · {ds.cols} columns
                </div>
              </div>

              <div className="flex gap-2">
                <button
                  onClick={() => onSelect(ds)}
                  className="px-3 py-2 bg-green-600 text-white rounded-md"
                >
                  View EDA
                </button>
                <button
                  onClick={() => onCompare(ds)}
                  className="px-3 py-2 bg-purple-600 text-white rounded-md"
                >
                  Compare Drift
                </button>
              </div>
            </div>
          </div>
        ))}

        {datasets.length === 0 && (
          <div className="text-gray-500">아직 데이터셋이 없습니다.</div>
        )}
      </div>
    </div>
  );
}