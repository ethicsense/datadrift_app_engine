import { lazy, Suspense } from "react";
import { Navigate, Route, Routes } from "react-router-dom";

import { AppShell } from "./components/AppShell";
import { OverviewPage } from "./pages/OverviewPage";

const CategoryPage = lazy(() => import("./pages/CategoryPage").then((m) => ({ default: m.CategoryPage })));
const EmbeddingPage = lazy(() => import("./pages/EmbeddingPage").then((m) => ({ default: m.EmbeddingPage })));
const MomentumPage = lazy(() => import("./pages/MomentumPage").then((m) => ({ default: m.MomentumPage })));
const PricePage = lazy(() => import("./pages/PricePage").then((m) => ({ default: m.PricePage })));
const SchemaExplorerPage = lazy(() =>
  import("./pages/SchemaExplorerPage").then((m) => ({ default: m.SchemaExplorerPage })),
);
const SemanticPage = lazy(() => import("./pages/SemanticPage").then((m) => ({ default: m.SemanticPage })));
const TextPage = lazy(() => import("./pages/TextPage").then((m) => ({ default: m.TextPage })));
const ThumbnailPage = lazy(() => import("./pages/ThumbnailPage").then((m) => ({ default: m.ThumbnailPage })));

function PageFallback() {
  return <div className="loading-state">페이지를 불러오는 중입니다.</div>;
}

export default function App() {
  return (
    <Suspense fallback={<PageFallback />}>
      <Routes>
        <Route path="/" element={<AppShell />}>
          <Route index element={<Navigate to="/overview" replace />} />
          <Route path="/overview" element={<OverviewPage />} />
          <Route path="/category" element={<CategoryPage />} />
          <Route path="/embedding" element={<EmbeddingPage />} />
          <Route path="/price" element={<PricePage />} />
          <Route path="/momentum" element={<MomentumPage />} />
          <Route path="/product-info" element={<SemanticPage />} />
          <Route path="/schema-explorer" element={<SchemaExplorerPage />} />
          <Route path="/text" element={<TextPage />} />
          <Route path="/thumbnails" element={<ThumbnailPage />} />
        </Route>
      </Routes>
    </Suspense>
  );
}
