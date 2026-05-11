import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

/** Docker 웹(Nginx)과 같이 상대 `/api`를 쓰므로, 개발 시 Vite가 백엔드로 넘김. */
const devApiProxyTarget = process.env.SILHOUETTE_VITE_PROXY_TARGET ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    host: "0.0.0.0",
    port: 4173,
    proxy: {
      "/api": {
        target: devApiProxyTarget,
        changeOrigin: true,
      },
      "/health": {
        target: devApiProxyTarget,
        changeOrigin: true,
      },
    },
  },
});
