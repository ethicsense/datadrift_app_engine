import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const apiTarget = process.env.VITE_API_PROXY_TARGET || "http://localhost:18765";
const wsTarget = process.env.VITE_WS_PROXY_TARGET || "ws://localhost:18765";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      // v1 nginx.conf의 /api rewrite 동작을 dev 환경에서 재현
      "/api": {
        target: apiTarget,
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
      // FastAPI WebSocket (/ws/*)
      "/ws": {
        target: wsTarget,
        ws: true,
      },
    },
  },
});