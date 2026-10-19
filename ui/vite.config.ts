import { defineConfig } from "vite";
import solid from "@solidjs/vite-plugin";
import tailwindcss from "@tailwindcss/vite";

// `pnpm dev` proxies the API to a running `drama serve` (DRAMA_URL, default 127.0.0.1:8080).
const api = process.env.DRAMA_URL ?? "http://127.0.0.1:8080";

export default defineConfig({
  plugins: [solid(), tailwindcss()],
  server: { port: 5278, proxy: { "/v1": api, "/health": api, "/openapi.json": api } },
  build: { target: "es2022", outDir: "dist", emptyOutDir: true },
});
