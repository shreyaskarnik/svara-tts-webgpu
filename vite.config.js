import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { viteStaticCopy } from "vite-plugin-static-copy";

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
    // ORT-Web's .wasm/.mjs runtime files aren't served by Vite by default.
    // Copy them from node_modules into the dev server + build output so the
    // worker can load them via /ort-wasm/<file>.
    viteStaticCopy({
      targets: [
        {
          src: "node_modules/onnxruntime-web/dist/*.{wasm,mjs}",
          dest: "ort-wasm",
        },
      ],
    }),
  ],
  worker: { format: "es" },
  build: {
    target: "esnext",
  },
});
