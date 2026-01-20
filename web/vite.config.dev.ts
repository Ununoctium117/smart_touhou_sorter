import { defineConfig } from "vite";
import wasm from "vite-plugin-wasm";
import topLevelAwait from "vite-plugin-top-level-await";
import wasmPackWatchPlugin from "vite-plugin-wasm-pack-watcher";
import { svelte } from '@sveltejs/vite-plugin-svelte'

export default defineConfig({
    build: {
        watch: {
            include: [
                "src/**/*.ts",
                "../engine/**/*.rs",
                "../engine/**/*.toml",
                "../web-engine/**/*.rs",
                "../web-engine/**/*.toml"
            ],
        },
    },
    plugins: [
        svelte(),
        wasmPackWatchPlugin(),
        wasm(),
        topLevelAwait()
    ],
});