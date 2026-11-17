import { defineConfig } from 'vite';

export default defineConfig({
    // Relative URLs: works on GitHub Pages under /<repo>/
    base: './',
    build: {
        // Built site, served by GitHub Pages ("Deploy from a branch", folder /docs)
        outDir: '../docs',
        emptyOutDir: true,
    },
    optimizeDeps: {
        exclude: ['onnxruntime-web'],
    },
});
