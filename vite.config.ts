import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Served from https://ishwar6.github.io/ishwar-books/. Override with BASE_PATH=/ for a custom domain.
export default defineConfig({
  base: process.env.BASE_PATH ?? '/ishwar-books/',
  plugins: [react()],
  // mermaid's diagram engines are large but only load on pages that contain a diagram
  build: { chunkSizeWarningLimit: 1600 },
})
