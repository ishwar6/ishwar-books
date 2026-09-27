import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Served from the custom domain https://ishwarj.com/ (see public/CNAME).
// Set BASE_PATH=/ishwar-books/ to serve from ishwar6.github.io/ishwar-books/ instead.
export default defineConfig({
  base: process.env.BASE_PATH ?? '/',
  plugins: [react()],
  // mermaid's diagram engines are large but only load on pages that contain a diagram
  build: { chunkSizeWarningLimit: 1600 },
})
