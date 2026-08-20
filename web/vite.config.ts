import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// The dev server proxies /api to FastAPI so the browser sees a single origin.
// That is what lets the API ship with no CORS middleware at all: in
// production FastAPI serves this app's build output itself, which is also one
// origin. Permissive CORS on an API that can run shell commands would mean any
// page you had open could drive the agent.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    // Loopback only, matching the API. Nothing here should be reachable from
    // another machine.
    host: '127.0.0.1',
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: false,
        // Server-sent events must not be buffered: a turn takes minutes and
        // the whole point is seeing it arrive.
        ws: false,
      },
    },
  },
})
