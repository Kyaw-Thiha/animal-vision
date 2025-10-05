import { defineConfig } from 'vite'
import { VitePWA } from "vite-plugin-pwa";
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite';

// https://vite.dev/config/
export default defineConfig({
  base: "./",
  plugins: [
    react(),
    tailwindcss(),
    VitePWA({
      injectRegister: 'auto',
      // generates 'manifest.webmanifest' file on build
      manifest: {
        // caches the assets/icons mentioned (assets/* includes all the assets present in your src/ directory) 
        name: 'Simplifying Progressive Web App',
        short_name: 'PWA Guide',
        "icons": [
          {
            "src": "src/assets/512.png",
            "type": "image/png",
            "sizes": "512x512"
          },
          {
            "src": "src/assets/192.png",
            "type": "image/png",
            "sizes": "192x192"
          }
        ]
        start_url: '/',
        background_color: '#ffffff',
        theme_color: '#000000',
        display: "fullscreen",
        prefer_related_applications : false
      },
      workbox: {
        // defining cached files formats
        globPatterns: ["**/*.{js,css,html,ico,png,svg,webmanifest}"],
      }
    })
  ],
  server:{
    allowedHosts : ["3bfb4efe4ad0.ngrok-free.app"],
    proxy: {
      "/getpic": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
        secure: false
      }
    }
  }
})
