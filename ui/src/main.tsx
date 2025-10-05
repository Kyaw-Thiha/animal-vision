import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './Video.tsx'
import Picture from './Picture.tsx'
import Home from './Home.tsx'
import { BrowserRouter, Routes, Route } from "react-router";

createRoot(document.getElementById('root')!).render(
  <StrictMode>
  <BrowserRouter>
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="video" element={<Video />} />
      <Route path="picture" element={<Picture />} />
    </Routes>
  </BrowserRouter>
  </StrictMode>,
)
