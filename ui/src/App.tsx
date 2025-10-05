//import { useState } from 'react'
import './App.css'
import React from 'react';
import logo from './logo.png'; // Tell webpack this JS file uses this image
import { BrowserRouter, Routes, Route } from "react-router";
import Video from './Video.tsx'
import Picture from './Picture.tsx'
import Home from './Home.tsx'

function App() {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="video" element={<Video />} />
      <Route path="picture" element={<Picture />} />
    </Routes>
  )
}

export default App
