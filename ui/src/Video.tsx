//import { useState } from 'react'
import Update from './Update.tsx'
import './App.css'
import React from 'react';
import logo from './logo.png'; // Tell webpack this JS file uses this image


function Video() {
  return (
    <>
      <Update></Update>
      <button id="installApp">Install</button>
    </>
  )
}

export default Video
