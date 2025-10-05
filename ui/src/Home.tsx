//import { useState } from 'react'
import './App.css'
import React from 'react';
import logo from './logo.png'; // Tell webpack this JS file uses this image

let navigate = useNavigate(); 

function Home() {
  return (
    <>
      <div className="app flex-row align-items-center">
      <h1 className="text-center">HOME PAGE</h1>
      <div>
          <button color="primary" className="px-4"
            onClick={() => {navigate("/video")}}
              >
              Video
            </button>
          <button color="primary" className="px-4"
            onClick={() => {navigate("/picture")}}
              >
              Picture
            </button>
       </div>
      </div>
      
    </>
  )
}

export default Home
