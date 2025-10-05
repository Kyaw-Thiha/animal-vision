//import { useState } from 'react'
import './App.css'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faCoffee, faVideo, faCamera, faDownload } from '@fortawesome/free-solid-svg-icons'
import React from 'react';
import { useNavigate } from 'react-router-dom';
import logo from './logo.png'; // Tell webpack this JS file uses this image

function Home() {
  const navigate = useNavigate();
  return (
    <>
      <div className="w-screen app flex flex-col items-center justify-center h-screen">
        <h1 className="text-center mb-8 text-4xl font-bold">Take Another View</h1>
        <div className="flex justify-center space-x-6">
            <button className="bg-amber-300 rounded-sm transition-all border-2 text-lg shadow-md hover:scale-105 hover:shadow-lg px-6 py-2 flex items-center gap-2"
              onClick={() => {navigate("/video")}}
                >
                <FontAwesomeIcon icon={faVideo} />
                Live View
              </button>
            <button className="bg-amber-300 rounded-sm transition-all border-2 text-lg shadow-md hover:scale-105 hover:shadow-lg px-6 py-2 flex items-center gap-2"
              onClick={() => {console.log("something")}}
                >
                <FontAwesomeIcon icon={faDownload} />
                Install PWA
              </button>
            <button className="bg-amber-300 rounded-sm transition-all border-2 text-lg shadow-md hover:scale-105 hover:shadow-lg px-6 py-2 flex items-center gap-2"
              onClick={() => {navigate("/picture")}}
                >
                <FontAwesomeIcon icon={faCamera} />
                Picture View
              </button>
        </div>
      </div>
      
    </>
  )
}

export default Home
