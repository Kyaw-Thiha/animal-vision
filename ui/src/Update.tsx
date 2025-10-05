import "./App.css"
import {useRef, useState, useEffect} from 'react'
import { io } from "socket.io-client"

function Update() {
    const socket = io('http://127.0.0.1:8000')
    const videoPlayerRef = useRef(null); 
    const canvasRef = useRef(null); 
    const [openCamera, setOpenCamera] = useState(true); 
    const [image, setImage] = useState(false); 
    const [animal, setAnimal] = useState("human"); 
    const [connection, setConnection] = useState(null);
   
    socket.on('connect', () => {
        console.log(`You connected to the server with id ${socket.id}`);
    })
    
    socket.on('getimage', (imagedata) => {
        console.log(`You get an image`, imagedata);
        setImage(imagedata);
        const ctx = canvas.getContext("2d");
        const img = new Image();
        img.src = image;
        img.onload = function() {
            ctx.drawImage(
                img,
                0,
                0,
                canvas.width,
                canvas.height
            );
        }
        }   
    );

    useEffect( () => {
        if (openCamera) {
            initializeMedia();
            setOpenCamera(false);
        }
        return () => {
                // hello
        };
    }, [openCamera]);


    const initializeMedia = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: 'environment' },
                audio: false
              });
            videoPlayerRef.current.srcObject = stream;
        } catch (error) {
            console.error("ERROR HER");
        }
    }

    useEffect(() => {
      const interval = setInterval(() => {
        setCurrentImage()
      }, 100);
    
      return () => clearInterval(interval); // This represents the unmount function, in which you need to clear your interval to prevent memory leaks.
    }, [])
    
    // update every 30 frames 
    const setCurrentImage = async () => {
        const image = captureImage();
        socket.emit('sendimage', image, animal);
        }

    const captureImage = () => {
        const canvas = canvasRef.current;
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
        const video = videoPlayerRef.current;
        if (video){
            const image = canvas.toDataURL("image/png") //.replace("image/png", "image/octet-stream");
            // const imageDataUrl = canvas.toDataURL("image/png");
            // setImage(imageDataUrl);
            return image
        }
    }

    return (
        <>
        <video
            className="!w-full border-2 border-black hidden"
            ref={videoPlayerRef}
            id="player"
            autoPlay
        ></video>
        <canvas
            className="border-2 border-amber-500"
            id="canvas"
            ref={canvasRef}
        ></canvas>
        <button
            className="bg-red-500 border-2 border-black absolute z-[1]"
            id="capture"
            onClick={captureImage}
        >Capture</button>
        </>
    )
}

export default Update;