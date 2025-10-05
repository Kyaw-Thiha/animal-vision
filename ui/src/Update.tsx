import "./App.css"
import {useRef, useState, useEffect} from 'react'
import { io } from "socket.io-client"

function Update() {
    const socket = io('http://127.0.0.1:8000')
    const videoPlayerRef = useRef(null); 
    const canvasRef = useRef(null); 
    const hiddencanvasRef = useRef(null); 
    const [openCamera, setOpenCamera] = useState(true); 
    const [image, setImage] = useState(""); 
    const [animal, setAnimal] = useState("human"); 
    const [connection, setConnection] = useState(null);
   
    socket.on('connect', () => {
        console.log(`You connected to the server with id ${socket.id}`);
    })

    socket.on('getimage', (imagedata) => {
        console.log(imagedata)
        if (imagedata['image'] != null){
            const imager = imagedata['image'];
            const canvas = canvasRef.current;
            const canvasctx = canvas.getContext("2d");
            const img = new Image();
            img.onload = function() {
                canvasctx.drawImage(img, 0, 0);
            };
            img.src = imager;
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
        captureImage()
      }, 100);
    
      return () => clearInterval(interval); // This represents the unmount function, in which you need to clear your interval to prevent memory leaks.
    }, [])
    

    const captureImage = () => {
        const hiddencanvas = hiddencanvasRef.current;
        const video = videoPlayerRef.current;
        if (video){
            const ctx = hiddencanvas.getContext("2d");
            ctx.drawImage(video, 0, 0, hiddencanvas.width, hiddencanvas.height); 
            const image = hiddencanvas.toDataURL("image/png") //.replace("image/png", "image/octet-stream");
            socket.emit('sendimage', image, animal);


            // canvas.toBlob((blob) => {
            //     socket.emit('sendimage', blob, animal);
            // }, "image/jpeg", 0.8); // JPEG at 80% quality for smaller size
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
            height={window.innerHeight}
            width={window.innerWidth}
        ></canvas>
        <canvas
            className="border-2 border-amber-500 hidden"
            id="hiddencanvas"
            ref={hiddencanvasRef}
            height={window.innerHeight}
            width={window.innerWidth}
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