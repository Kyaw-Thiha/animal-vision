import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faBackward } from '@fortawesome/free-solid-svg-icons'
import { useNavigate } from 'react-router-dom';
import "./App.css"
import {useRef, useState, useEffect} from 'react'

function Picture() {
    const navigate = useNavigate();
    const videoPlayerRef = useRef<HTMLVideoElement | null>(null); 
    const canvasRef = useRef<HTMLCanvasElement | null>(null); 
    const hiddencanvasRef = useRef<HTMLCanvasElement | null>(null); 
    const [openCamera, setOpenCamera] = useState(true); 
    const [animal, setAnimal] = useState("cat"); 
    
    // Keep canvas size constant from initial mount
    const [canvasWidth] = useState<number>(() => Math.max(0, window.innerWidth - 80));
    const [canvasHeight] = useState<number>(() => Math.max(0, window.innerHeight - 80));
    // Track orientation to rotate visible canvas accordingly
    const [orientationDeg, setOrientationDeg] = useState<number>(0);

    const getOrientationAngle = (): number => {
        const screenOrientation = window.screen?.orientation;
        let angle = 0;
        if (screenOrientation && typeof screenOrientation.angle === 'number') {
            angle = screenOrientation.angle;
        } else {
            const legacy = (window as unknown as { orientation?: number }).orientation;
            if (typeof legacy === 'number') {
                angle = legacy;
            } else {
                angle = window.innerWidth > window.innerHeight ? 90 : 0;
            }
        }
        angle = ((Math.round(angle / 90) * 90) % 360 + 360) % 360;
        return angle;
    };

    useEffect(() => {
        const updateOrientation = () => setOrientationDeg(getOrientationAngle());
        updateOrientation();
        window.addEventListener('orientationchange', updateOrientation);
        window.addEventListener('resize', updateOrientation);
        return () => {
            window.removeEventListener('orientationchange', updateOrientation);
            window.removeEventListener('resize', updateOrientation);
        };
    }, []);
 
    function drawImg(imageurl: { image?: string }) {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const canvasctx = canvas.getContext("2d");
        if (!canvasctx) return;
        const imager = imageurl['image'];
        const img = new Image();
        img.onload = function() {
            canvasctx.drawImage(img, 0, 0);
        };
        if (imager) img.src = imager;
    }

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
            if (videoPlayerRef.current) {
                videoPlayerRef.current.srcObject = stream;
            }
        } catch (error) {
            console.error("ERROR HER", error);
        }
    }

    const captureImage = () => {
        const hiddencanvas = hiddencanvasRef.current;
        const video = videoPlayerRef.current;
        if (video && hiddencanvas){
            const ctx = hiddencanvas.getContext("2d");
            if (!ctx) return;
            ctx.drawImage(video, 0, 0, hiddencanvas.width, hiddencanvas.height); 
            const image = hiddencanvas.toDataURL("image/png");
            fetch("https://animal.yoshixi.net/getpic", {
                method: "POST",
                headers: {
                  "Content-Type": "application/json"
                },
                body: JSON.stringify({image: image , animal : animal})
              })
              .then(async (response) => {
                  if (!response.ok) {
                      throw new Error(`HTTP ${response.status}`)
                  }
                  const data = await response.json()
                  drawImg(data)
              })
              .catch(err => {
                  console.error("Failed to fetch processed image", err)
              })
        }
    }

    function shootAnimal(animal: string){
        setAnimal(animal)
        captureImage()
    }

    return (
        <>
        <div className="flex justify-center">
        <video
            className="!w-full hidden"
            ref={videoPlayerRef}
            id="player"
            autoPlay
        ></video>
        <canvas
            className={`border-2 border-black rounded-lg absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 ${orientationDeg === 90 ? '-rotate-90' : orientationDeg === 180 ? 'rotate-180' : orientationDeg === 270 ? 'rotate-90' : 'rotate-0'}`}
            id="canvas"
            ref={canvasRef}
            height={canvasHeight}
            width={canvasWidth}
        ></canvas>
        <canvas
            className="hidden"
            id="hiddencanvas"
            ref={hiddencanvasRef}
            height={canvasHeight}
            width={canvasWidth}
        ></canvas>

        </div>
        <button className="absolute top-5 left-5 z-10 h-15 w-15 text-amber-500 bg-black border-4 border-amber-500 rounded-sm shadow-lg hover:shadow-xl transition-all duration-200 hover:scale-105" onClick={() => {navigate("/")}}>
            <FontAwesomeIcon icon={faBackward} />
            Back
        </button>
        <div className="w-screen py-4 grid grid-flow-col grid-rows-2 auto-cols-max gap-3 justify-center">
           <button 
               className={`z-10 w-20 rounded-sm border-2 shadow-lg hover:shadow-xl transition-all duration-200 hover:scale-105 ${
                   animal === "human" ? "bg-amber-300" : "bg-amber-500"
               }`} 
               onClick={ () => {shootAnimal("human")}}
           >
               Human
           </button> 
           <button 
               className={`z-10 w-20 rounded-sm border-2 shadow-lg hover:shadow-xl transition-all duration-200 hover:scale-105 ${
                   animal === "cat" ? "bg-amber-300" : "bg-amber-500"
               }`} 
               onClick={ () => {shootAnimal("cat")}}
           >
               Cat
           </button> 
           <button 
               className={`z-10 w-20 rounded-sm border-2 shadow-lg hover:shadow-xl transition-all duration-200 hover:scale-105 ${
                   animal === "dog" ? "bg-amber-300" : "bg-amber-500"
               }`} 
               onClick={ () => {shootAnimal("dog")}}
           >
               Dog
           </button> 
           <button 
               className={`z-10 w-20 rounded-sm border-2 shadow-lg hover:shadow-xl transition-all duration-200 hover:scale-105 ${
                   animal === "cow" ? "bg-amber-300" : "bg-amber-500"
               }`} 
               onClick={ () => {shootAnimal("cow")}}
           >
               Cow
           </button> 
           <button 
               className={`z-10 w-20 rounded-sm border-2 shadow-lg hover:shadow-xl transition-all duration-200 hover:scale-105 ${
                   animal === "goat" ? "bg-amber-300" : "bg-amber-500"
               }`} 
               onClick={ () => {shootAnimal("goat")}}
           >
               Goat
           </button> 
           <button 
               className={`z-10 w-20 rounded-sm border-2 shadow-lg hover:shadow-xl transition-all duration-200 hover:scale-105 ${
                   animal === "pig" ? "bg-amber-300" : "bg-amber-500"
               }`} 
               onClick={ () => {shootAnimal("pig")}}
           >
               Pig
           </button> 
           <button 
               className={`z-10 w-20 rounded-sm border-2 shadow-lg hover:shadow-xl transition-all duration-200 hover:scale-105 ${
                   animal === "sheep" ? "bg-amber-300" : "bg-amber-500"
               }`} 
               onClick={ () => {shootAnimal("sheep")}}
           >
               Sheep
           </button> 
        </div>
        </>
    )
}

export default Picture;