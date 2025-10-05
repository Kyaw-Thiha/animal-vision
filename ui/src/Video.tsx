import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faBackward } from '@fortawesome/free-solid-svg-icons'
import { useNavigate } from 'react-router-dom';
import "./App.css"
import {useRef, useState, useEffect, useCallback} from 'react'
import { io, Socket } from "socket.io-client"

function Video() {
    const navigate = useNavigate();
    const videoPlayerRef = useRef<HTMLVideoElement>(null); 
    const canvasRef = useRef<HTMLCanvasElement>(null); 
    const hiddencanvasRef = useRef<HTMLCanvasElement>(null); 
    const [openCamera, setOpenCamera] = useState(true); 
    const [animal, setAnimal] = useState("human"); 
    const [socket, setSocket] = useState<Socket | null>(null);

    useEffect(() => {
        const newSocket = io('https://animal.yoshixi.net/');
        setSocket(newSocket);

        newSocket.on('connect', () => {
            console.log(`You connected to the server with id ${newSocket.id}`);
        });

        newSocket.on('getimage', (imagedata) => {
            console.log(imagedata)
            if (imagedata['image'] != null){
                const imager = imagedata['image'];
                const canvas = canvasRef.current;
                if (canvas) {
                    const canvasctx = canvas.getContext("2d");
                    const img = new Image();
                    img.onload = function() {
                        if (canvasctx) {
                            canvasctx.drawImage(img, 0, 0);
                        }
                    };
                    img.src = imager;
                }
                }   
            }
        );

        return () => {
            newSocket.close();
        };
    }, []);

    const captureImage = useCallback(() => {
        const hiddencanvas = hiddencanvasRef.current;
        const video = videoPlayerRef.current;
        if (video && socket && hiddencanvas){
            const ctx = hiddencanvas.getContext("2d");
            if (ctx) {
                ctx.drawImage(video, 0, 0, hiddencanvas.width, hiddencanvas.height); 
                //const image = hiddencanvas.toDataURL("image/png") //.replace("image/png", "image/octet-stream");
                //socket.emit('sendimage', image, animal);

                hiddencanvas.toBlob((blob: Blob | null) => {
                    if (blob) {
                        socket.emit('sendimage', blob, animal);
                    }
                }, "image/jpeg", 0.8); // JPEG at 80% quality for smaller size
            }
        }
    }, [socket, animal]);

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

    useEffect(() => {
      const interval = setInterval(() => {
        captureImage()
      }, 200);
    
      return () => clearInterval(interval); // This represents the unmount function, in which you need to clear your interval to prevent memory leaks.
    }, [captureImage])

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
            className="border-2 border-black rounded-lg absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2"
            id="canvas"
            ref={canvasRef}
            height={window.innerHeight - 80}
            width={window.innerWidth - 80}
        ></canvas>
        <canvas
            className="hidden"
            id="hiddencanvas"
            ref={hiddencanvasRef}
            height={window.innerHeight - 80}
            width={window.innerWidth - 80}
        ></canvas>

        <button className="absolute top-5 left-5 z-10 h-15 w-15 text-amber-500 bg-black border-4 border-amber-500 rounded-sm shadow-lg hover:shadow-xl transition-all duration-200 hover:scale-105" onClick={() => {navigate("/")}}>
            <FontAwesomeIcon icon={faBackward} />
            Back
        </button>
        </div>
        <div className="bottom-0 h-20 flex flex-row absolute w-screen justify-center">
           <button 
               className={`w-20 rounded-sm border-2 mx-5 shadow-lg hover:shadow-xl transition-all duration-200 hover:scale-105 ${
                   animal === "human" ? "bg-amber-300" : "bg-amber-500"
               }`} 
               onClick={ () => {setAnimal("human")}}
           >
               Human
           </button> 
           <button 
               className={`w-20 rounded-sm border-2 mx-5 shadow-lg hover:shadow-xl transition-all duration-200 hover:scale-105 ${
                   animal === "cat" ? "bg-amber-300" : "bg-amber-500"
               }`} 
               onClick={ () => {setAnimal("cat")}}
           >
               Cat
           </button> 
           <button 
               className={`w-20 rounded-sm border-2 mx-5 shadow-lg hover:shadow-xl transition-all duration-200 hover:scale-105 ${
                   animal === "dog" ? "bg-amber-300" : "bg-amber-500"
               }`} 
               onClick={ () => {setAnimal("dog")}}
           >
               Dog
           </button> 
           <button 
               className={`w-20 rounded-sm border-2 mx-5 shadow-lg hover:shadow-xl transition-all duration-200 hover:scale-105 ${
                   animal === "cow" ? "bg-amber-300" : "bg-amber-500"
               }`} 
               onClick={ () => {setAnimal("cow")}}
           >
               Cow
           </button> 
           <button 
               className={`w-20 rounded-sm border-2 mx-5 shadow-lg hover:shadow-xl transition-all duration-200 hover:scale-105 ${
                   animal === "goat" ? "bg-amber-300" : "bg-amber-500"
               }`} 
               onClick={ () => {setAnimal("goat")}}
           >
               Goat
           </button> 
           <button 
               className={`w-20 rounded-sm border-2 mx-5 shadow-lg hover:shadow-xl transition-all duration-200 hover:scale-105 ${
                   animal === "pig" ? "bg-amber-300" : "bg-amber-500"
               }`} 
               onClick={ () => {setAnimal("pig")}}
           >
               Pig
           </button> 
           <button 
               className={`w-20 rounded-sm border-2 mx-5 shadow-lg hover:shadow-xl transition-all duration-200 hover:scale-105 ${
                   animal === "sheep" ? "bg-amber-300" : "bg-amber-500"
               }`} 
               onClick={ () => {setAnimal("sheep")}}
           >
               Sheep
           </button> 
           <button 
               className={`w-20 rounded-sm border-2 mx-5 shadow-lg hover:shadow-xl transition-all duration-200 hover:scale-105 ${
                   animal === "rat" ? "bg-amber-300" : "bg-amber-500"
               }`} 
               onClick={ () => {setAnimal("rat")}}
           >
               Rat
           </button> 
           <button 
               className={`w-20 rounded-sm border-2 mx-5 shadow-lg hover:shadow-xl transition-all duration-200 hover:scale-105 ${
                   animal === "horse" ? "bg-amber-300" : "bg-amber-500"
               }`} 
               onClick={ () => {setAnimal("horse")}}
           >
               Horse
           </button> 
           <button 
               className={`w-20 rounded-sm border-2 mx-5 shadow-lg hover:shadow-xl transition-all duration-200 hover:scale-105 ${
                   animal === "squirrel" ? "bg-amber-300" : "bg-amber-500"
               }`} 
               onClick={ () => {setAnimal("squirrel")}}
           >
               Squirrel
           </button> 
           <button 
               className={`w-20 rounded-sm border-2 mx-5 shadow-lg hover:shadow-xl transition-all duration-200 hover:scale-105 ${
                   animal === "elephant" ? "bg-amber-300" : "bg-amber-500"
               }`} 
               onClick={ () => {setAnimal("elephant")}}
           >
               Elephant
           </button> 
           <button 
               className={`w-20 rounded-sm border-2 mx-5 shadow-lg hover:shadow-xl transition-all duration-200 hover:scale-105 ${
                   animal === "lion" ? "bg-amber-300" : "bg-amber-500"
               }`} 
               onClick={ () => {setAnimal("lion")}}
           >
               Lion
           </button> 
           <button 
               className={`w-20 rounded-sm border-2 mx-5 shadow-lg hover:shadow-xl transition-all duration-200 hover:scale-105 ${
                   animal === "wolf" ? "bg-amber-300" : "bg-amber-500"
               }`} 
               onClick={ () => {setAnimal("wolf")}}
           >
               Wolf
           </button> 
           <button 
               className={`w-20 rounded-sm border-2 mx-5 shadow-lg hover:shadow-xl transition-all duration-200 hover:scale-105 ${
                   animal === "fox" ? "bg-amber-300" : "bg-amber-500"
               }`} 
               onClick={ () => {setAnimal("fox")}}
           >
               Fox
           </button> 
           <button 
               className={`w-20 rounded-sm border-2 mx-5 shadow-lg hover:shadow-xl transition-all duration-200 hover:scale-105 ${
                   animal === "bear" ? "bg-amber-300" : "bg-amber-500"
               }`} 
               onClick={ () => {setAnimal("bear")}}
           >
               Bear
           </button> 
           <button 
               className={`w-20 rounded-sm border-2 mx-5 shadow-lg hover:shadow-xl transition-all duration-200 hover:scale-105 ${
                   animal === "raccoon" ? "bg-amber-300" : "bg-amber-500"
               }`} 
               onClick={ () => {setAnimal("raccoon")}}
           >
               Raccon
           </button> 
           <button 
               className={`w-20 rounded-sm border-2 mx-5 shadow-lg hover:shadow-xl transition-all duration-200 hover:scale-105 ${
                   animal === "deer" ? "bg-amber-300" : "bg-amber-500"
               }`} 
               onClick={ () => {setAnimal("deer")}}
           >
               Deer
           </button> 
           <button 
               className={`w-20 rounded-sm border-2 mx-5 shadow-lg hover:shadow-xl transition-all duration-200 hover:scale-105 ${
                   animal === "kangaroo" ? "bg-amber-300" : "bg-amber-500"
               }`} 
               onClick={ () => {setAnimal("kangaroo")}}
           >
               Kangaroo
           </button> 
           <button 
               className={`w-20 rounded-sm border-2 mx-5 shadow-lg hover:shadow-xl transition-all duration-200 hover:scale-105 ${
                   animal === "tiger" ? "bg-amber-300" : "bg-amber-500"
               }`} 
               onClick={ () => {setAnimal("tiger")}}
           >
               Tiger
           </button> 
           <button 
               className={`w-20 rounded-sm border-2 mx-5 shadow-lg hover:shadow-xl transition-all duration-200 hover:scale-105 ${
                   animal === "rabbit" ? "bg-amber-300" : "bg-amber-500"
               }`} 
               onClick={ () => {setAnimal("rabbit")}}
           >
               Rabbit
           </button> 
           <button 
               className={`w-20 rounded-sm border-2 mx-5 shadow-lg hover:shadow-xl transition-all duration-200 hover:scale-105 ${
                   animal === "panda" ? "bg-amber-300" : "bg-amber-500"
               }`} 
               onClick={ () => {setAnimal("panda")}}
           >
               Panda
           </button> 
           <button 
               className={`w-20 rounded-sm border-2 mx-5 shadow-lg hover:shadow-xl transition-all duration-200 hover:scale-105 ${
                   animal === "honeybee" ? "bg-amber-300" : "bg-amber-500"
               }`} 
               onClick={ () => {setAnimal("honeybee")}}
           >
               Honeybee
           </button> 
        </div>
        </>
    )
}

export default Video;