import "./App.css"
import {useRef, useState, useEffect} from 'react'

function Picture() {
    const videoPlayerRef = useRef(null); 
    const canvasRef = useRef(null); 
    const hiddencanvasRef = useRef(null); 
    const [openCamera, setOpenCamera] = useState(true); 
    const [image, setImage] = useState(""); 
    const [animal, setAnimal] = useState("cat"); 
    const [connection, setConnection] = useState(null);
 
    function drawImg(imageurl) {
        const canvas = canvasRef.current;
        const canvasctx = canvas.getContext("2d");
        const imager = imageurl['image'];
        const img = new Image();
        img.onload = function() {
            canvasctx.drawImage(img, 0, 0);
        };
        img.src = imager;
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
            videoPlayerRef.current.srcObject = stream;
        } catch (error) {
            console.error("ERROR HER");
        }
    }

    const captureImage = () => {
        const hiddencanvas = hiddencanvasRef.current;
        const video = videoPlayerRef.current;
        if (video){
            const ctx = hiddencanvas.getContext("2d");
            ctx.drawImage(video, 0, 0, hiddencanvas.width, hiddencanvas.height); 
            //const image = hiddencanvas.toDataURL("image/png") //.replace("image/png", "image/octet-stream");
            //socket.emit('sendimage', image, animal);
            console.log('This aintturn out so well')
            hiddencanvas.toBlob((blob) => {
                console.log(blob)
                fetch("127.0.0.1/getpic", {
                    method: "POST",
                    headers: {
                      "Content-Type": "application/json"
                    },
                    body: JSON.stringify({image: blob , animal : animal})
                  })
                  .then(response => { drawImg(response.json()['image']) } )
            }, "image/jpeg", 0.8); // JPEG at 80% quality for smaller size
        }
    }

    function shootAnimal(animal){
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

        </div>
        <div className="bottom-0 h-20 flex flex-row absolute w-screen justify-center">
           <button 
               className={`w-20 rounded-sm border-2 mx-5 shadow-lg hover:shadow-xl transition-all duration-200 hover:scale-105 ${
                   animal === "human" ? "bg-amber-300" : "bg-amber-500"
               }`} 
               onClick={ () => {shootAnimal("human")}}
           >
               Human
           </button> 
           <button 
               className={`w-20 rounded-sm border-2 mx-5 shadow-lg hover:shadow-xl transition-all duration-200 hover:scale-105 ${
                   animal === "cat" ? "bg-amber-300" : "bg-amber-500"
               }`} 
               onClick={ () => {shootAnimal("cat")}}
           >
               Cat
           </button> 
           <button 
               className={`w-20 rounded-sm border-2 mx-5 shadow-lg hover:shadow-xl transition-all duration-200 hover:scale-105 ${
                   animal === "dog" ? "bg-amber-300" : "bg-amber-500"
               }`} 
               onClick={ () => {shootAnimal("dog")}}
           >
               Dog
           </button> 
           <button 
               className={`w-20 rounded-sm border-2 mx-5 shadow-lg hover:shadow-xl transition-all duration-200 hover:scale-105 ${
                   animal === "cow" ? "bg-amber-300" : "bg-amber-500"
               }`} 
               onClick={ () => {shootAnimal("cow")}}
           >
               Cow
           </button> 
           <button 
               className={`w-20 rounded-sm border-2 mx-5 shadow-lg hover:shadow-xl transition-all duration-200 hover:scale-105 ${
                   animal === "goat" ? "bg-amber-300" : "bg-amber-500"
               }`} 
               onClick={ () => {shootAnimal("goat")}}
           >
               Goat
           </button> 
           <button 
               className={`w-20 rounded-sm border-2 mx-5 shadow-lg hover:shadow-xl transition-all duration-200 hover:scale-105 ${
                   animal === "pig" ? "bg-amber-300" : "bg-amber-500"
               }`} 
               onClick={ () => {shootAnimal("pig")}}
           >
               Pig
           </button> 
           <button 
               className={`w-20 rounded-sm border-2 mx-5 shadow-lg hover:shadow-xl transition-all duration-200 hover:scale-105 ${
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