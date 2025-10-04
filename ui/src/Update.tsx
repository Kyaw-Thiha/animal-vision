import "./App.css"
import {useRef, useState, useEffect} from 'react'

function Update() {
    const videoPlayerRef = useRef(null); 
    const canvasRef = useRef(null); 
    const [openCamera, setOpenCamera] = useState(true); 
    const [image, setImage] = useState(false); 


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
            //videoPlayerRef.current.style.display = "block";
        } catch (error) {
            console.error("ERROR HER");
        }
    }
    const handleCapture = () => {
        console.log("CATURING");
        const canvas = canvasRef.current;
        const video = videoPlayerRef.current;
        console.log(video);
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext("2d");
        if (videoPlayerRef.current){
            ctx.drawImage(
                videoPlayerRef.current,
                0,
                0,
                canvas.width,
                videoPlayerRef.current.videoHeight / (videoPlayerRef.current.videoWidth / canvas.width)
            );
            const imageDataUrl = canvas.toDataURL("image/png");
            setImage(imageDataUrl);
            console.log(imageDataUrl);
            // videoPlayerRef.current.srcObject?.getVideoTracks().forEach((track : any) => {track.stop();});
            // videoPlayerRef.current.style.display = "none";
            // canvas.style.display = "block";
            // setOpenCamera(false);
        }
    }
    return (
        <>
        <video
            className="!w-full border-2 border-black"
            ref={videoPlayerRef}
            id="player"
            autoPlay
        ></video>
        <canvas
            id="canvas"
            ref={canvasRef}
        ></canvas>
        <button
            className="bg-red-500 border-2 border-black absolute z-[1]"
            id="capture"
            onClick={handleCapture}
        >Capture</button>
        </>

    )
}

export default Update;