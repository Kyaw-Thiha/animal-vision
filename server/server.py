from fastapi import FastAPI
import socketio
import asyncio
from typing import Dict, List, Tuple
from collections import deque

app = FastAPI()
sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')
socket_app = socketio.ASGIApp(sio, app)

conns: Dict[str, deque] = {}

@sio.event
async def connect(sid, environ):
    conns[sid] = deque(maxlen=100)  # Limit queue size to prevent memory issues

@sio.event
async def disconnect(sid):
    if sid in conns:
        del conns[sid]

@sio.event
async def sendimage(sid, image, animal):
    if sid not in conns:
        conns[sid] = deque(maxlen=100)
    
    if image and animal:
        conns[sid].append((image, animal))
        print(f"Queued image from {sid}, queue size: {len(conns[sid])}")
    else:
        await sio.emit('error', {'message': 'Invalid format'}, room=sid)

@sio.event
async def connect(sid, environ):
    print("hello")
    # Create background task when server starts accepting connections
    if not hasattr(sio, '_background_task_started'):
        sio.start_background_task(send_to_client)
        sio._background_task_started = True

def processimage(imagedata: bytes, animal: str) -> bytes:
    return imagedata

async def send_to_client():
    breakpoint()
    while True:
        try:
            for sid in list(conns.keys()):
                if sid in conns and len(conns[sid]) > 0:
                    image, animal = conns[sid].popleft()
                    processedimage = processimage(image, animal)
                    await sio.emit('getimage', {
                        'image': processedimage,
                        'animal': animal
                    }, room=sid)
            await asyncio.sleep(0.001)
        except Exception as e:
            print(f"Error in send_to_client: {e}")
            await asyncio.sleep(1)


# For running with uvicorn
# uvicorn filename:socket_app --reload
# app.mount("/", socket_app)

	 
@app.get("/")
def root():
	return {"conns" : conns}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(socket_app, host="0.0.0.0", port=8000)