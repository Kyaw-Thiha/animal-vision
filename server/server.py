from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import socketio
import sys
sys.path.append("../")
from utils import processimage, processsplitimage
import asyncio
from typing import Dict, List, Tuple
from collections import deque

app = FastAPI()

# Allow web UI to call our REST endpoints during development
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)
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
    # Create background task when server starts accepting connections
    if not hasattr(sio, '_background_task_started'):
        sio.start_background_task(send_to_client)
        sio._background_task_started = True


async def send_to_client():
    while True:
        try:
            for sid in list(conns.keys()):
                if sid in conns and len(conns[sid]) > 0:
                    image, animal = conns[sid].popleft()
                    processedimage = processimage(image, animal)
                    await sio.emit('getimage', {
                        'image': processedimage
                    }, room=sid)
            await asyncio.sleep(0.001)
        except Exception as e:
            print(f"Error in send_to_client: {e}")
            await asyncio.sleep(1)


# For running with uvicorn
# uvicorn filename:socket_app --reload
# app.mount("/", socket_app)

class PostImageRequest(BaseModel):
	image: str
	animal: str


@app.get("/")
def root():
	return {"conns" : conns}

@app.post("/getpic")
async def getpic(payload: PostImageRequest):
    print(payload)
    processed = processsplitimage(payload.image, payload.animal)
    return {"image": processed}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(socket_app, host="0.0.0.0", port=8000)