import os
import typer
import numpy as np
from InquirerPy import inquirer

from renderers.image import ImageRenderer
from renderers.video import VideoRenderer
from renderers.webcam import WebcamRenderer


app = typer.Typer()

INPUT_DIR = "input/images"
OUTPUT_DIR = "output"


@app.command()
def image(input_dir: str = INPUT_DIR):
    filename = os.path.join(input_dir, "filename")
    renderer = ImageRenderer(filename, show_window=True, save_to="out.png", wait_key=0)
    renderer.open()

    img = renderer.get_image()
    if img is not None:
        # out = process_img(img)
        out = img
        renderer.render(out)

    renderer.close()


@app.command()
def video():
    vr = VideoRenderer(read_path="in.mp4", write_path="out.mp4", window_name="Preview")
    vr.open()
    while True:
        frame = vr.get_image()
        if frame is None:
            break
        # out = animal.transform(frame, ctx)  # your per-frame processing
        out = frame
        vr.render(out)
    vr.close()


@app.command()
def webcam():
    wr = WebcamRenderer(
        index=0, width=1280, height=720, write_path="out.mp4", window_name="AnimalCam"
    )
    wr.open()
    try:
        while True:
            frame = wr.get_image()
            if frame is None:
                break
            # out = animal.transform(frame, ctx)  # your per-frame logic
            out = frame
            wr.render(out)
    finally:
        wr.close()


if __name__ == "__main__":
    app()
