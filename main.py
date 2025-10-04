import os
import typer
import numpy as np
from InquirerPy import inquirer

from renderers.image import ImageRenderer
from renderers.video import VideoRenderer
from renderers.webcam import WebcamRenderer
from utils import choose_file, choose_filename


app = typer.Typer()

IMAGES_INPUT = "input/images"
VIDEO_INPUT = "input/video"
IMAGES_OUTPUT = "output"
VIDEO_OUTPUT = "output"


@app.command()
def image(input_dir: str = IMAGES_INPUT, output_dir: str = IMAGES_OUTPUT):
    filename = choose_file(input_dir, (".png", ".jpg"))
    save_name = choose_filename(output_dir, ".png")
    renderer = ImageRenderer(filename, show_window=True, save_to=save_name, wait_key=0)
    renderer.open()

    img = renderer.get_image()
    if img is not None:
        # out = process_img(img)
        out = img
        renderer.render(out)

    renderer.close()


@app.command()
def video(input_dir: str = VIDEO_INPUT, output_dir: str = VIDEO_OUTPUT):
    filename = choose_file(input_dir, (".mp4", ".avi", ".mov"))
    save_name = choose_filename(output_dir, ".mp4")
    vr = VideoRenderer(read_path=filename, write_path=save_name, window_name="Preview")
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
    wr = WebcamRenderer(index=0, width=1280, height=720, write_path="out.mp4", window_name="AnimalCam")
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
