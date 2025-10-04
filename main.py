import os
import typer
import numpy as np
from InquirerPy import inquirer


app = typer.Typer()

INPUT_DIR = "input/images"
OUTPUT_DIR = "output"


@app.command()
def run_image(input_dir: str = INPUT_DIR):
    pass


@app.command()
def run_webcam():
    pass


if __name__ == "__main__":
    app()
