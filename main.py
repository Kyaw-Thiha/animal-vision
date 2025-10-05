import os
from typing import List, Optional, Tuple
import typer
import numpy as np
from InquirerPy import inquirer

from renderers.image import ImageRenderer
from renderers.video import VideoRenderer
from renderers.webcam import WebcamRenderer
from utils import choose_animal, choose_file, choose_filename

from typing import List, Tuple, Optional
from InquirerPy import inquirer
import cv2
import numpy as np
from datetime import datetime
import os

from gallery_grid import build_labeled_grid
from utils import choose_file, choose_filename
from renderers.image import ImageRenderer  # assuming your existing path


app = typer.Typer()

IMAGES_INPUT = "input/images"
VIDEO_INPUT = "input/video"
IMAGES_OUTPUT = "output"
VIDEO_OUTPUT = "output"


@app.command()
def image(input_dir: str = IMAGES_INPUT, output_dir: str = IMAGES_OUTPUT):
    filename = choose_file(input_dir, (".png", ".jpg"))
    save_name = choose_filename(output_dir, ".png")
    animal = choose_animal()

    renderer = ImageRenderer(filename, show_window=True, save_to=save_name, wait_key=0)
    renderer.open()

    img = renderer.get_image()
    if img is not None:
        result = animal.visualize(img)
        if result is not None:
            img, out = result
            if out is not None:
                # renderer.render(out)
                renderer.render_split_compare(img, out)

    renderer.close()


@app.command()
def video(input_dir: str = VIDEO_INPUT, output_dir: str = VIDEO_OUTPUT):
    filename = choose_file(input_dir, (".mp4", ".avi", ".mov"))
    save_name = choose_filename(output_dir, ".mp4")
    vr = VideoRenderer(read_path=filename, write_path=save_name, window_name="AnimalCam")
    animal = None
    vr.open()
    while True:
        frame = vr.get_image()
        if frame is None:
            break
        if animal == None:
            animal = choose_animal()  # per-frame processing
        result = animal.visualize(frame)
        if result is not None:
            img, out = result
            if out is not None:
                # vr.render(out)
                vr.render_split_compare(img, out)
    vr.close()


@app.command()
def webcam(output_dir: str = VIDEO_OUTPUT):
    save_name = choose_filename(output_dir, ".mp4")
    wr = WebcamRenderer(index=0, width=1280, height=720, write_path=save_name, window_name="AnimalCam")
    wr.open()
    animal = None
    try:
        while True:
            frame = wr.get_image()
            if frame is None:
                break
            if animal is None:
                animal = choose_animal()  # per-frame processing
            result = animal.visualize(frame)
            if result is not None:
                img, out = result
                if out is not None:
                    # wr.render(out)
                    wr.render_split_compare(img, out)
    finally:
        wr.close()


NON_UV_NAMES = [
    "Cat",
    "Dog",
    "Sheep",
    "Pig",
    "Goat",
    "Cow",
    "Horse",
    "Rabbit",
    "Panda",
    "Squirrel",
    "Elephant",
    "Lion",
    "Wolf",
    "Fox",
    "Bear",
    "Raccoon",
    "Deer",
    "Kangaroo",
    "Tiger",
    "Rat",
]
UV_NAMES = [
    "HoneyBee",
    "ReinDeer",
    "RatUV",
    "GoldFish",
    "DamselFish",
    "Anableps (Four-eyed fish)",
    "Northern Anchovy Fish",
    "Guppy Fish",
    "Morpho Butterfly",
    "Heliconius Butterfly",
    "Pieris Butterfly",
]
UNIQUE_UV_NAMES = ["Mantis Shrimp", "Kestrel", "Jumping Spider", "DragonFly", "HummingBird"]


# Build a lookup from your existing animal_choices list defined elsewhere
def _build_animal_lookup(animal_choices: List[dict]) -> dict[str, object]:
    return {entry["name"]: entry["value"] for entry in animal_choices}


def _pick_category() -> str:
    return inquirer.select(  # type: ignore[reportPrivateImportUsage]
        message="Choose a category:",
        choices=["Non-UV", "UV", "Unique-UV"],
        default="Non-UV",
    ).execute()


def _names_for_category(cat: str) -> List[str]:
    if cat == "Non-UV":
        return NON_UV_NAMES
    if cat == "UV":
        return UV_NAMES
    return UNIQUE_UV_NAMES


def _ensure_rgb_uint8(img: np.ndarray) -> np.ndarray:
    """Utility to keep behavior consistent with your pipeline."""
    if img.dtype != np.uint8:
        if np.issubdtype(img.dtype, np.floating):
            img = np.clip(img, 0.0, 1.0)
            img = (img * 255.0 + 0.5).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
    return img


def _run_visualize_get_output(animal_obj, rgb_image: np.ndarray) -> Optional[np.ndarray]:
    """
    Calls animal.visualize(image) and returns the 'visualized' frame (RGB),
    handling both return signatures:
      - np.ndarray
      - Tuple[np.ndarray, np.ndarray] -> (baseline, transformed)
    """
    try:
        res = animal_obj.visualize(rgb_image)
    except Exception as e:
        print(f"[WARN] {animal_obj.__class__.__name__}.visualize failed: {e}")
        return None

    if res is None:
        return None

    if isinstance(res, tuple) and len(res) == 2:
        base, out = res
        candidate = out if out is not None else base
    else:
        candidate = res

    if candidate is None:
        return None

    # normalize to RGB uint8 for montage
    return _ensure_rgb_uint8(candidate)


@app.command()
def gallery(input_dir: str = IMAGES_INPUT, output_dir: str = IMAGES_OUTPUT, tile_height: int = 256):
    """
    Build a labeled grid of animal visualizations for a chosen category and save it as a PNG.
    """
    # 1) Pick source image and output filename
    filename = choose_file(input_dir, (".png", ".jpg", ".jpeg"))
    if filename is None:
        print("No image selected.")
        return

    # Optional: you can still use choose_filename if you want your usual naming behavior
    save_path = choose_filename(output_dir, ".png")
    # If you prefer category in filename, we’ll rewrite the basename later.

    # 2) Choose category and collect animals
    category = _pick_category()
    wanted_names = _names_for_category(category)

    # Import your actual animal_choices from the module where it's defined
    from utils import animal_choices  # adjust import path if needed

    lookup = _build_animal_lookup(animal_choices)

    # 3) Load image via your ImageRenderer to keep behavior consistent
    renderer = ImageRenderer(filename, show_window=False, save_to=None, wait_key=0)
    renderer.open()
    src = renderer.get_image()
    renderer.close()

    if src is None:
        print("Failed to read the image.")
        return

    # Ensure RGB uint8
    src = _ensure_rgb_uint8(src)

    # 4) Run visualize for each animal in the category
    labeled_tiles: List[Tuple[str, np.ndarray]] = []
    for name in wanted_names:
        animal_obj = lookup.get(name)
        if animal_obj is None:
            print(f"[WARN] Animal '{name}' not found; skipping.")
            continue

        print(f"→ Rendering {name} ...")
        out_rgb = _run_visualize_get_output(animal_obj, src)
        if out_rgb is None:
            print(f"[WARN] {name} returned no output; skipping.")
            continue

        labeled_tiles.append((name, out_rgb))

    if not labeled_tiles:
        print("Nothing to render for this category.")
        return

    # 5) Build grid & save
    grid = build_labeled_grid(labeled_tiles, tile_height=tile_height, pad=8, bg=(20, 20, 20))
    if grid is None:
        print("Could not build grid.")
        return

    # Save as BGR for cv2.imwrite
    # If you want category in the filename, replace the chosen name’s stem.
    base_dir = os.path.dirname(save_path)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = f"gallery_{category.replace('-', '').replace(' ', '')}_{ts}.png"
    out_path = os.path.join(base_dir, out_name)

    ok = cv2.imwrite(out_path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    if not ok:
        print(f"Failed to save grid to {out_path}")
        return

    print(f"Saved gallery: {out_path}")


if __name__ == "__main__":
    app()
