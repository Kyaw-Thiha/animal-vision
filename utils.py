import os

import base64
from typing import List, Tuple
import cv2
from InquirerPy import inquirer

from animals.anchovy import Anchovy
from animals.animal import Animal
from animals.damselfish import Damselfish
from animals.goldfish import Goldfish
from animals.guppy import Guppy
from animals.heliconius import Heliconius
from animals.honeybee import HoneyBee
from animals.jumping_spider import JumpingSpider
from animals.kestrel import Kestrel
from animals.mantis_shrimp import MantisShrimp
from animals.morpho import Morpho
from animals.pieris import Pieris
from animals.reindeer import Reindeer
from renderers.video import VideoRenderer
from animals import (
    Cat,
    Dog,
    Sheep,
    Pig,
    Goat,
    Cow,
    Rat,
    Horse,
    Rabbit,
    Panda,
    Squirrel,
    Elephant,
    Lion,
    Wolf,
    Fox,
    Bear,
    Raccoon,
    Deer,
    Kangaroo,
    Tiger,
    Goldfish,
    RatUV,
    Damselfish,
    Anableps,
)


renderer = VideoRenderer()
catfilter = Cat()
dogfilter = Dog()
sheepfilter = Sheep()
pigfilter = Pig()
goatfilter = Goat()
cowfilter = Cow()
ratfilter = Rat()
horsefilter = Horse()
squirrelfilter = Squirrel()
elephantfilter = Elephant()
lionfilter = Lion()
wolffilter = Wolf()
foxfiler = Fox()
bearfilter = Bear()
raccoonfilter = Raccoon()
deerfilter = Deer()
kangaroofilter = Kangaroo()
tigerfilter = Tiger()
rabbitfilter = Rabbit()
pandafilter = Panda()


def processimage(imagedata: bytes, animal: str) -> str:
    """
    Takes the raw bytes of a image, and returns the specific animal it wants
    """
    # save image to file, and then convert that file into matrix
    f = open("temp.jpg", "wb")
    f.write(imagedata)
    f.close()
    img = cv2.imread("temp.jpg")
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # apply filters
    mmat = None
    match animal:
        case "human":
            mmat = img
        case "cat":
            mmat = catfilter.visualize(img)[1]
        case "cow":
            mmat = cowfilter.visualize(img)[1]
        case "goat":
            mmat = goatfilter.visualize(img)[1]
        case "pig":
            mmat = pigfilter.visualize(img)[1]
        case "sheep":
            mmat = sheepfilter.visualize(img)[1]
        case "dog":
            mmat = dogfilter.visualize(img)[1]
        case "rat":
            mmat = ratfilter.visualize(img)[1]
        case "horse":
            mmat = horsefilter.visualize(img)[1]
        case "rabbit":
            mmat = rabbitfilter.visualize(img)[1]
        case "panda":
            mmat = pandafilter.visualize(img)[1]
        case "squirrel":
            mmat = squirrelfilter.visualize(img)[1]
        case "elephant":
            mmat = elephantfilter.visualize(img)[1]
        case "lion":
            mmat = lionfilter.visualize(img)[1]
        case "wolf":
            mmat = wolffilter.visualize(img)[1]
        case "fox":
            mmat = foxfiler.visualize(img)[1]
        case "bear":
            mmat = bearfilter.visualize(img)[1]
        case "raccoon":
            mmat = raccoonfilter.visualize(img)[1]
        case "deer":
            mmat = deerfilter.visualize(img)[1]
        case "kangaroo":
            mmat = kangaroofilter.visualize(img)[1]
        case "tiger":
            mmat = tigerfilter.visualize(img)[1]
        case _:
            print("no case implemented here")
    # convert mmat into blob, to make base64 URI
    cv2.imwrite(f"tempexport.jpg", mmat)
    f = open("tempexport.jpg", "rb")
    imgdata = f.read()
    f.close()
    base64_encoded = base64.b64encode(imgdata).decode("utf-8")
    data_uri = f"data:image/jpeg;base64,{base64_encoded}"
    return data_uri


def processsplitimage(imagedata: bytes, animal: str) -> str:
    """
    Takes the data URL of a image, and returns the split of a single animal
    """
    header, encoded = imagedata.split(",", 1)
    contents = base64.b64decode(encoded)
    # save image to file, and then convert that file into matrix
    f = open("temp.jpg", "wb")
    f.write(contents)
    f.close()
    img = cv2.imread("temp.jpg")
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # apply filters
    mmat = None
    match animal:
        case "human":
            mmat = img
        case "cat":
            orig, modified = catfilter.visualize(img)
            mmat = renderer.make_split_frame(orig, modified)
        case "cow":
            orig, modified = cowfilter.visualize(img)
            mmat = renderer.make_split_frame(orig, modified)
        case "goat":
            orig, modified = goatfilter.visualize(img)
            mmat = renderer.make_split_frame(orig, modified)
        case "pig":
            orig, modified = pigfilter.visualize(img)
            mmat = renderer.make_split_frame(orig, modified)
        case "sheep":
            orig, modified = sheepfilter.visualize(img)
            mmat = renderer.make_split_frame(orig, modified)
        case "dog":
            orig, modified = dogfilter.visualize(img)
            mmat = renderer.make_split_frame(orig, modified)
        case "rat":
            orig, modified = ratfilter.visualize(img)
            mmat = renderer.make_split_frame(orig, modified)
        case "horse":
            orig, modified = horsefilter.visualize(img)
            mmat = renderer.make_split_frame(orig, modified)
        case "rabbit":
            orig, modified = rabbitfilter.visualize(img)
            mmat = renderer.make_split_frame(orig, modified)
        case "panda":
            orig, modified = pandafilter.visualize(img)
            mmat = renderer.make_split_frame(orig, modified)
        case "squirrel":
            orig, modified = squirrelfilter.visualize(img)
            mmat = renderer.make_split_frame(orig, modified)
        case "elephant":
            orig, modified = elephantfilter.visualize(img)
            mmat = renderer.make_split_frame(orig, modified)
        case "lion":
            orig, modified = lionfilter.visualize(img)
            mmat = renderer.make_split_frame(orig, modified)
        case "wolf":
            orig, modified = wolffilter.visualize(img)
            mmat = renderer.make_split_frame(orig, modified)
        case "fox":
            orig, modified = foxfiler.visualize(img)
            mmat = renderer.make_split_frame(orig, modified)
        case "bear":
            orig, modified = bearfilter.visualize(img)
            mmat = renderer.make_split_frame(orig, modified)
        case "raccoon":
            orig, modified = raccoonfilter.visualize(img)
            mmat = renderer.make_split_frame(orig, modified)
        case "deer":
            orig, modified = deerfilter.visualize(img)
            mmat = renderer.make_split_frame(orig, modified)
        case "kangaroo":
            orig, modified = kangaroofilter.visualize(img)
            mmat = renderer.make_split_frame(orig, modified)
        case "tiger":
            orig, modified = tigerfilter.visualize(img)
            mmat = renderer.make_split_frame(orig, modified)
        case "honeybee":
            orig, modified = honeybeefilter.visualize(img)
            mmat = renderer.make_split_frame(orig, modified)
        case _:
            print("no case implemented here")
    # convert mmat into blob, to make base64 URI
    cv2.imwrite(f"tempexport.jpg", mmat)
    f = open("tempexport.jpg", "rb")
    imgdata = f.read()
    f.close()
    base64_encoded = base64.b64encode(imgdata).decode("utf-8")
    data_uri = f"data:image/jpeg;base64,{base64_encoded}"
    return data_uri


def choose_file(input_dir: str, extensions: Tuple[str, ...]) -> str:
    """
    Interactively choose a file from a directory that matches given extensions.

    Scans `input_dir` for files ending with any of `extensions`, then shows
    a fuzzy-search menu for the user to select one.

    Args:
        input_dir (str): Directory to search for files.
        extensions (Tuple[str, ...]): File extensions to filter, e.g. (".jpg", ".png").

    Returns:
        str: Full path to the selected file.
    """
    files: List[str] = []
    for fname in os.listdir(input_dir):
        if fname.lower().endswith(extensions):
            files.append(os.path.join(input_dir, fname))

    choices = [{"name": f, "value": f} for f in files]
    selected_file = inquirer.fuzzy(  # type: ignore[reportPrivateImportUsage]
        message="Select which file you want to load: ",
        choices=choices,
        multiselect=False,
    ).execute()

    return selected_file


def choose_filename(input_dir: str, extension: str) -> str:
    """
    Prompt the user for a filename to save under a given extension.

    If the user presses Enter without typing anything, returns an empty string.

    Args:
        input_dir (str): Directory to save the file in.
        extension (str): File extension to append, e.g. ".png".

    Returns:
        str: Full save path with extension, or "" if skipped.
    """
    save_name: str = inquirer.text(  # type: ignore[reportPrivateImportUsage]
        message="Press [Enter] without inputing text to not save. \nEnter the filename to save as:", default=""
    ).execute()

    if not save_name or save_name == "":
        return ""
    save_path = os.path.join(input_dir, f"{save_name}{extension}")
    return save_path


def choose_animal() -> Animal:
    animal_choices = [
        {"name": "Cat", "value": Cat()},
        {"name": "Dog", "value": Dog()},
        {"name": "Sheep", "value": Sheep()},
        {"name": "Pig", "value": Pig()},
        {"name": "Goat", "value": Goat()},
        {"name": "Cow", "value": Cow()},
        {"name": "Horse", "value": Horse()},
        {"name": "Rabbit", "value": Rabbit()},
        {"name": "Panda", "value": Panda()},
        {"name": "Squirrel", "value": Squirrel()},
        {"name": "Elephant", "value": Elephant()},
        {"name": "Lion", "value": Lion()},
        {"name": "Wolf", "value": Wolf()},
        {"name": "Fox", "value": Fox()},
        {"name": "Bear", "value": Bear()},
        {"name": "Raccoon", "value": Raccoon()},
        {"name": "Deer", "value": Deer()},
        {"name": "Kangaroo", "value": Kangaroo()},
        {"name": "Tiger", "value": Tiger()},
        {"name": "Rat", "value": Rat()},
        # UV based animals
        {"name": "HoneyBee", "value": HoneyBee()},
        {"name": "ReinDeer", "value": Reindeer()},
        {"name": "RatUV", "value": RatUV()},
        {"name": "GoldFish", "value": Goldfish()},
        {"name": "DamselFish", "value": Damselfish()},
        {"name": "Anableps (Four-eyed fish)", "value": Anableps()},
        {"name": "Northern Anchovy Fish", "value": Anchovy()},
        {"name": "Guppy Fish", "value": Guppy()},
        {"name": "Morpho Butterfly", "value": Morpho()},
        {"name": "Heliconius Butterfly", "value": Heliconius()},
        {"name": "Pieris Butterfly", "value": Pieris()},
        # UV Unique Animals
        {"name": "Mantis Shrimp", "value": MantisShrimp()},
        {"name": "Kestrel", "value": Kestrel()},
        {"name": "Jumping Spider", "value": JumpingSpider()},
    ]
    animal_choice = inquirer.select(  # type: ignore[reportPrivateImportUsage]
        message="Select which animal you want to visualize:",
        choices=animal_choices,
        default=0,
    ).execute()

    return animal_choice
