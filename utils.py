import os
from typing import List, Tuple
import cv2
from InquirerPy import inquirer

from animals.animal import Animal
from animals import Cat, Dog

def processimage(imagedata: bytes, animal: str) -> bytes:
    """
    Takes the raw bytes of a image, and the specific animal it wants
    """
    # save image to file, and then convert that file into matrix
    f = open("temp.jpg", 'wb')
    f.write(imagedata)
    f.close()
    img = cv2.imread('a.jpg',0)

    return imagedata

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
    ]
    animal_choice = inquirer.select(  # type: ignore[reportPrivateImportUsage]
        message="Select which animal you want to visualize:",
        choices=animal_choices,
        default=0,
    ).execute()

    return animal_choice
