import os
from typing import List, Tuple

from InquirerPy import inquirer


def choose_file(input_dir: str, extensions: Tuple[str, ...]) -> str:
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
    save_name: str = inquirer.text(  # type: ignore[reportPrivateImportUsage]
        message="Press [Enter] without inputing text to not save. \nEnter the filename to save as:", default=""
    ).execute()

    if not save_name or save_name == "":
        return ""
    save_path = os.path.join(input_dir, f"{save_name}{extension}")
    return save_path
