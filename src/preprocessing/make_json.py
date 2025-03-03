from pathlib import Path
import json

accepted_file_types = {".jpg", ".jpeg", ".png"}


def make_json_file(train : Path, label: Path, json_path : Path, print_progress : bool = True) -> None:
    i = 0
    mapping : dict[int, str] = {}
    sorted_dir = sorted(train.iterdir())
    nbr_images = 0
    for img in sorted_dir:
        if img.suffix.lower() in accepted_file_types:
            nbr_images += 1
    
    for train in sorted_dir:
        if train.suffix.lower() in accepted_file_types: 
            mapping[i] = train.stem
            i += 1
            if print_progress:
                print(f"file {i}/{nbr_images}")
    output = json.dumps(mapping)
    with open(json_path, "w") as file:
        file.seek(0) ## to overwrite
        file.write(output)
    return

