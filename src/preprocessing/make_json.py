from pathlib import Path
import json
import argparse

accepted_file_types = {".jpg", ".jpeg", ".png"}


def main(train_path : Path, json_path : Path) -> None:
    i = 0
    mapping : dict[int, str] = {}
    sorted_dir = sorted(train_path.iterdir())
    for train in sorted_dir:
        if train.suffix.lower() in accepted_file_types: 
            mapping[i] = train.stem
            i += 1
    output = json.dumps(mapping)
    with open(json_path, "w") as file:
        file.write(output)
    return



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--json_path', default=None, type=str, help="path json file to overwrite and save mapping to")
    parser.add_argument('--data_path', default=None, type=str, help="path to your dataset")
    args = parser.parse_args()

    if args.json_path is None or args.data_path is None:
        raise ValueError("Must enter a path to json file, and a path to data directoy")
    main(Path(args.json_path), Path(args.data_path))