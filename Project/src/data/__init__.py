import pathlib


def get_dataset_dir(dataset_relpath: str):
    server_dir = pathlib.Path("/work/datasets", dataset_relpath)
    if server_dir.is_dir():
        print("Found dataset directory in:", server_dir)
        return str(server_dir)
    if server_dir.is_file():
        print("Found dataset file in:", server_dir)
        return str(server_dir)
    return str(pathlib.Path("data", dataset_relpath))
