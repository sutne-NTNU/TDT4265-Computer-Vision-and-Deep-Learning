import getpass
import pathlib


def get_dataset_dir(dataset_relpath: str):
    server_dir = pathlib.Path("/work/datasets", dataset_relpath)
    if server_dir.is_dir():
        return str(server_dir)
    if server_dir.is_file():
        return str(server_dir)
    return str(pathlib.Path("data", dataset_relpath))


def get_output_dir():
    work_dir = pathlib.Path("/work", "snotra", getpass.getuser())
    save_in_work = False
    if work_dir.is_dir() and save_in_work:
        return work_dir.joinpath("ssd_outputs")
    return pathlib.Path("../outputs")
