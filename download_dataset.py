import requests
import zipfile
import tqdm
import os


DATASET_URL = "https://dl.fbaipublicfiles.com/glue/data/QQP-clean.zip"
DATASET_FOLDER = "."


def main():
    filename = DATASET_URL.split("/")[-1]
    response = requests.get(DATASET_URL, stream=True)
    total_size = int(response.headers.get("content-length", 0))

    with open(filename, "wb") as file, tqdm.tqdm(
        desc=filename, total=total_size, unit="iB", unit_scale=True, unit_divisor=1024
    ) as progress_bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            progress_bar.update(size)

    with zipfile.ZipFile(filename, "r") as zip_ref:
        files = zip_ref.namelist()
        for file in tqdm.tqdm(files, desc="Extracting"):
            zip_ref.extract(file, DATASET_FOLDER)

    os.remove(filename)


if __name__ == "__main__":
    main()
