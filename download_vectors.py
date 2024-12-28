import requests
import zipfile
import shutil
import tqdm
import gzip
import os


GLOVE_6B_URL = "http://nlp.stanford.edu/data/glove.6B.zip"
GLOVE_TWITTER_URL = "http://nlp.stanford.edu/data/glove.twitter.27B.zip"
FASTTEXT_CC_URL = (
    "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.vec.gz"
)
FASTTEXT_WIKI_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.vec"
FASTTEXT_SIMPLE_URL = (
    "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.simple.vec"
)
VECTORS_FOLDER = "vectors"


def download_and_extract_zipfile(url):
    filename = url.split("/")[-1]
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))

    with open(filename, "wb") as file, tqdm.tqdm(
        desc=filename, total=total_size, unit="iB", unit_scale=True, unit_divisor=1024
    ) as progress_bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            progress_bar.update(size)

    with zipfile.ZipFile(filename, "r") as zip_ref:
        files = zip_ref.namelist()
        for file in tqdm.tqdm(files, desc=f"Extracting {filename}"):
            zip_ref.extract(file, VECTORS_FOLDER)

    os.remove(filename)


def download_and_extract_gz(url):
    filename = url.split("/")[-1]
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))

    with open(filename, "wb") as file, tqdm.tqdm(
        desc=filename, total=total_size, unit="iB", unit_scale=True, unit_divisor=1024
    ) as progress_bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            progress_bar.update(size)

    with gzip.open(filename, "rb") as f_in:
        with open(
            os.path.join(VECTORS_FOLDER, ".".join(filename.split(".")[:-1])), "wb"
        ) as f_out:
            shutil.copyfileobj(f_in, f_out)

    os.remove(filename)


def download_vectors(url):
    filename = url.split("/")[-1]
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))

    with open(os.path.join(VECTORS_FOLDER, filename), "wb") as file, tqdm.tqdm(
        desc=filename, total=total_size, unit="iB", unit_scale=True, unit_divisor=1024
    ) as progress_bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            progress_bar.update(size)


def main():
    os.makedirs(VECTORS_FOLDER, exist_ok=True)
    download_and_extract_zipfile(GLOVE_TWITTER_URL)
    download_and_extract_zipfile(GLOVE_6B_URL)
    download_and_extract_gz(FASTTEXT_CC_URL)
    download_vectors(FASTTEXT_SIMPLE_URL)
    download_vectors(FASTTEXT_WIKI_URL)


if __name__ == "__main__":
    main()
