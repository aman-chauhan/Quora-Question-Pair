import requests
import zipfile
import shutil
import click
import tqdm
import gzip
import os

# Location for storing vector embeddings
VECTORS_FOLDER = "vectors"
# Available embeddings with their URLs and processing functions
EMBEDDINGS = {
    "glove-twitter": {
        "url": "http://nlp.stanford.edu/data/glove.twitter.27B.zip",
        "processor": "zip",
    },
    "glove-6b": {
        "url": "http://nlp.stanford.edu/data/glove.6B.zip",
        "processor": "zip",
    },
    "fasttext-cc": {
        "url": "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.vec.gz",
        "processor": "gz",
    },
    "fasttext-wiki": {
        "url": "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.vec",
        "processor": "direct",
    },
    "fasttext-simple": {
        "url": "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.simple.vec",
        "processor": "direct",
    },
}


def download_file(url: str) -> tuple[str, bytes]:
    """Download a file and show progress."""
    filename = url.split("/")[-1]
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))

    with open(filename, "wb") as file, tqdm.tqdm(
        desc=filename, total=total_size, unit="iB", unit_scale=True, unit_divisor=1024
    ) as progress_bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            progress_bar.update(size)
    return filename


def process_zip(filename: str):
    """Extract a zip file to the vectors folder."""
    with zipfile.ZipFile(filename, "r") as zip_ref:
        files = zip_ref.namelist()
        for file in tqdm.tqdm(files, desc=f"Extracting {filename}"):
            zip_ref.extract(file, VECTORS_FOLDER)
    os.remove(filename)


def process_gz(filename: str):
    """Extract a gz file to the vectors folder."""
    with gzip.open(filename, "rb") as f_in:
        output_file = os.path.join(VECTORS_FOLDER, ".".join(filename.split(".")[:-1]))
        with open(output_file, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    os.remove(filename)


def process_direct(filename: str):
    """Move the file directly to the vectors folder."""
    shutil.move(filename, os.path.join(VECTORS_FOLDER, filename))


PROCESSORS = {"zip": process_zip, "gz": process_gz, "direct": process_direct}


def download_and_process(name: str, url: str, processor: str):
    """Download and process a single embedding."""
    click.echo(f"\nDownloading {name}...")
    filename = download_file(url)
    click.echo(f"Processing {name}...")
    PROCESSORS[processor](filename)
    click.echo(f"Completed {name}")


@click.command()
@click.option(
    "--embeddings",
    "-e",
    multiple=True,
    type=click.Choice(list(EMBEDDINGS.keys()) + ["all"]),
    help='Embedding(s) to download. Use multiple times for multiple embeddings or "all" for all embeddings.',
)
def main(embeddings):
    """Download word embeddings from various sources."""
    if not embeddings:
        click.echo(
            "Please specify at least one embedding to download using the -e option"
        )
        return

    os.makedirs(VECTORS_FOLDER, exist_ok=True)

    if "all" in embeddings:
        embeddings_to_process = EMBEDDINGS.keys()
    else:
        embeddings_to_process = embeddings

    for name in embeddings_to_process:
        embedding = EMBEDDINGS[name]
        try:
            download_and_process(name, embedding["url"], embedding["processor"])
        except Exception as e:
            click.echo(f"Error processing {name}: {str(e)}", err=True)


if __name__ == "__main__":
    main()
