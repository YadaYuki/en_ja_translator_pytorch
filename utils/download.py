import io
import tarfile
from os.path import exists, join

import requests
from const.path import CORPUS_PATH


def download_corpus() -> None:
    url = "http://www.phontron.com/kftt/download/kftt-data-1.0.tar.gz"
    if not exists(join(CORPUS_PATH, "kftt-data-1.0")):
        print("Downloading corpus...")
        data = io.BytesIO(requests.get(url).content)
        with tarfile.open(fileobj=data) as tar:
            tar.extractall(CORPUS_PATH)
    else:
        print("Corpus already exists.")


if __name__ == "__main__":
    download_corpus()
