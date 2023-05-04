from pathlib import Path
from urllib.error import URLError

import torch
from torchvision.datasets.utils import download_and_extract_archive, download_url

from imagined_handwriting.settings import DOWNLOAD_DIR


class HandwritingDownload:
    """A base class for pytorch datasets which handle downloading the raw data."""

    mirrors = [
        "https://datadryad.org/stash/downloads/file_stream/662811",
    ]
    resources = [
        ("handwritingBCIData.tar.gz", "38b804d8bd271bd5981c7fda7f925eb8"),
    ]

    corpus_mirrors = [
        "https://raw.githubusercontent.com/fwillett/handwritingBCI/main/wordList/google-10000-english-usa.txt"  # noqa
    ]
    corpus_resources = [
        ("google-10000-english-usa.txt", "b05d7e9f8a05a70fec8cd54101fe19ad")
    ]

    def __init__(
        self,
        root: str,
        download: bool = False,
    ):
        """Initialize the dataset.

        Args:
            root: The root directory of the dataset.
            train: Whether to load the training or test set. The splits are
                according to the "HeldOutBlocks" scheme of the original dataset.
                See the paper for details.
            transform: A function to apply to the data.
            download: Whether to download the dataset if it doesn't exist.
        """
        if isinstance(root, torch._six.string_classes):
            root = str(Path(root).resolve())
        self.root = root
        if download:
            self.download()

        if not self.check_exists():
            raise RuntimeError(
                "Dataset not found. You can use download=True to download it"
            )
        if not self._check_corpus_exists():
            raise RuntimeError(
                "Corpus not found. You can use download=True to download it"
            )

    @property
    def raw_folder(self) -> str:
        return str(Path(self.root) / DOWNLOAD_DIR)

    def check_exists(self) -> bool:
        return all(
            (Path(self.raw_folder) / _true_stem(url)).exists()
            for url, _ in self.resources
        )

    def download(self) -> None:
        self.download_raw()
        self.download_corpus()

    def download_raw(self) -> None:
        """Download the Handwriting data if it doesn't exist already."""

        if self.check_exists():
            return

        Path(self.raw_folder).mkdir(parents=True, exist_ok=True)

        # download files
        for filename, md5 in self.resources:
            for mirror in self.mirrors:
                url = mirror
                try:
                    print(f"Downloading {url}")
                    download_and_extract_archive(
                        url, download_root=self.raw_folder, filename=filename, md5=md5
                    )
                except URLError as error:
                    print(f"Failed to download (trying next):\n{error}")
                    continue
                finally:
                    print()
                break
            else:
                raise RuntimeError(f"Error downloading {filename}")

    def download_corpus(self):
        """Download the corpus."""
        if self._check_corpus_exists():
            return

        Path(self.raw_folder).mkdir(parents=True, exist_ok=True)

        # download files
        for filename, md5 in self.corpus_resources:
            for mirror in self.corpus_mirrors:
                url = mirror
                try:
                    print(f"Downloading {url}")
                    download_url(url, root=self.raw_folder, filename=filename, md5=md5)
                except URLError as error:
                    print(f"Failed to download (trying next):\n{error}")
                    continue
                finally:
                    print()
                break
            else:
                raise RuntimeError(f"Error downloading {filename}")

    def _check_corpus_exists(self):
        """Check that the corpus exists."""
        return all(
            Path(self.raw_folder, filename).exists()
            for filename, _ in self.corpus_resources
        )


def _true_stem(url: str) -> str:
    stem = Path(url).stem
    return stem if stem == url else _true_stem(stem)


def download(root: str) -> None:
    HandwritingDownload(root, download=True)
