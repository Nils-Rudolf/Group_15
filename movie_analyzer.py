import os
import zipfile
import requests
import pandas as pd
from pathlib import Path
from pydantic import BaseModel

class MovieAnalyzer(BaseModel):
    """
    A class to handle downloading, unzipping, and loading a movie dataset.
    ---
    Attributes: 
    data_url (str): URL of the dataset to be downloaded.
    downloads_dir (Path): Directory where the dataset will be stored.
    file_name (Path): Path to the downloaded dataset file.
    extracted_dir (Path): Path to the extracted dataset folder.
    character_metadata_df (pd.DataFrame): DataFrame containing character metadata.
    movie_metadata_df (pd.DataFrame): DataFrame containing movie metadata.
    name_clusters_df (pd.DataFrame): DataFrame containing name clusters.
    plot_summaries_df (pd.DataFrame): DataFrame containing plot summaries.
    tvtropes_clusters_df (pd.DataFrame): DataFrame containing TV tropes clusters.
    ---
    Methods:
    ensure_downloaded(): Ensures the dataset is downloaded.
    unzip_dataset(): Extracts the dataset if it is compressed.
    load_dataset(): Loads the dataset into Pandas DataFrames.
    """
    data_url: str
    downloads_dir: Path = Path("downloads")
    file_name: Path = downloads_dir / "movie_dataset.zip"
    extracted_dir: Path = downloads_dir / "movie_data"

    class Config:
        arbitrary_types_allowed = True
        
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ensure_downloaded()
        self.unzip_dataset()
        self.load_dataset()

    def ensure_downloaded(self) -> None:
        """
        Ensures the dataset is downloaded. If not, downloads it.
        ---
        Parameters:
        None
        ---
        Returns:
        None
        """
        if not self.file_name.exists():
            print("Downloading dataset, this will not take long...")
            response = requests.get(self.data_url, stream=True)
            with open(self.file_name, "wb") as x:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        x.write(chunk)
            print("Download complete")
        else:
            print("Dataset already exists. Skipping download.")

    def unzip_dataset(self) -> None:
        """
        Extracts the dataset if it has not been extracted yet.
        ---
        Parameters:
        None
        ---
        Returns:
        None
        """
        if not self.extracted_dir.exists():
            print("Extracting dataset, this will not take long...")
            with zipfile.ZipFile(self.file_name, "r") as zip_ref:
                zip_ref.extractall(self.downloads_dir)
            print("Extraction complete.")
        else:
            print("Dataset already extracted.")

    def load_dataset(self) -> None:
        """
        Loads dataset files into Pandas DataFrames as class attributes.
        ---
        Parameters:
        None
        ---
        Returns:
        None
        """
        self.character_metadata_df = pd.read_csv(self.extracted_dir / "character.metadata.tsv", sep='\t', header=None)
        self.movie_metadata_df = pd.read_csv(self.extracted_dir / "movie.metadata.tsv", sep='\t', header=None)
        self.name_clusters_df = pd.read_csv(self.extracted_dir / "name.clusters", sep='\t', header=None, on_bad_lines='skip')
        self.plot_summaries_df = pd.read_csv(self.extracted_dir / "plot_summaries", sep='\t', header=None, on_bad_lines='skip')
        self.tvtropes_clusters_df = pd.read_csv(self.extracted_dir / "tvtropes.clusters", sep='\t', header=None, on_bad_lines='skip')
        print("Data successfully loaded into DataFrames.")
