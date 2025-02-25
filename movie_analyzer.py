import os
import requests
import zipfile
import pandas as pd
from typing import Dict
from pydantic import BaseModel


class MovieAnalyzer(BaseModel):
    """
    A brief summary of the class's purpose and behavior.

    Attributes:
        attr1 (type): Description of attr1.
        attr2 (type): Description of attr2.

    Methods:
        method1: Brief description of method1.
        method2: Brief description of method2.
    """
    
    data_url: str
    file_name: str
    downloads_dir: str = "downloads"
    dataframes: Dict[str, pd.DataFrame] = {}

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, data_url: str, file_name: str):
        """Initializes the MovieAnalyzer instance."""
        super().__init__(data_url=data_url, file_name=file_name)

        self.file_path = os.path.join(self.downloads_dir, self.file_name)
        self.unzip_folder = os.path.join(self.downloads_dir, self.file_name.replace(".zip", ""))

        if not os.path.exists(self.file_path):
            self.download_data()

        if self.file_name.endswith(".zip") and not os.path.exists(self.unzip_folder):
            self.unzip_data()

        self.load_data()

    def download_data(self) -> None:
        """Downloads the dataset if it is not already present."""
        print(f"Downloading {self.file_name}...")

        try:
            response = requests.get(self.data_url, stream=True)
            response.raise_for_status()

            with open(self.file_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)

            print("Download completed.")

        except requests.RequestException as e:
            print(f"Error downloading file: {e}")

    def unzip_data(self) -> None:
        """Unzips the dataset if it is a ZIP archive."""
        print(f"Unzipping {self.file_name}...")

        try:
            with zipfile.ZipFile(self.file_path, "r") as zip_ref:
                zip_ref.extractall(self.unzip_folder)
            print("Unzipping completed.")

        except zipfile.BadZipFile:
            print(f"Error: {self.file_name} is not a valid ZIP file.")

    def load_data(self) -> None:
        """Loads CSV files into pandas DataFrames."""
        self.dataframes = {}

        if os.path.exists(self.unzip_folder):
            for file in os.listdir(self.unzip_folder):
                if file.endswith(".csv"):
                    file_path = os.path.join(self.unzip_folder, file)
                    df_name = os.path.splitext(file)[0]

                    try:
                        self.dataframes[df_name] = pd.read_csv(file_path)
                        print(f"Loaded {file} into DataFrame.")

                    except Exception as e:
                        print(f"Error loading {file}: {e}")
