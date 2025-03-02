import os
import logging
import urllib.request
import tarfile
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Union, Dict, List, Tuple
import numpy as np
from pydantic import BaseModel, Field,field_validator, HttpUrl
from typing import Optional, Union, Dict, List, Tuple, ClassVar


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

#pydantic model for configuration
class MovieCorpusAnalyzerConfig(BaseModel): 
    """
    Configuration model for MovieCorpusAnalyzer.
    
    This Pydantic model validates the configuration parameters for the MovieCorpusAnalyzer.
    """
    data_url: HttpUrl = Field(
        default="http://www.cs.cmu.edu/~ark/personas/data/MovieSummaries.tar.gz",
        description="URL to download the dataset from"
    )
    download_dir: str = Field(
        default="downloads", 
        description="Directory to store downloaded and extracted files"
    )
    
    @field_validator('download_dir')
    def directory_must_be_valid(cls, v):
        if not isinstance(v, str) or not v:
            raise ValueError("download_dir must be a non-empty string")
        return v

class MovieCorpusAnalyzer:
    """
    A class for analyzing the CMU Movie Corpus dataset.
    
    This class provides functionality to download, extract and analyze data from
    the CMU Movie Corpus, particularly focusing on character metadata.
    
    Attributes:
        data_url (str): URL to download the dataset.
        download_dir (str): Directory path to store downloaded files.
        archive_path (str): Path to the downloaded archive file.
        data_path (str): Path to the extracted character metadata file.
        data (pd.DataFrame): DataFrame containing the character metadata.
        column_names (List[str]): Names of columns in the character metadata.
    """
    #column names as ClassVar for type safety
    column_names: ClassVar[List[str]] = [
        "wikipedia_movie_id", "freebase_movie_id", "release_date", "character_name",
        "actor_dob", "actor_gender", "actor_height", "actor_ethnicity",
        "actor_name", "actor_age_at_release", "freebase_char_actor_map_id",
        "freebase_character_id", "freebase_actor_id" 
    ]
    
    def __init__(self, 
                config: Optional[MovieCorpusAnalyzerConfig] = None,
                **kwargs) -> None:
        """
        Initialize the MovieCorpusAnalyzer with data from the CMU Movie Corpus.
        
        Args:
            config (MovieCorpusAnalyzerConfig, optional): Configuration for the analyzer.
            **kwargs: Additional keyword arguments to create or override the configuration.
        """
        # Create config from provided parameters or defaults
        if config is None:
            config_dict = kwargs
            self.config = MovieCorpusAnalyzerConfig(**config_dict)
        else:
            self.config = config
            
        # Set paths based on config
        self.archive_path = os.path.join(self.config.download_dir, "MovieSummaries.tar.gz")
        self.data_path = os.path.join(self.config.download_dir, "character.metadata.tsv")

        # Create download directory if it doesn't exist
        os.makedirs(self.config.download_dir, exist_ok=True)
        
        # Download data if needed
        self._download_data()
        
        # Extract data if needed
        self._extract_data()
        
        # Load data
        self._load_data()
    
    def _download_data(self) -> None:
        """
        Download the dataset archive if it doesn't exist locally.
        """
        if not os.path.exists(self.archive_path):
            logger.info(f"Downloading dataset from {self.data_url}...")
            try:
                urllib.request.urlretrieve(self.data_url, self.archive_path)
                logger.info("Download completed successfully.")
            except Exception as e:
                logger.error(f"Failed to download dataset: {e}")
                raise
        else:
            logger.info("Dataset archive already exists. Skipping download.")
    
    def _extract_data(self) -> None:
        """
        Extract the required dataset file from the archive if it doesn't exist.
        """
        if not os.path.exists(self.data_path):
            logger.info("Extracting character metadata from archive...")
            try:
                with tarfile.open(self.archive_path, "r:gz") as tar:
                    # Extract only the needed file
                    for member in tar.getmembers():
                        if member.name.endswith("character.metadata.tsv"):
                            member.name = os.path.basename(member.name)  # Remove directory structure
                            tar.extract(member, path=self.config.download_dir)
                            logger.info("Extraction completed successfully.")
                            break
                    else:
                        raise FileNotFoundError("character.metadata.tsv not found in archive")
            except Exception as e:
                logger.error(f"Failed to extract dataset: {e}")
                raise
        else:
            logger.info("Character metadata file already exists. Skipping extraction.")
    
    def _load_data(self) -> None:
        """
        Load the character metadata into a pandas DataFrame.
        """
        logger.info("Loading character metadata into DataFrame...")
        try:
            self.data = pd.read_csv(
                self.data_path, 
                sep="\t", 
                header=None, 
                names=self.column_names,
                na_values=["", "NA", "N/A"],
                encoding='utf-8'
            )
            
            # Convert data types
            self.data["release_date"] = pd.to_datetime(self.data["release_date"], errors="coerce")
            self.data["actor_dob"] = pd.to_datetime(self.data["actor_dob"], errors="coerce")
            self.data["actor_height"] = pd.to_numeric(self.data["actor_height"], errors="coerce")
            self.data["actor_age_at_release"] = pd.to_numeric(self.data["actor_age_at_release"], errors="coerce")
            
            logger.info(f"Loaded {len(self.data)} records.")
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise
    
class MovieTypeResult(BaseModel):
    """Model for movie type analysis results."""
    movie_type: str = Field(..., alias="Movie_Type")
    count: int = Field(..., alias="Count")
    
    class Config:
        allow_population_by_field_name = True

class ActorCountResult(BaseModel):
    """Model for actor count results."""
    number_of_actors: int = Field(..., alias="Number_of_Actors")
    movie_count: int = Field(..., alias="Movie_Count")
    
    class Config:
        allow_population_by_field_name = True

class HeightDistributionResult(BaseModel):
    """Model for height distribution results."""
    height: float
    count: int
    
    class Config:
        allow_population_by_field_name = True

class MovieCorpusAnalyzer(MovieCorpusAnalyzer):  # Extend the previous class
    """Extended MovieCorpusAnalyzer with Pydantic-validated methods."""

    def movie_type(self, N: int = 10) -> pd.DataFrame:
        """
        Calculate the N most common types of movies in the dataset.
        
        Args:
            N (int): Number of most common movie types to return. Defaults to 10.
            
        Returns:
            pd.DataFrame: DataFrame with columns "Movie_Type" and "Count".
            
        Raises:
            TypeError: If N is not an integer.
        """
        if not isinstance(N, int):
            raise TypeError("Parameter N must be an integer")
        
        # Count movies by their Wikipedia ID
        movie_counts = self.data.groupby("wikipedia_movie_id").size().reset_index()
        movie_counts.columns = ["wikipedia_movie_id", "Count"]
        
        # Get the N most common types
        top_movies = movie_counts.sort_values("Count", ascending=False).head(N)
        
        # Create a DataFrame with movie types
        movie_types = pd.DataFrame({
            "Movie_Type": top_movies["wikipedia_movie_id"].astype(str),
            "Count": top_movies["Count"]
        })
        
        return movie_types
    
    def actor_count(self) -> pd.DataFrame:
        """
        Calculate a histogram of number of actors per movie.
        
        Returns:
            pd.DataFrame: DataFrame with columns "Number_of_Actors" and "Movie_Count".
        """
        # Count the number of actors per movie
        actors_per_movie = self.data.groupby("wikipedia_movie_id").size().reset_index()
        actors_per_movie.columns = ["wikipedia_movie_id", "Number_of_Actors"]
        
        # Create a histogram
        actor_count_histogram = actors_per_movie.groupby("Number_of_Actors").size().reset_index()
        actor_count_histogram.columns = ["Number_of_Actors", "Movie_Count"]
        
        # Sort by number of actors
        actor_count_histogram = actor_count_histogram.sort_values("Number_of_Actors")
        
        return actor_count_histogram
    
    def actor_distributions(self, 
                          gender: str = "All", 
                          min_height: float = 0, 
                          max_height: float = 3, 
                          plot: bool = False) -> pd.DataFrame:
        """
        Calculate height distributions for actors based on gender and height range.
        
        Args:
            gender (str): Filter by actor gender. Use "All" for all genders, "F" for female and "M" for male.
            min_height (float): Minimum actor height in meters.
            max_height (float): Maximum actor height in meters.
            plot (bool): Whether to generate a matplotlib plot. Defaults to False.
            
        Returns:
            pd.DataFrame: DataFrame with height distribution data.
            
        Raises:
            TypeError: If gender is not a string or if heights are not numeric.
            ValueError: If min_height is greater than max_height or if height values are unrealistic.
        """
        # Type checking
        if not isinstance(gender, str):
            raise TypeError("Parameter gender must be a string")
        if not (isinstance(min_height, (int, float)) and isinstance(max_height, (int, float))):
            raise TypeError("Height parameters must be numeric values")
        
        # Validate height values (realistic human heights in meters)
        if min_height < 0:
            raise ValueError("Minimum height cannot be negative")
        if max_height > 3:
            raise ValueError("Maximum height exceeds realistic human height (3 meters)")
        if min_height > max_height:
            raise ValueError("Minimum height cannot be greater than maximum height")
        
        # Filter data based on parameters
        filtered_data = self.data.copy()
        filtered_data = filtered_data.dropna(subset=["actor_height"])
        
        if gender != "All":
            filtered_data = filtered_data[filtered_data["actor_gender"] == gender]
        
        filtered_data = filtered_data[
            (filtered_data["actor_height"] >= min_height) & 
            (filtered_data["actor_height"] <= max_height)
        ]
        
        # Create height bins
        bins = np.linspace(min_height, max_height, 20)
        labels = [(bins[i] + bins[i+1])/2 for i in range(len(bins)-1)]
        
        # Create distribution DataFrame
        filtered_data["height_bin"] = pd.cut(filtered_data["actor_height"], bins=bins, labels=labels)
        height_dist = filtered_data.groupby("height_bin").size().reset_index()
        height_dist.columns = ["Height", "Count"]
        
        # Generate plot if requested
        if plot:
            plt.figure(figsize=(10, 6))
            plt.bar(height_dist["Height"].astype(float), height_dist["Count"], width=0.05)
            plt.xlabel("Height (meters)")
            plt.ylabel("Count")
            plt.title(f"Actor Height Distribution - Gender: {gender}")
            plt.grid(axis="y", alpha=0.3)
            plt.show()
        
        return height_dist
        