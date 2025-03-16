import os
import logging
import urllib.request
import tarfile
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Union, Dict, List, Tuple
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

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
    
    def __init__(self, 
            data_url: str = "http://www.cs.cmu.edu/~ark/personas/data/MovieSummaries.tar.gz",
            download_dir: str = "downloads") -> None:
   
        self.data_url = data_url
        self.download_dir = download_dir
        self.archive_path = os.path.join(download_dir, "MovieSummaries.tar.gz")
        self.data_path = os.path.join(download_dir, "character.metadata.tsv")
        self.movie_metadata_path = os.path.join(download_dir, "movie.metadata.tsv")
        self.plot_summaries_path = os.path.join(download_dir, "plot_summaries.txt")
        self.column_names = [
            "wikipedia_movie_id", "freebase_movie_id", "release_date", "character_name",
            "actor_dob", "actor_gender", "actor_height", "actor_ethnicity",
            "actor_name", "actor_age_at_release", "freebase_char_actor_map_id",
            "freebase_character_id", "freebase_actor_id"
        ]
        
        self.movie_metadata_columns = [
            "wikipedia_movie_id", "freebase_movie_id", "movie_name", "movie_release_date",
            "box_office", "runtime", "languages", "countries", "genres"
        ]
        
        # Create download directory if it doesn't exist
        os.makedirs(download_dir, exist_ok=True)

        # Download data if needed
        try:
            self._download_data()
        except Exception as e:
            logger.warning(f"Standard download failed: {str(e)}")
            if not self._download_with_wget():
                raise
        
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

    def _download_with_wget(self) -> bool:
        """
        Alternative download method using wget if available
        """
        try:
            import subprocess
            logger.info("Attempting download with wget...")
            subprocess.run(["wget", "-O", self.archive_path, self.data_url], check=True)
            return True
        except (ImportError, subprocess.SubprocessError):
            logger.warning("wget download failed or not available")
            return False
    
    def _extract_data(self) -> None:
        """
        Extract the required dataset files from the archive if they don't exist.
        """
        needed_files = ["character.metadata.tsv", "movie.metadata.tsv", "plot_summaries.txt"]
        files_to_extract = []
        
        for file in needed_files:
            file_path = os.path.join(self.download_dir, file)
            if not os.path.exists(file_path):
                files_to_extract.append(file)
                
        if files_to_extract:
            logger.info(f"Extracting files from archive: {files_to_extract}")
            try:
                with tarfile.open(self.archive_path, "r:gz") as tar:
                    # Print available files for debugging
                    logger.info("Files in archive:")
                    for member in tar.getmembers():
                        logger.info(f"  - {member.name}")
                        
                    # More robust extraction logic
                    for member in tar.getmembers():
                        basename = os.path.basename(member.name)
                        if basename in needed_files:
                            # Set the extraction path to just the basename
                            extracted_name = basename
                            member.name = basename
                            tar.extract(member, path=self.download_dir)
                            logger.info(f"Extracted {extracted_name} successfully.")
            except Exception as e:
                logger.error(f"Failed to extract dataset: {e}")
                raise
        else:
            logger.info("All required files already exist. Skipping extraction.")
    
    def _load_data(self) -> None:
        """
        Load the character and movie metadata into pandas DataFrames.
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
            
            logger.info(f"Loaded {len(self.data)} character records.")
            
            # Load movie metadata
            logger.info("Loading movie metadata into DataFrame...")
            self.movie_metadata = pd.read_csv(
                self.movie_metadata_path,
                sep="\t",
                header=None,
                names=self.movie_metadata_columns,
                na_values=["", "NA", "N/A"],
                encoding='utf-8'
            )
            
            # Convert data types for movie metadata
            self.movie_metadata["movie_release_date"] = pd.to_datetime(
                self.movie_metadata["movie_release_date"], 
                errors="coerce"
            )
            
            logger.info(f"Loaded {len(self.movie_metadata)} movie records.")
            
            # Load plot summaries
            if os.path.exists(self.plot_summaries_path):
                logger.info("Loading plot summaries into DataFrame...")
                self.summaries = pd.read_csv(
                    self.plot_summaries_path, 
                    sep="\t", 
                    header=None, 
                    names=["wikipedia_movie_id", "summary"],
                    encoding='utf-8'
                )
                logger.info(f"Loaded {len(self.summaries)} plot summaries.")
            else:
                logger.warning("Plot summaries file not found. Summaries will be unavailable.")
                self.summaries = pd.DataFrame(columns=["wikipedia_movie_id", "summary"])
                
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise
    
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
    
    def releases(self, genre: Optional[str] = None) -> pd.DataFrame:
        """
        Calculate movie releases per year, optionally filtered by genre.
        
        Args:
            genre (str, optional): The genre to filter by. If None, includes all movies.
            
        Returns:
            pd.DataFrame: DataFrame with years and movie counts
        """
        # Extract release years from dates
        self.movie_metadata['year'] = pd.to_datetime(
            self.movie_metadata['movie_release_date'], 
            errors='coerce'
        ).dt.year
        
        if genre is None:
            # Count all movies per year
            releases_df = self.movie_metadata['year'].value_counts().reset_index()
            releases_df.columns = ['Year', 'Count']
        else:
            # Convert to string and do simple text matching
            # This is more reliable than trying to parse the dictionary structure
            genre_lower = genre.lower()
            genre_movies = self.movie_metadata[
                self.movie_metadata['genres'].astype(str).str.lower().str.contains(genre_lower, na=False)
            ]
            
            # Create the dataframe
            if len(genre_movies) > 0:
                releases_df = genre_movies['year'].value_counts().reset_index()
                releases_df.columns = ['Year', 'Count']
            else:
                # Return empty DataFrame with correct columns
                releases_df = pd.DataFrame(columns=['Year', 'Count'])
                
        # Sort by year and remove NaN years
        releases_df = releases_df.dropna().sort_values('Year')
        if not releases_df.empty:
            releases_df['Year'] = releases_df['Year'].astype(int)
            
        return releases_df
        
    def ages(self, time_unit='Y'):
        """
        Calculate births per year or month.
        
        Args:
            time_unit (str): 'Y' for year, 'M' for month
                
        Returns:
            pd.DataFrame: DataFrame with time units and birth counts
        """
        # Make a copy of relevant data
        actor_data = self.data.copy()
        
        # Convert 'actor_dob' to datetime
        actor_data['birth_date'] = pd.to_datetime(actor_data['actor_dob'], errors='coerce')
        
        # Drop rows with missing birth dates
        birth_dates = actor_data['birth_date'].dropna()
        
        if time_unit == 'Y':
            # Group by year
            births_by_time = birth_dates.dt.year.value_counts().reset_index()
            births_by_time.columns = ['Year', 'Births']
            births_by_time = births_by_time.sort_values('Year')
            return births_by_time
        
        elif time_unit == 'M':
            # Group by month (regardless of year)
            births_by_time = birth_dates.dt.month.value_counts().reset_index()
            births_by_time.columns = ['Month', 'Births']
            births_by_time = births_by_time.sort_values('Month')
            
            # Convert month numbers to names
            month_names = {
                1: 'January', 2: 'February', 3: 'March', 4: 'April',
                5: 'May', 6: 'June', 7: 'July', 8: 'August',
                9: 'September', 10: 'October', 11: 'November', 12: 'December'
            }
            births_by_time['Month'] = births_by_time['Month'].map(month_names)
            
            return births_by_time
        
        else:
            # Default to year if invalid option
            return self.ages('Y')
            
    def get_movie_details(self, movie_id: int) -> Dict[str, Union[str, List[str]]]:
        """
        Retrieve movie title, summary, and genres for a given movie ID.
        
        Args:
            movie_id (int): The Wikipedia movie ID.
            
        Returns:
            Dict[str, Union[str, List[str]]]: Dictionary containing title, summary, and genres.
        """
        # Find the movie in the metadata
        movie_row = self.movie_metadata[self.movie_metadata["wikipedia_movie_id"] == movie_id]
        
        if movie_row.empty:
            return {"title": "Unknown Title", "summary": "Summary not available", "genres": []}
            
        # Extract the title
        title = movie_row["movie_name"].values[0] if pd.notna(movie_row["movie_name"].values[0]) else "Unknown Title"
        
        # Fetch the summary from plot_summaries.tsv
        summary_row = self.summaries[self.summaries["wikipedia_movie_id"] == movie_id]
        summary = summary_row["summary"].values[0] if not summary_row.empty else "Summary not available"
        
        # Extract genres and format them properly
        genres_raw = movie_row["genres"].values[0] if pd.notna(movie_row["genres"].values[0]) else "{}"
        
        try:
            genres_dict = eval(genres_raw) if isinstance(genres_raw, str) else genres_raw
            genres_list = list(genres_dict.values()) if isinstance(genres_dict, dict) else []
        except Exception as e:
            genres_list = []
            logger.warning(f"Error parsing genres for movie ID {movie_id}: {e}")
            
        return {"title": title, "summary": summary, "genres": genres_list}
