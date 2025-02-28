import pytest
import pandas as pd
import numpy as np
from src.movie_analyzer import MovieCorpusAnalyzer


class TestMovieCorpusAnalyzer:
    """Tests for the MovieCorpusAnalyzer class."""
    
    @pytest.fixture
    def mock_analyzer(self, monkeypatch):
        """Create a mock analyzer with dummy data for testing."""
        def mock_load_data(self):
            self.data = pd.DataFrame({
                'wikipedia_movie_id': [54166] * 15,
                'freebase_movie_id': ['/m/0f4yh'] * 15,
                'release_date': [None] * 15,
                'character_name': [
                    "Dr. Marcus Brody", "Simon Katanga", "Dr. René Belloq", "Major Arnold Toht", "Indiana Jones",
                    "Marion Ravenwood", "Colonel Dietrich", "Sallah", "Satipo", "Giant Sherpa",
                    "Gobler", "Col. Musgrove", "Major Eaton", "Barranca", "Bureaucrat"
                ],
                'actor_dob': [
                    "1922-05-31", "1949-10-20", "1943-01-18", "1935-09-28", "1942-07-13",
                    "1951-10-05", "1946-04-26", "1944-05-05", "1953-05-24", "1937-05-19",
                    "1947-05-09", "1922-12-02", "1948-07-05", None, None
                ],
                'actor_gender': ["M", "M", "M", "M", "M", "F", "M", "M", "M", "M", "M", "M", "M", "M", "M"],
                'actor_height': [
                    1.816, 1.87, 1.77, None, 1.85,
                    1.7, None, 1.85, 1.89, 1.95,
                    None, None, None, None, None
                ],
                'actor_ethnicity': [
                    "/m/02w7gg", None, None, None, "/m/01qhm_",
                    "/m/07bch9", None, "/m/06gbnc", "/m/02w7gg", "/m/0g96wd",
                    None, None, None, None, None
                ],
                'actor_name': [
                    "Denholm Elliott", "George Harris", "Paul Freeman", "Ronald Lacey", "Harrison Ford",
                    "Karen Allen", "Wolf Kahler", "John Rhys-Davies", "Alfred Molina", "Pat Roach",
                    "Anthony Higgins", "Don Fellows", "William Hootkins", "Vic Tablian", "Bill Reimbold"
                ],
                'actor_age_at_release': [
                    59, 31, 38, 45, 38,
                    29, 35, 37, 28, 44,
                    34, 58, 32, None, None
                ],
                'freebase_char_actor_map_id': [
                    "/m/02nwzzv", "/m/02nw_18", "/m/02nwzzg", "/m/02nwzyz", "/m/0k294p",
                    "/m/0k294v", "/m/02vd7bq", "/m/03jr4n_", "/m/03jrl6s", "/m/03mk_th",
                    "/m/0cf_wlf", "/m/0cg2gvx", "/m/0cg3259", "/m/0gc54kz", "/m/0gcd5f9"
                ],
                'freebase_character_id': [None] * 15,
                'freebase_actor_id': [None] * 15
            })
        monkeypatch.setattr(MovieCorpusAnalyzer, "_download_data", lambda self: None)
        monkeypatch.setattr(MovieCorpusAnalyzer, "_extract_data", lambda self: None)
        monkeypatch.setattr(MovieCorpusAnalyzer, "_load_data", mock_load_data)
        
        return MovieCorpusAnalyzer()
    
    def test_movie_type_invalid_type(self, mock_analyzer):
        """Test that movie_type raises TypeError for non-integer N."""
        with pytest.raises(TypeError):
            mock_analyzer.movie_type(N="10")
        
        with pytest.raises(TypeError):
            mock_analyzer.movie_type(N=10.5)
    
    def test_movie_type_valid_input(self, mock_analyzer):
        """Test that movie_type works with valid input."""
        result = mock_analyzer.movie_type(N=2)
        assert isinstance(result, pd.DataFrame)
        assert len(result) <= 2
        assert "Movie_Type" in result.columns
        assert "Count" in result.columns
    
    def test_actor_distributions_invalid_gender_type(self, mock_analyzer):
        """Test that actor_distributions raises TypeError for non-string gender."""
        with pytest.raises(TypeError):
            mock_analyzer.actor_distributions(gender=123)
    
    def test_actor_distributions_invalid_height_type(self, mock_analyzer):
        """Test that actor_distributions raises TypeError for non-numeric heights."""
        with pytest.raises(TypeError):
            mock_analyzer.actor_distributions(min_height="1.5")
        
        with pytest.raises(TypeError):
            mock_analyzer.actor_distributions(max_height="2.0")
    
    def test_actor_distributions_invalid_height_values(self, mock_analyzer):
        """Test that actor_distributions raises ValueError for invalid height values."""
        with pytest.raises(ValueError):
            mock_analyzer.actor_distributions(min_height=-1)
        
        with pytest.raises(ValueError):
            mock_analyzer.actor_distributions(max_height=4)
        
        with pytest.raises(ValueError):
            mock_analyzer.actor_distributions(min_height=2, max_height=1)
    
    def test_actor_distributions_valid_input(self, mock_analyzer):
        """Test that actor_distributions works with valid input."""
        result = mock_analyzer.actor_distributions(gender="male", min_height=1.5, max_height=2.0)
        assert isinstance(result, pd.DataFrame)
        assert "Height" in result.columns
        assert "Count" in result.columns