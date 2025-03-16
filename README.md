# Group_15
 Nova Advanced Programming Group Project

Student Infos:
Nils Rudolf, 63855, 63855@novasbe.pt; Paulina Gründel, 63943, 63943@novasbe.pt; Niklas Keckeisen, 63488, 63488@novasbe.pt


# CMU Movie Corpus Analyzer

A Streamlit application for analyzing the CMU Movie Corpus dataset, specifically the character metadata.

## Project Structure

```
Group_15/
├── .gitignore
├── README.md
├── requirements.txt
├── app.py
├── pytest.ini
├── src/
│   ├── __init__.py
│   ├── movie_analyzer.py
│   └── utils.py
├── tests/
│   ├── __init__.py
│   └── test_movie_analyzer.py
└── downloads/
    └── .gitkeep
```

## Features

- **Movie Type Analysis**: Visualize the most common types of movies in the dataset
- **Actor Count Analysis**: See the distribution of number of actors per movie
- **Height Distribution**: Analyze actor heights filtered by gender and height range

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/movie_corpus_analysis.git
   cd movie_corpus_analysis
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

The application will automatically download and extract the required dataset on first run.

## Testing

Run the tests using pytest:
```bash
pytest
```

## Dataset

The application uses the [CMU Movie Corpus dataset](https://www.cs.cmu.edu/~ark/personas/data/MovieSummaries.tar.gz), specifically the `character.metadata.tsv` file which contains the following columns ([Documentation](https://www.cs.cmu.edu/~ark/personas/)):

1. Wikipedia movie ID
2. Freebase movie ID
3. Movie release date
4. Character name
5. Actor date of birth
6. Actor gender
7. Actor height (in meters)
8. Actor ethnicity (Freebase ID)
9. Actor name
10. Actor age at movie release
11. Freebase character/actor map ID
12. Freebase character ID
13. Freebase actor ID
