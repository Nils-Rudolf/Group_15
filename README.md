# Group_15
 Nova Advanced Programming Group Project

Student Infos:
Nils Rudolf, 63855, 63855@novasbe.pt; Paulina Gründel, 63943, 63943@novasbe.pt; Niklas Keckeisen, 63488, 63488@novasbe.pt


# CMU Movie Corpus Analyzer

A Streamlit application for analyzing the CMU Movie Corpus dataset, specifically the character metadata.

## Project Structure

```
Group_15/
├── LICENSE
├── README.md
├── app.py
├── downloads
├── pytest.ini
├── requirements.txt
├── src
│   ├── __init__.py
│   ├── movie_analyzer.py
│   └── utils.py
├── structure.txt
└── tests
    ├── __init__.py
    └── test_movie_analyzer.py
```

## Features

**Movie Analysis**
- **Movie Type Analysis**: Visualize the most common types of movies in the dataset
- **Actor Count Analysis**: See the distribution of number of actors per movie
- **Height Distribution**: Analyze actor heights filtered by gender and height range

**Chronological Information**
- **Movie Releases by Year**: Examine release trends with optional genre filtering
- **Birth Statistics**: View actor birth distribution by year or month

**LLM Classification**
- **Displaying random movie information**: Visualize title, summary and genre of a random movie
- **Genre Classification**: Use a local LLM to classify movie genres based on summaries
- **Genre Comparison**: Automatically compare LLM classifications with database genres

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Nils-Rudolf/Group_15.git
   cd Group_15
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Download [Ollama](https://ollama.com/) for your machine.

2. Run 
```bash
ollama serve 
```

3. Run a LLM model of your choice ([Ollama](https://github.com/ollama/ollama)):
```bash
ollama run deepseek-r1:1.5b
```

4. Run the Streamlit app:
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

## Essay:
Connecting a local LLM text classification project to the United Nations' Sustainable Development Goals feels like a stretch. The SDGs focus on global challenges like poverty, education, climate change, and inequality, while this project is a technical exercise centered on analyzing movie metadata and predicting genres. We are wondering if this is some kind of LLM check. All that aside, there are potential indirect links that can be thought about with some imagination.

One of them is to SDG 4: Quality Education. Text classification using LLMs like Ollama is a case in point of affordable, user-friendly AI technology that can be applied for education purposes. By integrating such tools into an educational platform, students can explore data science, natural language processing, and prompt engineering in an engaging way—e.g., through analysis of movies or other cultural data sets. This initiative could result in open-source education tools, making access to AI literacy a leveler for everyone and developing skills in accordance with inclusive, equitable education's vision.

A weak but plausible link is to SDG 9: Industry, Innovation, and Infrastructure. Building and documenting a pipeline that employs local LLMs promotes innovation in lean, scalable AI. Compared to cloud models that require high infrastructure, Ollama runs locally, thereby making it deployable in resource-constrained environments. This can contribute to sustainable technology development, reducing reliance on energy-guzzling data centers and contributing to resilient infrastructure—a humble but meaningful recognition of SDG 9.

Finally, consider SDG 16: Peace, Justice, and Strong Institutions. Proper text classification can, in a broader context, be applied to examine media material (e.g., movies) to identify violence themes, discrimination, or social justice themes. While this project is genre-focused, the technology itself can be utilized to monitor cultural narratives and assist in promoting peaceful and inclusive societies by being aware of how media influences thinking.