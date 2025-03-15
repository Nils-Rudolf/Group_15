import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from src.movie_analyzer import MovieCorpusAnalyzer
from src.utils import (
    create_movie_type_plot,
    create_actor_count_plot,
    create_height_distribution_plot
)
import seaborn as sns
import random
import ollama


# Set page config
st.set_page_config(
    page_title="CMU Movie Corpus Analyzer",
    page_icon="🎬",
    layout="wide"
)

# App title
st.title("CMU Movie Corpus Analyzer")
st.markdown("Analysis of the CMU Movie Corpus character metadata")

# Initialize the analyzer
@st.cache_resource
def get_analyzer():
    with st.spinner("Loading data... This might take a while for the first time."):
        return MovieCorpusAnalyzer()

try:
    analyzer = get_analyzer()
    
    # Create a multipage app
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Movie Analysis", "Chronological Info", "Text Classification"])

    # Page logic
    if page == "Movie Analysis":
        # Create tabs for different analyses
        tab1, tab2, tab3 = st.tabs(["Movie Types", "Actor Count", "Height Distribution"])
        
        # Tab 1: Movie Types
        with tab1:
            st.header("Movie Type Analysis")
            st.markdown("Shows the most common types of movies in the dataset.")
            
            # Slider for N
            n_value = st.slider(
                "Number of movie types to display", 
                min_value=5, 
                max_value=50, 
                value=10,
                step=5
            )
            
            # Get data and create plot
            movie_types_df = analyzer.movie_type(N=n_value)
            
            # Show plot
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(movie_types_df["Movie_Type"], movie_types_df["Count"])
            ax.set_xlabel("Movie Type")
            ax.set_ylabel("Count")
            ax.set_title(f"Top {n_value} Most Common Movie Types")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            
            st.pyplot(fig)
            
            # Show data
            st.subheader("Data")
            st.dataframe(movie_types_df)
        
        # Tab 2: Actor Count
        with tab2:
            st.header("Actor Count Analysis")
            st.markdown("Shows the distribution of number of actors per movie.")
            
            # Get data and create plot
            actor_count_df = analyzer.actor_count()
            
            # Show plot
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(actor_count_df["Number_of_Actors"], actor_count_df["Movie_Count"])
            ax.set_xlabel("Number of Actors")
            ax.set_ylabel("Movie Count")
            ax.set_title("Distribution of Actors per Movie")
            plt.tight_layout()
            
            st.pyplot(fig)
            
            # Show data
            st.subheader("Data")
            st.dataframe(actor_count_df)
        
        # Tab 3: Height Distribution
        with tab3:
            st.header("Actor Height Distribution")
            st.markdown("Shows the distribution of actor heights filtered by gender and height range.")
            
            # Create a column layout
            col1, col2, col3 = st.columns(3)
            
            # Gender selection
            with col1:
                # Get unique gender values
                gender_options = ["All"] + list(analyzer.data["actor_gender"].dropna().unique())
                gender = st.selectbox("Select Gender", options=gender_options)
            
            # Height range selection
            with col2:
                min_height = st.number_input(
                    "Minimum Height (m)", 
                    min_value=0.0, 
                    max_value=3.0, 
                    value=1.5, 
                    step=0.1
                )
            
            with col3:
                max_height = st.number_input(
                    "Maximum Height (m)", 
                    min_value=0.0, 
                    max_value=3.0, 
                    value=2.0, 
                    step=0.1
                )
            
            # Validate height inputs
            if min_height > max_height:
                st.error("Minimum height cannot be greater than maximum height.")
            else:
                # Get data and create plot
                try:
                    height_dist_df = analyzer.actor_distributions(
                        gender=gender, 
                        min_height=min_height, 
                        max_height=max_height
                    )
                    
                    # Show plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.bar(height_dist_df["Height"].astype(float), height_dist_df["Count"], width=0.05)
                    ax.set_xlabel("Height (meters)")
                    ax.set_ylabel("Count")
                    ax.set_title(f"Actor Height Distribution - Gender: {gender}")
                    ax.grid(axis="y", alpha=0.3)
                    plt.tight_layout()
                    
                    st.pyplot(fig)
                    
                    # Show data
                    st.subheader("Data")
                    st.dataframe(height_dist_df)
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    elif page == "Chronological Info":
        st.title("Chronological Information")
        
        # Section for releases by year
        st.header("Movie Releases by Year")
        
        # Create a dropdown for genre selection
        top_genres = ["None", "Drama", "Comedy", "Action", "Romance", "Thriller"]  # Add your top genres
        selected_genre = st.selectbox("Select a genre:", top_genres)
        
        # Process the genre selection
        genre_param = None if selected_genre == "None" else selected_genre
        
        # Get the dataframe using your releases method
        releases_df = analyzer.releases(genre=genre_param)

        # Check if we have data to plot
        if releases_df.empty:
            st.warning(f"No data found for genre: {selected_genre}. Try a different genre.")
        else:
            # Plot the data
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(x='Year', y='Count', data=releases_df, ax=ax)
            ax.set_title(f"Movie Releases by Year {f'({selected_genre})' if genre_param else ''}")
            ax.set_xlabel("Year")
            ax.set_ylabel("Number of Movies")

            # Sample the years to avoid overcrowding
            n = max(1, len(releases_df) // 25)  # Show approximately 25 labels
            ticks = range(0, len(releases_df), n)
            tick_labels = [str(releases_df.iloc[i]['Year']) for i in ticks]

            # Set the selected ticks and rotated labels
            plt.xticks(ticks, tick_labels, rotation=90)

            #layout
            st.pyplot(fig)

        #new section for birth statistics
        st.header("Birth Statistics")

        # Create a dropdown for time unit selection
        time_unit = st.selectbox("Select time unit:", ["Year", "Month"])

        # Process the selection
        time_unit_param = 'Y' if time_unit == "Year" else 'M'

        # Get the dataframe using your ages method
        ages_df = analyzer.ages(time_unit=time_unit_param)

        # Plot the data
        if not ages_df.empty:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # The column names vary based on the selected time unit
            x_col = 'Year' if time_unit == 'Year' else 'Month'
            
            sns.barplot(x=x_col, y='Births', data=ages_df, ax=ax)
            ax.set_title(f"Births by {time_unit}")
            ax.set_xlabel(time_unit)
            ax.set_ylabel("Number of Births")
            
            if time_unit == "Year":
                # Sample the years to avoid overcrowding
                n = max(1, len(ages_df) // 25)  # Show approximately 25 labels
                ticks = range(0, len(ages_df), n)
                tick_labels = [str(ages_df.iloc[i][x_col]) for i in ticks]

                # Set the selected ticks and rotated labels
                plt.xticks(ticks, tick_labels, rotation=90)
                
            #layout    
            st.pyplot(fig)
        else:
            st.warning("No birth data available to display.")

    elif page == "Text Classification":
        st.title("Movie Genre Classification with LLM")
                
        def get_random_movie():
            """
            Select a random movie from the dataset and extract relevant details.
            """
            # Select a random movie row
            random_idx = random.randint(0, len(analyzer.movie_metadata) - 1)
            movie = analyzer.movie_metadata.iloc[random_idx]
            
            # Extract title
            title = movie.get('movie_name', 'Unknown Title')
            
            # Extract summary (check if a summary column exists)
            summary = movie.get('summary', None)  # Replace 'summary' with the actual column name if available
            
            # If no summary, use placeholder
            if not summary or pd.isna(summary):
                summary = "Summary not available in the dataset"
                
            # Extract genres (handling dictionary format)
            genres_raw = movie.get('genres', {})
            
            # Ensure it's a dictionary before extracting values
            if isinstance(genres_raw, dict):
                genres_list = list(genres_raw.values())
            elif isinstance(genres_raw, str):  # If stored as a string, try converting it
                try:
                    genres_dict = eval(genres_raw)  # Safe conversion
                    genres_list = list(genres_dict.values()) if isinstance(genres_dict, dict) else []
                except:
                    genres_list = []
            else:
                genres_list = []
                
            return {
                'title': title,
                'summary': summary,
                'genres': genres_list
            }
            
        # Initialize session state for the movie data if it doesn't exist
        if 'current_movie' not in st.session_state:
            st.session_state.current_movie = get_random_movie()
            st.session_state.llm_genres = []
            st.session_state.comparison_result = ""
        
        # Display information about using Ollama
        st.info("""
        This feature requires Ollama to be installed and running locally.
        Please make sure you have Ollama installed and a model like 'llama3' or 'mistral' is available.
        """)
        
        # Create the shuffle button
        if st.button("Shuffle"):
            with st.spinner("Getting a random movie..."):
                st.session_state.current_movie = get_random_movie()
                st.session_state.llm_genres = []
                st.session_state.comparison_result = ""
        
        # Display the movie information
        movie = st.session_state.current_movie
        
        # First text box: Movie Title and Summary
        st.subheader("Movie Information")
        st.text_area("Title and Summary", 
                     f"Title: {movie['title']}\n\nSummary: {movie['summary']}", 
                     height=200, 
                     key="movie_info")
                    
        # Second text box: Database Genres
        st.subheader("Database Genres")
        st.text_area("Genres from Database", 
                     ", ".join(movie['genres']) if movie['genres'] else "No genres available", 
                     height=100, 
                     key="db_genres")
                    
        # Third text box: LLM Classification
        st.subheader("LLM Genre Classification")
        
        # Button to run the LLM classification
        if st.button("Classify with LLM"):
            with st.spinner("Classifying genres using LLM..."):
                try:
                    # Craft a prompt for the LLM
                    prompt = f"""
                    You are a movie genre classifier. Based on the title and summary, identify the genres of the following movie.
                    Only respond with a comma-separated list of genres, nothing else.
                    
                    Title: {movie['title']}
                    Summary: {movie['summary']}
                    
                    Genres:
                    """
                    
                    # Call Ollama with the prompt
                    response = ollama.chat(model="deepseek-r1:1.5b", messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ])
                    
                    # Extract the genres from the response
                    llm_genres = response['message']['content'].strip().split(", ")
                    st.session_state.llm_genres = llm_genres
                    
                    # Display the LLM genres
                    st.text_area("LLM Classified Genres", 
                                 ", ".join(llm_genres), 
                                 height=100, 
                                 key="llm_genres")
                    
                    # Now let's do the comparison
                    second_prompt = f"""
                    Compare these two lists of movie genres:
                    
                    Database genres: {', '.join(movie['genres'])}
                    LLM classified genres: {', '.join(llm_genres)}
                    
                    Are the LLM's identified genres contained in the database genres list? 
                    Answer only Yes or No, followed by a brief explanation.
                    """
                    
                    comparison_response = ollama.chat(model="deepseek-r1:1.5b", messages=[
                        {
                            "role": "user",
                            "content": second_prompt
                        }
                    ])
                    
                    comparison_result = comparison_response['message']['content'].strip()
                    st.session_state.comparison_result = comparison_result
                    
                    # Determine if it's a positive or negative comparison
                    is_positive = "yes" in comparison_result.lower()
                    
                    # Display the comparison result
                    st.subheader("Comparison Result")
                    
                    if is_positive:
                        st.success(comparison_result)
                    else:
                        st.error(comparison_result)
                    
                except Exception as e:
                    st.error(f"Error connecting to Ollama: {str(e)}")
                    st.info("Make sure Ollama is installed and running, and that you have the required model.")
        
        # Display the previously computed results if available
        if st.session_state.llm_genres:
            st.text_area("LLM Classified Genres", 
                         ", ".join(st.session_state.llm_genres), 
                         height=100, 
                         key="stored_llm_genres")
        
        if st.session_state.comparison_result:
            st.subheader("Comparison Result")
            is_positive = "yes" in st.session_state.comparison_result.lower()
            
            if is_positive:
                st.success(st.session_state.comparison_result)
            else:
                st.error(st.session_state.comparison_result)

except Exception as e:
    st.error(f"Failed to initialize the analyzer: {str(e)}")
    st.info("Please check your internet connection and try again.")
