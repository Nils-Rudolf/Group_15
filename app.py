import random
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import ollama
from src.movie_analyzer import MovieCorpusAnalyzer

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
            movie_row = analyzer.movie_metadata.iloc[random_idx]
            
            # Get the movie ID
            movie_id = movie_row['wikipedia_movie_id']
            
            # Use the analyzer's get_movie_details method to get complete information
            return analyzer.get_movie_details(movie_id)
            
        # Initialize session state for the movie data if it doesn't exist
        if 'current_movie' not in st.session_state:
            st.session_state.current_movie = get_random_movie()
            st.session_state.llm_genres = []
            st.session_state.comparison_result = ""
        
        # Display information about using Ollama
        st.info("""
        This feature requires Ollama to be installed and running locally.
        Please make sure you have Ollama installed and the model 'deepseek-r1:1.5b' is available.
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
                    # Even more explicit prompt with examples
                    prompt = f"""
                    Classify this movie into standard film genres only.
                    OUTPUT FORMAT: Return ONLY a simple comma-separated list of genres.
                    
                    Example input: "A tale of an orphan who discovers he's a wizard and goes to magic school"
                    Example output: Fantasy, Adventure, Family
                    
                    Input: {movie['summary']}
                    Output:
                    """
                    
                    # Call Ollama with your installed model
                    response = ollama.chat(
                        model="deepseek-r1:1.5b",
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a genre classifier that outputs ONLY genres as a comma-separated list with no explanation or thinking process."
                            },
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        options={"temperature": 0.1}
                    )
                    
                    # Get raw response for debugging only (don't show this to users in final version)
                    raw_response = response['message']['content'].strip()
                    
                    # Store raw response for debugging purposes
                    debug_response = raw_response  # Keep this for debugging
                    
                    # AGGRESSIVE CLEANING:
                    # Remove thinking part
                    if "<think>" in raw_response:
                        thinking_parts = raw_response.split("<think>")
                        if len(thinking_parts) > 1:
                            raw_response = thinking_parts[1]
                            if "\n" in raw_response:
                                lines = raw_response.split("\n")
                                # Take the last non-empty line as the genres
                                for line in reversed(lines):
                                    if line.strip() and any(c.isalpha() for c in line):
                                        raw_response = line
                                        break
                                        
                    # Extract just a list of common genre words
                    common_genres = ["Comedy", "Drama", "Action", "Adventure", "Fantasy", "Horror", 
                                  "Thriller", "Romance", "Science Fiction", "Sci-Fi", "Animation", 
                                  "Family", "Musical", "Documentary", "Short Film", "War", "Western", 
                                  "Mystery", "Crime", "Biography"]
                                    
                    # Find any common genre words in the text
                    found_genres = []
                    for genre in common_genres:
                        if genre.lower() in raw_response.lower():
                            found_genres.append(genre)
                            
                    # If we found genres this way, use them
                    if found_genres:
                        llm_genres = found_genres
                    else:
                        # Otherwise, try to extract a comma-separated list
                        # Look for the most likely comma-separated part
                        for line in raw_response.split("\n"):
                            if "," in line and not line.startswith("<") and not "example" in line.lower():
                                raw_response = line
                                break
                                
                        # Remove any prefixes like "Genres:" or "Output:"
                        for prefix in ["Genres:", "genres:", "Genre:", "genre:", "Classifications:", 
                                      "The genres are:", "Output:", "My classification:"]:
                            if prefix in raw_response:
                                raw_response = raw_response.split(prefix, 1)[1].strip()
                                
                        # Get clean list of genres from the comma-separated list
                        llm_genres = [genre.strip() for genre in raw_response.split(",") if genre.strip()]
                        
                    # Store genres in session state
                    st.session_state.llm_genres = llm_genres
                    
                    # Display the LLM genres in a clear way
                    st.text_area("LLM Classified Genres", 
                                 ", ".join(llm_genres) if llm_genres else "No genres identified", 
                                 height=100, 
                                 key="llm_genres_display")
                                
                    # Now let's do the comparison
                    second_prompt = f"""
                    Compare these two lists of movie genres:
                    
                    Database genres: {', '.join(movie['genres'])}
                    LLM classified genres: {', '.join(llm_genres)}
                    
                    Are any of the LLM classified genres contained in the database genres list?
                    Answer exactly as you can see in OUTPUT format and in the examples.
                    OUTPUT FORMAT: Your response must start with "YES" or "NO" followed by a brief explanation that compares specific genres.
                    
                    For example:
                    If database genres = "Comedy, Romance, Action, Thriller" and LLM genres = "Comedy, Drama, Action, Family, Thriller"
                    Respond with: "YES - Comedy, Action, Thriller is present in both lists, but Drama, Family is not in the database genres."
                    
                    If database genres = "Action, Adventure" and LLM genres = "Comedy, Drama"
                    Respond with: "NO - None of the LLM genres (Comedy, Drama) appear in the database genres (Action, Adventure)."
                    """
                    
                    comparison_response = ollama.chat(model="deepseek-r1:1.5b", messages=[
                        {
                            "role": "user",
                            "content": second_prompt
                        }
                    ])
                    
                    comparison_result = comparison_response['message']['content'].strip()
                    
                    # Clean up comparison result to remove thinking
                    if "<think>" in comparison_result:
                        parts = comparison_result.split("</think>")
                        if len(parts) > 1:
                            comparison_result = parts[1].strip()
                        else:
                            parts = comparison_result.split("<think>")
                            if len(parts) > 1:
                                lines = parts[1].split("\n")
                                for line in reversed(lines):
                                    if line.strip() and (line.strip().startswith("YES") or line.strip().startswith("NO")):
                                        comparison_result = line.strip()
                                        break
                                        
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
                    
except Exception as e:
    st.error(f"Failed to initialize the analyzer: {str(e)}")
    st.info("Please check your internet connection and try again.")
    
