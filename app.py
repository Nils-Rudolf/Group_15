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

except Exception as e:
    st.error(f"Failed to initialize the analyzer: {str(e)}")
    st.info("Please check your internet connection and try again.")