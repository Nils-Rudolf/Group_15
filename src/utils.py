"""
Plotting Utilities for the CMU Movie Corpus Analyzer

This module provides helper functions for creating standardized plots used in the
CMU Movie Corpus analysis. The functions generate matplotlib figures and axes
based on the dataframes produced by the MovieCorpusAnalyzer class.

Functions:
    create_movie_type_plot: Creates a bar chart for movie type distribution
    create_actor_count_plot: Creates a histogram for actor count distribution
    create_height_distribution_plot: Creates a bar chart for actor height distribution

Each function returns a dictionary containing the figure and axes objects,
which can be used for further customization or rendering in various contexts.

These utility functions help maintain consistent visualization styles across
the application and separate the plotting logic from the data analysis.
"""


import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any

def create_movie_type_plot(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Create a plot for movie type distribution.
    
    Args:
        df (pd.DataFrame): DataFrame with columns "Movie_Type" and "Count".
        
    Returns:
        Dict[str, Any]: Dictionary with figure and axes objects.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(df["Movie_Type"], df["Count"])
    ax.set_xlabel("Movie Type")
    ax.set_ylabel("Count")
    ax.set_title("Most Common Movie Types")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    return {"fig": fig, "ax": ax}

def create_actor_count_plot(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Create a plot for actor count histogram.
    
    Args:
        df (pd.DataFrame): DataFrame with columns "Number_of_Actors" and "Movie_Count".
        
    Returns:
        Dict[str, Any]: Dictionary with figure and axes objects.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(df["Number_of_Actors"], df["Movie_Count"])
    ax.set_xlabel("Number of Actors")
    ax.set_ylabel("Movie Count")
    ax.set_title("Distribution of Actors per Movie")
    plt.tight_layout()
    
    return {"fig": fig, "ax": ax}

def create_height_distribution_plot(df: pd.DataFrame, gender: str) -> Dict[str, Any]:
    """
    Create a plot for actor height distribution.
    
    Args:
        df (pd.DataFrame): DataFrame with columns "Height" and "Count".
        gender (str): Selected gender for the plot title.
        
    Returns:
        Dict[str, Any]: Dictionary with figure and axes objects.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(df["Height"].astype(float), df["Count"], width=0.05)
    ax.set_xlabel("Height (meters)")
    ax.set_ylabel("Count")
    ax.set_title(f"Actor Height Distribution - Gender: {gender}")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    
    return {"fig": fig, "ax": ax}
