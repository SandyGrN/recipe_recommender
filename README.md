# recipe_recommender

  The question often arises as to what to cook with the products that are already in the fridge. Or how to combine products that were bought at a discount into one dish. 
  
# Project Overview

- This tool recommends recipes based on the ingredients entered by the user. The model is aimed both at finding the most accurate matches and, in the   absence of a database, offering the most similar recipes to the entered set of recipes.
- Scraped  3000  recipes from Simple Recipes(https://www.simplyrecipes.com). 
- Parsed recipe ingredients and created word embeddings using Word2Vec and TF-IDF.
- The model achieves its best results through a combined approach: 1) text encoding and comparison of semantic similarity 2) prescribed rules           established experimentally
- Built app with Streamlit (______________)

Adding jupiter notebook with feature engineering and model experiments in progress. 
