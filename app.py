import streamlit as st
import pandas as pd
import numpy as np

import gensim
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict


st.set_page_config(
    layout = "centered",
    initial_sidebar_state = "auto")
    
style = "<style>.row-widget.stButton {text-align: center;}</style>"

@st.cache_data 
def load_data(name):        
    df = pd.read_csv(name, index_col=[0])  
    return df


@st.cache_resource
def load_model_cbow(data):
    model_cbow = Word2Vec(
      data, sg=0, workers=8, window=11, min_count=1, vector_size=600
    )
    return model_cbow


@st.cache_resource
def load_model_tfidf():
    tfidf = TfidfVectorizer(tokenizer=lambda x: [word.strip() for word in x.split(',') if word.strip() != ''])
    return tfidf


def fit_tfidf_embedding(docs, model_cbow, tfidf_model):
    '''
    Fits TF-IDF model  on the preprocessed documents, and calculates the IDF weights of each word 
    in the vocabulary. Returns the IDF weights of each word and the vector size of the CBOW model. 
    '''

    text_docs = []
    for doc in docs:
        text_docs.append(", ".join(doc))

    tfidf = tfidf_model #TfidfVectorizer(tokenizer=lambda x: [word.strip() for word in x.split(',') if word.strip() != ''])
    tfidf.fit(text_docs)  
    # if a word was never seen it is given idf of the max of known idf value
    max_idf = max(tfidf.idf_)  
    
    word_idf_weight = defaultdict(
        lambda: max_idf,
        [(word, tfidf.idf_[i]) for word, i in tfidf.vocabulary_.items()],
    )
    vector_size = model_cbow.wv.vector_size
    
    return word_idf_weight, vector_size


def doc_average(doc, model_cbow, word_idf_weight, vector_size):
    '''
    Compute weighted mean of documents word embeddings
    '''
    mean = []
    for word in doc:
        if word in model_cbow.wv.index_to_key:
            mean.append(
                model_cbow.wv.get_vector(word) * word_idf_weight[word]
            ) 

    if not mean:  
        return np.zeros(vector_size)
    else:
        mean = np.array(mean).mean(axis=0)
        return mean



def transform_tfidf_embedding(docs, model_cbow, word_idf_weight, vector_size, tfidf_model):
    '''
    Compute average word vector for multiple docs, where docs had been tokenized.    
    '''

    doc_word_vector = np.vstack([doc_average(doc, model_cbow, word_idf_weight, vector_size) for doc in docs])
    return doc_word_vector



def check_ingredients(df, ingredients:list):
    '''
    Counts the number of ingredients that match 
    for each recipe and stores the indices of these recipes
    '''

    result = []

    if len(ingredients) == 1:
        for i in range(len(df)):
            if df[ingredients[0]][i] == 1:
                result.append(i)
    elif len(ingredients) <= 2:
        for i in range(len(df)):
            if df[ingredients[0]][i] == 1 and df[ingredients[1]][i] == 1:
                result.append(i)
    elif len(ingredients) >= 3:
        for i in range(len(df)):
            match = 0
            for ing in ingredients:
                if df[ing][i] == 1:
                    match += 1
            if match >= len(ingredients)*0.75:
                result.append(i)

    return result


def format_output(result, user_input, user_vector, recipe_vectors):
    '''
    Depending on the length of the result, we determine which output to form
    '''
    result_length = len(result)
    if (result_length == 0):
        st.info("Oops, we don't have any recipes with this combination in our base yet, but there are a few recipes with some similar items", icon="ℹ️") 
        return  recipes_with_top_scores(user_input, recipe_vectors)
    else:
        return sort_recipes_by_similarity(result, user_vector, user_input, recipe_vectors)
#    elif (result_length < 3):
#        user_vector = doc_average(user_input, model_cbow, word_idf_weight, vector_size)
#        similarities = cosine_similarity([user_vector], recipe_vectors)[0]
#        top_indices = similarities.argsort()[::-1][:5]
#
#        top_recipes_needed = 5 - result_length
#        indices = result + top_indices[:top_recipes_needed].tolist()
#        return df.iloc[top_indices][['url', 'name', 'prep_time', 'calories']]
#    else:
#        return sort_recipes_by_similarity(result, user_vector, user_input, recipe_vectors)
#     #return df.iloc[top_indices][['url', 'name', 'prep_time', 'calories']]
#


  
def recipes_with_top_scores(user_input, recipe_vectors):
    '''
    якщо довжина відсортованого списку = 0, то повертає найвищий скор з усього датасету
    '''

    user_vector = doc_average(user_input, model_cbow, word_idf_weight, vector_size)
    similarities = cosine_similarity([user_vector], recipe_vectors)[0]
    top_indices = similarities.argsort()[::-1][:5]

    # Create a dataframe with the recipe information
    recipes_df = df.iloc[top_indices][['url', 'name', 'prep_time', 'calories']]
    recipes_df.reset_index(inplace=True, drop=True)

    return recipes_df[:5]


def sort_recipes_by_similarity(results, user_vector, user_input, recipe_vectors):
    '''
    Сортує рядки з індексами, що подані в аргументі, за схожістю з вектором користувача 
    та повертає новий датафрейм з інформацією про рецепти,
    відсортований за зменшенням схожості.
    '''
    # Calculate similarities between user vector and recipe vectors
    user_vector = doc_average(user_input, model_cbow, word_idf_weight, vector_size)
    similarities = cosine_similarity([user_vector], recipe_vectors[results])[0]
    
    # Sort the results by similarity
    sorted_similarities = np.argsort(similarities)[::-1]
    sorted_result = [value for value in sorted_similarities if value in results]
    # Create a dataframe with the recipe information
    recipes_df = df.iloc[sorted_result][['url', 'name', 'prep_time', 'calories']]
    recipes_df.reset_index(inplace=True, drop=True)
    results.reverse()
    if (len(sorted_result) < 5):
        return df.iloc[results][['url', 'name', 'prep_time', 'calories']][:6]

    return recipes_df[:6]


#def additional_recipes(result_length, user_input):
#  '''
#  якщо довжина менша за 3, то повертає відсортовані зі списку + найвищий скор
#  '''
#   need to fix this 
#   



@st.cache_resource
def get_rec_vect():
    word_idf_weight, vector_size = fit_tfidf_embedding(data, model_cbow, tfidf_model)
    recipe_vectors = transform_tfidf_embedding(data, model_cbow, word_idf_weight, vector_size, tfidf_model)
    return recipe_vectors

df = load_data('main.csv')
options = sorted(df.columns[7:953]) 
#data = df['ingredients_list']
data = df['ingredients_list'].apply(lambda x: x.strip("[]").replace("'", "").split(", ") if isinstance(x, str) else [])
model_cbow = load_model_cbow(data)
tfidf_model = load_model_tfidf()
word_idf_weight, vector_size = fit_tfidf_embedding(data, model_cbow, tfidf_model)
recipe_vectors = get_rec_vect()



def main():

    st.title('What to cook if I have..?🧑‍🍳')
    st.write("""
            If you like to shop at the grocery store for discounts, 
            but then don't understand how to combine these items into 
            one dish, this app will do everything for you.
            """)
    st.markdown(style, unsafe_allow_html=True)
    
    
    user_input = sorted(st.multiselect(
                    '\n'.join ([
                                'Find out what you can cook' 
                                ' with food you already have in your fridge.' 
                                'Write the ingredients in the field ⬇️',
                                ]),
                                options,
                                help = 'Enter at least 3 items',
                                )) 
    
    
    if (st.button(':green[Find recipes]',)):

        user_vector = doc_average(user_input, model_cbow, word_idf_weight, vector_size)
        recipe_vectors = get_rec_vect()
        result_indexes = check_ingredients(df, user_input)
        recipes_df = format_output(result_indexes, user_input, user_vector, recipe_vectors)

        
        counter = 1
        for index, row in recipes_df.iterrows():
            recipe_name = row['name']
            recipe_url = row['url']
            recipe_time = int(row['prep_time']) 
            st.write(f"{counter}. {recipe_name} (est. cooking time - {recipe_time} mins ⏳): {recipe_url}")
            counter += 1
        


if __name__ == "__main__":
    main()
