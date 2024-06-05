#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the saved LSTM model
try:
    model = load_model('next_word_predictor.h5')
    st.success("Model loaded successfully")
except Exception as e:
    st.error(f"Error loading model: {e}")

# Initialize the tokenizer
tokenizer = Tokenizer()

# Read and process the text data
try:
    text = open('Data.txt', encoding="utf-8").read().lower()
    tokenizer.fit_on_texts([text])
except Exception as e:
    st.error(f"Error loading text data: {e}")

# Define the maximum sequence length
max_sequence_len = 1233

# Define the function to predict the next words
def predict_next_words(model, tokenizer, text, num_words):
    for _ in range(num_words):
        token_list = tokenizer.texts_to_sequences([text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        pos = np.argmax(model.predict(token_list), axis=-1)[0]

        for word, index in tokenizer.word_index.items():
            if index == pos:
                text = text + " " + word
                break  # Exit the loop once the word is found
    return text

# Streamlit app interface
st.title('Next Word Predictor')

st.write('Enter a starting text to predict the next words:')

user_input = st.text_area('Input Text')
num_words = st.number_input('Number of words to predict', min_value=1, max_value=100, value=5)

if st.button('Predict'):
    if user_input:
        try:
            prediction = predict_next_words(model, tokenizer, user_input, num_words)
            st.write(f'Predicted Text: {prediction}')
        except Exception as e:
            st.error(f"Error making prediction: {e}")
    else:
        st.write('Please enter some input text.')


# In[ ]:




