# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 14:29:20 2022

@author: Sergio
"""

import streamlit as st
import io
import os
from os import path
import spacy
from spacy import displacy
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
import nltk
from collections import Counter
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
import requests
from bs4 import BeautifulSoup


# Create a WordNet lemmatizer
lemmatizer = WordNetLemmatizer()

# Download necessary NLTK packages and corpora
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')


@st.cache_data
def verikeybert(text, hits):
    from keybert import KeyBERT
    from flair.embeddings import TransformerDocumentEmbeddings
    from keyphrase_vectorizers import KeyphraseCountVectorizer
    import nltk
    nltk.download('stopwords')
     
    # Init KeyBERT
    kw_model = KeyBERT()
    
    roberta = TransformerDocumentEmbeddings('roberta-base')
    kw_model = KeyBERT(model=roberta)  
    
    keywords = kw_model.extract_keywords(
        text, 
        vectorizer=KeyphraseCountVectorizer(),  
        use_mmr=True,
        diversity=0.5,
        top_n=hits
        )
    
    return keywords

    

def open_file(file):
    # To convert to a string based IO:
    stringio = io.StringIO(file.getvalue().decode("utf-8"))
    # To read file as string:
    string_data = stringio.read()
    clean_text = " ".join(string_data.split())
    return clean_text



#########################################################
#
# The following functions are for text analysis
#
#########################################################

# Load the English language model in SpaCy
nlp = spacy.load("en_core_web_sm")


# Define a function to display the dataframe in the app
def show_magic_dataframe(df):
    # Get the list of columns in the dataframe
    columns = list(df.columns)

    col1, col2 = st.columns([1,2])
    with col1:
        st.caption("**:orange[The Magic NLP Dataframe]**")
        # Allow the user to select which columns to display
        selected_columns = st.multiselect("Select columns to display", columns)   
    with col2:
        # Set the maximum width for the text in the dataframe
        pd.set_option("display.max_colwidth", 200)
    
        # Display the selected columns in the table
        st.write(df[selected_columns])
        
        

# Define a function to count the number of words in the text
def count_words(text):
    doc = nlp(text)
    return len([token for token in doc if token.is_alpha])


# Define a function to count the number of each part of speech
def count_pos(doc):
    pos_counts = {
        "VERB": 0,
        "NOUN": 0,
        "ADJ": 0,
        "ADV": 0,
        "PROPN": 0,
        "PRON": 0,
        "ADP": 0
    }
    for token in doc:
    # Collect POS statistics
      try:
        pos_counts[token.pos_] += 1
      except KeyError:
        pos_counts[token.pos_] = 1

    return pos_counts

# Define a function to count the number of named entities
def count_ner(doc):
    ner_counts = {}
    for ent in doc.ents:
        if ent.label_ not in ner_counts:
            ner_counts[ent.label_] = 1
        else:
            ner_counts[ent.label_] += 1
    return ner_counts

# Define a function to visualize a dependency parse
def visualize_dep(doc):
    # Generate the SVG visualization
    svg = displacy.render(doc, style="dep", jupyter=False)
    
    # Display the entity visualization in the browser:
    st.markdown(svg, unsafe_allow_html=True)
    
    # Return the SVG as a string
    return 

# Define a function to visualize named entities
def visualize_ner(doc):
    # Generate the SVG visualization
    svg = displacy.render(doc, style="ent", jupyter=False)
    
    # Display the entity visualization in the browser:
    st.markdown(svg, unsafe_allow_html=True)
    
    # Return the SVG as a string
    return 

def num_sentences(text):
    """Returns the number of sentences in the given text."""
    # process the text using spaCy's nlp() function
    doc = nlp(text)
    # return the number of sentences
    return len([sent for sent in doc.sents])

def avg_num_words_per_sentence(text):
    """Returns the average number of words per sentence in the given text."""
    # process the text using spaCy's nlp() function
    doc = nlp(text)
    # calculate the average number of words per sentence
    avg_num_words = sum([len(sent) for sent in doc.sents]) / num_sentences(text)
    # return the result
    return avg_num_words

def num_words_longest_sentence(text):
    """Returns the number of words in the longest sentence in the given text."""
    # process the text using spaCy's nlp() function
    doc = nlp(text)
    # find the longest sentence
    longest_sentence = ""
    for sentence in doc.sents:
        if len(sentence.text) > len(longest_sentence):
            longest_sentence = sentence.text
    # return the number of words in the longest sentence
    return len(longest_sentence.split())

def num_words_shortest_sentence(text):
    """Returns the number of words in the shortest sentence in the given text."""
    # process the text using spaCy's nlp() function
    doc = nlp(text)
    # find the shortest sentence
    shortest_sentence = ""
    for sentence in doc.sents:
        if len(shortest_sentence) == 0 or len(sentence.text) < len(shortest_sentence):
            shortest_sentence = sentence.text
    # return the number of words in the shortest sentence
    return len(shortest_sentence.split())


def most_repeated_tokens(text, nlp, token_type):
    doc = nlp(text)
    token_counts = {}
    
    for token in doc:
        if token.pos_ == token_type:
            if token.text in token_counts:
                token_counts[token.lemma_] += 1
            else:
                token_counts[token.lemma_] = 1
    
    sorted_token_counts = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
    #Just the token
    #most_repeated_tokens = [token[0] for token in sorted_token_counts[:3]]
    #dictionary where the keys are the most repeated tokens and the values are their respective counts
    most_repeated_tokens = {token[0]: token[1] for token in sorted_token_counts[:3]}
    return most_repeated_tokens

def most_repeated_nouns(text, nlp):
    return most_repeated_tokens(text, nlp, "NOUN")

def most_repeated_verbs(text, nlp):
    return most_repeated_tokens(text, nlp, "VERB")

def most_repeated_adjectives(text, nlp):
    return most_repeated_tokens(text, nlp, "ADJ")

def most_repeated_adverbs(text, nlp):
    return most_repeated_tokens(text, nlp, "ADV")


def most_repeated_named_entities(text, nlp):
    doc = nlp(text)
    ner_counts = {}
    for ent in doc.ents:
        label = ent.label_
        if label not in ner_counts:
            ner_counts[label] = {}
        if ent.text in ner_counts[label]:
            ner_counts[label][ent.text] += 1
        else:
            ner_counts[label][ent.text] = 1
    return ner_counts




# Define a function to tokenize, lemmatize, and extract dependencies, POs, and ents
def process_text(text):
    # Tokenize and lemmatize the text
    doc = nlp(text)
    tokens = [token.text for token in doc]
    lemmas = [token.lemma_ for token in doc]

    # Extract dependencies, POs, and ents
    dependencies = [token.dep_ for token in doc]
    part_of_speech = [token.pos_ for token in doc]
    entities = [token.ent_type_ for token in doc]

    # Create a dataframe with the processed text
    df = pd.DataFrame(
        {
            "token": tokens,
            "lemma": lemmas,
            "dependency": dependencies,
            "part_of_speech": part_of_speech,
            "entity": entities,
        }
    )
    return df



# Define a function to visualize the data: columns
def show_visualizations_1_POS(df):
    # Calculate the counts of each part of speech
    pos_counts = df["part_of_speech"].value_counts()

    # Plot the part of speech counts as a bar chart
    st.bar_chart(pos_counts)

# Define a function to visualize the data: columns
def show_visualizations_1_NER(df):
    # Filter out blank entities
    ner_entities = df[df["entity"] != ""]
    
    # Calculate the counts of each named entity
    ner_counts = ner_entities["entity"].value_counts()

    # Plot the named entity counts as a bar chart
    st.bar_chart(ner_counts)

    
# Define a function to visualize the data: pie
def show_visualizations_2(df):
    # Calculate the counts of each category
    counts = df["part_of_speech"].value_counts()
    
    # Get the labels and sizes for the pie chart
    labels = counts.index
    sizes = counts.values
    
    # Create the pie chart
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct="%1.1f%%")
    ax.axis("equal")
    
    # Show the pie chart in the app
    st.pyplot(fig)

# Define a function to visualize the data: scatter
def show_visualizations_3(df):
    # Create a dataframe with the lemma counts
    lemma_counts = df["lemma"].value_counts()
    lemma_df = pd.DataFrame({"lemma": lemma_counts.index, "count": lemma_counts.values})
    
    # Create the scatter chart
    chart = (
        alt.Chart(lemma_df)
        .mark_circle()
        .encode(x="lemma", y="count")
    )
    
    # Display the scatter chart in the app
    st.altair_chart(chart, use_container_width=True) 
    
def show_visualizations_4(df):
    import plotly.express as px
    import streamlit as st
    
    df = px.data.gapminder()
    
    fig = px.scatter(
        df.query("year==2007"),
        x="gdpPercap",
        y="lifeExp",
        size="pop",
        color="continent",
        hover_name="country",
        log_x=True,
        size_max=60,
    )
    
    tab1, tab2 = st.tabs(["Streamlit theme (default)", "Plotly native theme"])
    with tab1:
        # Use the Streamlit theme.
        # This is the default. So you can also omit the theme argument.
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    with tab2:
        # Use the native Plotly theme.
        st.plotly_chart(fig, theme=None, use_container_width=True)


    
# Define a function to create a dataframe with linguistic statistics
def create_linguistic_df(text):
    # Tokenize the text
    tokens = nltk.word_tokenize(text)

    # Count the tokens
    token_counts = Counter(tokens)

    # Calculate the token frequencies
    total_tokens = len(tokens)
    token_frequencies = {token: count / total_tokens for token, count in token_counts.items()}

    # Create a dataframe with the token counts, frequencies, lengths, and part-of-speech tags
    df = pd.DataFrame({"token": list(token_counts.keys()), "count": list(token_counts.values()), "frequency": list(token_frequencies.values()), "length": [len(token) for token in token_counts.keys()], "pos_": [nltk.pos_tag([token])[0][1] for token in token_counts.keys()]})
    
    return df
def create_linguistic_df2(text):
    import spacy
    
    # Load the English model
    nlp = spacy.load("en_core_web_sm")
    
    # Tokenize and tag the text
    doc = nlp(text)
    
    # Create a dataframe with the tokens and their POS tags
    data = []
    for token in doc:
        data.append({"token": token.text, "pos": token.pos_})
    df = pd.DataFrame(data)
    return df

def show_visualizations_5(df):
    import plotly.express as px
    import streamlit as st
    

    
    # Create the scatter chart using the dataframe
    fig = px.scatter(
        df,
        x="frequency",
        y="count",
        size="length",
        color="token",
        hover_name="token",
        log_x=True,
        size_max=60,
    )
    
    tab1, tab2 = st.tabs(["Streamlit theme (default)", "Plotly native theme"])
    with tab1:    
        # Use the Streamlit theme.
        # This is the default. So you can also omit the theme argument.
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    with tab2:
        # Use the native Plotly theme.
        st.plotly_chart(fig, theme=None, use_container_width=True)
        
def show_visualizations_6(df):
    import plotly.express as px
    import streamlit as st
    
    # Calculate the word density
    df["density"] = df["count"] / df["length"]
    
    # Create the scatter chart using the dataframe
    fig = px.scatter(
        df,
        x="frequency",
        y="density",
        size="count",
        color="token",
        hover_name="token",
        log_x=True,
        size_max=60,
    )
    
    tab1, tab2 = st.tabs(["Streamlit theme (default)", "Plotly native theme"])
    with tab1:    
        # Use the Streamlit theme.
        # This is the default. So you can also omit the theme argument.
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    with tab2:
        # Use the native Plotly theme.
        st.plotly_chart(fig, theme=None, use_container_width=True)
        
def calculate_tfidf(text):
    # Tokenize the text
    tokens = nltk.word_tokenize(text)

    # Count the tokens
    token_counts = Counter(tokens)

    # Calculate the token frequencies
    total_tokens = len(tokens)
    token_frequencies = {token: count / total_tokens for token, count in token_counts.items()}

    # Create a dataframe with the token counts, frequencies, lengths, and part-of-speech tags
    df = pd.DataFrame({"token": list(token_counts.keys()), "count": list(token_counts.values()), "frequency": list(token_frequencies.values()), "length": [len(token) for token in token_counts.keys()], "pos_": [nltk.pos_tag([token])[0][1] for token in token_counts.keys()]})
        
    # Load the English model
    nlp = spacy.load("en_core_web_sm")
    
    # Tokenize and tag the text
    doc = nlp(text)
    
    # Create a dataframe with the tokens and their POS tags
    #data = []
    for token in df["token"]:
        df.append({"pos": token.pos_})
    #df = pd.DataFrame(data)
    
    
    # Calculate the TF-IDF for each word in the text
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names()
    
    # Create a new column in the dataframe for the TF-IDF values
    df['tfidf'] = [tfidf_matrix[i, j] for i, j in enumerate(df['token'].index)]
    return df

def show_visualizations_7(df):
    import plotly.express as px
    import streamlit as st
    
    # Create the scatter chart using the dataframe
    fig = px.scatter(
        df,
        x="frequency",
        y="count",
        size="length",
        color="token",
        hover_name="token",
        log_x=True,
        size_max=60,
    )
    
    tab1, tab2 = st.tabs(["Streamlit theme (default)", "Plotly native theme"])
    with tab1:    
        # Use the Streamlit theme.
        # This is the default. So you can also omit the theme argument.
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    with tab2:
        # Use the native Plotly theme.
        st.plotly_chart(fig, theme=None, use_container_width=True)
        
        
        
        
# Create a WordNet Lemmatizer
lemmatizer = WordNetLemmatizer()


# Define a function to display the dataframe in the app
@st.cache_data
def get_term_definitions(df):
    st.title("Terminology definitions")
    
    # Create the initial dataframe
    #df = pd.DataFrame(keywords, columns=["Keyword/Keyphrase", "Relevancy"])

    # Create two columns for layout
    col1, col2 = st.columns(2)

    # Display the dataframe in the first column
    col1.dataframe(df)
    #col2.table(df.format(format_dictionary).data)
    

    # Select terms using multiselect in the second column
    selected_terms = col2.multiselect(
        "Select Terms to Generate Definitions For",
        options=list(df["Keyword/Keyphrase"]),
    )
    
    if st.button("Generate Definitions"):
        st.subheader("Definitions")

        # Check if any terms are selected
        if len(selected_terms) > 0:
            # Create a new dataframe for definitions
            definitions_df = pd.DataFrame(selected_terms, columns=["Term"])

            # Add columns for WordNet and Merriam-Webster definitions
            definitions_df["Merriam-Webster Definition"] = definitions_df["Term"].apply(get_merriam_webster_definition)
            definitions_df["WordNet Definition"] = definitions_df["Term"].apply(get_wordnet_definition)

            # Display the definitions dataframe
            st.dataframe(definitions_df)
        else:
            st.write("No terms selected.")


# Function to retrieve the definition and source reference of a term using WordNet
def get_wordnet_definition(term):
    definitions = []
    for token in term.split():
        lemma = lemmatizer.lemmatize(token)
        synsets = wn.synsets(lemma)
        if synsets:
            definitions.append(synsets[0].definition())
    if definitions:
        definition = ', '.join(definitions)
        source = "WordNet"
        return f"{definition} ({source})"
    else:
        return "Definition not found"

# Function to retrieve the definition of a term from Merriam-Webster
def get_merriam_webster_definition(term):
    url = f"https://www.merriam-webster.com/dictionary/{term}"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    definition = soup.find(class_="dtText")
    if definition:
        return definition.text.strip()
    else:
        return "Definition not found"
    
# Function to retrieve the part of speech (POS) of a term using spaCy
def get_pos(term):
    doc = nlp(term)
    if doc and len(doc) > 0:
        return doc[0].pos_
    else:
        return "POS not found"

# Function to retrieve the lemma of a term using spaCy
def get_lemma(term):
    doc = nlp(term)
    if doc and len(doc) > 0:
        return doc[0].lemma_
    else:
        return "Lemma not found"

















