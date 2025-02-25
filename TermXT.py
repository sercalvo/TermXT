# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 12:59:10 2022

@author: Sergio
"""

# --- Imports ---
import time
import os
from os import path
import streamlit as st
import streamlit.components.v1 as components
from vfunctions import *
import pandas as pd
from pandas import DataFrame
import re
import seaborn as sns
import random
import wikipediaapi
from translate import Translator
import requests
from bs4 import BeautifulSoup
import spacy
from spacy import displacy
from spacy.tokens import Span
from annotated_text import annotated_text
import io


# Instantiate Wikipedia API client
wiki_api = wikipediaapi.Wikipedia('MyProjectName (merlin@example.com)','en')


# --- Set page configuration ---
st.set_page_config(
    page_title="TermXT - Terminology Mining Tool",
    page_icon="img//V-Logo-icon48.png",
)

# Google Analytics
GA_ID = "G-2SV7DZ5WMM"  # Reemplázalo con tu ID de Google Analytics
GA_SCRIPT = f"""
    <!-- Google tag (gtag.js) -->
    <script async src="https://www.googletagmanager.com/gtag/js?id={GA_ID}"></script>
    <script>
        window.dataLayer = window.dataLayer || [];
        function gtag(){{dataLayer.push(arguments);}}
        gtag('js', new Date());

        gtag('config', '{GA_ID}');
    </script>
"""

components.html(GA_SCRIPT, height=0, scrolling=False)

# --- Load custom CSS if available ---
if os.path.exists("styles.css"):
    with open("styles.css") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# --- Determine working directory ---
d = path.dirname(__file__) if "__file__" in locals() else os.getcwd()

# --- SVG Logo ---
svg_logo = """
<svg width="400" height="110" viewBox="50 50 400 120" xmlns="http://www.w3.org/2000/svg">
  <!-- Background -->
  <rect width="100%" height="100%" fill="none" />
  <!-- Speech Bubble -->
  <g transform="translate(50,50)">
    <path d="M0 0 h140 a10 10 0 0 1 10 10 v60 a10 10 0 0 1 -10 10 h-90 l-12 12 l4 -12 a10 10 0 0 0 -4 -2 h-38 a10 10 0 0 1 -10 -10 v-60 a10 10 0 0 1 10 -10 z" fill="#34495E"/>
    <!-- Line connecting nodes (light gray) -->
    <line x1="45" y1="30" x2="95" y2="45" stroke="#BDC3C7" stroke-width="3"/>
    <!-- NLP Nodes -->
    <circle cx="95" cy="45" r="6" fill="#1ABC9C"/>
    <circle cx="45" cy="30" r="6" fill="#d55e00"/>
  </g>
  <!-- Text "LocNLP" -->
  <text x="210" y="85" font-family="Montserrat, sans-serif" font-weight="bold" font-size="35" fill="#2ac1b5">Loc</text>
  <text x="271" y="85" font-family="Montserrat, sans-serif" font-weight="bold" font-size="35" fill="#34495E">NLP</text>
  <!-- Text "Lab23" -->
  <text x="210" y="125" font-family="Montserrat, sans-serif" font-size="38" font-weight="bold" fill="#34495E">Lab</text>
  <text x="277" y="125" font-family="Montserrat, sans-serif" font-size="32" font-weight="bold" fill="#d55e00">23</text>
  <!-- Slogan -->
  <text x="40" y="162" font-family="Akronim, sans-serif" font-size="22" fill="#34495E">LANGUAGE AUTOMATION</text>
</svg>
"""

# Save the SVG to a file
with open("logo.svg", "w") as f:
    f.write(svg_logo)

# Display the SVG logo in the sidebar
st.sidebar.image("logo.svg", width=150)



# --- Download necessary NLTK data ---
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# --- Load spaCy model for English ---
nlp = spacy.load("en_core_web_sm")

# --- Load the Flair model for English ---


####################################
# --- Initialize functions ---
####################################
# Define the global DataFrame variable
# This will store the extracted terms and their metadata
df = None

#
###### Function to extract terms from text#############
@st.cache_resource
def get_random_context(keyword, text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    keyword_sentences = [sentence for sentence in sentences if keyword.lower() in sentence.lower()]
    
    if len(keyword_sentences) >= 2:
        first_sentence = random.choice(keyword_sentences)
        remaining_sentences = [sentence for sentence in keyword_sentences if sentence != first_sentence]
        second_sentence = random.choice(remaining_sentences) if remaining_sentences else "No other sentence"
    elif len(keyword_sentences) == 1:
        first_sentence = keyword_sentences[0]
        second_sentence = "No other sentences found"
    else:
        first_sentence = "No other sentences found"
        second_sentence = "No other sentences found"
    
    return first_sentence, second_sentence






def show_term_extraction_results(text, hits):
    global df  # Use the global DataFrame variable
    
    keywords = verikeybert(text, hits)
    st.subheader("Terminology extraction results\n")    
    
    st.write("##### Please see a list of ", len(keywords)," candidate terms and keywords.")
    
    df = (
        DataFrame(keywords, columns=["Keyword/Keyphrase", "Relevancy"])
        .sort_values(by="Relevancy", ascending=False)
        .reset_index(drop=True)
    )
    
    if add_POS:
        df.insert(1, "POS", df["Keyword/Keyphrase"].apply(lambda x: " ".join(get_pos(word) for word in x.split())))

    if add_lemma:
        #df.insert(2, "lemma", df['Keyword/Keyphrase'].apply(get_lemma))
        df.insert(2, "Lemma", df["Keyword/Keyphrase"].apply(lambda x: " ".join(get_lemma(word) for word in x.split())))
    if add_definition:
        # Add columns for WordNet and Merriam-Webster definitions
        df.insert(3, "WordNet Definition", df["Keyword/Keyphrase"].apply(get_wordnet_definition) )
        df.insert(4, "Merriam-Webster Definition", df["Keyword/Keyphrase"].apply(get_merriam_webster_definition) )
    if add_context:
        df.insert(3, "Context Sentence 2", df["Keyword/Keyphrase"].apply(lambda x: get_random_context(x, text)[1]))
        df.insert(3, "Context Sentence 1", df["Keyword/Keyphrase"].apply(lambda x: get_random_context(x, text)[0]))
        
    # Adjust the index to start at 1
    df.index += 1      
            
    # Add styling
    cmGreen = sns.light_palette("green", as_cmap=True)
    styled_df = df.style.background_gradient(
        cmap=cmGreen,
        subset=[
            "Relevancy",
        ],
    )
    
    c1, c2, c3 = st.columns([1, 3, 1])
    
    format_dictionary = {
        "Relevancy": "{:.1%}",
    }
    
    styled_df = styled_df.format(format_dictionary)
    
    st.table(styled_df)
    st.balloons()
    
    if df is not None:
        st.header("Save the terms")
        
        @st.cache_resource
        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')
        
        csv = convert_df(df)
    

        with st.popover("Save terms as..."):
            st.write("You can save the extracted terms as a CSV or Excel file.")
            c1, c2 = st.columns(2)
            with c1:
                st.download_button(
                    label="Save as CSV",
                    data=csv,
                    file_name='extracted_terms.csv',
                    mime='text/csv',
                )
            with c2:
                st.download_button(
                    label="Save as Excel",
                    data=csv,
                    file_name='extracted_terms.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                )
        

        return styled_df, df
    
@st.cache_resource    
def get_term_definitions():
    global df  # Use the global DataFrame variable
    
    st.header("Generate definitions")
    selected_terms = st.multiselect(
        "Select terms to generate definitions for",
        df["Keyword/Keyphrase"].tolist(),
        default=st.session_state.get("selected_terms", [])
    )
    st.session_state["selected_terms"] = selected_terms
    
    if selected_terms:
        definitions = get_term_definitions(selected_terms)
        if definitions:
            st.table(definitions)
        else:
            st.write("No definitions found for the selected terms.")

### Harvesting metadata for terms ############
@st.cache_resource
def create_df(terms):
    # Perform term extraction using your preferred method
    # Replace this with your actual term extraction code
    keywords = list(term for term in terms)  # Dummy list of extracted terms
    
    # Combine terms into a DataFrame
    df = pd.DataFrame({"Term": keywords})
    
    # Sort DataFrame by term in ascending order
    df = df.sort_values(by="Term", ascending=True).reset_index(drop=True)
    
    # Adjust the index to start at 1
    df.index += 1
    
    return df

@st.cache_resource
def retrieve_definitions(terms, target_language="en", add_POS=False, add_lemma=False, add_translation=False, add_Wikipedia_context=False, add_WordNet_definition=False, add_Merriam_definition=False, add_Wiktionary_definition=False):
    # Retrieve definitions for the selected terms
    # Replace this with your actual definition retrieval code
    definitions = pd.DataFrame(terms, columns=["Term"]).sort_values(by="Term", ascending=False).reset_index(drop=True)
    
    # Add additional columns based on user preferences
    if add_POS:
        definitions["POS"] = definitions["Term"].apply(lambda x: " ".join(get_pos(word) for word in x.split()))
    
    if add_lemma:
        definitions["Lemma"] = definitions["Term"].apply(lambda x: " ".join(get_lemma(word) for word in x.split()))

    if add_translation:
        definitions["Translation"] = definitions["Term"].apply(lambda x: translate_term(x, target_language))
    
    if add_Wikipedia_context:
        # Retrieve context sentences from Wikipedia
        definitions["Wikipedia Context"] = definitions["Term"].apply(get_wikipedia_context)
    
    if add_WordNet_definition:
        definitions["WordNet Definition"] = definitions["Term"].apply(get_wordnet_definition)
    if add_Merriam_definition:
        definitions["Merriam-Webster Definition"] = definitions["Term"].apply(get_merriam_webster_definition)
    if add_Wiktionary_definition:
        definitions["Wiktionary Definition"] = definitions["Term"].apply(get_wiktionary_definition)
    
    # Adjust the index to start at 1
    definitions.index += 1
    
    return definitions

@st.cache_resource
def get_oxford_definition(term):
    # Retrieve definition from Oxford Dictionary
    # Replace with your code to fetch definition from Oxford Dictionary API
    return "Definition of " + term + " from Oxford Dictionary"

### Definition retrieval ############

@st.cache_resource
def get_wiktionary_definition(term):
    # Retrieve definition from Wiktionary
    # Replace with your code to fetch definition from Wiktionary API
    url = f"https://en.wiktionary.org/wiki/{term}"
    
    response = requests.get(url)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find the definition section on the page
        definition_section = soup.find('div', {'class': 'mw-parser-output'})
        
        if definition_section:
            # Extract the first paragraph within the definition section
            first_paragraph = definition_section.find('p')
            
            if first_paragraph:
                # Clean the text by removing HTML tags and extra whitespaces
                cleaned_definition = ' '.join(first_paragraph.stripped_strings)
                cleaned_definition = re.sub(r'\s+([.,!?])', r'\1', cleaned_definition)
                return cleaned_definition.strip()
    
    return "Definition of '" + term + "' not found in Wiktionary"


@st.cache_resource
def get_wikipedia_context(term):
    # Retrieve context sentence from Wikipedia
    page = wiki_api.page(term)
    
    if page.exists():
        return page.summary[0:350]  # Extract the first 200 characters from the Wikipedia summary
    
    return ""

@st.cache_resource
def get_random_context(keyword, text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    keyword_sentences = [sentence for sentence in sentences if keyword.lower() in sentence.lower()]
    
    if len(keyword_sentences) >= 2:
        first_sentence = random.choice(keyword_sentences)
        remaining_sentences = [sentence for sentence in keyword_sentences if sentence != first_sentence]
        second_sentence = random.choice(remaining_sentences) if remaining_sentences else "No other sentence"
    elif len(keyword_sentences) == 1:
        first_sentence = keyword_sentences[0]
        second_sentence = "No other sentences found"
    else:
        first_sentence = "No other sentences found"
        second_sentence = "No other sentences found"
    
    return first_sentence, second_sentence


@st.cache_resource
def get_google_definition(term):
    # Retrieve definition from Google search using "define:" operator
    query = f"define:{term}"
    url = f"https://www.google.com/search?q={query}"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.182 Safari/537.36"
    }
    
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find the div containing the definition
        definition_div = soup.find('div', {'class': 'BNeawe iBp4i AP7Wnd'})
        
        if definition_div:
            definition = definition_div.get_text(separator=' ')
            return definition.strip()
    
    return "Definition of " + term + " from Google"

@st.cache_resource
def translate_term(term, target_language):
    translator = Translator(to_lang=target_language)
    translation = translator.translate(term)
    return translation



### Term annotation ############
@st.cache_resource
def annotate_keyphrases(text, phrases):
    # Load the English language model from spaCy
    nlp = spacy.load("en_core_web_sm")

    # Create a spaCy doc object from the text
    doc = nlp(text)

    # Initialize a list to store the positions and part-of-speech tags of the phrases
    phrase_list = []

    # Use the re.finditer method to find the phrases in the text
    for match in re.finditer(r"\b(" + "|".join(phrases) + r")\b", text):
        # Get the matched phrase
        phrase = match.group()
        # Get the start and end positions of the phrase in the text
        start_pos = match.start()
        end_pos = match.end()
        # Initialize the spans
        spans = []
        # Use the doc.char_span method to get the tokens in the span of the matched phrase
        def find_spans(doc, phrases):
            for phrase in phrases:
                start = doc.text.find(phrase)
                end = start + len(phrase)
                span = doc.char_span(start, end)
                if span is not None:
                    # Add the phrase, its start and end positions, and its part-of-speech tag to the phrase list spans
                    spans.append(Span(doc, span.start, span.end, "TERM"))
            return spans

        # Compile the spans for displacy span visualization
        doc.spans["sc"] = find_spans(doc, phrases)

    # Get the text of the document
    final_text = doc.text

    # Replace the matched phrases with their annotated versions
    for phrase in phrases:
        #final_text = final_text.replace(phrase, "<atg>{}</tag>".format(phrase))
        final_text = final_text.replace(phrase, f"`<term>`**{phrase}**`</term>`")

    # Print to screen the annotated texts
    st.subheader("Term mark-up using Displacy spans")
    st.success("This visualization uses Spacy and the visualizer Displacy to mark up the terms given the list.")
    options = {"ents": ["TERM"], "colors": {"TERM": "#fabc02"}}
    ent_html = displacy.render(doc, style="span", options=options, jupyter=False)
    # Display the entity visualization in the browser:
    st.markdown(ent_html, unsafe_allow_html=True)

    st.write("## Tagged text")
    st.success("This visualization provides a markup with tags `<term>`text`</term>` using Python.")
    st.caption('This is a string that explains something above.')
    # Print the final text and the phrase list
    st.markdown(final_text)

@st.cache_resource
def load_term(chorradas):
    st.write("Extract keywords")
    prueba = verikeybert(chorradas, 10)
    df = DataFrame(prueba, columns=["Keyword", "Relevancy"])
    st.dataframe(df)
    terms = [x for x in df['Keyword']]
    st.write(prueba)
    return terms

@st.cache_resource
def extract_10_terms(text):
    prueba = verikeybert(text, 10)
    df = DataFrame(prueba, columns=["Keyword", "Relevancy"])
    terms = [x for x in df['Keyword']]
    return terms




###############################################################################
# Sidebar Navigation
###############################################################################
st.sidebar.title("TermXT - Terminology Mining Tool")
sections = [
    "🚀 Introduction",
    "✨ Terminology Extraction",
    "🌿 Metadata Harvesting",
    "🖍️ Term Annotation",
    "🎯 Conclusion",
    "☕ Buy Me a Coffee"
]
choice = st.sidebar.radio("Go to Section:", sections)

##################################
# About the author
##################################
with st.sidebar:
    with st.expander("ℹ️ - About this app", expanded=False):
        st.write(
            """     
            - This app is an easy-to-use interface built in Streamlit that uses [KeyBERT](https://github.com/MaartenGr/KeyBERT) library from Maarten Grootendorst!
            - It uses a minimal keyword extraction technique that leverages multiple NLP embeddings and relies on `Transformers` from Hugging Face 🤗 to extract the most relevant keywords/keyphrases, that is to say, the terms in the text!
            - It also uses `Flair` to help adding a pipeline for the Roberta language model from HuggingFace.
            - And it also integrates `keyphrase-vectorizers` to automatically select the best approach regarding how many n-grams to include.
            - Finally, as a translator would suggest, it also has the option to save the terms in CSV.   
            """
        )
    st.caption("By **Sergio Calvo** :sunglasses: :smile: ")


###############################################################################
# 1. Introduction
###############################################################################
if choice == "🚀 Introduction":
    st.image("logo.svg")
    st.title("TermXT - Terminology Mining Tool")
    st.markdown(
        """
        Welcome to **TermXT**, your go-to tool for extracting and analyzing terminology from text. This interactive guide will help you understand the process of terminology extraction and metadata harvesting, providing you with the tools to enhance your text analysis projects.
        """
    )
    annotated_text(
        ("Terminology Mining", "Important!"),
        " is a crucial task in ",
        ("Natural Language Processing", "NLP"),
        " that involves identifying and extracting key terms from text. These terms can be used for various applications such as ",
        ("document summarization", "task"),
        ", ",
        ("keyword extraction", "task"),
        ", and ",
        ("topic modeling", "task"),
        ".",
        "\nUsing Python and various libraries, we can automate the extraction and analysis of terminology. This app is built using **Streamlit**, designed specifically for ",
        ("Machine Learning", "ML"),
        " and ",
        ("Data Science", "DS"),
        " projects."
    )
    st.markdown(
        """              
        Behind the scenes, the code leverages libraries such as `Spacy`, `NLTK`, `Transformers`, `Pandas`, `scikit-learn`, `Beautiful Soup`, `TextBlob`, `Gensim`, `PyTorch`, and many more.
        \n**The main features demonstrated in this app include:**
        """
    )
    nlp_tasks = [
        "Terminology Extraction", "Metadata Harvesting", "Term Annotation",
        "Context Sentence Retrieval", "Definition Retrieval", "POS Tagging",
        "Lemmatization", "Translation", "Wikipedia Context Extraction",
        "WordNet Definition Retrieval", "Merriam-Webster Definition Retrieval",
        "Wiktionary Definition Retrieval"
    ]
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.title(":rainbow: Terminology Mining Features")
        with col2:
            st.markdown(
                """
                `Terminology Extraction`, `Metadata Harvesting`, `Term Annotation`, `Context Sentence Retrieval`, 
                `Definition Retrieval`, `POS Tagging`, `Lemmatization`, `Translation`, `Wikipedia Context Extraction`, 
                `WordNet Definition Retrieval`, `Merriam-Webster Definition Retrieval`, `Wiktionary Definition Retrieval`
                """
            )
    st.markdown(
        """
        \n### Want to learn more?
        \n**👈 Select a demo from the sidebar** to see some interactive terminology mining examples!
        """
    )
    with st.popover("About Sergio"):
        st.markdown(
            """
            <div style="text-align: center;">
            <h3>About the Author</h3>
            <p>
            🙋‍♂️ <strong>Sergio Calvo</strong>
            </p>
            <p>
            🌐 Translator, Reviewer, Computational Linguist, Terminologist, and Localization Engineer with 20+ years of experience in translation, localization, and NLP.
            </p>
            <p>
            💬 Passionate about diving deep into the intricacies of language – whether human or computer – to unveil the beauty of communication.
            </p>
            <p>
            <a href="https://www.veriloquium.com" target="_blank">
            🔗 www.veriloquium.com
            </a>
            </p>
            </div>
            """,
            unsafe_allow_html=True
        )

###############################################################################
# 2. Terminology Extraction
###############################################################################
elif choice == "✨ Terminology Extraction":
    st.title("✨ Terminology Extraction using NLP :bookmark_tabs:")
    st.markdown(
        """
        Terminology extraction is a subtask of information extraction that aims to automatically extract relevant terms from a given text. The extracted terms can be used for various purposes, such as document summarization, keyword extraction, and topic modeling.
        """
    )

    st.markdown(
        """
        It's simple: extract in seconds an accurate list of keywords from a text. Are they terms too? Well, strictly speaking, not in all cases, but you will definitely get a great bunch of the main phrases according to their relevance within the document.
        """
    )


    # Add a header
    st.header("Add your text to extract term candidates")
    #st.subheader("Add some text to extract your terminology candidates")
    st.write("This app will do the rest, that is to say, tokenize the text, remove stopwords and identify the most relevant candidates terms.")

    # get text input from user
    input_type = st.radio('Choose input type:', ['Paste text', 'Select sample data', 'Upload file'], help="Only clean text format (.txt file)")

    if input_type == 'Paste text':
        text = st.text_area('Enter text to analyze')
    elif input_type == 'Select sample data':
        sample_data = {
            "Sample text 1 - Audio interfaces": "An interface allows one thing to interact with another. One of our most common uses of the word is in computing; a human requires a “user interface” to interact with a computer. Likewise, an “audio interface” is a device capable of passing multiple channels of audio to and from a computer in real time. That definition is intentionally broad – many units also contain microphone preamplifiers, basic mixing capabilities, onboard processing, and other features. Smaller interfaces typically carry four channels or less, and are often “bus-powered,” meaning that the USB (or similar) cable from the computer supplies both data and power connectivity. Most larger interfaces carry at least eight channels and require their own power supply.",
            "Sample text 2 - Philosophy": """Jean-Paul Sartre belongs to the existentialists. For him, ultimately humans are "condemned to be free". There is no divine creator and therefore there is no plan for human beings. But what does this mean for love, which is so entwined with ideas of fate and destiny? Love must come from freedom, it must be blissful and mutual and a merging of freedom. But for Sartre, it isn't: love implies conflict. The problem occurs in the seeking of the lover's approval, one wants to be loved, wants the lover to see them as their best possible self. But in doing so one risks transforming into an object under the gaze of the lover, removing subjectivity and the ability to choose, becoming a "loved one". """,
            "Sample text 3 - Wind energy": "Wind is used to produce electricity by converting the kinetic energy of air in motion into electricity. In modern wind turbines, wind rotates the rotor blades, which convert kinetic energy into rotational energy. This rotational energy is transferred by a shaft which to the generator, thereby producing electrical energy. Wind power has grown rapidly since 2000, driven by R&D, supportive policies and falling costs. Global installed wind generation capacity – both onshore and offshore – has increased by a factor of 98 in the past two decades, jumping from 7.5 GW in 1997 to some 733 GW by 2018 according to IRENA’s data. Onshore wind capacity grew from 178 GW in 2010 to 699 GW in 2020, while offshore wind has grown proportionately more, but from a lower base, from 3.1 GW in 2010 to 34.4 GW in 2020. Production of wind power increased by a factor of 5.2 between 2009 and 2019 to reach 1412 TWh.",
            "Sample text 4 - Electronics": "In electronics and telecommunications, modulation is the process of varying one or more properties of a periodic waveform, called the carrier signal, with a separate signal called the modulation signal that typically contains information to be transmitted. For example, the modulation signal might be an audio signal representing sound from a microphone, a video signal representing moving images from a video camera, or a digital signal representing a sequence of binary digits, a bitstream from a computer."
        }
        selected_sample = st.selectbox('Select sample data', list(sample_data.keys()))
        text = sample_data[selected_sample]
    else:
        uploaded_file = st.file_uploader('Upload file', type=['txt'])
        if uploaded_file is not None:
            text = uploaded_file.read().decode('utf-8')
        else:
            text = ''
            

    num_words = count_words(text)

    if text:
        st.subheader('Text to analyze')
        with st.expander("Preview text"):
            st.caption("Showing only first 1000 words in the text")
            words = text.split()[:1000]
            limited_text = ' '.join(words)
            st.markdown(f'<div style="height: 300px; overflow-y: scroll;">{limited_text}</div>', unsafe_allow_html=True)

        # display term extraction
        st.header("Extract the candidate terms")  

        with st.form('extract'):
            
        
            #preview = st.text_area("**Text Preview**", "", height=150, key="preview")
            st.write(f"""#### The text contains `{num_words}` words. Guess how many terms? 
                    \nLet's try to find some terms and keywords. Magic is one click away... Go for it! :dart: !""")
            
            c1, c2 = st.columns(2)
            with c1:
                hits = st.number_input(label='Select the maximum number of terms', min_value=10)
                submit_extract = st.form_submit_button('Extract terms')
            with c2:
                st.caption("**Add metadata fields**")
                # Use st.checkbox() to create checkboxes for enabling stop word removal and lemmatization
                add_POS = st.checkbox(":green[Add POS tags]", help="It will add the Part Of Speech to each term.")
                add_lemma = st.checkbox(":green[Add lemma]", help="It will add the lemma or cannonical form of the word.")
                add_definition = st.checkbox(":green[Add definition]", help="It will add Merriam-Webster and WordNet definitions to each term.")
                add_context = st.checkbox(":green[Add context sentences]", help="It will add random context sentences to each term.")

            
            
        
        
        if submit_extract:
            styled_df, df = show_term_extraction_results(text, hits)
            st.session_state['df'] = df


###############################################################################
# 3. Terminology harvesting
###############################################################################
elif choice == "🌿 Metadata Harvesting":
    # Streamlit app code
    st.title("🌿 Metadata generation for term harvesting 	:face_with_monocle:")
    st.write("This app aims at providing contextual information for term harvesting. It retrieves a range of details for the terms, such as grammatical information, context sentences and several definitions from diferent sources.")

    st.markdown(
        """
        Terminology harvesting is the process of collecting, extracting, and organizing terminology from a variety of sources. This process is essential for building and maintaining terminology resources such as glossaries, dictionaries, and ontologies.
        """
    )
    st.write("This section will help you generate definitions for the extracted terms.")
    


    # Terms input for metadata populating
    st.subheader('Add your terms for metadata generation')
    input_terms= "".join(st.text_area('Add your own keywords, phrases or terms separated by comma', 'harvest, terminology', key="my_input_kw_area", help="Use comma without space to separate the terms like: electrical energy,kinetic energy,modern wind turbines,rotational energy,wind,rotor blades,motion,generator")).split(sep=",")

    # Term extraction settings
    #hits = st.number_input("Maximum number of terms", min_value=1, value=5)

    # Perform term extraction
    if st.button("Load terms for analysis"):
        if input_terms:
            with st.expander("See table with terms"):
                df_terms = create_df(input_terms)
                st.write("Terms provided:")
                st.table(df_terms)
                #st.success("Term extraction completed")
            
            # Store df_terms in session state
            st.session_state.df_terms = df_terms
        else:
            st.warning("Please enter text to extract terms")
            
        

    # Select terms for definition retrieval
    if "df_terms" in st.session_state:
        
        st.subheader("**Add metadata fields**")
        target_language_code = "en"
        c1, c2 = st.columns(2)
        with c1:
            # Use st.checkbox() to create checkboxes for enabling stop word removal and lemmatization
            add_POS = st.checkbox(":green[Add POS tags]", help="It will add the Part Of Speech to each term.")
            add_lemma = st.checkbox(":green[Add lemma]", help="It will add the lemma or cannonical form of the word.")
            
            add_translation = st.checkbox(":green[Add Translation]", help="It will add the translation of each term to the selected language.")
            if add_translation:
                target_language = st.selectbox("Select Target Language", ["English", "French", "Spanish", "German"])
                target_language_code = "en"  # Default to English
            
                # Map target language to language code
                language_mapping = {
                    "English": "en",
                    "French": "fr",
                    "Spanish": "es",
                    "German": "de"
                }

                target_language_code = language_mapping.get(target_language, target_language_code)

            
            
        with c2:
            add_Wikipedia_context = st.checkbox(":green[Add Wikipedia context sentence]", help="It will add Wikipedia context sentences to each term.")
            add_WordNet_definition = st.checkbox(":green[Add WordNet definition]", help="It will add a WordNet definition to each term.")
            add_Merriam_definition = st.checkbox(":green[Add Merriam-Webster Definition]", help="It will add a Merriam-Webster definition to each term.")
            add_Wiktionary_definition = st.checkbox(":green[Add Wiktionary definition]", help="It will add a Wiktionary definition to each term.")
            #add_Google_definition = st.checkbox(":green[Add Google 'define:']", help="It will add random context sentences to each term.")

        selected_terms = st.multiselect("Select terms for metadata generation", st.session_state.df_terms["Term"].tolist())

        # Retrieve definitions for selected terms
        if selected_terms:
            # Pass the metadata options to the retrieve_definitions function
            df_definitions = retrieve_definitions(selected_terms, target_language_code, add_POS, add_lemma, add_translation, add_Wikipedia_context, add_WordNet_definition, add_Merriam_definition, add_Wiktionary_definition)

            st.write("Term Definitions:")
            st.table(df_definitions)

            # Add download buttons for CSV and Excel
            csv = df_definitions.to_csv(index=False).encode('utf-8')
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                df_definitions.to_excel(writer, index=False, sheet_name='Sheet1')
                writer.close()
            excel_data = excel_buffer.getvalue()

            c1, c2 = st.columns(2)
            with c1:
                st.download_button(
                    label="Save as CSV",
                    data=csv,
                    file_name='term_metadata.csv',
                    mime='text/csv',
                )
            with c2:
                st.download_button(
                    label="Save as Excel",
                    data=excel_data,
                    file_name='term_metadata.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                )

###############################################################################
# 4. Term Annotation    
###############################################################################
elif choice == "🖍️ Term Annotation":
    # Add a page title for the app
    st.title('🖍️ Term Annotation :male-detective:')
    st.markdown('This app makes use of `Spacy` library to provide nice visualizations of terms and keywords.')

    st.markdown(
        """
        Term annotation is the process of marking up text with annotations that provide additional information about the terms and phrases in the text. This process is useful for highlighting key terms, providing definitions, and adding context to the text.
        """)


    # Store the initial value of widgets in session state
    if "checkbox" not in st.session_state:
        st.session_state.visibility = "visible"
        st.session_state.disabled = False
        st.session_state["disabled"] = False


    # Add a header for the first section: Select text
    st.header("Add your text and terms to annotate")

    # Define session_state function to later hide the checkbox if pasted text is passed in or clear the fields
    def clear_form():
        st.session_state["my_input_area"] = ""
        st.session_state["my_input_kw_area"] = ""
        st.session_state["disabled"] = False
        st.session_state["checkbox"] = False

    # get text input from user
    input_type = st.radio('Choose input type:', ['Paste text', 'Select sample data', 'Upload file'], help="Only clean text format (.txt file)")
    if input_type == 'Paste text':
        text = st.text_area('Enter text to analyze')
    elif input_type == 'Select sample data':
        sample_data = {
            "Sample text 1 - Audio interfaces": "An interface allows one thing to interact with another. One of our most common uses of the word is in computing; a human requires a 'user interface' to interact with a computer. Likewise, an 'audio interface' is a device capable of passing multiple channels of audio to and from a computer in real time. That definition is intentionally broad – many units also contain microphone preamplifiers, basic mixing capabilities, onboard processing, and other features. Smaller interfaces typically carry four channels or less, and are often 'bus-powered,' meaning that the USB (or similar) cable from the computer supplies both data and power connectivity. Most larger interfaces carry at least eight channels and require their own power supply.",
            "Sample text 2 - Philosophy": "Jean-Paul Sartre belongs to the existentialists. For him, ultimately humans are 'condemned to be free.' There is no divine creator and therefore no plan for human beings. But what does this mean for love, which is so entwined with ideas of fate and destiny? Love must come from freedom, it must be blissful and mutual and a merging of freedom. But for Sartre, it isn't: love implies conflict. The problem occurs in the seeking of the lover's approval; one wants to be loved, wants the lover to see them as their best possible self. But in doing so, one risks transforming into an object under the gaze of the lover, removing subjectivity and the ability to choose, becoming a 'loved one.'",
            "Sample text 3 - Wind energy": "Wind is used to produce electricity by converting the kinetic energy of air in motion into electricity. In modern wind turbines, wind rotates the rotor blades, which convert kinetic energy into rotational energy. This rotational energy is transferred by a shaft to the generator, thereby producing electrical energy. Wind power has grown rapidly since 2000, driven by R&D, supportive policies, and falling costs. Global installed wind generation capacity – both onshore and offshore – has increased by a factor of 98 in the past two decades, jumping from 7.5 GW in 1997 to some 733 GW by 2018 according to IRENA’s data. Onshore wind capacity grew from 178 GW in 2010 to 699 GW in 2020, while offshore wind has grown proportionately more, but from a lower base, from 3.1 GW in 2010 to 34.4 GW in 2020. Production of wind power increased by a factor of 5.2 between 2009 and 2019 to reach 1412 TWh.",
            "Sample text 4 - Electronics": "In electronics and telecommunications, modulation is the process of varying one or more properties of a periodic waveform, called the carrier signal, with a separate signal called the modulation signal that typically contains information to be transmitted. For example, the modulation signal might be an audio signal representing sound from a microphone, a video signal representing moving images from a video camera, or a digital signal representing a sequence of binary digits, a bitstream from a computer."
        }
        selected_sample = st.selectbox('Select sample data', list(sample_data.keys()))
        text = sample_data[selected_sample]
    else:
        uploaded_file = st.file_uploader('Upload file', type=['txt'])
        if uploaded_file is not None:
            text = uploaded_file.read().decode('utf-8')
        else:
            text = ''

    if text:
        st.subheader('Text to analyze')
        st.caption("Showing only first 1000 words in the text")
        words = text.split()[:1000]
        limited_text = ' '.join(words)
        st.markdown(f'<div style="height: 300px; overflow-y: scroll;">{limited_text}</div>', unsafe_allow_html=True)

    # display the annotated text
    terms = []

    # Add a form for the user to paste a text
    with st.form(key='my_annotation'):
        # Section header for terms the annotated text
        st.subheader('Add your terms to annotate')
        input_phrases = "".join(st.text_input('Add your own keywords, phrases or terms separated by comma', 'interface,computing,user interface,computer,audio interface,channel,microphone preamplifiers,mixing,bus-powered,USB,power supply', key="my_input_kw_area", help="Use comma without space to separate the terms like: electrical energy,kinetic energy,modern wind turbines,rotational energy,wind,rotor blades,motion,generator")).split(sep=",")

        # Create two columns for two buttons
        f1, f2 = st.columns(2)
        with f1:
            # Button to send text and phrases to print on screen
            gettext_button = st.form_submit_button(label='Annotate terms in the text')

        with f2:
            agree = st.checkbox(':orange[Extract terms automatically]', help="Automatic terminology extraction will extract the best 10 candidates using several algorithms to identify the most relevant collocations or single words.")

            if agree:
                st.write('Great!')
                terms = extract_10_terms(text)

    def ui_message(message):
        placeholder = st.empty()
        placeholder.success(f'{message}')
        time.sleep(1)
        placeholder.empty()
        return

    def ui_warning(message):
        placeholder = st.empty()
        placeholder.warning(f'{message}')
        time.sleep(1)
        placeholder.empty()
        return

    if gettext_button:
        if agree == False and input_phrases == "":
            ui_warning("Hey, no term! Mark at least the automatic extraction :smile:")
        elif text == "":
            ui_warning("No text, my friend!")
        else:
            st.subheader("Preview text and terms ")
            st.write('**The text to be processed :point_down:**')
            c5, c6 = st.columns([1, 3])
            with c5:
                st.write(pd.DataFrame({'Keyphrases': input_phrases + terms })) 
            with c6:
                st.markdown(f":green[{text}]")
            
            final_phrases = input_phrases + terms
                
            st.header("Visualize and annotate the text")
            annotate_keyphrases(text, final_phrases)

            
###############################################################################
# 5. Conclusion
###############################################################################
elif choice == "🎯 Conclusion":
    st.header("🎯 Conclusion")
    st.write(
        """
        Terminology extraction and metadata harvesting are crucial tasks in Natural Language Processing (NLP) that help in identifying and analyzing key terms from text.
        
        In this app, we explored essential features, including:
        
        🔹 **Terminology Extraction:**  
           - Extracting relevant terms and keywords from text  
           - Adding metadata such as POS tags, lemmas, definitions, and context sentences  
        
        🔹 **Metadata Harvesting:**  
           - Collecting additional information for terms  
           - Retrieving definitions from various sources  
           - Translating terms to different languages  
        
        🔹 **Term Annotation:**  
           - Annotating text with key terms  
           - Visualizing terms using spaCy and Displacy  
        
        These features serve as building blocks for powerful text analysis and terminology management applications.
        
        🚀 **Next Steps:**  
        Feel free to experiment with the interactive examples and explore how these techniques can be applied in real-world scenarios. Mastering these core concepts will help you unlock the full potential of terminology extraction and metadata harvesting.
        
        🔗 **Continue Learning:**  
        For a deeper dive into terminology extraction and NLP, check out more resources and tutorials available online.
        """
    )
    st.markdown("---")

###############################################################################
# 6. Buy Me a Coffee
###############################################################################
elif choice == "☕ Buy Me a Coffee":
    st.title("☕ Buy Me a Coffee")
    st.write(
        """
        If you found this terminology mining tool helpful and would like to support my work, it's not a bad day to consider **buying me a coffee**! Your support helps me create more useful content, tutorials, and projects for the community. 
        Every coffee fuels more research, coding, and open-source contributions! 🚀🔥
        """
    )
    coffee_button = """
    <div style="display: flex; justify-content: center;">
        <a href="https://www.buymeacoffee.com/sergiocalvc" target="_blank">
            <img src="https://img.buymeacoffee.com/button-api/?text=Buy me a coffee&emoji=☕&slug=sergiocalvc&button_colour=FFDD00&font_colour=000000&font_family=Arial&outline_colour=000000&coffee_colour=ffffff" alt="Buy Me a Coffee">
        </a>
    </div>
    """
    st.markdown(coffee_button, unsafe_allow_html=True)
    st.markdown("---")
    st.write(
        """
        🔗 **Other ways to support:**  
        - Share this project with your network  
        - Provide feedback and suggestions  
        - Connect with me on [LinkedIn](https://www.linkedin.com/in/sergiocalvopaez)
        """
    )
    st.markdown(
        """
        ### 💛 Your support makes a difference! [![Buy Me a Coffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-Support%20Me%20☕-yellow)](https://www.buymeacoffee.com/sergiocalvc)
        """
    )

# --- Footer ---
st.markdown(
    """<p style='text-align: center;'> Brought to you with <span style='color:red'>❤</span> by <a href='https://www.veriloquium.com/'>Sergio Calvo</a> | Veriloquium © LocNLP Lab23 </p>""",
    unsafe_allow_html=True
)
