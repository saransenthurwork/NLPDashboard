import streamlit as st
import re
import nltk
import pandas as pd
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from transformers import pipeline

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except:
    nltk.download('averaged_perceptron_tagger_eng')

try:
    nltk.data.find('corpora/wordnet')
except:
    nltk.download('wordnet')

# Page configuration
st.set_page_config(page_title="NLP & LLM Playground", page_icon="🤖", layout="wide")

st.title("NLP and Large Language Model Demo")

# ---------------------------------------------------------
# Compact Sidebar
# ---------------------------------------------------------

st.sidebar.title("App Features")

st.sidebar.markdown("""
**NLP Processing**
- Lowercase text
- Remove punctuation
- Tokenization
- Stemming (Snowball)
- Lemmatization with POS

**Word Frequency**
- Count word occurrences

**LLM Text Generator**
- Prompt based text generation
- Pretrained model (DistilGPT-2)
""")

st.sidebar.markdown("---")
st.sidebar.caption("Built with Streamlit, NLTK & Transformers")

# ---------------------------------------------------------
# Tabs
# ---------------------------------------------------------

tab1, tab2, tab3 = st.tabs(["NLP Processing", "Word Frequency", "LLM Text Generation"])

stemmer = SnowballStemmer("english")
lemmatizer = WordNetLemmatizer()

# POS full names
pos_full = {
    "NN": "Noun",
    "NNS": "Plural Noun",
    "NNP": "Proper Noun",
    "NNPS": "Proper Plural Noun",
    "VB": "Verb",
    "VBD": "Verb Past",
    "VBG": "Verb Gerund",
    "VBN": "Verb Past Participle",
    "VBP": "Verb Present",
    "VBZ": "Verb Present Singular",
    "JJ": "Adjective",
    "JJR": "Adjective Comparative",
    "JJS": "Adjective Superlative",
    "RB": "Adverb",
    "RBR": "Adverb Comparative",
    "RBS": "Adverb Superlative",
    "IN": "Preposition",
    "CC": "Conjunction",
    "PRP": "Pronoun",
    "DT": "Determiner"
}

# ---------------------------------------------------------
# TAB 1 : NLP PROCESSING
# ---------------------------------------------------------

with tab1:

    st.header("NLP Text Processing")

    text = st.text_area("Enter Text")

    option = st.selectbox(
        "Select NLP Operation",
        (
            "Lowercase",
            "Remove Punctuation",
            "Tokenization",
            "Stemming (Snowball)",
            "Lemmatization with POS"
        )
    )

    if st.button("Process Text"):

        if text.strip() == "":
            st.warning("Please enter text")

        else:

            if option == "Lowercase":
                st.subheader("Lowercase Output")
                st.write(text.lower())

            elif option == "Remove Punctuation":
                clean = re.sub(r"[^\w\s]", "", text)
                st.subheader("Text without Punctuation")
                st.write(clean)

            elif option == "Tokenization":
                tokens = word_tokenize(text)
                st.subheader("Tokens")
                st.write(tokens)

            elif option == "Stemming (Snowball)":
                tokens = word_tokenize(text.lower())
                stemmed = [stemmer.stem(word) for word in tokens]
                st.subheader("Stemmed Words")
                st.write(stemmed)

            elif option == "Lemmatization with POS":

                tokens = word_tokenize(text)
                pos_tags = nltk.pos_tag(tokens)

                data = []

                for word, tag in pos_tags:
                    lemma = lemmatizer.lemmatize(word)
                    meaning = pos_full.get(tag, "Other")
                    data.append([word, tag, meaning, lemma])

                df = pd.DataFrame(data, columns=["Word", "POS Tag", "POS Meaning", "Lemma"])

                st.subheader("Lemmatization Result")
                st.dataframe(df)

# ---------------------------------------------------------
# TAB 2 : WORD FREQUENCY
# ---------------------------------------------------------

with tab2:

    st.header("Word Frequency Analysis")

    text2 = st.text_area("Enter Text for Analysis")

    if st.button("Analyze Frequency"):

        if text2.strip() == "":
            st.warning("Please enter text")

        else:

            clean = re.sub(r"[^\w\s]", "", text2.lower())
            tokens = word_tokenize(clean)

            freq = Counter(tokens)

            df = pd.DataFrame(freq.items(), columns=["Word", "Frequency"])

            st.subheader("Word Frequency Table")
            st.dataframe(df)

# ---------------------------------------------------------
# TAB 3 : LLM TEXT GENERATION
# ---------------------------------------------------------

with tab3:

    st.header("Text Generation using Pretrained Language Model")

    prompt = st.text_input("Enter Prompt")

    @st.cache_resource
    def load_model():
        generator = pipeline("text-generation", model="distilgpt2")
        return generator

    generator = load_model()

    if st.button("Generate Text"):

        if prompt.strip() == "":
            st.warning("Please enter a prompt")

        else:

            result = generator(prompt, max_length=50, num_return_sequences=1)

            st.subheader("Generated Text")
            st.write(result[0]["generated_text"])