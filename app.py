import streamlit as st
import nltk
import string
import pandas as pd
import matplotlib.pyplot as plt

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from wordcloud import WordCloud

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

st.set_page_config(page_title="AI NLP Explorer", layout="wide")

# ---------------- UI STYLING ---------------- #

st.markdown("""
<style>
.stButton>button {
    width:100%;
    border-radius:8px;
    height:45px;
    font-weight:bold;
}

.llm-btn button {
    background-color:#ff4b4b;
    color:white;
}
</style>
""", unsafe_allow_html=True)

st.title("AI NLP Processing Dashboard")
st.caption("Explore each NLP step interactively using the sidebar.")
st.divider()

# ---------------- DEFINITIONS ---------------- #

def def_tokenization():
    return "Tokenization is the process of breaking text into smaller units called tokens. These tokens are usually words used for further NLP analysis."

def def_stopwords():
    return "Stopwords are common words such as 'the', 'is', and 'and' that usually do not add significant meaning."

def def_stemming():
    return "Stemming reduces words to their base or root form by removing suffixes."

def def_ngram():
    return "N-grams are sequences of consecutive words used to capture context in text."

def def_bow():
    return "Bag of Words converts text into numerical features by counting word occurrences."

def def_tfidf():
    return "TF-IDF measures how important a word is compared to other documents."

def def_wordcloud():
    return "Word Cloud visually displays word frequency in text."

def def_llm():
    return "Large Language Models generate text by predicting the next word in a sequence."

# ---------------- SESSION STATE ---------------- #

if "step" not in st.session_state:
    st.session_state.step = "Input Text"

# ---------------- SIDEBAR BUTTONS ---------------- #

st.sidebar.title("NLP Steps")

col1, col2 = st.sidebar.columns(2)

with col1:
    if st.button("Input Text"):
        st.session_state.step = "Input Text"

    if st.button("Stopword Removal"):
        st.session_state.step = "Stopword Removal"

    if st.button("N-Grams"):
        st.session_state.step = "N-Grams"

    if st.button("TF-IDF"):
        st.session_state.step = "TF-IDF"

with col2:
    if st.button("Tokenization"):
        st.session_state.step = "Tokenization"

    if st.button("Stemming"):
        st.session_state.step = "Stemming"

    if st.button("Bag of Words"):
        st.session_state.step = "Bag of Words"

    if st.button("Word Cloud"):
        st.session_state.step = "Word Cloud"

# Highlighted LLM button
st.sidebar.markdown('<div class="llm-btn">', unsafe_allow_html=True)
if st.sidebar.button("LLM Generator"):
    st.session_state.step = "LLM Generator"
st.sidebar.markdown('</div>', unsafe_allow_html=True)

# ---------------- INPUT BOX AT BOTTOM ---------------- #

st.sidebar.markdown("---")

text = st.sidebar.text_area(
    "Enter Text",
    "Natural Language Processing helps computers understand human language and build intelligent AI systems."
)

# ---------------- PREPROCESS ---------------- #

text_clean = text.lower()
text_clean = text_clean.translate(str.maketrans('', '', string.punctuation))

tokens = word_tokenize(text_clean)

stop_words = set(stopwords.words('english'))
filtered = [w for w in tokens if w not in stop_words]

stemmer = PorterStemmer()
stemmed = [stemmer.stem(w) for w in filtered]

# ---------------- NGRAM FUNCTION ---------------- #

def generate_ngrams(words, n):
    ngrams = zip(*[words[i:] for i in range(n)])
    return [" ".join(ngram) for ngram in ngrams]

step = st.session_state.step

# ---------------- MAIN PAGE ---------------- #

if step == "Input Text":

    st.header("Original Input Text")
    st.write(text)

elif step == "Tokenization":

    st.header("Tokenization")
    st.info(def_tokenization())

    st.subheader("Tokens")
    st.write(tokens)

elif step == "Stopword Removal":

    st.header("Stopword Removal")
    st.info(def_stopwords())

    st.subheader("Filtered Words")
    st.write(filtered)

elif step == "Stemming":

    st.header("Stemming")
    st.info(def_stemming())

    st.subheader("Stemmed Words")
    st.write(stemmed)

elif step == "N-Grams":

    st.header("N-Gram Explorer")
    st.info(def_ngram())

    tab1, tab2, tab3 = st.tabs(["Unigram", "Bigram", "Trigram"])

    with tab1:
        st.write(tokens)

    with tab2:
        st.write(generate_ngrams(tokens, 2))

    with tab3:
        st.write(generate_ngrams(tokens, 3))

elif step == "Bag of Words":

    st.header("Bag of Words")
    st.info(def_bow())

    vectorizer = CountVectorizer()
    bow = vectorizer.fit_transform([" ".join(stemmed)])

    vocab = vectorizer.get_feature_names_out()

    df = pd.DataFrame(bow.toarray(), columns=vocab)

    st.dataframe(df)

    st.subheader("Word Frequency Chart")

    fig, ax = plt.subplots()
    df.iloc[0].plot(kind="bar", ax=ax)
    plt.xticks(rotation=45)

    st.pyplot(fig)

elif step == "TF-IDF":

    st.header("TF-IDF Analyzer")
    st.info(def_tfidf())

    tfidf = TfidfVectorizer()
    matrix = tfidf.fit_transform([" ".join(stemmed)])

    features = tfidf.get_feature_names_out()
    scores = matrix.toarray()[0]

    df = pd.DataFrame({
        "Word": features,
        "Score": scores
    }).sort_values(by="Score", ascending=False)

    st.dataframe(df)

    fig, ax = plt.subplots()
    ax.bar(df["Word"], df["Score"])
    plt.xticks(rotation=45)

    st.pyplot(fig)

elif step == "Word Cloud":

    st.header("Word Cloud")
    st.info(def_wordcloud())

    wordcloud = WordCloud(
        background_color="white",
        width=900,
        height=400
    ).generate(" ".join(stemmed))

    fig, ax = plt.subplots(figsize=(10,5))
    ax.imshow(wordcloud)
    ax.axis("off")

    st.pyplot(fig)

elif step == "LLM Generator":

    st.header("LLM Text Generation")
    st.info(def_llm())

    if st.button("Generate AI Text"):

        from transformers import pipeline

        generator = pipeline("text-generation", model="gpt2")

        result = generator(text, max_length=60)

        st.success(result[0]['generated_text'])