import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
from PIL import Image

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Comment Toxicity Detection System",
    page_icon=None,
    layout="centered"
)

st.markdown("""
<style>

    
header {
    visibility: hidden;
}

div[data-baseweb="select"] > div {
    border: 2px solid #4A90E2 !important;
    border-radius: 12px !important;
    background-color: white !important;
    box-shadow: 0px 2px 10px rgba(0,0,0,0.08);
}

@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700&display=swap');
                   
[data-testid="stToolbar"] {
    display: none;
}

[data-testid="stHeader"] {
    display: none;
}
            
.main, .block-container {
    position: relative;
    z-index: 1;
}
            
body {
    background-color: #f5f5f5;
}

.stApp {
    background-color: #f5f5f5;
}

section.main > div {
    padding-top: 1rem;
}

.block-container {
    padding-top: 1rem;
    padding-bottom: 0rem;
}
            
/* Watermark Background Logos */
.stApp::before {
    content: "";

    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;

    background-image:
        url("https://cdn-icons-png.flaticon.com/512/733/733547.png"),
        url("https://cdn-icons-png.flaticon.com/512/2111/2111463.png"),
        url("https://cdn-icons-png.flaticon.com/512/733/733579.png"),
        url("https://cdn-icons-png.flaticon.com/512/1384/1384060.png");

    background-repeat: no-repeat;

    background-size:
        180px,
        180px,
        180px,
        180px;

    background-position:
        left 40px top 180px,
        right 40px top 180px,
        left 40px bottom 120px,
        right 40px bottom 120px;

    opacity: 0.12;

    z-index: 0;
    pointer-events: none;
}
            

/* TEXT AREA OUTLINE */
textarea {
    border: 2px solid #7aa7ff !important;
    border-radius: 12px !important;
    background-color: #f5f7fb !important;
    padding: 12px !important;
}

/* WHEN CLICKED */
textarea:focus {
    border: 2px solid #3b82f6 !important;
    box-shadow: 0 0 10px rgba(59,130,246,0.3) !important;
}

</style>
""", unsafe_allow_html=True)


import re
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.base import BaseEstimator, TransformerMixin

# NLTK Downloads
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('punkt_tab')

# Custom Transformer

class TextPreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self):

        self.stop_word = set(stopwords.words('english'))

        self.lemmatizer = WordNetLemmatizer()

        self.word_tokenize = word_tokenize

    def text_cleaning(self, text):

        text = str(text).lower()

        # remove links
        text = re.sub(r'https?://\S+|www\.\S+', ' ', text)

        # remove hashtags
        text = re.sub(r'#\w+', ' ', text)

        # remove mentions
        text = re.sub(r'@\w+', ' ', text)

        # remove emails
        text = re.sub(r'\S+@\S+', ' ', text)

        # remove special chars
        text = re.sub(r'[^a-z0-9\s]', ' ', text)

        # remove extra spaces
        text = ' '.join(
            [word for word in text.split() if word.isalpha()]
        )

        return text

    def tokenize_lemmatize(self, text):

        tokens = self.word_tokenize(text)

        return ' '.join([
            self.lemmatizer.lemmatize(word)
            for word in tokens
            if word not in self.stop_word
        ])

    def fit(self, X, y=None):

        return self

    def transform(self, X, y=None):

        return [
            self.tokenize_lemmatize(
                self.text_cleaning(text)
            )
            for text in X
        ]
    

# =========================
# LOAD MODELS
# =========================

with open("logistic_regression_pipeline.pkl", "rb") as file:
    lr_model = pickle.load(file)

with open("linear_svc_pipeline.pkl", "rb") as file:
    svc_model = pickle.load(file)

# =========================
# TITLE SECTION
# =========================

st.markdown("""
<h1 style='
    text-align: center;
    font-size: 58px;
    color: black;
    margin-top: 10px;
    line-height: 1.2;
    font-weight: 700;
    font-family: "Montserrat", sans-serif;
'>
Comment Toxicity Detection<br>
System
</h1>
""", unsafe_allow_html=True)

# SUBTITLE
st.markdown("""
<h3 style='
    text-align: center;
    color: #555555;
    margin-top: 10px;
    margin-bottom: 35px;
'>
NLP • Machine Learning • Transformers
</h3>
""", unsafe_allow_html=True)

# =========================
# DROPDOWN NAVIGATION
# =========================

from streamlit_option_menu import option_menu

page = option_menu(
    menu_title=None,
    options=[
        "Home",
        "Data Insights & Model Comparison",
        "Live Toxicity Detection",
        "CSV Bulk Prediction"
    ],
    orientation="horizontal"
)


# =========================
# HOME PAGE
# =========================

if page == "Home":


# SOCIAL MEDIA LOGOS
    st.markdown("""
        <div style='
        display:flex;
        justify-content:center;
        gap:60px;
        margin-top:40px;
'>

<img src="https://cdn-icons-png.flaticon.com/512/733/733547.png" width="65">

<img src="https://cdn-icons-png.flaticon.com/512/2111/2111463.png" width="65">

<img src="https://cdn-icons-png.flaticon.com/512/733/733579.png" width="65">

<img src="https://cdn-icons-png.flaticon.com/512/1384/1384060.png" width="65">

</div>
""", unsafe_allow_html=True)

# SOCIAL MEDIA NAMES
st.markdown("""
<h4 style='
    text-align:center;
    color:#666666;
    margin-top:15px;
'>
Facebook • Instagram • Twitter • YouTube
</h4>
""", unsafe_allow_html=True)

# =========================
# DATA INSIGHTS PAGE
# =========================

if page == "Data Insights & Model Comparison":

    st.title("Data Insights")

    st.write("")

    # CLASS IMBALANCE
    st.subheader("Class Imbalance")

    image = Image.open("class_imbalance.png")
    st.image(image, use_container_width=True)

    st.success("""
    The dataset is highly imbalanced with a majority of non-toxic comments.
    This makes F1-score and Recall important evaluation metrics.
    """)

    # COMMENT LENGTH DISTRIBUTION
    st.subheader("Comment Length Distribution")

    image = Image.open("comment_length_distribution.png")
    st.image(image, use_container_width=True)

    st.success("""
Most comments are short in length, while a small number of comments are extremely long.
The distribution is highly right-skewed, showing the presence of outliers.
""")

    # COMMENT LENGTH BY TOXICITY
    st.subheader("Comment Length by Toxicity")

    image = Image.open("comment_length_toxicity.png")
    st.image(image, use_container_width=True)

    st.success("""
Toxic comments generally have a wider spread and more extreme outliers compared to non-toxic comments.
This indicates that toxic comments can vary significantly in length.
""")

    # WORDCLOUD
    st.subheader("WordCloud Visualization")

    image = Image.open("wordcloud.png")
    st.image(image, use_container_width=True)

    st.success("""
Frequently occurring words such as 'article', 'page', 'Wikipedia', and 'source'
appear prominently in the dataset. This helps identify commonly used terms in comments.
""")

    st.write("---")

    # MODEL COMPARISON
    st.title("Model Comparison")

    model_data = pd.DataFrame({
        "Model": [
            "Multinomial NB",
            "Logistic Regression",
            "Linear SVC"
        ],
        "F1 Score": [
            0.21,
            0.67,
            0.72
        ]
    })

    fig = px.bar(
        model_data,
        x="Model",
        y="F1 Score",
        color="Model",
        text="F1 Score",
        title="Model Performance Comparison"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.write("---")

    # CLASSIFICATION REPORTS

    st.subheader("📌 Logistic Regression")

    st.code("""
Precision : 0.87
Recall    : 0.55
F1 Score  : 0.67
""")

    st.subheader("📌 Linear SVC")

    st.code("""
Precision : 0.83
Recall    : 0.63
F1 Score  : 0.72
""")

    st.subheader("📌 Multinomial NB")

    st.code("""
Precision : 0.88
Recall    : 0.12
F1 Score  : 0.21
""")


    st.success("""
Final Observation:

Linear SVC achieved the best traditional machine learning
performance with balanced precision and recall.
""")

# =========================
# LIVE TOXICITY DETECTION
# =========================

elif page == "Live Toxicity Detection":

    st.title("⚡ Live Toxicity Detection")

    st.markdown("""
    <h4 style='text-align:center; color:#666666;'>
    Facebook • Instagram • Twitter • YouTube
    </h4>
    """, unsafe_allow_html=True)

    st.markdown("""
### 💡 Example Comments

✅ Non-Toxic:
- "Great video, very informative!"
- "I really enjoyed this content."
- "Amazing explanation, thank you!"

🚨 Toxic:
- "You are stupid."
- "This is the worst thing ever."
- "Nobody likes you."
""")
    

    # COMMENT INPUT
    comment = st.text_area(
        "Enter a comment",
        height=150
    )

    # PREDICT BUTTON
    if st.button("Predict Toxicity"):

        # EMPTY INPUT CHECK
        if comment.strip() == "":
            st.warning("Please enter a comment.")

        else:

# =========================
# TOXICITY PROBABILITIES
# =========================

            probability = lr_model.predict_proba([comment])[0]

# Convert to percentages
            all_probabilities = probability * 100

# Maximum probability
            toxic_probability = round(
                max(all_probabilities),
                2
)

# =========================
# LABELS
# =========================

            labels = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]

            detected_labels = []

# =========================
# DETECT LABELS
# =========================

            for i, score in enumerate(all_probabilities):

                if score >= 30:

                    detected_labels.append(
                    f"{labels[i]} ({round(score,2)}%)"
        )
            # =========================
            # FINAL OUTPUT
            # =========================

            if detected_labels:

                st.error("🚨 Toxic Comment Detected")

                st.write("### Detected Labels")

                for label in detected_labels:
                    st.write(f"• {label}")

            else:

                st.success("✅ Non-Toxic Comment")

            # =========================
            # TOXICITY METRIC
            # =========================

            st.metric(
                label="Toxicity Probability",
                value=f"{toxic_probability}%"
            )

            # =========================
            # PROBABILITY BREAKDOWN
            # =========================

            st.write("### Category-wise Toxicity Scores")

            for i, score in enumerate(all_probabilities):

                st.write(
                f"**{labels[i]}** → {round(score,2)}%"
                )

                st.progress(min(score / 100, 1.0))


# =========================
# CSV BULK PREDICTION
# =========================

elif page == "CSV Bulk Prediction":

    st.title("CSV Bulk Prediction")

    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=["csv"]
    )

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Dataset")

        st.dataframe(df.head())

        if st.button("Predict Comments"):

            with st.spinner("Predicting comments... Please wait ⏳"):

                # MODEL PREDICTIONS
                predictions = lr_model.predict(
                    df["comment_text"]
                )

                # ADD PREDICTIONS TO DATAFRAME
                df["Prediction"] = predictions.max(axis=1)

                # CONVERT 0/1 TO TEXT
                df["Prediction"] = df["Prediction"].map({
                    0: "Non-Toxic",
                    1: "Toxic"
                })

                st.success("Prediction Completed ✅")

                st.subheader("Prediction Results")

                st.dataframe(df.head())

                # DOWNLOAD CSV
                csv = df.to_csv(index=False).encode("utf-8")

                st.download_button(
                    label="Download Predictions",
                    data=csv,
                    file_name="toxicity_predictions.csv",
                    mime="text/csv"
                )