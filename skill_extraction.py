import spacy
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

# Download the stopwords from NLTK
nltk.download('punkt')
nltk.download('stopwords')

# Load the spaCy English model
nlp = spacy.load("en_core_web_sm")

# Define the clean_text function for text cleaning
def clean_text(text):
    # Your cleaning logic here
    return text

# Define the remove_stopwords function
def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    filtered_text = ' '.join([word for word in tokens if word.lower() not in stop_words])
    return filtered_text

# Extract skills from a user's paragraph
def extract_skills(text):
    doc = nlp(text)
    #need to put skill dictionary insted of this list
    #the skill dictionary that we created
    skill_df = pd.read_csv('skill2vec_50K.csv')
    skill_keywords = pd.Series(skill_df.values.ravel('F')).unique().tolist()
    skills = []

    for token in doc:
        if token.text.lower() in skill_keywords or "skill" in token.text.lower() or token.ent_type_ == "SKILL":
            if token.text.lower() not in ["me", "and", "or", "i", "myself","strong","experience","Excellent"] and len(token.text) > 1:
                skills.append(token.text)

    unique_skills = list(set(skills))
    return '|'.join(unique_skills)

user_paragraph = "I have experience in Python and Java, with strong skills in machine learning. Excellent communication and teamwork skills."
user_skills = extract_skills(user_paragraph)
print ("Users's extracted skills are: ",user_skills)