import spacy
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams

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
    skill_df = pd.read_csv('cleaned_skill_dictionary.csv')
    skill_df = skill_df.applymap(lambda x: str(x) if pd.notnull(x) else '')
    skill_keywords = pd.Series(skill_df.values.ravel('F')).unique().tolist()
    ignore_words = set(["me", "and", "or", "i", "myself", "experience", "strong", "skills"])
    skills = []

    # Extract named entities
    for ent in doc.ents:
        # Check if the named entity is a skill
        if ent.text.lower() in skill_keywords:
            # Exclude common words and add the preprocessed skill to the list
            if ent.text.lower() not in ignore_words and len(ent.text) > 1:
                skills.append(ent.text.lower().replace(" ", "_").replace(".", "_"))

    # Extract noun chunks
    for chunk in doc.noun_chunks:
        # Check if the chunk is a skill
        if chunk.text.lower() in skill_keywords:
            # Exclude common words and add the preprocessed skill to the list
            if chunk.text.lower() not in ignore_words and len(chunk.text) > 1:
                skills.append(chunk.text.lower().replace(" ", "_").replace(".", "_"))

    # Extract n-grams (bi-grams)
    for token1, token2 in zip(doc[:-1], doc[1:]):
        ngram_text = f"{token1.text.lower().replace(' ', '_')}_{token2.text.lower().replace(' ', '_')}"
        # Check if the n-gram is a skill
        if ngram_text in skill_keywords:
            # Exclude common words and add the preprocessed skill to the list
            if ngram_text not in ignore_words and len(ngram_text) > 1:
                skills.append(ngram_text)

    


    # Remove duplicates and join skills into a string
    unique_skills = list(set(skills))
    return '|'.join(unique_skills)

user_paragraph = "I have experience in Python and Java, with strong skills in machine learning. Excellent communication and teamwork skills."
user_skills = extract_skills(user_paragraph)
print("User's extracted skills are:", user_skills)

