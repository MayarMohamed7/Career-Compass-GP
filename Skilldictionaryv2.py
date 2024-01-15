#Skill Dictionary new code version 1
import re

import nltk
import pandas as pd
import spacy
import wordninja
from nltk.tokenize import word_tokenize
from nltk.util import ngrams

# Download the stopwords from NLTK
nltk.download('punkt')
nltk.download('stopwords')

# Load the spaCy English model
nlp = spacy.load("en_core_web_sm")

# Function to segment text without spaces
def segment_text(text):
    return ' '.join(wordninja.split(text))

# Extract skills from a user's paragraph
def extract_skills(text):
    # Segment text without spaces
    segmented_text = segment_text(text)
    doc = nlp(segmented_text)

    # Load the skill dictionary with case-insensitive comparison
    skill_df = pd.read_csv('cleaned_skill_dictionary.csv')
    skill_keywords = skill_df['Skills'].astype(str).str.lower().tolist()
    ignore_words = set(["me", "and", "or", "i", "myself", "experience", "excellent", "skills"])
    skills = []

    # Extract named entities
    for ent in doc.ents:
        # Check if the named entity is a skill
        if ent.text.lower() in skill_keywords:
            # Exclude common words and add the preprocessed skill to the list
            if ent.text.lower() not in ignore_words and len(ent.text) > 1:
                skills.append(ent.text.lower().replace(" ", "_").replace(".", "_"))

    # Iterate over noun chunks
    for chunk in doc.noun_chunks:
        # Check if the chunk is a skill
        chunk_text = chunk.text.lower()
        if chunk_text in skill_keywords:
            # Exclude common words and add the preprocessed skill to the list
            if chunk_text not in ignore_words and len(chunk_text) > 1:
                skills.append(chunk_text.replace(" ", "_").replace(".", "_"))

    # Extract n-grams (bi-grams)
    for token1, token2 in zip(doc[:-1], doc[1:]):
        ngram_text = f"{token1.text} {token2.text}"
        # Check if any part of the n-gram is a skill
        if ngram_text.lower() in skill_keywords:
            # Exclude common words and add the preprocessed skill to the list
            if ngram_text.lower() not in ignore_words and len(ngram_text) > 1:
                skills.append(ngram_text.replace(" ", "_").replace(".", "_"))

    # Remove duplicates and join skills into a string
    unique_skills = list(set(skills))
    return '|'.join(unique_skills)

# Read the dataset, skipping the first column if it contains IDs
df = pd.read_csv('skill2vec_50K.csv', usecols=lambda x: x != 0)

# Function to clean each skill
def clean_skill(skill):
    if isinstance(skill, str):
        cleaned_skill = re.sub(r'[^a-zA-Z#.\s]', '', skill, flags=re.I)
        return cleaned_skill.lower().strip()
    else:
        return ''  # Return an empty string for non-string data types

# Flatten all columns into a single list of skills and apply cleaning
flattened_skills = df.applymap(clean_skill).stack().dropna().unique()

# Convert the array to a set to remove duplicates
skill_dictionary = set(flattened_skills)

# Save the cleaned skill dictionary to a CSV file
pd.DataFrame(list(skill_dictionary), columns=['Skills']).to_csv('cleaned_skill_dictionary.csv', index=False)

#print(skill_dictionary)

user_paragraph = "I possess extensive expertise in Python scripting, showcasing proficiency in crafting robust scripts. My skill set extends to encompass advanced capabilities in both machine learning and web development. Additionally, I have a solid command of HTML, contributing to my comprehensive proficiency in creating dynamic and interactive web applications."
user_skills = extract_skills(user_paragraph)
print("User's extracted skills are:", user_skills)
