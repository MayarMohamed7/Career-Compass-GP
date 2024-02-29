import pandas as pd
import spacy
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
merged_data = pd.read_csv("/content/drive/MyDrive/dataset_grad/mergeddata_outerjoin.csv")

# Load the spaCy English model
nlp = spacy.load("en_core_web_sm")

# Define the remove_stopwords function
def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    filtered_text = ' '.join([word for word in tokens if word.lower() not in stop_words])
    return filtered_text

# Extract skills from a user's paragraph
def extract_skills(text):
    return "my skills are digital marketing|online marketing|sem|search engine marketing|pay per click|google adwords professional|media planning|social media marketing|programmatic|paid media|digital campaigns|campaign management|campaign planning|Google Analytics|DCB"

# Convert skills to vectors
def skills_to_vector(skills, word2vec_model):
    vector = []
    for skill in skills.split('|'):
        if skill in word2vec_model.wv:
            vector.append(word2vec_model.wv[skill])
    return vector

# Load the trained Word2Vec model
word2vec_model = Word2Vec.load("/content/drive/MyDrive/word2vec_model.model")

# User paragraph
user_input = "My skills are digital marketing, online marketing, sem, search engine marketing, pay per click, google adwords professional, media planning, social media marketing, programmatic, paid media, digital campaigns, campaign management, campaign planning, Google Analytics, DCB."

# Remove stop words and punctuation from the user input
user_input_cleaned = remove_stopwords(user_input)

# Extract skills from the user input
user_skills = extract_skills(user_input_cleaned)

# Print user skills
print("User's extracted skills are:", user_skills)

# Check if skills from user input exist in Word2Vec model vocabulary
for skill in user_skills.split('|'):
    if skill not in word2vec_model.wv:
        print(f"Skill '{skill}' not found in Word2Vec model vocabulary")

# Print Word2Vec model vocabulary
print("Word2Vec model vocabulary:", word2vec_model.wv.index_to_key)

# Calculate user skills vector
user_skills_vector = skills_to_vector(user_skills, word2vec_model)

# Print user skills vector
print("User's skills vector:")
for skill, vector in zip(user_skills.split('|'), user_skills_vector):
    print(skill + ":", vector)

# Calculate cosine similarity with all jobs
merged_data['cosine_similarity'] = merged_data['Skills'].apply(lambda x: cosine_similarity(skills_to_vector(x, word2vec_model), user_skills_vector).mean())

# Get recommended job
recommended_job = merged_data.loc[merged_data['cosine_similarity'].idxmax()]

# Print recommended job
print("Recommended job is:", recommended_job['Job Title'])

# Print cosine similarity
print("Cosine similarity:", recommended_job['cosine_similarity'])
