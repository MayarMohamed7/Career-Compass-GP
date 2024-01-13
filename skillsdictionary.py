import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re

df = pd.read_csv('F:\shithole\4th year-1st semester\Graduation project\skill2vec_10K.csv')

#initialize NLP tools 
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

#clean skills ba 
def clean_skills(skill):
    #remove special characters
    skill = re.sub(r'[^a-zA-Z0-9#.\s]', '', skill, flags=re.I)
    #remove whitespaces
    skill = re.sub(r'\s+', ' ', skill).strip()
    #remove stop words
    skill = ' '.join([word for word in skill.split() if word not in stop_words])
    #lemmatize: reduce words to their root form
    #lemma involves understandong the context its more complex than stemming
    skill = ' '.join([lemmatizer.lemmatize(word) for word in skill.split()])
    return skill

skill_columns = df.columns

skill_dictionary = set()

for column in skill_columns:
    #drop NAN vallues
    skills = df[column].dropna().str.lower()
    skills = skills.apply(clean_skills)
    skill_dictionary.update(skills)

print(skill_dictionary)

# Example usage:
# Print information about a specific skill
print("Information about 'Python':", skills_dict.get('Python', 'Skill not found'))

# Print all skills and their information
for skill, info in skills_dict.items():
    print(f"Skill: {skill}, Information: {info}")
