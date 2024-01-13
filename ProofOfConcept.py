import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk


# Download the stopwords from NLTK
nltk.download('punkt')
nltk.download('stopwords')

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    filtered_text = ' '.join([word for word in tokens if word not in stop_words])
    return filtered_text

df = pd.read_csv("jobs.csv")

df['Skills'] = df['Key Skills'].str.replace('|', ' ', regex=False)
df['Skills'] = df['Skills'].str.lower()
df['Skills'] = df['Skills'].apply(remove_stopwords)

selected_columns = ['Skills', 'Job Title']
print(df[selected_columns].head())

data = df[['Skills', 'Job Title']].dropna()

inputs = data['Skills'].astype(str)
labels = data['Job Title'].astype(str)

X_train, X_test, y_train, y_test = train_test_split(inputs, labels, test_size=0.2, random_state=0)

vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)