import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Download the stopwords from NLTK
nltk.download('punkt')
nltk.download('stopwords')

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

# Read the CSV file with the correct encoding
df = pd.read_csv("/datasetwuzzuf.csv", encoding='latin1')

# Apply text cleaning and preprocessing to 'Key Skills' column
df['Skills'] = df['Skills'].apply(clean_text)
df['Skills'] = df['Skills'].str.replace('|', ' ', regex=False)
df['Skills'] = df['Skills'].str.lower()
df['Skills'] = df['Skills'].apply(remove_stopwords)

# Select relevant columns and drop NaN values
data = df[['Skills', 'Job Title']].dropna()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['Key Skills'], data['Job Title'], test_size=0.2, random_state=0)

# Vectorize the text data using CountVectorizer
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train a simple classifier (you can choose either SVC or Logistic Regression)
classifier = LogisticRegression()
classifier.fit(X_train_vectorized, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test_vectorized)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='weighted'))
print("Recall:", recall_score(y_test, y_pred, average='weighted'))
