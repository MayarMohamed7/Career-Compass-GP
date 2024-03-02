import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense
from keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv('/content/drive/MyDrive/DataGrad/CareerCoach_Dataset.csv')

# Drop NaN values from the 'Skills' column
data = data.dropna(subset=['Skills'])

# Convert 'Skills' column to strings using .loc
data.loc[:, 'Skills'] = data['Skills'].astype(str)

# Extract texts and labels
texts = data['Skills']
labels = data['Job Title']

# Hyperparameters
max_words = 10000  # max number of words to use in the vocabulary
max_len = 100       # max length of each text (in terms of number of words)
embedding_dim = 100 # dimension of word embeddings
lstm_units = 64     # number of units in the LSTM layer

# Tokenize the texts and create a vocabulary
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# Pad the sequences so they all have the same length
x = pad_sequences(sequences, maxlen=max_len)

# Encode the labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
num_classes = len(label_encoder.classes_)

# Convert labels to one-hot encoded format
y = to_categorical(encoded_labels, num_classes=num_classes)

# Splitting the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Build the model
model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=max_len))
model.add(LSTM(lstm_units))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, batch_size=32, epochs=10, validation_split=0.2)

# Predict job titles for the test set
y_pred = model.predict_classes(x_test)

# Convert encoded job titles back to original labels
predicted_job_titles = label_encoder.inverse_transform(y_pred)

# Convert encoded job titles of the test set back to original labels
true_job_titles = label_encoder.inverse_transform([y.argmax() for y in y_test])

# Generate classification report
report = classification_report(true_job_titles, predicted_job_titles)
print(report)
