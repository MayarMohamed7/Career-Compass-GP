#new dataset

import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv('/content/drive/MyDrive/DataGrad/CareerCoach_Dataset.csv')

# Ensure there are no missing values
data = data.dropna()

# Preprocessing Skills
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['Skills'])
sequences = tokenizer.texts_to_sequences(data['Skills'])

# Padding sequences
max_sequence_length = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

# Encoding Job Titles
label_encoder = LabelEncoder()
encoded_job_titles = label_encoder.fit_transform(data['Job Title'])

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, encoded_job_titles, test_size=0.2, random_state=42)

# Define LSTM Model
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100, input_length=max_sequence_length))
model.add(LSTM(100))
model.add(Dense(len(set(encoded_job_titles)), activation='softmax'))

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2)

# Evaluation
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Displaying the first 5 rows of the converted data
converted_data = pd.DataFrame({'Skills ': padded_sequences.tolist(), 'Job Title (Encoded)': encoded_job_titles})
print(converted_data.head())
