import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
import re

# Sample text data
text_data = """
Your sample text data goes here. Add more sentences to create a larger dataset.
For example, you can add some lines of poetry or prose.
"""

# Adding start and stop tokens using regex for more robust sentence splitting
text_data = re.sub(r'([.!?])', r' \1 stopseq startseq ', text_data).strip()
text_data = 'startseq ' + text_data + ' stopseq'

# Tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text_data])
total_words = len(tokenizer.word_index) + 1

# Create input sequences using the tokens
input_sequences = []
for line in text_data.split('startseq'):
    line = line.strip()
    if line:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)

# Pad sequences
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# Create predictors and labels
xs, labels = input_sequences[:,:-1], input_sequences[:,-1]
ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)

# Build the Model
model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))
model.add(Bidirectional(LSTM(150)))
model.add(Dense(total_words, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Train the Model
history = model.fit(xs, ys, epochs=100, verbose=1)

# Generate Text
def generate_text(seed_text, next_words, max_sequence_len):
    seed_text = 'startseq ' + seed_text
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict(token_list, verbose=0)
        predicted_word_index = np.argmax(predicted, axis=1)[0]
        output_word = tokenizer.index_word.get(predicted_word_index, '')
        if output_word in ['stopseq', '']:
            break
        seed_text += " " + output_word
    return seed_text.replace('startseq ', '').replace(' stopseq', '')

seed_text = "Your starting seed text"
generated_text = generate_text(seed_text, 20, max_sequence_len)
print(generated_text)
