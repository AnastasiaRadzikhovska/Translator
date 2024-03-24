import numpy as np  
import tensorflow as tf
import os
import shutil
import zipfile
import pandas as pd

# Шлях до директорії, де знаходиться "por.txt"
file_path = r"F:\\Lab\\archive\\por.txt"

# Зчитування даних за допомогою pandas
data = pd.read_csv(file_path, sep='\t', header=None, names=['input', 'target', 'other_column'])

# Виведення перших декількох рядків даних
print(data.head())

batch_size = 64  # Встановлення розміру пакета на 64
latent_dim = 256  # Встановлення розмірності латентного простору на 256
num_samples = 10000  # Встановлення кількості вибірок на 10000 для тренування

# Підготовка даних
input_texts = []  # Ініціалізація списку для збереження вхідних текстів
target_texts = []  # Ініціалізація списку для збереження цільових текстів
input_characters = set()  # Ініціалізація множини для унікальних символів вводу
target_characters = set()  # Ініціалізація множини для унікальних символів виводу

# Визначення шляху до директорії, де знаходиться "por.txt"
destination_directory = r"F:\\Lab\\"
archive_directory = os.path.join(destination_directory, "archive")

with open(os.path.join(archive_directory, "por.txt"), "r", encoding="utf-8") as f:
    lines = f.read().split("\n")

# Ітерація по кожному рядку у файлі
for line in lines[: min(num_samples, len(lines) - 1)]:
    input_text, target_text, _ = line.split("\t")  # Розділення рядка на вхідний та цільовий текст
    target_text = "\t" + target_text + "\n"  # Додавання токенів початку та кінця до цільового тексту
    input_texts.append(input_text)  # Додавання вхідного тексту до списку
    target_texts.append(target_text)  # Додавання цільового тексту до списку
    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)  # Додавання унікальних символів до множини вводу
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)  # Додавання унікальних символів до множини виводу

# Перетворення символів у відсортовані списки та отримання їхніх довжин
input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])
input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])
encoder_input_data = np.zeros((len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype="float32")
decoder_input_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32")
decoder_target_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32")

# Ітерація по кожній парі вхідного та цільового тексту
for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    # Кодування вхідної послідовності 
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.0
    encoder_input_data[i, t + 1 :, input_token_index[" "]] = 1.0
    # Кодування цільової послідовності 
    for t, char in enumerate(target_text):
        decoder_input_data[i, t, target_token_index[char]] = 1.0
        if t > 0:
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.0
    decoder_input_data[i, t + 1 :, target_token_index[" "]] = 1.0
    decoder_target_data[i, t:, target_token_index[" "]] = 1.0

# Визначення архітектури моделі кодера-декодера 
encoder_inputs = tf.keras.Input(shape=(None, num_encoder_tokens))
encoder = tf.keras.layers.GRU(latent_dim, return_state=True)
encoder_outputs, state_h = encoder(encoder_inputs)
decoder_inputs = tf.keras.Input(shape=(None, num_decoder_tokens))
decoder_gru = tf.keras.layers.GRU(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _ = decoder_gru(decoder_inputs, initial_state=state_h)
decoder_dense = tf.keras.layers.Dense(num_decoder_tokens, activation="softmax")
decoder_outputs = decoder_dense(decoder_outputs)
model = tf.keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
encoder_inputs = model.input[0]
encoder_outputs, state_h_enc = model.layers[2].output
encoder_model = tf.keras.Model(encoder_inputs, state_h_enc)
model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
decoder_inputs = model.input[1]
decoder_state_input_h = tf.keras.Input(shape=(latent_dim,))
decoder_gru = model.layers[3]
decoder_outputs, state_h_dec = decoder_gru(decoder_inputs, initial_state=decoder_state_input_h)
decoder_dense = model.layers[4]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = tf.keras.Model([decoder_inputs] + [decoder_state_input_h], [decoder_outputs] + [state_h_dec])

# Тренування моделі
model.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size=batch_size, epochs=100, validation_split=0.2)

# Декодування тестових послідовностей
reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())
for seq_index in range(20):
    input_seq = encoder_input_data[seq_index : seq_index + 1]
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    target_seq[0, 0, target_token_index["\t"]] = 1.0
    stop_condition = False
    decoded_sentence = ""
    while not stop_condition:
        output_tokens, h = decoder_model.predict([target_seq] + [states_value])
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char
        if sampled_char == "\n" or len(decoded_sentence) > max_decoder_seq_length:
            stop_condition = True
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.0
        states_value = [h]
    print("-")
    print("Input sentence:", input_texts[seq_index])
    print("Decoded sentence:", decoded_sentence)
