import tensorflow as tf
from tensorflow import keras
import numpy as np

data = keras.datasets.imdb

(train_data, train_labels), (test_data,
                             test_labels) = data.load_data(num_words=88000)


# get word to index list
word_index = data.get_word_index()

word_index = {k: (v+3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED"] = 3

# numbers point to words
reverse_word_index = dict([(value, key)
                          for (key, value) in word_index.items()])


# make input length to equal max review length and add padding to fill smaller texts
train_data = keras.preprocessing.sequence.pad_sequences(
    train_data, value=word_index["<PAD>"], padding="post", maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(
    test_data, value=word_index["<PAD>"], padding="post", maxlen=250)


# translate text from numbers to words
def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])


# train model
'''
best_acc = 0
for i in range(10):
    (train_data, train_labels), (test_data,
                                 test_labels) = data.load_data(num_words=88000)

    # make input length to equal max review length and add padding to fill smaller texts
    train_data = keras.preprocessing.sequence.pad_sequences(
        train_data, value=word_index["<PAD>"], padding="post", maxlen=250)
    test_data = keras.preprocessing.sequence.pad_sequences(
        test_data, value=word_index["<PAD>"], padding="post", maxlen=250)

    # initialize model
    model = keras.Sequential()
    # embedding layer translates word indices to vectors
    model.add(keras.layers.Embedding(88000, 16))
    # averaged layer (shrinks vector data down)
    model.add(keras.layers.GlobalAveragePooling1D())
    # hidden fully connected layer
    model.add(keras.layers.Dense(16, activation="relu"))
    # output neuron (good or bad review)
    model.add(keras.layers.Dense(1, activation="sigmoid"))

    model.summary()

    # configure model for training
    model.compile(optimizer="adam", loss="binary_crossentropy",
                  metrics=["accuracy"])

    x_val = train_data[:10000]
    x_train = train_data[10000:]

    y_val = train_labels[:10000]
    y_train = train_labels[10000:]

    fitModel = model.fit(x_train, y_train, epochs=40,
                         batch_size=512, validation_data=(x_val, y_val), verbose=1)

    results = model.evaluate(test_data, test_labels)
    acc = results[1]
    print(results)

    if acc >= best_acc:
        best_acc = acc
        model.save("model.h5")
'''

# test_review = np.array([test_data[1]])
# prediction = model.predict(test_review)

# print(decode_review(test_data[1]))
# print(f'Prediction: {prediction}')
# print(f'actual: {test_labels[1]}')


def review_encode(s):
    encoded = [1]

    for word in s:
        if word.lower() in word_index:
            encoded.append(word_index[word.lower()])  # convert word to index
        else:
            encoded.append(2)  # if not in vocab -> mark as UNK (unknown)

    return encoded


model = keras.models.load_model("model.h5")

# with keyword closes file automatically after code block
with open("test.txt", encoding="utf-8") as f:
    for line in f.readlines():
        # clean text
        nline = line.replace(",", "").replace(".", "").replace(
            "(", "").replace(")", "").replace("\"", "").replace(":", "").strip().split(" ")
        # encode word to indexes
        encode = review_encode(nline)
        encode = train_data = keras.preprocessing.sequence.pad_sequences(
            [encode], value=word_index["<PAD>"], padding="post", maxlen=250)
        predict = model.predict(encode)
        print(line)
        print(encode)

        if predict > 0.7:
            print("Good review")
        elif predict < 0.3:
            print("Bad review")
        else:
            print("Neutral review")
