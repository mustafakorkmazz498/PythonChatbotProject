import nltk
from nltk.stem.lancaster import LancasterStemmer
##nltk.download("punkt")
from flask import Flask, render_template, request, jsonify
import numpy
import tflearn
import tensorflow
import random
import json
import pickle
stemmer = LancasterStemmer()


app = Flask(__name__)

#open metodu ve utf 8 olarak json dosyasını açmaya yarıyor
with open("intents.json", encoding="utf-8") as file:
    data = json.load(file)
#data pickle var mı bakmak için yazıyoruz

#yukarıda açtığımız json dosyasını tensorflow kelime kelime ayrıştırıyor bu data pickle olayı
try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []  #words adında boş bir dizi oluşturdum kelimeler bu dizinin içine doluyor
    labels = []
    docs_x = []
    docs_y = []

#verilerin içindeki kelimeler intents ve pattern başlığı altında ayrıştırıyor
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)     #x ve y olarak eğitiyor ve ekliyor
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])
#kelimeleri düzenli şekilde listeliyor.
    words = [stemmer.stem(w.lower()) for w in words if w !="?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append((output_row))
#ayrıştırılmış kelimeleri cevap için açıyor
        with open("data.pickle", "wb") as f:
            pickle.dump((words, labels, training, output), f)

#bu kısım modeli eğitip kaydediyor.
tensorflow.compat.v1.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)


model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
model.save("model.tflearn")




#CHATBOT karşılıklı cevaplaşmaları sağlıyor

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return  numpy.array(bag)


def chat(message):
    print("params "+message)


    inp = message

    results = model.predict([bag_of_words(inp, words)])
    results_index = numpy.argmax(results)
    tag = labels[results_index]

    for tg in data["intents"]:
        if tg["tag"] == tag:
            responses = tg["responses"]


    print("ChatBot: " + random.choice(responses))


    return random.choice(responses)

@app.route('/get_response', methods=['POST'])
def get_response():
    message = request.form.get('user_input')
    res = chat(message)
    return jsonify({'response': res})

@app.route('/')
def index():
    return render_template('chat.html')

if __name__ == "__main__":
    app.run(debug=False)
















