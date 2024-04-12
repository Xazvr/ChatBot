import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Cargar el modelo entrenado
model = load_model('chatbot_model.h5')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    sorted_indexes = np.argsort(res)[::-1]
    categories = [classes[i] for i in sorted_indexes]
    probabilities = [res[i] for i in sorted_indexes]
    return categories, probabilities

def get_responses(tag, intents_json, pattern_indexes):
    list_of_intents = intents_json['intents']
    responses = []
    for i in list_of_intents:
        if i["tag"] == tag:
            if 'responses' in i:
                for index in pattern_indexes:
                    if len(i['responses']) > index:
                        responses.append(i['responses'][index])
            break
    return responses

previous_responses = set()

while True:
    message = input("")
    categories, probabilities = predict_class(message)
    responses = get_responses(categories[0], intents, range(len(categories)))

    # Filtrar respuestas previas y seleccionar respuestas ponderadas por probabilidad
    available_responses = [response for response in responses if response not in previous_responses]

    if available_responses:
        # Selección diversificada de respuestas basada en la probabilidad
        selected_responses = random.choices(available_responses, weights=probabilities, k=min(2, len(available_responses)))
        
        for selected_response in selected_responses:
            previous_responses.add(selected_response)  # Agregar la respuesta a respuestas previas
            print(selected_response)
    else:
        print("Lo siento, no tengo una respuesta para eso.")
