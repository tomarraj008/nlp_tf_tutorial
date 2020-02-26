import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer


sentences = [
    'i love my dog' ,
    'i love my cat'
   'you' , 'love' , 'dog'
]

tokernizer = Tokenizer(num_words=100)


print(tokernizer.fit_on_texts(sentences))


word_index = tokernizer.word_index

print(word_index)
