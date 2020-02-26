from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences




sentences = [
    'i love my dog' ,
    'i love my cat',
   'you  love dog!' ,
'Do you  think my dog is amazing?'
]

tokenizer = Tokenizer(num_words = 100 , oov_token = "<RAJ> ")

tokenizer.fit_on_texts(sentences)

word_index = tokenizer.word_index

#create sequences
sequences = tokenizer.texts_to_sequences(sentences)

#print(word_index)
#print(sequences)

test_data = ['i really love my dog',
'my dog loves my menatee']

test_seq = tokenizer.texts_to_sequences(test_data)

#print(test_seq)

#padding: post ,pre and truncating option : maxlen
padded  =  pad_sequences(sequences , padding='post' , maxlen=8)


print(word_index)
print(sentences)
print(sequences)
print(padded)

#test_padded = pad_sequences(test_seq)

#print(test_seq)
#print(test_padded)
