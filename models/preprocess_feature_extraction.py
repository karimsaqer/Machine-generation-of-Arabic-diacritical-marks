import re
from collections import Counter
import pandas as pd
import numpy as np
import tensorflow as tf
import nltk, re
from keras.preprocessing.text import Tokenizer
from datetime import datetime
from gensim.models import *
import logging
# from rnn_utils import *
%matplotlib inline


''' D_NAMES: This is a list containing names of various Arabic diacritics. Each
 element of the list represents a specific diacritic type. '''
D_NAMES = ['Fathatan', 'Dammatan', 'Kasratan', 'Fatha', 'Damma', 'Kasra', 'Shadda', 'Sukun']

##############################################################################################

''' NAME2DIACRITIC: This uses a dictionary comprehension to create a mapping 
from diacritic names to their corresponding Unicode characters.'''
NAME2DIACRITIC = dict((name, chr(code)) for name, code in zip(D_NAMES, range(0x064B, 0x0653)))

##############################################################################################

''' DIACRITIC2NAME: This is the inverse of the previous dictionary.'''
DIACRITIC2NAME = dict((code, name) for name, code in NAME2DIACRITIC.items())

##############################################################################################

''' ARABIC_DIACRITICS: This creates a frozenset containing the Unicode
 characters of all the diacritics.'''
ARABIC_DIACRITICS = frozenset(NAME2DIACRITIC.values())


# Remove all standard diacritics from the text, leaving the letters only.
def clear_diacritics(text):
    assert isinstance(text, str)
    return ''.join([l for l in text if l not in ARABIC_DIACRITICS])


# Return the diacritics from the text while keeping their original positions.
def extract_diacritics(text):
    assert isinstance(text, str)
    diacritics = []
    classes = []
    temp = ''
    for i in range(1, len(text)):
        temp = ''
        if text[i] in ARABIC_DIACRITICS:
            if text[i-1] == NAME2DIACRITIC['Shadda']:
                diacritics[-1] = (DIACRITIC2NAME[text[i-1]], DIACRITIC2NAME[text[i]])
                temp = (DIACRITIC2NAME[text[i-1]], DIACRITIC2NAME[text[i]])
                if (temp == ('Shadda', 'Fatha')):
                    classes.pop()
                    classes.append(8)
                elif (temp == ('Shadda', 'Fathatan')):
                    classes.pop()
                    classes.append(9)
                elif (temp == ('Shadda', 'Damma')):
                    classes.pop()
                    classes.append(10)
                elif (temp == ('Shadda', 'Dammatan')):
                    classes.pop()
                    classes.append(11)
                elif (temp == ('Shadda', 'Kasra')):
                    classes.pop()
                    classes.append(12)
                elif (temp == ('Shadda', 'Kasratan')):
                    classes.pop()
                    classes.append(13)
            else:
                diacritics.append(DIACRITIC2NAME[text[i]])
                temp = DIACRITIC2NAME[text[i]]
                if (temp == 'Fatha'):
                    classes.append(0)
                elif (temp == 'Fathatan'):
                    classes.append(1)
                elif (temp == 'Damma'):
                    classes.append(2)
                elif (temp == 'Dammatan'):
                    classes.append(3)
                elif (temp == 'Kasra'):
                    classes.append(4)
                elif (temp == 'Kasratan'):
                    classes.append(5)
                elif (temp == 'Sukun'):
                    classes.append(6)
                elif (temp == 'Shadda'):
                    classes.append(7)
        elif text[i - 1] not in ARABIC_DIACRITICS:
            diacritics.append('')
            classes.append(14)
        
    if text[-1] not in ARABIC_DIACRITICS:
        diacritics.append('')
        classes.append(14)
    return diacritics, classes


def extract_arabic_words2(text):
    arabic_pattern = re.compile('[\u0600-\u06FF]+')
    arabic_matches = arabic_pattern.findall(text)
    result = ' '.join(arabic_matches)
    processed_text = re.sub(r'[؛،\.]+', '', result)
    final_processed_text = re.sub(r'\s+', ' ', processed_text)
    return final_processed_text


input_file_path = "train.txt"  # Replace with your input file path
with open(input_file_path, "r", encoding="utf-8") as input_file:
    input_text = input_file.read()

arabic_words = extract_arabic_words2(input_text)

output_words = clear_diacritics(arabic_words)
words = output_words.split()
words2 = arabic_words.split()
words_array = [list(word) for word in words]
words_array2 = [list(word2) for word2 in words2]

output_without_spaces = arabic_words.replace(" ", "")
output_without_spaces2 = output_words.replace(" ", "")
array_of_chars = [char for char in output_without_spaces]
_,classes_extraction = extract_diacritics (output_without_spaces)


num_feature = 100
min_word_count = 1
num_thread = 5
window_size = 10
down_sampling = 0.001
iteration = 20

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
model_fastText = FastText(words_array, 
                          vector_size=num_feature, 
                          window=window_size, 
                          min_count=min_word_count, 
                          workers=num_thread)


j=0
chars =[]
char_vectors =[]
char_classes=[]
for word in words_array:
  i=i+1
  for char in word:
    chars.append(char)
    char_classes.append(classes_extraction[j])
    vector = model_fastText.wv[char]
    char_vectors.append(vector)
    j=j+1

print (j)
print(chars[1])
print(char_classes[1])
print(char_vectors[1])
