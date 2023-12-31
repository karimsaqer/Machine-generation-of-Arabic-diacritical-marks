# %%
import re
import numpy as np
from collections import Counter


# %% [markdown]
# # Constants

# %%
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

# %% [markdown]
# ## Functions

# %%
# Extract The Arabic Words From The Text and neglect the un-needed characters, words and numbers.  
def extract_arabic_words(text):
    arabic_pattern = re.compile('[\u0600-\u06FF]+')
    arabic_matches = arabic_pattern.findall(text)
    result = ' '.join(arabic_matches)
    return result

# %%
# Replace {؛, ،, .} with <\s> and append <s> after each replacement
def preprocess_text(text):
    processed_text = re.sub(r'[؛،\.]+', '</s><s>', text)
    return processed_text

# %%
# Return the diacritics from the text while keeping their original positions.
def extract_diacritics(text):
    assert isinstance(text, str)
    diacritics = []
    for i in range(1, len(text)):
        if text[i] in ARABIC_DIACRITICS:
            if text[i-1] == NAME2DIACRITIC['Shadda']:
                diacritics[-1] = (text[i-1], text[i])
            else:
                diacritics.append(text[i])
        elif text[i - 1] not in ARABIC_DIACRITICS:
            diacritics.append('')
    if text[-1] not in ARABIC_DIACRITICS:
        diacritics.append('')
    return diacritics

# %%
# Remove all standard diacritics from the text, leaving the letters only.
def clear_diacritics(text):
    assert isinstance(text, str)
    return ''.join([l for l in text if l not in ARABIC_DIACRITICS])

# %%
# Byte Pair Encoding

def get_stats(vocab): # Get the frequency of each pair of characters in the vocabulary
    pairs = Counter()
    # Loop over the words in the vocabulary
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i + 1]] += freq
    return pairs

def merge_vocab(pair, vocab): # Merge the most frequent pair of characters in the vocabulary
    new_vocab = {}
    # Convert the pair of words into a single word
    bigram = ' '.join(pair)
    replacement = ''.join(pair)
    # Loop over the words in the vocabulary
    for word in vocab:
        # Replace the most frequent pair of characters with the new merged word
        new_word = word.replace(bigram, replacement)
        new_vocab[new_word] = vocab[word]
    
    # Return the new vocabulary
    return new_vocab

def byte_pair_encoding(text, num_merges): # Apply the Byte Pair Encoding algorithm
    # Tokenize the text into Arabic words
    vocab = Counter(text.split())

    # loop for the number of merges
    for i in range(num_merges):
        pairs = get_stats(vocab)
        # Break if there are no more pairs to merge
        if not pairs:
            break
        # Merge the most frequent pair of characters in the vocabulary
        best_pair = max(pairs, key=pairs.get)
        vocab = merge_vocab(best_pair, vocab)

    # Convert the final vocabulary into a list of tokens (Arabic words)
    tokens = list(vocab.keys())

    return tokens

# %% [markdown]
# # Example 
# Here we just try (extract_arabic_words) to a sample data

# %%
# sample data to see the results
sample_Data = '''قَوْلُهُ : ( أَوْ قَطَعَ الْأَوَّلُ يَدَهُ إلَخْ ) قَالَ الزَّرْكَشِيُّ( 14 / 123 )
ابْنُ عَرَفَةَ : قَوْلُهُ : بِلَفْظٍ يَقْتَضِيه كَإِنْكَارِ غَيْرِ حَدِيثٍ بِالْإِسْلَامِ وُجُوبَ مَا عُلِمَ وُجُوبُهُ مِنْ الدِّينِ ضَرُورَةً ( كَإِلْقَاءِ مُصْحَفٍ بِقَذَرٍ وَشَدِّ زُنَّارٍ ) ابْنُ عَرَفَةَ : قَوْلُ ابْنِ شَاسٍ : أَوْ بِفِعْلٍ يَتَضَمَّنُهُ هُوَ كَلُبْسِ الزُّنَّارِ وَإِلْقَاءِ الْمُصْحَفِ فِي صَرِيحِ النَّجَاسَةِ وَالسُّجُودِ لِلصَّنَمِ وَنَحْوِ ذَلِكَ ( وَسِحْرٍ ) مُحَمَّدٌ : قَوْلُ مَالِكٍ وَأَصْحَابِهِ أَنَّ السَّاحِرَ كَافِرٌ بِاَللَّهِ تَعَالَى قَالَ مَالِكٌ : هُوَ كَالزِّنْدِيقِ إذَا عَمِلَ السِّحْرَ بِنَفْسِهِ قُتِلَ وَلَمْ يُسْتَتَبْ .
( قَوْلُهُ لِعَدَمِ مَا تَتَعَلَّقُ إلَخْ ) أَيْ الْوَصِيَّةُ ( قَوْلُهُ مَا مَرَّ ) أَيْ قُبَيْلَ قَوْلِ الْمَتْنِ لَغَتْ وَلَوْ اقْتَصَرَ عَلَى أَوْصَيْت لَهُ بِشَاةٍ أَوْ أَعْطُوهُ شَاةً وَلَا غَنَمَ لَهُ عِنْدَ الْمَوْتِ هَلْ تَبْطُلُ الْوَصِيَّةُ أَوْ يُشْتَرَى لَهُ شَاةٌ وَيُؤْخَذُ مِنْ قَوْلِهِ الْآتِي كَمَا لَوْ لَمْ يَقُلْ مِنْ مَالِي وَلَا مِنْ غَنَمِي أَنَّهَا لَا تَبْطُلُ ، وَعِبَارَةُ الْكَنْزِ وَلَوْ لَمْ يَقُلْ مِنْ مَالِي وَلَا مِنْ غَنَمِي لَمْ يَتَعَيَّنْ غَنَمُهُ إنْ كَانَتْ انْتَهَتْ ا ه سم ( قَوْلُهُ فَيُعْطَى وَاحِدَةً مِنْهَا إلَخْ ) كَمَا لَوْ كَانَتْ مَوْجُودَةً عِنْدَ الْوَصِيَّةِ وَالْمَوْتِ ، وَلَا يَجُوزُ أَنْ يُعْطَى وَاحِدَةً مِنْ غَيْرِ غَنَمِهِ فِي الصُّورَتَيْنِ وَإِنْ تَرَاضَيَا ؛ لِأَنَّهُ صُلْحٌ عَلَى مَجْهُولٍ مُغْنِي وَنِهَايَةٌ قَالَ ع ش قَوْلُهُ وَاحِدَةً مِنْهَا أَيْ كَامِلَةً ، وَلَا يَجُوزُ أَنْ يُعْطَى نِصْفَيْنِ مِنْ شَاتَيْنِ ؛ لِأَنَّهُ لَا يُسَمَّى شَاةً وَقَوْلُهُ وَلَا يَجُوزُ أَنْ يُعْطَى وَاحِدَةً مِنْ غَيْرِ غَنَمِهِ وَيَنْبَغِي أَنْ يُقَالَ مِثْلُ ذَلِكَ فِي الْأَرِقَّاءِ ا ه .'''

# %%
# Extract Arabic words
arabic_words = extract_arabic_words(sample_Data)

# Specify the output file path
output_file_path = "filterd_output.txt"

# Write the Arabic words to the output file
with open(output_file_path, "w", encoding="utf-8") as output_file:
    output_file.write(arabic_words)


# %% [markdown]
# # Pre-process The Training Data
# using our four functions we will read the 'train.txt' to extract just Arabic words and process these words.

# %%
# Read data from input file
input_file_path = "../train.txt"  # Replace with your input file path
with open(input_file_path, "r", encoding="utf-8") as input_file:
    input_text = input_file.read()

# %%
# Extract Arabic words
arabic_words = extract_arabic_words(input_text)

# Specify the output file path
output_file_path = "filtered_output.txt"

# Write the Arabic words to the output file
with open(output_file_path, "w", encoding="utf-8") as output_file:
    output_file.write(arabic_words)


# %%
# Read data from input file
path_filtered = "./filtered_output.txt"  # Replace with your input file path
with open(path_filtered, "r", encoding="utf-8") as input_file:
    input_read = input_file.read()

# %%
# Preprocess the text
processed_text = preprocess_text(input_read)

# Write the preprocessed text to the output file
output_file_path = "processed_output.txt"

with open(output_file_path, "w", encoding="utf-8") as output_file:
    output_file.write(processed_text)

print(f"Preprocessed text has been written to {output_file_path}")


# %%
# Open and read the input text file
input_file_path = 'processed_output.txt'  # Change this to the path of your input file
with open(input_file_path, 'r', encoding='utf-8') as file:
    input_text = file.read()

# Call the extract_diacritics function
output_diacritics = extract_diacritics(input_text)

# Write the output to a new text file
output_file_path = 'diacritics2.txt'  # Change this to the path of your output file
with open(output_file_path, 'w', encoding='utf-8') as file:
    for diacritic in output_diacritics:
        if isinstance(diacritic, tuple):
            file.write(''.join(diacritic) + ' ')
        else:
            file.write(diacritic + ' ')

print(f"Output diacritics written to {output_file_path}")

# %%
# Open and read the input text file
input_file_path = 'processed_output.txt'  # Change this to the path of your input file
with open(input_file_path, 'r', encoding='utf-8') as file:
    input_text = file.read()

# Call the clear_diacritics function
output_words = clear_diacritics(input_text)

# Write the output to a new text file
output_file_path = 'words.txt'  # Change this to the path of your output file
with open(output_file_path, 'w', encoding='utf-8') as file:
    for words in output_words:
        if isinstance(words, tuple):
            file.write(''.join(words))
        else:
            file.write(words)

print(f"Output words written to {output_file_path}")

# %%
# Open and read the input text file
input_file_path = 'filtered_output.txt'  # Change this to the path of your input file
with open(input_file_path, 'r', encoding='utf-8') as file:
    input_text = file.read()

# Call the clear_diacritics function
output_words = clear_diacritics(input_text)

# Write the output to a new text file
output_file_path = 'words_with_separators.txt'  # Change this to the path of your output file
with open(output_file_path, 'w', encoding='utf-8') as file:
    for words in output_words:
        if isinstance(words, tuple):
            file.write(''.join(words))
        else:
            file.write(words)

print(f"Output words_with_separators written to {output_file_path}")

# %%
# Open and read the input text file
input_file_path = 'filtered_output.txt'  # Change this to the path of your input file
with open(input_file_path, 'r', encoding='utf-8') as file:
    input_text = file.read()

tokens = byte_pair_encoding(input_text, 10000000)

# Write the output to a new text file
output_file_path = 'tokens.txt'  
with open(output_file_path, 'w', encoding='utf-8') as file:
    for token in tokens:
        file.write(token + '\n')

print(f"Output tokens written to {output_file_path}")

# %%


# %%



