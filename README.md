# Machine-generation-of-Arabic-diacritical-marks

This project aims to automatically generate Arabic diacritical marks using three different models: RNN (Recurrent Neural Network), Encoder-Decoder RNN, and LSTM (Long Short-Term Memory). The models are designed to predict diacritical marks for Arabic text, enhancing the accuracy and efficiency of diacritical mark placement.

## Models Used

### 1. RNN (Recurrent Neural Network)
- The RNN model is employed for sequential data processing, making it suitable for predicting diacritical marks in Arabic text.

### 2. Encoder-Decoder RNN
- The Encoder-Decoder RNN architecture is utilized to capture contextual information and improve the accuracy of diacritical mark predictions.

### 3. LSTM (Long Short-Term Memory)
- The LSTM model is incorporated for its ability to capture long-range dependencies in sequences, contributing to the precise placement of diacritical marks.

## Feature Extraction

Various features are extracted to enhance the performance of the models. These features include:

### 1. One-Hot Encoding
- One-hot encoding is employed to represent each Arabic character as a binary vector, providing a categorical representation for the model input.

### 2. Word Embedding
- Word embedding is utilized to capture semantic relationships between words in the Arabic text, enhancing the model's understanding of the context.

### 3. Character Embedding
- Character embedding is applied to represent individual Arabic characters as continuous vectors, contributing to the model's ability to recognize character-level patterns.

### 4. Concatenation of Word Embedding and One-Hot Encoding
- The concatenation of word embedding and one-hot encoding features is used to combine both categorical and semantic information, improving the overall performance of the models.

