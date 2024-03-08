import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk import pos_tag

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def morphological_analysis(text):
    # Tokenization
    words = word_tokenize(text)

    # Stemming
    porter_stemmer = PorterStemmer()
    stemmed_words = [porter_stemmer.stem(word) for word in words]

    # Part-of-speech tagging
    pos_tags = pos_tag(words)

    # Display results
    print("Original Text:", text)
    print("\nTokenized Words:", words)
    print("\nStemmed Words:", stemmed_words)
    print("\nPart-of-Speech Tags:", pos_tags)

if __name__ == "__main__":
    example_text = "Morphological analysis helps in understanding the structure of words and their forms."
    morphological_analysis(example_text)
