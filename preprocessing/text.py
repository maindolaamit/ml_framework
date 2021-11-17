import re
import string

import nltk
import emoji

stopwords = None
try:
    stopwords = nltk.corpus.stopwords.words('english')
except LookupError as le:
    print('Downloading stopwords ...')
    nltk.download('stopwords')
    stopwords = nltk.corpus.stopwords.words('english')

# Download the Wordnet package if not available, it is required by Lemmatizer
nltk.download('wordnet')
wn = nltk.WordNetLemmatizer()
ps = nltk.PorterStemmer()


class Clean:

    def __init__(self, ):
        pass

    @staticmethod
    def remove_punctuation(input_txt):
        """
        Function to remove whitespace and punctuation from the passed text
        :param input_txt: Text to remove punctuation
        :return: Text with punctuation removed
        """
        return ' '.join([word for word in re.sub(r'\s+', ' ', input_txt).split() if word not in string.punctuation])

    @staticmethod
    def tokenize_stopwords(input_txt):
        """
        Function the stopwords from the passed text
        :param input_txt: Text to remove stopwords
        :return: tokens of words
        """
        tokens = [word for word in input_txt.split() if word not in stopwords]
        return tokens

    @staticmethod
    def stem_tokens(tokens):
        """
        Function to apply stemming on the passed token of words
        :param tokens:
        :return: stemming results
        """
        return [ps.stem(word) for word in tokens]

    @staticmethod
    def lammatize_tokens(tokens):
        """
        # Function to perform stemming on the passed token of Words
        :param tokens:
        :return: lamentized results
        """
        return [wn.lemmatize(word) for word in tokens]

    @staticmethod
    def remove_emoji(input_txt):
        """
        Remove any emoji characters from the text
        :param input_txt: Text to clean
        :return: Clean text
        """
        return re.sub(r':', '', emoji.demojize(input_txt))

    @staticmethod
    def remove_url(input_txt):
        """
        Removes any URL headers from the input text. For e.g. input string 'http//google.co.in'
        will be returned as //google.co.in
        :param input_txt: Input string
        :return: transformed string
        """
        return re.sub(r'http\S+', '', input_txt)  # Remove URLs

    @staticmethod
    def remove_dollar(input_txt):
        return re.sub(r'\$\S+', 'dollar', input_txt)  # Change dollar amounts to dollar

    @staticmethod
    def remove_special_chars(input_txt):
        """
        Remove any characters except words and number from passed text.
        :param input_txt: Text to remove characters
        :return: cleaned text.
        """
        return re.sub(r'[^\w\s]', '', input_txt).strip()

    # Function to remove punctuation and Tokenize
    @staticmethod
    def clean_text(input_txt, root_selection='lammetize'):
        input_txt = Clean.remove_punctuation(input_txt)
        input_txt = Clean.remove_special_chars(input_txt)
        tokens = Clean.tokenize_stopwords(input_txt)
        if root_selection == 'lammetize':
            input_txt = ' '.join(Clean.lammatize_tokens(tokens))
        else:
            input_txt = ' '.join(Clean.stem_tokens(tokens))
        return input_txt


if __name__ == '__main__':
    text = "Hello, how are you doing. I am doing gr8 !!! @ ; "
    print(text)
    print(Clean.tokenize_stopwords(Clean.clean_text(text)))

