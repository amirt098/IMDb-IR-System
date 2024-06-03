import re
import string


class Preprocessor:

    def __init__(self, documents: [str]):
        """
        Initialize the class.

        Parameters
        ----------
        documents : list
            The list of documents to be preprocessed, path to stop words, or other parameters.
        """
        # TODO
        self.documents = documents

        path = '../Logic/core/utility/stopwords.txt'
        with open(path, 'r') as file:
            stopwords = file.read().splitlines()

        self.stopwords = stopwords

    def preprocess(self):
        """
        Preprocess the text using the methods in the class.

        Returns
        ----------
        List[str]
            The preprocessed documents.
        """

        preprocessed_documents = []
        for doc in self.documents:
            doc = self.remove_links(doc)
            doc = self.remove_punctuations(doc)
            doc = self.normalize(doc)
            words = self.tokenize(doc)
            words = self.remove_stopwords(' '.join(words))
            preprocessed_documents.append(' '.join(words))
        return preprocessed_documents

    def normalize(self, text: str):
        """
        Normalize the text by converting it to a lower case, stemming, lemmatization, etc.

        Parameters
        ----------
        text : str
            The text to be normalized.

        Returns
        ----------
        str
            The normalized text.
        """

        text = text.lower()
        text = self.remove_punctuations(text)
        words = self.tokenize(text)
        lemmatized_words = [self.simple_lemmatizer(word) for word in words]
        return ' '.join(lemmatized_words)

    def simple_lemmatizer(self, word: str):
        lemma_dict = {
            "am": "be", "are": "be", "is": "be",
            "was": "be", "were": "be",
            "has": "have", "have": "have", "had": "have",
            "do": "do", "does": "do", "did": "do",
            "car": "car", "cars": "car", "car's": "car", "cars'": "car",
            "boy": "boy", "boys": "boy", "boy's": "boy", "boys'": "boy",
            "girl": "girl", "girls": "girl", "girl's": "girl", "girls'": "girl",
            "give": "give", "gives": "give", "give's": "give",
            "human": "human", "human's": "human", "humans": "human",
            "house": "house", "house's": "house", "houses": "house",
            "cats": "cat", "cat`s": "cat", "dogs": "dog", "dog's": "dog",
            "child": "child", "children": "child", "child's": "child", "children's": "child",
            "run": "run", "runs": "run", "running": "run", "ran": "run",
            "walk": "walk", "walks": "walk", "walking": "walk", "walked": "walk",
            "go": "go", "goes": "go", "going": "go", "went": "go", "gone": "go",
            "be": "be", "being": "be", "been": "be",
            "color": "color", "colors": "color", "color's": "color", "colors'": "color",
            "different": "different", "differently": "different"
        }

        suffix_rules = [
            ('ing', ''), ('ly', ''), ('ed', ''), ('ious', ''), ('ies', 'y'),
            ('ive', ''), ('es', ''), ('ment', '')
        ]

        if word in lemma_dict:
            return lemma_dict[word]

        for suffix, replacement in suffix_rules:
            if word.endswith(suffix):
                base_form = word[:-len(suffix)] + replacement
                if base_form in lemma_dict:
                    return lemma_dict[base_form]
                return base_form

        return word

    def remove_links(self, text: str):
        """
        Remove links from the text.

        Parameters
        ----------
        text : str
            The text to be processed.

        Returns
        ----------
        str
            The text with links removed.
        """
        patterns = [r'\S*http\S*', r'\S*www\S*', r'\S+\.ir\S*', r'\S+\.com\S*', r'\S+\.org\S*', r'\S*@\S*']
        for pattern in patterns:
            text = re.sub(pattern, '', text)
        return text

    def remove_punctuations(self, text: str):
        """
        Remove punctuations from the text.

        Parameters
        ----------
        text : str
            The text to be processed.

        Returns
        ----------
        str
            The text with punctuations removed.
        """
        return text.translate(str.maketrans('', '', string.punctuation))

    def tokenize(self, text: str):
        """
        Tokenize the words in the text.

        Parameters
        ----------
        text : str
            The text to be tokenized.

        Returns
        ----------
        list
            The list of words.
        """
        return re.findall(r'\b\w+\b', text)

    def remove_stopwords(self, text: str):
        """
        Remove stopwords from the text.

        Parameters
        ----------
        text : str
            The text to remove stopwords from.

        Returns
        ----------
        list
            The list of words with stopwords removed.
        """
        words = text.split()
        return [word for word in words if word not in self.stopwords]


if __name__ == '__main__':
    documents = [
        "Check out this link: http://example.com and this one too www.example.org!",
        "The quick brown fox jumps over the lazy dog.",
        "A completely different document with some unique words."
    ]

    preprocessor = Preprocessor(documents)
    processed_documents = preprocessor.preprocess()
    for doc in processed_documents:
        print(doc)
