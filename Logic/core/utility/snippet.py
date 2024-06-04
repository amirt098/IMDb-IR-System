class Snippet:
    def __init__(self, number_of_words_on_each_side=5):
        """
        Initialize the Snippet

        Parameters
        ----------
        number_of_words_on_each_side : int
            The number of words on each side of the query word in the doc to be presented in the snippet.
        """
        self.number_of_words_on_each_side = number_of_words_on_each_side

        self.stop_words = frozenset([
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
            'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself',
            'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
            'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
            'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for',
            'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
            'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
            'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
            'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just',
            'don', 'should', 'now'
        ])

    def remove_stop_words_from_query(self, query):
        """
        Remove stop words from the input string.

        Parameters
        ----------
        query : str
            The query that you need to delete stop words from.

        Returns
        -------
        str
            The query without stop words.
        """

        # TODO: remove stop words from the query.

        query_tokens = query.lower().split()
        filtered_query = [token for token in query_tokens if token not in self.stop_words]
        return ' '.join(filtered_query)

    def find_snippet(self, doc, query):
        """
        Find snippet in a doc based on a query.

        Parameters
        ----------
        doc : str
            The retrieved doc which the snippet should be extracted from that.
        query : str
            The query which the snippet should be extracted based on that.

        Returns
        -------
        final_snippet : str
            The final extracted snippet. IMPORTANT: The keyword should be wrapped by *** on both sides.
            For example: Sahwshank ***redemption*** is one of ... (for query: redemption)
        not_exist_words : list
            Words in the query which don't exist in the doc.
        """
        final_snippet = ""

        # TODO: Extract snippet and the tokens which are not present in the doc.
        # print(doc, query)
        filtered_query = self.remove_stop_words_from_query(query)
        doc_tokens = doc.split()

        snippets = []
        not_exist_words = []

        for token in filtered_query.split(' '):
            if token in doc_tokens:
                token_indices = [i for i, x in enumerate(doc_tokens) if x == token]
                for index in token_indices:
                    start = max(index - self.number_of_words_on_each_side, 0)
                    end = min(index + self.number_of_words_on_each_side + 1, len(doc_tokens))
                    snippet = doc_tokens[start:index] + [f'***{token}***'] + doc_tokens[index + 1:end]
                    snippets.append(" ".join(snippet))
            else:
                not_exist_words.append(token)

        final_snippet = ' ... '.join(snippets)

        return final_snippet, not_exist_words

        # filtered_query = self.remove_stop_words_from_query(query).split()
        # doc_tokens = doc.split()
        #
        # snippets = []
        # not_exist_words = []
        # token_positions = {token: [] for token in filtered_query}
        #
        # # Gather all positions for each query token in the document
        # for i, word in enumerate(doc_tokens):
        #     if word in token_positions:
        #         token_positions[word].append(i)
        #
        # for token in filtered_query:
        #     if token_positions[token]:
        #         # Find the best window for each token considering other tokens
        #         best_window = None
        #         best_window_score = -1
        #
        #         for index in token_positions[token]:
        #             start = max(index - self.number_of_words_on_each_side, 0)
        #             end = min(index + self.number_of_words_on_each_side + 1, len(doc_tokens))
        #             window_tokens = doc_tokens[start:end]
        #
        #             # Score the window based on how many query tokens it includes
        #             score = sum(1 for t in filtered_query if t in window_tokens)
        #
        #             if score > best_window_score:
        #                 best_window_score = score
        #                 best_window = window_tokens
        #
        #         # Highlight the current token in the best window
        #         if best_window:
        #             highlighted_window = [f'***{word}***' if word == token else word for word in best_window]
        #             snippets.append(" ".join(highlighted_window))
        #     else:
        #         not_exist_words.append(token)
        #
        # final_snippet = '...'.join(snippets)
        #

if __name__ == '__main__':
    snippet_extractor = Snippet(number_of_words_on_each_side=5)

    filtered_query = snippet_extractor.remove_stop_words_from_query("quick brown fox typing")
    print(filtered_query, 'filtered_query')

    doc = ("The  brown fox brown jumps over the lazy dog. This classic sentence has been used to "
           "demonstrate typing and font samples because it includes every letter of the alphabet. "
           "The movements of the fox and the lethargic demeanor of the dog make for an interesting juxtaposition.")

    final_snippet, not_exist_words = snippet_extractor.find_snippet(doc, "quick brown fox typing")
    print("Final Snippet:", final_snippet)
    print("Not Exist Words:", not_exist_words)
