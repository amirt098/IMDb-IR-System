import numpy as np


class Scorer:
    def __init__(self, index, number_of_documents):
        """
        Initializes the Scorer.

        Parameters
        ----------
        index : dict
            The index to score the documents with.
        number_of_documents : int
            The number of documents in the index.
        """

        self.index = index
        self.idf = {}
        self.N = number_of_documents

    def get_list_of_documents(self, query):
        """
        Returns a list of documents that contain at least one of the terms in the query.

        Parameters
        ----------
        query: List[str]
            The query to be scored

        Returns
        -------
        list
            A list of documents that contain at least one of the terms in the query.

        Note
        ---------
            The current approach is not optimal but we use it due to the indexing structure of the dict we're using.
            If we had pairs of (document_id, tf) sorted by document_id, we could improve this.
                We could initialize a list of pointers, each pointing to the first element of each list.
                Then, we could iterate through the lists in parallel.

        """
        list_of_documents = []
        for term in query:
            if term in self.index.keys():
                list_of_documents.extend(self.index[term].keys())
        return list(set(list_of_documents))

    def get_idf(self, term):
        """
        Returns the inverse document frequency of a term.

        Parameters
        ----------
        term : str
            The term to get the inverse document frequency for.

        Returns
        -------
        float
            The inverse document frequency of the term.

        Note
        -------
            It was better to store dfs in a separate dict in preprocessing.
        """
        idf = self.idf.get(term, None)
        if idf is None:

            df = len(self.index.get(term, {}))

            if df > 0:
                idf = np.log((self.N - df + 0.5) / (df + 0.5)) + 1

            else:
                idf = 0

            self.idf[term] = idf
        return idf

    def get_query_tfs(self, query):
        """
        Returns the term frequencies of the terms in the query.

        Parameters
        ----------
        query : List[str]
            The query to get the term frequencies for.

        Returns
        -------
        dict
            A dictionary of the term frequencies of the terms in the query.
        """

        tfs = {}

        for term in query:

            # tfs[term] = 1 if term not in tfs else tfs[term] + 1
            if term in tfs:
                tfs[term] += 1
            else:
                tfs[term] = 1
        return tfs

    def compute_scores_with_vector_space_model(self, query, method):
        """
        compute scores with vector space model

        Parameters
        ----------
        query: List[str]
            The query to be scored
        method : str ((n|l)(n|t)(n|c).(n|l)(n|t)(n|c))
            The method to use for searching.

        Returns
        -------
        dict
            A dictionary of the document IDs and their scores.
        """

        query_tfs = self.get_query_tfs(query)
        scores = {}

        documents = self.get_list_of_documents(query)
        doc_method, query_method = method.split('.')

        for document_id in documents:
            score = self.get_vector_space_model_score(query, query_tfs, document_id, doc_method, query_method)
            scores[document_id] = score

        return scores

    def get_vector_space_model_score(
            self, query, query_tfs, document_id, document_method, query_method
    ):
        """
        Returns the Vector Space Model score of a document for a query.

        Parameters
        ----------
        query: List[str]
            The query to be scored
        query_tfs : dict
            The term frequencies of the terms in the query.
        document_id : str
            The document to calculate the score for.
        document_method : str (n|l)(n|t)(n|c)
            The method to use for the document.
        query_method : str (n|l)(n|t)(n|c)
            The method to use for the query.

        Returns
        -------
        float
            The Vector Space Model score of the document for the query.
        """

        score = 0
        doc_vector = self.index.get(document_id, {})

        for term in query:
            if term in doc_vector:
                tf = doc_vector[term]
                idf = self.get_idf(term)
                query_tf = query_tfs.get(term, 0)

                if document_method[1] == 'n':
                    doc_weight = tf
                elif document_method[1] == 'l':
                    doc_weight = 1 + np.log(tf) if tf > 0 else 0
                elif document_method[1] == 't':
                    doc_weight = (1 + np.log(tf)) * idf if tf > 0 else 0
                elif document_method[1] == 'c':
                    max_tf = max(doc_vector.values())
                    doc_weight = (0.5 + 0.5 * tf / max_tf) * idf if tf > 0 else 0

                if query_method[1] == 'n':
                    query_weight = query_tf
                elif query_method[1] == 'l':
                    query_weight = 1 + np.log(query_tf) if query_tf > 0 else 0
                elif query_method[1] == 't':
                    query_weight = (1 + np.log(query_tf)) * idf if query_tf > 0 else 0
                elif query_method[1] == 'c':
                    max_query_tf = max(query_tfs.values())
                    query_weight = (0.5 + 0.5 * query_tf / max_query_tf) * idf if query_tf > 0 else 0

                score += doc_weight * query_weight

        return score

    def compute_socres_with_okapi_bm25(
            self, query, average_document_field_length, document_lengths
    ):
        """
        compute scores with okapi bm25

        Parameters
        ----------
        query: List[str]
            The query to be scored
        average_document_field_length : float
            The average length of the documents in the index.
        document_lengths : dict
            A dictionary of the document lengths. The keys are the document IDs, and the values are
            the document's length in that field.

        Returns
        -------
        dict
            A dictionary of the document IDs and their scores.
        """
        scores = {}
        documents = self.get_list_of_documents(query)

        for document_id in documents:
            score = self.get_okapi_bm25_score(query, document_id, average_document_field_length, document_lengths)
            scores[document_id] = score

        return scores

    def get_okapi_bm25_score(
            self, query, document_id, average_document_field_length, document_lengths
    ):
        """
        Returns the Okapi BM25 score of a document for a query.

        Parameters
        ----------
        query: List[str]
            The query to be scored
        document_id : str
            The document to calculate the score for.
        average_document_field_length : float
            The average length of the documents in the index.
        document_lengths : dict
            A dictionary of the document lengths. The keys are the document IDs, and the values are
            the document's length in that field.

        Returns
        -------
        float
            The Okapi BM25 score of the document for the query.
        """

        k1 = 1.5
        b = 0.75
        score = 0

        doc_length = document_lengths.get(document_id, 0)

        for term in query:
            if term in self.index:
                term_freq = self.index[term].get(document_id, 0)
                idf = self.get_idf(term)
                term_score = idf * ((term_freq * (k1 + 1)) /
                                    (term_freq + k1 * (1 - b + b * (doc_length /
                                                                    average_document_field_length))))
                score += term_score

        return score

    def compute_scores_with_unigram_model(
            self, query, smoothing_method, document_lengths=None, alpha=0.5, lamda=0.5
    ):
        """
        Calculates the scores for each document based on the unigram model.

        Parameters
        ----------
        query : str
            The query to search for.
        smoothing_method : str (bayes | naive | mixture)
            The method used for smoothing the probabilities in the unigram model.
        document_lengths : dict
            A dictionary of the document lengths. The keys are the document IDs, and the values are
            the document's length in that field.
        alpha : float, optional
            The parameter used in bayesian smoothing method. Defaults to 0.5.
        lamda : float, optional
            The parameter used in some smoothing methods to balance between the document
            probability and the collection probability. Defaults to 0.5.

        Returns
        -------
        float
            A dictionary of the document IDs and their scores.
        """
        scors = {}
        documents = self.get_list_of_documents(query)

        for document_id in documents:
            score = self.compute_score_with_unigram_model(query, document_id, smoothing_method, document_lengths, alpha,
                                                          lamda)
            scors[document_id] = score

        return scors

    def compute_score_with_unigram_model(
            self, query, document_id, smoothing_method, document_lengths, alpha, lamda
    ):
        """
        Calculates the scores for each document based on the unigram model.

        Parameters
        ----------
        query : str
            The query to search for.
        document_id : str
            The document to calculate the score for.
        smoothing_method : str (bayes | naive | mixture)
            The method used for smoothing the probabilities in the unigram model.
        document_lengths : dict
            A dictionary of the document lengths. The keys are the document IDs, and the values are
            the document's length in that field.
        alpha : float, optional
            The parameter used in bayesian smoothing method. Defaults to 0.5.
        lamda : float, optional
            The parameter used in some smoothing methods to balance between the document
            probability and the collection probability. Defaults to 0.5.

        Returns
        -------
        float
            The Unigram score of the document for the query.
        """

        score = 0
        doc_length = document_lengths.get(document_id, 0)
        collection_length = sum(document_lengths.values())
        for term in query:
            term_freq = self.index.get(term, {}).get(document_id, 0)
            collection_term_freq = sum([self.index.get(term, {}).get(doc, 0) for doc in document_lengths.keys()])

            if smoothing_method == 'bayes':
                term_prob = (term_freq + alpha) / (doc_length + alpha * len(self.index))
            elif smoothing_method == 'naive':
                term_prob = term_freq / doc_length
            elif smoothing_method == 'mixture':
                term_prob = lamda * (term_freq / doc_length) + (1 - lamda) * (collection_term_freq / collection_length)

            score += np.log(term_prob) if term_prob > 0 else 0

        return score


# if __name__ == "__main__":
#     index = {
#         "term1": {"doc1": 2, "doc2": 3},
#         "term2": {"doc2": 1, "doc3": 4},
#         "term3": {"doc1": 1, "doc3": 5},
#     }
#     number_of_documents = 3
#     document_lengths = {"doc1": 100, "doc2": 150, "doc3": 200}
#     average_document_length = sum(document_lengths.values()) / len(document_lengths)
#
#     scorer = Scorer(index, number_of_documents)
#     query = ["term1", "term2"]
#     list_of_docs = scorer.get_list_of_documents(query)
#     assert set(list_of_docs) == {"doc1", "doc2", "doc3"}
#
#     idf_term1 = scorer.get_idf("term1")
#     assert np.isclose(idf_term1, np.log((3 - 2 + 0.5) / (2 + 0.5)) + 1)
#
#     query_tfs = scorer.get_query_tfs(query)
#     assert query_tfs == {"term1": 1, "term2": 1}
#     vsm_scores = scorer.compute_scores_with_vector_space_model(query, "lnc.ltc")
#     assert all(doc in vsm_scores for doc in list_of_docs)
#
#     bm25_scores = scorer.compute_socres_with_okapi_bm25(query, average_document_length, document_lengths)
#     assert all(doc in bm25_scores for doc in list_of_docs)
#
#     unigram_scores = scorer.compute_scores_with_unigram_model(query, "mixture", document_lengths)
#     assert all(doc in unigram_scores for doc in list_of_docs)
#
#     print(" passe")

