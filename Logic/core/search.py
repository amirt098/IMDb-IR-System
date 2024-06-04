import json
import numpy as np
from .utility import Preprocessor, Scorer
from .indexer import Indexes, Index_types, Index_reader


class SearchEngine:
    def __init__(self):
        """
        Initializes the search engine.

        """
        path = "../Logic/core/indexer/index"
        self.document_indexes = {
            Indexes.STARS: Index_reader(path, Indexes.STARS),
            Indexes.GENRES: Index_reader(path, Indexes.GENRES),
            Indexes.SUMMARIES: Index_reader(path, Indexes.SUMMARIES),
        }
        self.tiered_index = {
            Indexes.STARS: Index_reader(path, Indexes.STARS, Index_types.TIERED),
            Indexes.GENRES: Index_reader(path, Indexes.GENRES, Index_types.TIERED),
            Indexes.SUMMARIES: Index_reader(
                path, Indexes.SUMMARIES, Index_types.TIERED
            ),
        }
        self.document_lengths_index = {
            Indexes.STARS: Index_reader(
                path, Indexes.STARS, Index_types.DOCUMENT_LENGTH
            ),
            Indexes.GENRES: Index_reader(
                path, Indexes.GENRES, Index_types.DOCUMENT_LENGTH
            ),
            Indexes.SUMMARIES: Index_reader(
                path, Indexes.SUMMARIES, Index_types.DOCUMENT_LENGTH
            ),
        }
        self.metadata_index = Index_reader(
            path, Indexes.DOCUMENTS, Index_types.METADATA
        )
        self.document = Index_reader(
            path, Indexes.DOCUMENTS
        )

    def search(
        self,
        query,
        method,
        weights,
        safe_ranking=True,
        max_results=10,
        smoothing_method=None,
        alpha=0.5,
        lamda=0.5,
    ):
        """
        searches for the query in the indexes.

        Parameters
        ----------
        query : str
            The query to search for.
        method : str ((n|l)(n|t)(n|c).(n|l)(n|t)(n|c)) | OkapiBM25 | Unigram
            The method to use for searching.
        weights: dict
            The weights of the fields.
        safe_ranking : bool
            If True, the search engine will search in whole index and then rank the results.
            If False, the search engine will search in tiered index.
        max_results : int
            The maximum number of results to return. If None, all results are returned.
        smoothing_method : str (bayes | naive | mixture)
            The method used for smoothing the probabilities in the unigram model.
        alpha : float, optional
            The parameter used in bayesian smoothing method. Defaults to 0.5.
        lamda : float, optional
            The parameter used in some smoothing methods to balance between the document
            probability and the collection probability. Defaults to 0.5.

        Returns
        -------
        list
            A list of tuples containing the document IDs and their scores sorted by their scores.
        """
        preprocessor = Preprocessor([query])
        query = preprocessor.preprocess()[0]

        scores = {}
        if method == "unigram":
            self.find_scores_with_unigram_model(
                query, smoothing_method, weights, scores, alpha, lamda
            )
        elif safe_ranking:
            self.find_scores_with_safe_ranking(query, method, weights, scores)
        else:
            self.find_scores_with_unsafe_ranking(
                query, method, weights, max_results, scores
            )

        final_scores = {}

        self.aggregate_scores(weights, scores, final_scores)

        result = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        if max_results is not None:
            result = result[:max_results]

        return result

    def aggregate_scores(self, weights, scores, final_scores):
        """
        Aggregates the scores of the fields.

        Parameters
        ----------
        weights : dict
            The weights of the fields.
        scores : dict
            The scores of the fields.
        final_scores : dict
            The final scores of the documents.
        """


        # print("Weights:", weights)
        # print("Scores:", scores)

        for doc_id in set.union(*[set(scores[field].keys()) for field in scores]):
            final_score = 0
            for field, weight in weights.items():
                final_score += weight * scores[field].get(doc_id, 0)
            final_scores[doc_id] = final_score

    def find_scores_with_unsafe_ranking(
        self, query, method, weights, max_results, scores
    ):
        """
        Finds the scores of the documents using the unsafe ranking method using the tiered index.

        Parameters
        ----------
        query: List[str]
            The query to be scored
        method : str ((n|l)(n|t)(n|c).(n|l)(n|t)(n|c)) | OkapiBM25
            The method to use for searching.
        weights: dict
            The weights of the fields.
        max_results : int
            The maximum number of results to return.
        scores : dict
            The scores of the documents.
        """
        for field in weights:
            tiered_index = self.tiered_index[field]
            scorer = Scorer(tiered_index, len(tiered_index))
            if method == "OkapiBM25":
                doc_lengths = self.document_lengths_index[field].index
                avg_doc_len = sum(doc_lengths.values()) / len(doc_lengths)
                for tier in ["first_tier", "second_tier", "third_tier"]:
                    tier_index = tiered_index[tier]
                    if len(scores[field]) < max_results:
                        scores[field].update(
                            scorer.compute_socres_with_okapi_bm25(query, avg_doc_len, doc_lengths, tier_index))
                    else:
                        break
            else:
                for tier in ["first_tier", "second_tier", "third_tier"]:
                    tier_index = tiered_index[tier]
                    if len(scores[field]) < max_results:
                        scores[field].update(scorer.compute_scores_with_vector_space_model(query, method, tier_index))
                    else:
                        break

    def find_scores_with_safe_ranking(self, query, method, weights, scores):
        """
        Finds the scores of the documents using the safe ranking method.

        Parameters
        ----------
        query: List[str]
            The query to be scored
        method : str ((n|l)(n|t)(n|c).(n|l)(n|t)(n|c)) | OkapiBM25
            The method to use for searching.
        weights: dict
            The weights of the fields.
        scores : dict
            The scores of the documents.
        """

        for field in weights:
            index = self.document_indexes[field].index
            scorer = Scorer(index, len(index))
            if method == "OkapiBM25":
                doc_lengths = self.document_lengths_index[field].index
                avg_doc_len = sum(doc_lengths.values()) / len(doc_lengths)
                scores[field] = scorer.compute_socres_with_okapi_bm25(query, avg_doc_len, doc_lengths)
            else:
                scores[field] = scorer.compute_scores_with_vector_space_model(query, method)

    def find_scores_with_unigram_model(
        self, query, smoothing_method, weights, scores, alpha=0.5, lamda=0.5
    ):
        """
        Calculates the scores for each document based on the unigram model.

        Parameters
        ----------
        query : str
            The query to search for.
        smoothing_method : str (bayes | naive | mixture)
            The method used for smoothing the probabilities in the unigram model.
        weights : dict
            A dictionary mapping each field (e.g., 'stars', 'genres', 'summaries') to its weight in the final score. Fields with a weight of 0 are ignored.
        scores : dict
            The scores of the documents.
        alpha : float, optional
            The parameter used in bayesian smoothing method. Defaults to 0.5.
        lamda : float, optional
            The parameter used in some smoothing methods to balance between the document
            probability and the collection probability. Defaults to 0.5.
        """
        for field in weights:
            if weights[field] == 0:
                continue
            scorer = Scorer(self.document_indexes[field].index, len(self.metadata_index.index))
            field_scores = scorer.compute_scores_with_unigram_model(query, smoothing_method, alpha=alpha, lamda=lamda)
            for doc_id, score in field_scores.items():
                if doc_id not in scores:
                    scores[doc_id] = {}
                scores[doc_id][field] = score
        return scores

    def merge_scores(self, scores1, scores2):
        """
        Merges two dictionaries of scores.

        Parameters
        ----------
        scores1 : dict
            The first dictionary of scores.
        scores2 : dict
            The second dictionary of scores.

        Returns
        -------
        dict
            The merged dictionary of scores.
        """

        merged_scores = scores1.copy()
        for doc_id, score in scores2.items():
            if doc_id in merged_scores:
                merged_scores[doc_id] += score
            else:
                merged_scores[doc_id] = score
        return merged_scores


if __name__ == "__main__":
    search_engine = SearchEngine()
    query = "spider man in wonderland"
    method = "lnc.ltc"
    weights = {Indexes.STARS: 1, Indexes.GENRES: 1, Indexes.SUMMARIES: 1}
    result = search_engine.search(query, method, weights)

    print(result)
