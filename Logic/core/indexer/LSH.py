import hashlib
from collections import defaultdict

import numpy as np
import itertools
import random


class MinHashLSH:
    def __init__(
            self,
            documents: [str],
            num_hashes: int,
    ):
        """
        Initialize the MinHashLSH

        Parameters
        ----------
        documents : list of str
            The input documents for similarity analysis.
        num_hashes : int
            Number of hashes for mini-hashing.
        """
        self.documents = documents
        self.num_hashes = num_hashes

    def shingle_document(self, document, k=2):
        """
        Convert a document into a set of shingles.

        Parameters
        ----------
        document : str
            The input document.
        k : int
            The size of each shingle.

        Returns
        ----------
        set
            A set of shingles.
        """

        words = document.split()

        # Ensure the document length is sufficient for at least one shingle
        if len(words) < k:
            return set()

        # Create the set of shingles
        shingles = set()
        for i in range(len(words) - k + 1):
            shingle = tuple(words[i:i + k])
            shingles.add(shingle)

        return shingles

    def build_characteristic_matrix(self):
        """
        Build the characteristic matrix representing the presence of shingles in documents.

        Returns
        ----------
        numpy.ndarray
            The binary characteristic matrix.
        """
        all_shingles = set()
        shingles_per_document = []

        for document in self.documents:
            shingles = self.shingle_document(document, 2)
            shingles_per_document.append(shingles)
            all_shingles.update(shingles)

        all_shingles = list(all_shingles)
        num_documents = len(self.documents)
        num_shingles = len(all_shingles)
        characteristic_matrix = np.zeros((num_shingles, num_documents), dtype=int)

        # Fill the characteristic matrix
        for doc_idx, shingles in enumerate(shingles_per_document):
            for shingle in shingles:
                shingle_idx = all_shingles.index(shingle)
                characteristic_matrix[shingle_idx, doc_idx] = 1

        return characteristic_matrix

    def min_hash_signature(self):
        """
        Perform Min-Hashing to generate hash signatures for documents.

        Returns
        ----------
        numpy.ndarray
            The Min-Hash signatures matrix.
        """
        num_documents = len(self.documents)
        characteristic_matrix = self.build_characteristic_matrix()
        num_shingles = characteristic_matrix.shape[0]
        signature_matrix = np.full((self.num_hashes, num_documents), np.inf)

        for i in range(self.num_hashes):
            permutation = np.random.permutation(num_shingles)
            for row in range(num_shingles):
                permuted_row = permutation[row]
                for col in range(num_documents):
                    if characteristic_matrix[permuted_row, col] == 1:
                        signature_matrix[i, col] = min(signature_matrix[i, col], row)

        return signature_matrix

    def lsh_buckets(self, signature, bands=10, rows_per_band=10):
        """
        Group documents into Locality-Sensitive Hashing (LSH) buckets based on Min-Hash signatures.

        Parameters
        ----------
        signature : numpy.ndarray
            Min-Hash signatures for documents.
        bands : int
            Number of bands for LSH.
        rows_per_band : int
            Number of rows per band.

        Returns
        ----------
        dict
            A dictionary mapping bucket IDs to lists of document indices.
        """
        assert bands * rows_per_band == self.num_hashes, "bands * rows_per_band must equal num_hashes"

        buckets = defaultdict(list)

        for doc_idx in range(signature.shape[1]):
            for band_idx in range(bands):
                start_row = band_idx * rows_per_band
                end_row = start_row + rows_per_band
                band_signature = tuple(signature[start_row:end_row, doc_idx])
                bucket_id = hashlib.md5(str(band_signature).encode('utf8')).hexdigest()
                buckets[bucket_id].append(doc_idx)

        return buckets

    def perform_lsh(self):
        """
        Perform the entire Locality-Sensitive Hashing (LSH) process.

        Returns
        ----------
        dict
            A dictionary mapping bucket IDs to lists of document indices.
        """
        signature = self.min_hash_signature()
        bands = 10
        rows_per_band = self.num_hashes // bands
        return self.lsh_buckets(signature, bands, rows_per_band)

    def jaccard_score(self, first_set: (), second_set: ()):
        """
        Calculate jaccard score for two sets.

        Parameters
        ----------
        first_set : set
            Set of first shingled document.
        second_set : set
            Set of second shingled document.

        Returns
        ----------
        float
            Jaccard score.
        """
        intersection = len(first_set.intersection(second_set))
        union = len(first_set.union(second_set))

        return float(intersection) / float(union) if union != 0 else 0.0

    def jaccard_similarity_test(self, buckets, all_documents):
        """
        Test your near duplicate detection code based on jaccard similarity.

        Parameters
        ----------
        buckets : dict
            A dictionary mapping bucket IDs to lists of document indices.
        all_documents : list
            The input documents for similarity analysis.
        """
        correct_near_duplicates = 0
        all_near_duplicates = 0

        for bucket_id in buckets.keys():
            docs_in_this_bucket = buckets[bucket_id]
            unique_doc_ids = set(docs_in_this_bucket)
            if len(unique_doc_ids) > 1:
                combinations = list(itertools.combinations(unique_doc_ids, 2))
                for comb in combinations:
                    all_near_duplicates += 1

                    first_doc_id = comb[0]
                    second_doc_id = comb[1]

                    first_shingled_doc = self.shingle_document(all_documents[first_doc_id], 2)
                    second_shingled_doc = self.shingle_document(all_documents[second_doc_id], 2)

                    near_duplicated_jaccard_score = self.jaccard_score(first_shingled_doc, second_shingled_doc)
                    current_score = 0

                    for _ in range(5):
                        random_doc_id = first_doc_id
                        while random_doc_id == first_doc_id or random_doc_id == second_doc_id:
                            random_doc_id = random.randint(0, len(all_documents) - 1)
                        random_shingled_doc = self.shingle_document(all_documents[random_doc_id], 2)

                        random_jaccard_score = self.jaccard_score(first_shingled_doc, random_shingled_doc)

                        if near_duplicated_jaccard_score > random_jaccard_score:
                            current_score += 1

                    if current_score == 5:
                        correct_near_duplicates += 1

        # a good score is around 0.8
        print("your final score in near duplicate detection:", correct_near_duplicates / all_near_duplicates)

# if __name__ == "__main__":
#     documents = [
#         "the quick brown fox jumps over the lazy dog",
#         "the fast brown fox leaps over the lazy dog",
#         "the quick brown fox jumps over the lazy dog",
#         "a completely different document with unique words",
#         "another document that is entirely different",
#         "the quick brown fox jumps over the lazy dog again",
#         "fox jumps over the lazy dog quickly and swiftly",
#         "the fast brown fox leaps over the lazy dog again",
#         "document with some overlap but not much",
#         "entirely unique document with no similarities"
#     ]
#
#     num_hashes = 100
#
#     minhash_lsh = MinHashLSH(documents, num_hashes)
#     characteristic_matrix = minhash_lsh.build_characteristic_matrix()
#     min_hash_signatures = minhash_lsh.min_hash_signature()
#     buckets = minhash_lsh.perform_lsh()
#
#     print("Characteristic Matrix:\n", characteristic_matrix)
#     print("Min-Hash Signatures:\n", min_hash_signatures)
#     print("LSH Buckets:\n", buckets)
#
#     # Perform near duplicate detection test
#     minhash_lsh.jaccard_similarity_test(buckets, documents)
