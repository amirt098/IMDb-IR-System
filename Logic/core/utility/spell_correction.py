from collections import Counter


class SpellCorrection:
    def __init__(self, all_documents):
        """
        Initialize the SpellCorrection

        Parameters
        ----------
        all_documents : list of str
            The input documents.
        """
        self.all_shingled_words, self.word_counter = self.shingling_and_counting(all_documents)

    def shingle_word(self, word, k=2):
        """
        Convert a word into a set of shingles.

        Parameters
        ----------
        word : str
            The input word.
        k : int
            The size of each shingle.

        Returns
        -------
        set
            A set of shingles.
        """
        shingles = set()

        for i in range(len(word) - k + 1):
            shingle = word[i:i + k]
            shingles.add(shingle)

        return shingles


    def jaccard_score(self, first_set, second_set):
        """
        Calculate jaccard score.

        Parameters
        ----------
        first_set : set
            First set of shingles.
        second_set : set
            Second set of shingles.

        Returns
        -------
        float
            Jaccard score.
        """

        intersection = first_set.intersection(second_set)
        union = first_set.union(second_set)
        if not union:
            return 0.0
        return len(intersection) / len(union)

    def shingling_and_counting(self, all_documents):
        """
        Shingle all words of the corpus and count TF of each word.

        Parameters
        ----------
        all_documents : list of str
            The input documents.

        Returns
        -------
        all_shingled_words : dict
            A dictionary from words to their shingle sets.
        word_counter : dict
            A dictionary from words to their TFs.
        """
        # TODO: Create shingled words dictionary and word counter dictionary here.

        all_shingled_words = {}
        word_counter = Counter()

        for document in all_documents:
            words = document.split()
            for word in words:
                shingled_word = self.shingle_word(word)
                all_shingled_words[word] = shingled_word
                word_counter[word] += 1

        return all_shingled_words, word_counter

    def find_nearest_words(self, word):
        """
        Find correct form of a misspelled word.

        Parameters
        ----------
        word : stf
            The misspelled word.

        Returns
        -------
        list of str
            5 nearest words.
        """
        top5_candidates = list()

        shingled_word = self.shingle_word(word)
        candidates = []

        for candidate_word, candidate_shingles in self.all_shingled_words.items():
            score = self.jaccard_score(shingled_word, candidate_shingles)
            if score > 0:
                candidates.append((candidate_word, score))

        candidates.sort(key=lambda x: x[1], reverse=True)
        print(candidates, 'candidates')
        top5_candidates = candidates[:5]

        top5_candidates = sorted(top5_candidates, key=lambda x: x[1], reverse=True)

        top5_candidates = [t[0] for t in top5_candidates]

        return top5_candidates

    def spell_check(self, query):
        """
        Find correct form of a misspelled query.

        Parameters
        ----------
        query : stf
            The misspelled query.

        Returns
        -------
        str
            Correct form of the query.
        """
        final_result = ""

        # TODO: Do spell correction here.
        words = query.split()
        corrected_query = []

        for word in words:
            nearest_words = self.find_nearest_words(word)
            if nearest_words:
                corrected_word = nearest_words[0]
            else:
                corrected_word = word
            corrected_query.append(corrected_word)
        print(corrected_query, 'corrected_query')
        final_result = " ".join(corrected_query)

        return final_result


# if __name__ == "__main__":
#
#     all_documents = [
#         "This is a sample document. working",
#         "We are going to correct the spelling.",
#         "Spell correction is an important task.",
#         "Sometimes words are misspelled.",
#         "The correct word should be found.",
#         'while we have WHils why should use whales?'
#
#     ]
#
#     spell_correction = SpellCorrection(all_documents)
#     query = "whle we are working"
#     corrected_query = spell_correction.spell_check(query)
#     print(corrected_query)
