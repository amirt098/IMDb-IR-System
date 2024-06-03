import time
import os
import json
import copy
from .indexes_enum import Indexes, Index_types


class Index:
    def __init__(self, preprocessed_documents: list):
        """
        Create a class for indexing.
        """

        self.preprocessed_documents = preprocessed_documents

        self.index = {
            Indexes.DOCUMENTS.value: self.index_documents(),
            Indexes.STARS.value: self.index_stars(),
            Indexes.GENRES.value: self.index_genres(),
            Indexes.SUMMARIES.value: self.index_summaries(),
            Index_types.DOCUMENT_LENGTH.value: self.index_document_lengths(),

        }

    def index_document_lengths(self):
        """
        Index the documents based on their lengths.
        """
        current_index = {}
        for doc in self.preprocessed_documents:
            doc_length = sum(len(summary.split()) for summary in doc['summaries'])
            current_index[doc['id']] = doc_length
        return current_index

    def index_documents(self):
        """
        Index the documents based on the document ID. In other words, create a dictionary
        where the key is the document ID and the value is the document.

        Returns
        ----------
        dict
            The index of the documents based on the document ID.
        """

        current_index = {}
        for doc in self.preprocessed_documents:
            current_index[doc['id']] = doc
        return current_index

    def index_stars(self):
        """
        Index the documents based on the stars.

        Returns
        ----------
        dict
            The index of the documents based on the stars. You should also store each terms' tf in each document.
            So the index type is: {term: {document_id: tf}}
        """

        current_index = {}

        for doc in self.preprocessed_documents:
            # print(doc)
            # print('***')
            if not doc.get('stars'):
                print('no stars found for')
                print(doc.get('title'))
                print(doc.get('id'))
                continue
            for star in doc['stars']:

                if star not in current_index:
                    current_index[star] = {}

                if doc['id'] not in current_index[star]:
                    current_index[star][doc['id']] = 0

                current_index[star][doc['id']] += 1

        return current_index

    def index_genres(self):
        """
        Index the documents based on the genres.

        Returns
        ----------
        dict
            The index of the documents based on the genres. You should also store each terms' tf in each document.
            So the index type is: {term: {document_id: tf}}
        """

        current_index = {}

        for doc in self.preprocessed_documents:
            if not doc.get('genres'):
                print('no genres found for')
                print(doc.get('title'))
                print(doc.get('id'))
                continue
            for genre in doc['genres']:

                if genre not in current_index:
                    current_index[genre] = {}

                if doc['id'] not in current_index[genre]:
                    current_index[genre][doc['id']] = 0

                current_index[genre][doc['id']] += 1

        return current_index

    def index_summaries(self):
        """
        Index the documents based on the summaries (not first_page_summary).

        Returns
        ----------
        dict
            The index of the documents based on the summaries. You should also store each terms' tf in each document.
            So the index type is: {term: {document_id: tf}}
        """

        current_index = {}

        for doc in self.preprocessed_documents:

            if not doc.get('summaries'):
                print('no summaries found for')
                print(doc.get('title'))
                print(doc.get('id'))
                continue

            for summary in doc['summaries']:

                words = summary.split()

                for word in words:
                    if word not in current_index:
                        current_index[word] = {}

                    if doc['id'] not in current_index[word]:
                        current_index[word][doc['id']] = 0

                    current_index[word][doc['id']] += 1

        return current_index

    def get_posting_list(self, word: str, index_type: str):
        """
        get posting_list of a word

        Parameters
        ----------
        word: str
            word we want to check
        index_type: str
            type of index we want to check (documents, stars, genres, summaries)

        Return
        ----------
        list
            posting list of the word (you should return the list of document IDs that contain the word and ignore the tf)
        """

        try:
            return list(self.index[index_type][word].keys())
        except:
            return []

    def add_document_to_index(self, document: dict):
        """
        Add a document to all the indexes

        Parameters
        ----------
        document : dict
            Document to add to all the indexes
        """

        self.preprocessed_documents.append(document)
        self.index[Indexes.DOCUMENTS.value][document['id']] = document

        for star in document['stars']:
            if star not in self.index[Indexes.STARS.value]:
                self.index[Indexes.STARS.value][star] = {}
            self.index[Indexes.STARS.value][star][document['id']] = self.index[Indexes.STARS.value][star].get(document['id'], 0) + 1

        for genre in document['genres']:
            if genre not in self.index[Indexes.GENRES.value]:
                self.index[Indexes.GENRES.value][genre] = {}
            self.index[Indexes.GENRES.value][genre][document['id']] = self.index[Indexes.GENRES.value][genre].get(document['id'], 0) + 1

        for summary in document['summaries']:
            words = summary.split()
            for word in words:
                if word not in self.index[Indexes.SUMMARIES.value]:
                    self.index[Indexes.SUMMARIES.value][word] = {}
                self.index[Indexes.SUMMARIES.value][word][document['id']] = self.index[Indexes.SUMMARIES.value][word].get(document['id'], 0) + 1


    def remove_document_from_index(self, document_id: str):
        """
        Remove a document from all the indexes

        Parameters
        ----------
        document_id : str
            ID of the document to remove from all the indexes
        """
        document = self.index[Indexes.DOCUMENTS.value].pop(document_id, None)

        if document:
            for star in document['stars']:
                if document_id in self.index[Indexes.STARS.value].get(star, {}):
                    del self.index[Indexes.STARS.value][star][document_id]
                    if not self.index[Indexes.STARS.value][star]:
                        del self.index[Indexes.STARS.value][star]

            for genre in document['genres']:
                if document_id in self.index[Indexes.GENRES.value].get(genre, {}):
                    del self.index[Indexes.GENRES.value][genre][document_id]
                    if not self.index[Indexes.GENRES.value][genre]:
                        del self.index[Indexes.GENRES.value][genre]

            for summary in document['summaries']:
                words = summary.split()
                for word in words:
                    if document_id in self.index[Indexes.SUMMARIES.value].get(word, {}):
                        del self.index[Indexes.SUMMARIES.value][word][document_id]
                        if not self.index[Indexes.SUMMARIES.value][word]:
                            del self.index[Indexes.SUMMARIES.value][word]


    def check_add_remove_is_correct(self):
        """
        Check if the add and remove is correct
        """

        dummy_document = {
            'id': '100',
            'stars': ['tim', 'henry'],
            'genres': ['drama', 'crime'],
            'summaries': ['good']
        }

        index_before_add = copy.deepcopy(self.index)
        self.add_document_to_index(dummy_document)
        index_after_add = copy.deepcopy(self.index)

        if index_after_add[Indexes.DOCUMENTS.value]['100'] != dummy_document:
            print('Add is incorrect, document')
            return

        if (set(index_after_add[Indexes.STARS.value]['tim']).difference(set(index_before_add[Indexes.STARS.value].get('tim', [])))
                != {dummy_document['id']}):
            print('Add is incorrect, tim')
            return

        if (set(index_after_add[Indexes.STARS.value]['henry']).difference(set(index_before_add[Indexes.STARS.value].get('henry', [])))
                != {dummy_document['id']}):
            print('Add is incorrect, henry')
            return
        if (set(index_after_add[Indexes.GENRES.value]['drama']).difference(set(index_before_add[Indexes.GENRES.value].get('drama', [])))
                != {dummy_document['id']}):
            print('Add is incorrect, drama')
            return

        if (set(index_after_add[Indexes.GENRES.value]['crime']).difference(set(index_before_add[Indexes.GENRES.value].get('crime', [])))
                != {dummy_document['id']}):
            print('Add is incorrect, crime')
            return

        if (set(index_after_add[Indexes.SUMMARIES.value]['good']).difference(set(index_before_add[Indexes.SUMMARIES.value].get('good', [])))
                != {dummy_document['id']}):
            print('Add is incorrect, good')
            return

        print('Add is correct')

        self.remove_document_from_index('100')
        index_after_remove = copy.deepcopy(self.index)

        if index_after_remove == index_before_add:
            print('Remove is correct')
        else:
            print('Remove is incorrect')

    def store_index(self, path: str, index_name: str = None):
        """
        Stores the index in a file (such as a JSON file)

        Parameters
        ----------
        path : str
            Path to store the file
        index_name: str
            name of index we want to store (documents, stars, genres, summaries)
        """

        if not os.path.exists(path):
            os.makedirs(path)

        if index_name not in self.index:
            raise ValueError('Invalid index name')

        with open(os.path.join(path, f"{index_name}.json"), 'w') as f:
            json.dump(self.index[index_name], f)


    def load_index(self, path: str):
        """
        Loads the index from a file (such as a JSON file)

        Parameters
        ----------
        path : str
            Path to load the file
        """
        for index_name in self.index.keys():
            with open(os.path.join(path, f"{index_name}.json"), 'r') as f:
                self.index[index_name] = json.load(f)

    def check_if_index_loaded_correctly(self, index_type: str, loaded_index: dict):
        """
        Check if the index is loaded correctly

        Parameters
        ----------
        index_type : str
            Type of index to check (documents, stars, genres, summaries)
        loaded_index : dict
            The loaded index

        Returns
        ----------
        bool
            True if index is loaded correctly, False otherwise
        """

        return self.index[index_type] == loaded_index

    def check_if_indexing_is_good(self, index_type: str, check_word: str = 'good'):
        """
        Checks if the indexing is good. Do not change this function. You can use this
        function to check if your indexing is correct.

        Parameters
        ----------
        index_type : str
            Type of index to check (documents, stars, genres, summaries)
        check_word : str
            The word to check in the index

        Returns
        ----------
        bool
            True if indexing is good, False otherwise
        """

        # brute force to check check_word in the summaries
        start = time.time()
        docs = []
        for document in self.preprocessed_documents:
            if index_type not in document or document[index_type] is None:
                continue

            for field in document[index_type]:
                if check_word in field:
                    docs.append(document['id'])
                    break

            # if we have found 3 documents with the word, we can break
            if len(docs) == 3:
                break

        end = time.time()
        brute_force_time = end - start

        # check by getting the posting list of the word
        start = time.time()
        # TODO: based on your implementation, you may need to change the following line
        posting_list = self.get_posting_list(check_word, index_type)

        end = time.time()
        implemented_time = end - start

        print('Brute force time: ', brute_force_time)
        print('Implemented time: ', implemented_time)

        if set(docs).issubset(set(posting_list)):
            print('Indexing is correct')

            if implemented_time < brute_force_time:
                print('Indexing is good')
                return True
            else:
                print('Indexing is bad')
                return False
        else:
            print('Indexing is wrong')
            return False

# TODO: Run the class with needed parameters, then run check methods and finally report the results of check methods


if __name__ == '__main__':
    with open('IMDB_crawled.json', 'r') as file:
        preprocessed_documents = json.load(file)

    # for movie in preprocessed_documents:
    #     movie_title = movie['title']
    #     movie_summary = movie['first_page_summary']
    #     # Print the desired value
    #     print(f"Title: {movie_title}")
    #     print(f"Summary: {movie_summary}")
    #     print(f"stars: {movie['stars']}")
    #
    #     print("-" * 20)

    # preprocessed_documents = [
    #     {'id': '1', 'stars': ['tim1', 'henry'], 'genres': ['drama'], 'summaries': ['This is a good movie.']},
    #     {'id': '2', 'stars': ['tim1'], 'genres': ['action'], 'summaries': ['A great action movie.']},
    #     {'id': '3', 'stars': ['henry'], 'genres': ['drama', 'crime'], 'summaries': ['An excellent crime drama.']},
    #     {'id': '4', 'stars': ['tim1', 'henry', 'jane'], 'genres': ['comedy', 'romance'],
    #      'summaries': ['A hilarious romantic comedy.', 'Funny and heartwarming.']},
    #     {'id': '5', 'stars': ['bob', 'alice'], 'genres': ['sci-fi', 'adventure'],
    #      'summaries': ['A thrilling sci-fi adventure.', 'Out of this world!']},
    #     {'id': '6', 'stars': ['jane', 'bob'], 'genres': ['horror'],
    #      'summaries': ['A terrifying horror movie.', 'Prepare to be scared!']},
    #     {'id': '7', 'stars': ['alice'], 'genres': ['documentary'],
    #      'summaries': ['An eye-opening documentary.', 'Informative and thought-provoking.']},
    #     {'id': '8', 'stars': ['tim1', 'jane', 'alice'], 'genres': ['drama', 'mystery'],
    #      'summaries': ['A gripping mystery drama.', 'Keep you guessing until the end.']},
    #     {'id': '9', 'stars': ['henry', 'bob'], 'genres': ['action', 'thriller'],
    #      'summaries': ['An intense action thriller.', 'Edge-of-your-seat excitement.']},
    #     {'id': '10', 'stars': ['tim1', 'alice'], 'genres': ['fantasy', 'adventure'],
    #      'summaries': ['An epic fantasy adventure.', 'Escape to a magical world.']},
    #     {'id': '11', 'stars': ['jane', 'henry'], 'genres': ['drama', 'romance'],
    #      'summaries': ['A heartwarming romantic drama.', 'Love conquers all.']},
    #     {'id': '12', 'stars': ['bob'], 'genres': ['comedy'],
    #      'summaries': ['A laugh-out-loud comedy.', 'Hilarious from start to finish.']},
    #     {'id': '13', 'stars': ['tim1', 'henry', 'jane', 'bob', 'alice'], 'genres': ['ensemble'],
    #      'summaries': ['An ensemble cast at their best.', 'Star-studded and entertaining.']},
    #     {'id': '14', 'stars': [], 'genres': ['experimental'],
    #      'summaries': ['A unique and thought-provoking film.', 'Challenging but rewarding.']},
    #     {'id': '15', 'stars': ['jane', 'alice'], 'genres': ['biography', 'drama'],
    #      'summaries': ['An inspiring biographical drama.', 'A true story of courage and perseverance.']},
    #     {'id': '16', 'stars': ['tim1', 'henry', 'jane', 'bob', 'alice', 'charlie', 'eve', 'david'],
    #      'genres': ['action', 'adventure', 'sci-fi'],
    #      'summaries': ['An epic sci-fi action adventure with a star-studded cast.',
    #                    'Prepare for mind-blowing special effects and non-stop thrills.']},
    #     {'id': '17', 'stars': [], 'genres': ['silent'],
    #      'summaries': ['A classic silent film that speaks volumes through its visuals.']},
    #     {'id': '18', 'stars': ['tim1', 'henry'], 'genres': ['western', 'drama'],
    #      'summaries': ['A gritty western drama about the harsh realities of frontier life.',
    #                    'Powerful performances and stunning cinematography.']},
    #     {'id': '19', 'stars': ['jane', 'alice', 'eve'], 'genres': ['musical', 'comedy'],
    #      'summaries': ['A delightful musical comedy with catchy tunes and side-splitting humor.',
    #                    'You\'ll be tapping your feet and laughing out loud.']},
    #     {'id': '20', 'stars': ['bob', 'david', 'charlie'], 'genres': ['war', 'historical'],
    #      'summaries': ['An intense and realistic portrayal of the horrors of war.',
    #                    'A sobering reminder of the sacrifices made by those who served.']},
    #     {'id': '21', 'stars': ['tim1', 'jane', 'eve'], 'genres': ['animation', 'family'],
    #      'summaries': ['A heartwarming animated film for the whole family.',
    #                    'Colorful characters and valuable life lessons.']},
    #     {'id': '22', 'stars': ['henry', 'alice', 'charlie'], 'genres': ['sports', 'drama'],
    #      'summaries': ['An inspiring sports drama about overcoming adversity and achieving greatness.',
    #                    'Get ready to cheer for the underdogs.']},
    #     {'id': '23', 'stars': ['bob', 'david'], 'genres': ['noir', 'mystery'],
    #      'summaries': ['A gritty noir mystery with twists and turns that will keep you guessing.',
    #                    'Atmospheric and suspenseful.']},
    #     {'id': '24', 'stars': ['tim1', 'jane', 'alice', 'eve', 'charlie'], 'genres': ['ensemble', 'comedy'],
    #      'summaries': ['A hilarious ensemble comedy with a talented cast of comedic actors.',
    #                    'Prepare for non-stop laughter and side-splitting humor.']},
    #     {'id': '25', 'stars': ['henry', 'bob', 'david'], 'genres': ['political', 'thriller'],
    #      'summaries': ['A gripping political thriller that exposes the dark underbelly of power and corruption.',
    #                    'Edge-of-your-seat suspense and intrigue.']},
    #     {'id': '26', 'stars': ['tim1', 'jane', 'alice', 'eve', 'charlie', 'david'], 'genres': ['period', 'drama'],
    #      'summaries': ['A lavish period drama that transports you to a bygone era.',
    #                    'Stunning costumes, sets, and performances that bring history to life.']},
    #     {'id': '27', 'stars': ['henry', 'bob'], 'genres': ['superhero', 'action'],
    #      'summaries': ['A high-octane superhero action film with jaw-dropping special effects.',
    #                    'Get ready for epic battles and larger-than-life heroes.']},
    #     {'id': '28', 'stars': ['tim1', 'jane', 'alice'], 'genres': ['romantic', 'comedy'],
    #      'summaries': ['A charming romantic comedy that will warm your heart and make you laugh.',
    #                    'A delightful exploration of love, friendship, and finding happiness.']},
    #     {'id': '29', 'stars': ['henry', 'bob', 'david', 'charlie', 'eve'], 'genres': ['heist', 'crime'],
    #      'summaries': ['A slick and suspenseful heist film with a star-studded cast.',
    #                    'Get ready for twists, turns, and high-stakes action.']},
    #     {'id': '30', 'stars': ['tim1', 'jane', 'alice', 'eve'], 'genres': ['independent', 'drama'],
    #      'summaries': ['An indie drama that explores the complexities of human relationships.',
    #                    'Powerful performances and thought-provoking themes.']},
    # ]

    index = Index(preprocessed_documents)

    index_path = "index/"

    # Store the indexes in JSON files
    for index_type in Indexes:
        index.store_index(index_path, index_type.value)

    index.check_add_remove_is_correct()

    index.store_index('./index_storage', Indexes.DOCUMENTS.value)
    index.store_index('./index_storage', Indexes.STARS.value)
    index.store_index('./index_storage', Indexes.GENRES.value)
    index.store_index('./index_storage', Indexes.SUMMARIES.value)

    # Load index from files
    index.load_index('./index_storage')

    # Check if indexes are loaded correctly
    for index_type in [Indexes.DOCUMENTS.value, Indexes.STARS.value, Indexes.GENRES.value, Indexes.SUMMARIES.value]:
        loaded_index = index.index[index_type]
        print(f"Checking if {index_type} index is loaded correctly: ", index.check_if_index_loaded_correctly(index_type, loaded_index))

    # Check if indexing is good
    for index_type in [Indexes.DOCUMENTS.value, Indexes.STARS.value, Indexes.GENRES.value, Indexes.SUMMARIES.value]:
        print(f"Checking if indexing is good for {index_type}: ", index.check_if_indexing_is_good(index_type, 'good'))
