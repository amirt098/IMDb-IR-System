from .graph import LinkGraph
from ..indexer.indexes_enum import Indexes
from ..indexer.index_reader import Index_reader

class LinkAnalyzer:
    def __init__(self, root_set):
        """
        Initialize the Link Analyzer attributes:

        Parameters
        ----------
        root_set: list
            A list of movie dictionaries with the following keys:
            "id": A unique ID for the movie
            "title": string of movie title
            "stars": A list of movie star names
        """
        self.root_set = root_set
        self.graph = LinkGraph()
        self.hubs = []
        self.authorities = []
        self.initiate_params()

    def initiate_params(self):
        """
        Initialize links graph, hubs list and authorities list based of root set

        Parameters
        ----------
        This function has no parameters. You can use self to get or change attributes
        """
        for movie in self.root_set:
            # print(movie)
            self.graph.add_node(movie["id"])
            for star in movie["stars"]:
                self.graph.add_node(star)

            for star in movie["stars"]:
                self.graph.add_edge(movie["id"], star)

    def expand_graph(self, corpus):
        """
        expand hubs, authorities and graph using given corpus

        Parameters
        ----------
        corpus: list
            A list of movie dictionaries with the following keys:
            "id": A unique ID for the movie
            "stars": A list of movie star names

        Note
        ---------
        To build the base set, we need to add the hubs and authorities that are inside the corpus
        and refer to the nodes in the root set to the graph and to the list of hubs and authorities.
        """
        #print(corpus)
        for movie in corpus:
            # print(movie)
            self.graph.add_node(movie["id"])

            for star in movie["stars"]:
                self.graph.add_node(star)
            for star in movie["stars"]:
                self.graph.add_edge(movie["id"], star)
            if movie["id"] not in self.hubs:
                self.hubs.append(movie["id"])
            for star in movie["stars"]:
                if star not in self.authorities:
                    self.authorities.append(star)

    def hits(self, num_iteration=5, max_result=10):
        """
        Return the top movies and actors using the Hits algorithm

        Parameters
        ----------
        num_iteration: int
            Number of algorithm execution iterations
        max_result: int
            The maximum number of results to return. If None, all results are returned.

        Returns
        -------
        list
            List of names of 10 actors with the most scores obtained by Hits algorithm in descending order
        list
            List of names of 10 movies with the most scores obtained by Hits algorithm in descending order
        """
        a_s = []
        h_s = []

        #print(self.authorities)
        for i in range(num_iteration):
            new_authorities = {}
            new_hubs = {}

            for node in self.authorities:
                auth_score = 0
                for neighbor in self.graph.get_predecessors(node):
                    auth_score += self.hubs.count(neighbor)
                new_authorities[node] = auth_score

            for node in self.hubs:
                hub_score = 0
                for neighbor in self.graph.get_successors(node):
                    hub_score += new_authorities[neighbor]
                new_hubs[node] = hub_score


            total_auth = sum(new_authorities.values())
            total_hub = sum(new_hubs.values())
            for node in self.authorities:
                new_authorities[node] /= total_auth
            for node in self.hubs:
                new_hubs[node] /= total_hub

            self.authorities = list(new_authorities.keys())
            self.hubs = list(new_hubs.keys())
            #print(self.hubs)
        sorted_authorities = sorted(self.authorities, key=lambda x: new_authorities[x], reverse=True)

        sorted_hubs = sorted(self.hubs, key=lambda x: new_hubs[x], reverse=True)
        # print(sorted_hubs, sorted_authorities)

        if max_result is not None:
            a_s = sorted_authorities[:max_result]
            h_s = sorted_hubs[:max_result]
        else:
            a_s = sorted_authorities
            h_s = sorted_hubs

        return a_s, h_s

#
# if __name__ == "__main__":
#     corpus = [
#         {
#             "id": "movie1",
#             "stars": ["actor1", "actor2"]
#         },
#         {
#             "id": "movie2",
#             "stars": ["actor2", "actor3"]
#         },
#     ]
#
#     root_set = [
#         {
#             "id": "root_movie1",
#             "title": "Root Movie 1",
#             "stars": ["root_actor1", "root_actor2"]
#         },
#         {
#             "id": "root_movie2",
#             "title": "Root Movie 2",
#             "stars": ["root_actor2", "root_actor3"]
#         },
#     ]
#
#     analyzer = LinkAnalyzer(root_set=root_set)
#     analyzer.expand_graph(corpus=corpus)
#     actors, movies = analyzer.hits(max_result=5)
#     print("Top Actors:")
#     print(*actors, sep=' - ')
#     print("Top Movies:")
#     print(*movies, sep=' - ')
if __name__ == "__main__":
    # You can use this section to run and test the results of your link analyzer
    corpus = []    # TODO: it shoud be your crawled data
    root_set = []   # TODO: it shoud be a subset of your corpus

    analyzer = LinkAnalyzer(root_set=root_set)
    analyzer.expand_graph(corpus=corpus)
    actors, movies = analyzer.hits(max_result=5)
    print("Top Actors:")
    print(*actors, sep=' - ')
    print("Top Movies:")
    print(*movies, sep=' - ')
