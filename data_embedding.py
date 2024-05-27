import pandas as pd
import rdflib
import networkx as nx
from node2vec import Node2Vec
from gensim.models import Word2Vec
from pyrdf2vec.graphs import KG
import numpy as np

def load_graph_from_ttl(file_path):
    rdf_graph = rdflib.Graph()
    rdf_graph.parse(file_path, format='turtle')

    nx_graph = nx.Graph()

    for s, p, o in rdf_graph:
        nx_graph.add_edge(str(s), str(o), label=str(p))

    return nx_graph

def generate_random_walks(graph, num_walks, walk_length, p, q):
    node2vec = Node2Vec(graph, dimensions=64, walk_length=walk_length, num_walks=num_walks, p=p, q=q, workers=4)
    return node2vec.walks
def learn_embeddings_from_walks(random_walks, embedding_dim):
    model = Word2Vec(sentences=random_walks, vector_size=embedding_dim, window=5, sg=1, hs=0, negative=10, workers=4)
    return model

def save_embeddings_to_file(model_path, output_file):
    # Load the model
    model = Word2Vec.load(model_path)

    # Extract embeddings and words
    embeddings = model.wv.vectors
    words = model.wv.index_to_key  # List of node IDs corresponding to the vectors

    # Save to CSV file
    with open(output_file, "w") as f:
        for word, vec in zip(words, embeddings):
            f.write(f"{word}," + ",".join(map(str, vec)) + "\n")
    print(f'Embeddings saved to {output_file}')

def main():
    file_path = '/content/drive/My Drive/article_ontology_with_data.ttl'
    graph = load_graph_from_ttl(file_path)
    num_walks = 100
    walk_length = 10
    p = 1
    q = 1
    random_walks = generate_random_walks(graph, num_walks, walk_length, p, q)
    print('walks done')

    embedding_dim = 64
    model = learn_embeddings_from_walks(random_walks, embedding_dim)
    model.save("node2vec_embeddings.model")

if __name__ == "__main__":
    main()


