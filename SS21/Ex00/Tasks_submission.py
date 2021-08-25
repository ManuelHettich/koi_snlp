from gensim.models import KeyedVectors
import numpy as np

vectors = KeyedVectors.load("word_embeddings.vec")


def count_similar_word_vector_positions(word_vector_1, word_vector_2, delta: float):
    assert word_vector_1.size == word_vector_2.size
    return sum([abs(w_i - v_i) <= delta for w_i, v_i in zip(word_vector_1, word_vector_2)])


def get_most_similar(word: str):
    try:
        word_vector = vectors.get_vector(word)
        return [word for word, _ in vectors.similar_by_word(word, topn=10)]
    except KeyError:
        return []


def get_D_of_A_is_related_to_B_like_C_to_D(A, B, C):
    try:
        vector_A = vectors.get_vector(A)
        vector_B = vectors.get_vector(B)
        vector_C = vectors.get_vector(C)

        return [vectors.index2word[idx] for idx
                in np.argsort(vectors.distances(vector_C - (vector_A - vector_B)))][1:15]

    except KeyError:
        return None


print("The similar word-vector-fields of \"Israel\" and \"Isreali\" are {}".format(
    count_similar_word_vector_positions(word_vector_1=vectors.get_vector("Israel"),
                                        word_vector_2=vectors.get_vector("Israeli"), delta=0.025)))
print("The 10 most similar words to \"Israel\" (sorted, most similar word at first position) are: {}".format(
    ", ".join(get_most_similar("Israel"))
))
print("Isreal is related to Jerusalem like U.S. to {}".format(
    get_D_of_A_is_related_to_B_like_C_to_D(A="Israel", B="Jerusalem", C="U.S.")))
