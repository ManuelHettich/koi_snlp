from gensim.models import KeyedVectors

# YOUR METHODS HERE

vectors = KeyedVectors.load("word_embeddings.vec")

print("The similar word-vector-fields of \"Israel\" and \"Isreali\" are {}".format(
    count_similar_word_vector_positions(word_vector_1=vectors.get_vector("Israel"),
                                        word_vector_2=vectors.get_vector("Israeli"), delta=0.025)))
print("The 10 most similar words to \"Israel\" (sorted, most similar word at first position) are: {}".format(
    ", ".join(get_most_similar("Israel"))
))
print("Isreal is related to Jerusalem like U.S. to {}".format(
    get_D_of_A_ist_related_to_B_like_C_to_D(A="Israel", B="Jerusalem", C="U.S.")))
