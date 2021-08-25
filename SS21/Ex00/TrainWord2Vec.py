from gensim.models import Word2Vec

import logging
import sys

logging.basicConfig(stream=sys.stdout, format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from pathlib import Path
from import_corpus import import_conll2004
from import_corpus import Sentence, AnnotatedToken

sentences_train, sentences_test = import_conll2004(str(Path(".").joinpath("conll2004").absolute()))

sentences_train.extend(sentences_test)
sentences = [[token.token for token in sen.annotated_tokens if isinstance(token, AnnotatedToken)]
             for sen in sentences_train if isinstance(sen, Sentence) and isinstance(sen.annotated_tokens, list)]

model = Word2Vec(sentences=sentences, corpus_file=None, size=32, window=5, min_count=2, workers=4, iter=3,
                 compute_loss=True)
#model.build_vocab(sentences=common_texts, corpus_file=None)
model.train(sentences=sentences, corpus_file=None, total_examples=len(sentences), epochs=2)
model.save("test.model")
model = model.wv
model.save("word_embeddings.vec")
word_vector_isreal = model.get_vector("Israel")
print("{} ({})".format(word_vector_isreal, type(word_vector_isreal)))
