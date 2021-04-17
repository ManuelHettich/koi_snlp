"""
Ich habe die Sachen, die ihr wahrscheinlich sowieso nicht brauchen werdet, rausgenommen.

Hinzugekommen ist die Klasse AnnotatedToken. Diese enthält das Token selbst, das Entity-Label im
IOB-Schema und das POS-Label

Die Klasse Sentence enthält eine Liste annotated_tokens, die Instanzen dieser Klasse erhält.

Kernstück ist die Funktion import_conll2004, welche ein Paar von Listen zurückgibt. Die erste Liste enthält
die Sätze der Trainingsmenge und die zweite die der Testmenge.

Am Ende des Codes gebe ich den ersten Satz der Trainingsmenge aus (mit Annotationen).
Schau's dir einfach mal an.

Was noch fehlt, sind die Relationen. Hast du einen Wunsch, wie ich die Information repräsentieren soll?
"""

import itertools
import copy


# string representing 'no relation' class
NO_RELATION_LABEL = 'N'


        
class AnnotatedToken:
	def __init__(self):
		self.token = None
		self.pos_tag = None
		self.entity_tag = None
        

        
class Sentence:
    def __init__(self, phrase_rows, relation_rows):
        self.sentence_id = None # id of sentence
        self.annotated_tokens = []
        
        # extract phrases
        for i,row in enumerate(phrase_rows):
            cols = row.strip().split('\t')
            
            if not self.sentence_id:
                self.sentence_id = int(cols[0].strip())
                
            phrase = [token for token in cols[5].split('/')]
            pos_tags = [token for token in cols[4].split('/')]
            entity_tag  = cols[1]
            #print(phrase)
            #print(pos_tags)
            
            # extract annotations
            for i,token in enumerate(phrase):
            	annotated_token = AnnotatedToken()
            	annotated_token.token = token
            	
            	# pos tag
            	if i >= len(pos_tags): # WARNING: Missing POS tag!!!
            		annotated_token.pos_tag = '?'
            	else:
            		annotated_token.pos_tag = pos_tags[i]
            	
            	# entity tag in IOB-schema
            	if entity_tag == 'O':
            		annotated_token.entity_tag = 'O'
            	else:
            		if i == 0:
            			annotated_token.entity_tag = 'B-' + entity_tag
            		else:
            			annotated_token.entity_tag = 'I-' + entity_tag
            			
            	self.annotated_tokens.append(annotated_token)

        
        
def import_sentences(path):
    PHRASE_MODE = 1
    RELATION_MODE = 2
    mode = PHRASE_MODE
    sentences = []
    phrase_rows = []
    relation_rows = []
    
    f = open(path)
    for line in f:
        line = line.strip()
        
        # check if mode changes
        if not line:
            if mode == PHRASE_MODE:
                mode = RELATION_MODE
            else:
                mode = PHRASE_MODE
                
                # save current sentence
                sentence = Sentence(phrase_rows, relation_rows)
                sentences.append( sentence )
                phrase_rows = []
                relation_rows = []
                
            continue
            
        if mode == PHRASE_MODE:
            phrase_rows.append(line)
        else:
            relation_rows.append(line)
            
    return sentences



# function for importing indices of train/test split
def import_sentence_indices(path):
    indices = []
    f = open(path)
    
    for line in f:
        cols = line.split(':')
        index = int(cols[0])
        indices.append(index)
        
    return indices



def extract_sentences_subset(all_sentences, indices):
    sentences_subset = []
    
    for sentence in all_sentences:
        if sentence.sentence_id in indices:
            sentences_subset.append(sentence)
            
    return sentences_subset

    
    
#####################################################################################
# Function for importing the CONLL2004 dataset
# Parameters: path: String pointing to the directory which contains the files of the corpus   
# Returns: pair of lists; first list contains training triples, second one test triples
def import_conll2004(path):
	RELATIONS_PATH = path + '/relations.txt'
	SENTENCES_TRAIN_PATH = path + '/sentences_train.txt'
	SENTENCES_TEST_PATH = path + '/sentences_test.txt'
	
	# import sentences
	sentences = import_sentences(RELATIONS_PATH)

	# import train/test split
	indices_train = import_sentence_indices(SENTENCES_TRAIN_PATH)
	indices_test = import_sentence_indices(SENTENCES_TEST_PATH)

	# split sentences into train/test split
	sentences_train = extract_sentences_subset(sentences, indices_train)
	sentences_test = extract_sentences_subset(sentences, indices_test)
	
	return sentences_train, sentences_test
	

train,test = import_conll2004('conll2004/')
sent = train[0]
for anno in sent.annotated_tokens:
	print(anno.token + '  ' + anno.pos_tag + '  ' + anno.entity_tag)
