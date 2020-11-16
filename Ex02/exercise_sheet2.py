################################################################################
## SNLP exercise sheet 2
################################################################################

'''
This function can be used for importing the corpus.
Parameters: path_to_file: string; path to the file containing the corpus
Returns: list of list; the second layer list contains tuples (token,label);
'''
def import_corpus(path_to_file):
    sentences = []
    sentence = []
    f = open(path_to_file)
    
    while True:
        line = f.readline()
        if not line: break
            
        line = line.strip()
        if len(line) == 0:
            sentences.append(sentence)    
            sentence = []
            continue
                
        parts = line.split(' ')
        sentence.append((parts[0], parts[-1]))
        
    f.close()        
    return sentences
    



# Exercise 1 ###################################################################
'''
Implement the probability distribution of the initial states.
Parameters:	state: string
            internal_representation: data structure representing the parameterization of this probability distribuion;
                this data structure is returned by the function estimate_initial_state_probabilities
Returns: float; initial probability of the given state
'''
def initial_state_probabilities(state, internal_representation):
    pass
    
    
    
    
'''
Implement the matrix of transition probabilities.
Parameters:	from_state: string;
            to_state: string;
            internal_representation: data structure representing the parameterization of the matrix of transition probabilities;
                this data structure is returned by the function estimate_transition_probabilities
Returns: float; probability of transition from_state -> to_state
'''
def transition_probabilities(from_state, to_state, internal_representation):
    pass
    
    
    
    
'''
Implement the matrix of emmision probabilities.
Parameters:	state: string;
            emission_symbol: string;
            internal_representation: data structure representing the parameterization of the matrix of emission probabilities;
                this data structure is returned by the function estimate_emission_probabilities
Returns: float; emission probability of the symbol emission_symbol if the current state is state
'''
def emission_probabilities(state, emission_symbol, internal_representation):
    pass
    
    
    
    
'''
Implement a function for estimating the parameters of the probability distribution of the initial states.
Parameters: corpus: list returned by the function import_corpus
Returns: data structure containing the parameters of the probability distribution of the initial states;
            use this data structure for the argument internal_representation of the function initial_state_probabilities
'''
def estimate_initial_state_probabilities(corpus):
    pass
    
    
    
    
'''
Implement a function for estimating the parameters of the matrix of transition probabilities
Parameters: corpus: list returned by the function import_corpus
Returns: data structure containing the parameters of the matrix of transition probabilities;
            use this data structure for the argument internal_representation of the function transition_probabilities
'''
def estimate_transition_probabilities(corpus):
    pass
    
    
    
    
'''
Implement a function for estimating the parameters of the matrix of emission probabilities
Parameters: corpus: list returned by the function import_corpus
Returns: data structure containing the parameters of the matrix of emission probabilities;
            use this data structure for the argument internal_representation of the function emission_probabilities
'''
def estimate_transition_probabilities(corpus):
    pass
    
    
    
    
    
    
# Exercise 2 ###################################################################
''''
Implement the Viterbi algorithm for computing the most likely state sequence given a sequence of observed symbols.
Parameters: observed_smbols: list of strings; the sequence of observed symbols
            initial_state_probabilities_parameters: data structure containing the parameters of the probability distribution of the initial states, returned by estimate_initial_state_probabilities
            transition_probabilities_parameters: data structure containing the parameters of the matrix of transition probabilities, returned by estimate_transition_probabilities
            emission_probabilities_parameters: data structure containing the parameters of the matrix of emission probabilities, returned by estimate_emission_probabilities
Returns: list of strings; the most likely state sequence
'''
def most_likely_state_sequence(observed_smbols, initial_state_probabilities_parameters, transition_probabilities_parameters, emission_probabilities_parameters):
    pass















