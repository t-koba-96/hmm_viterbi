import numpy

class HmmViterbi:
    
    def __init__(self,
                observed_set_list,   # observed class list
                hidden_set_list,     # hidden class list 
                start_prob,          # start probablity
                transition_prob_mat, # transition probability matrix
                emission_prob        # emmision probability matrix
                ):
        self.observed_set = observed_set_list
        self.hidden_set = hidden_set_list
        self.start_prob = start_prob
        self.transition_prob_mat = transition_prob_mat
        self.emission_prob = emission_prob

    def viterbi(self,
                sequence # input sequence
                ):

        probabilities = []
        
        #convert given sequence of strings to indices
        targets_dict = dict(zip(self.observed_set, list(range(len(self.observed_set)))))    
        targets_sequence = []
        for item in sequence:
            targets_sequence.append(targets_dict[item])

        #first maximal probability
        probabilities.append(
            tuple(
                self.start_prob[state]*self.emission_prob[state, targets_sequence[0]]
                for state in range(len(self.hidden_set)))
        )

        for i in range(1, len(targets_sequence)):
            previous_probabilities = probabilities[-1]
            current_probabilities = []
            for col in range(len(self.transition_prob_mat[0,:])):
                p = max(
                    previous_probabilities[state]*self.transition_prob_mat[state,col]*self.emission_prob[col,targets_sequence[i]]
                    for state in range(len(self.hidden_set))
                )
                current_probabilities.append(p)
            probabilities.append(tuple(current_probabilities))

        #find the sequence of hidden states
        hidden_states_sequence = []
        for i in probabilities:
            hidden_state = self.hidden_set[numpy.argmax(i)]
            hidden_states_sequence.append(hidden_state)

        return probabilities, hidden_states_sequence