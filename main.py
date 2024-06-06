import os
import yaml
import argparse
import numpy as np
from collections import Counter

from metrics import accuracy
from viterbi import HmmViterbi

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_directory', type=str, default='sample_datas')
    parser.add_argument('--train_files',    type=str, default='01.yml,02.yml')
    parser.add_argument('--test_files',     type=str, default='03.yml')
    return parser.parse_args()

def get_probs(train_dict):
    observed_set = sorted(list(set([item for sublist in train_dict['observed_states'] for item in sublist])))
    hidden_set = sorted(list(set([item for sublist in train_dict['hidden_states'] for item in sublist])))

    hidden_stt_list = hidden_set + [states[0] for states in train_dict['hidden_states']]
    hidden_stt_count = Counter(hidden_stt_list)
    hidden_stt_count = {state: count - 1 for state, count in hidden_stt_count.items()}
    total_stt_count = sum(hidden_stt_count.values())
    start_prob_dict = {hidden_state: count / total_stt_count for hidden_state, count in hidden_stt_count.items()}
    start_prob = [prob for prob in start_prob_dict.values()]

    transition_counts = {hidden: {hidden: 0 for hidden in hidden_set} for hidden in hidden_set}
    for states in train_dict['hidden_states']:
        for i in range(len(states) - 1):
            current_state = states[i]
            next_state = states[i + 1]
            transition_counts[current_state][next_state] += 1
    transition_prob_matrix = np.zeros((3, 3))
    for i, hidden in enumerate(hidden_set):
        for j, next_hidden in enumerate(hidden_set):
            transition_prob_matrix[i, j] = transition_counts[hidden][next_hidden] / sum(transition_counts[hidden].values())

    observed_states = [item for sublist in train_dict['observed_states'] for item in sublist]
    hidden_states = [item for sublist in train_dict['hidden_states'] for item in sublist]
    emission_counts = {hidden: {observed: 0 for observed in observed_set} for hidden in hidden_set}
    for i in range(len(hidden_states)):
        observed_state = observed_states[i]
        hidden_state = hidden_states[i]
        emission_counts[hidden_state][observed_state] += 1
    emission_prob_matrix = np.zeros((len(emission_counts), len(observed_set)))
    for i, hidden_state in enumerate(list(emission_counts.keys())):
        total_count = sum(emission_counts[hidden_state].values())
        for j, observed_state in enumerate(observed_set):
            emission_prob_matrix[i, j] = emission_counts[hidden_state][observed_state] / total_count

    return observed_set, hidden_set, start_prob, transition_prob_matrix, emission_prob_matrix

def main():
    args = get_arguments()
    train_dict = {"observed_states": [], "hidden_states": []}
    for train_file in args.train_files.split(','):
        file_path = os.path.join(args.data_directory, train_file)
        with open(file_path, "r") as file:
            data = yaml.safe_load(file)
            train_dict['observed_states'].append(data['observed'])
            train_dict['hidden_states'].append(data['hidden'])

    observed_set, hidden_set, start_prob, transition_prob_matrix, emission_prob_matrix = get_probs(train_dict)
    viterbi = HmmViterbi(observed_set, hidden_set, start_prob, transition_prob_matrix, emission_prob_matrix)
    
    for test_file in args.test_files.split(','):
        file_path = os.path.join(args.data_directory, test_file)
        with open(file_path, "r") as file:
            data = yaml.safe_load(file)
            pred_probs, pred_seq = viterbi.viterbi(data['observed'])
            print('====================================================')
            print(test_file)
            print('Pred', pred_seq)
            print('Target', data['hidden'])
            print('[Acc] ', accuracy(pred_seq, data['hidden']))
            print('====================================================')


if __name__ == '__main__':
    main()