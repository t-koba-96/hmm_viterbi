
def accuracy(list1, list2):
    if len(list1) != len(list2):
        return "Error: Lists must have the same length."
    
    correct_count = sum(1 for x, y in zip(list1, list2) if x == y)
    total_count = len(list1)
    accuracy_percentage = (correct_count / total_count) * 100
    
    return accuracy_percentage