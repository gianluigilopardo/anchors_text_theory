import re


# Jaccard similarity between two lists
def jaccard_similarity(list1, list2):
    list1 = [re.sub(r'[^\w]', ' ', s.lower()) for s in list1]
    list2 = [re.sub(r'[^\w]', ' ', s.lower()) for s in list2]
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(set(list1)) + len(set(list2))) - intersection
    return float(intersection) / union

