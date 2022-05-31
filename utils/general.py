import re


# Jaccard similarity between two lists
def jaccard_similarity(list1, list2):
    list1 = [re.sub(r'[^a-zA-Z]', '', s.lower()) for s in list1]
    list2 = [re.sub(r'[^a-zA-Z]', '', s.lower()) for s in list2]
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(set(list1)) + len(set(list2))) - intersection
    return float(intersection) / union


# preprocess new words as vectorizer
def get_words(vectorizer, doc):
    if vectorizer:
        p = vectorizer.build_preprocessor()
        p_doc = p(doc)
        t = vectorizer.build_tokenizer()
        words = t(p_doc)
    else:
        words = doc.split()
    return words


# count multiplicities
def count_multiplicities(doc, vectorizer=None):
    counter = {}
    words = get_words(vectorizer, doc)
    for word in words:
        if word not in counter:
            counter[word] = 0
        counter[word] += 1
    return dict(sorted(counter.items(), key=lambda item: item[1], reverse=True))


# returns linear coefficient for each word
def get_coefficients(model, doc, vectorizer):
    dic = vectorizer.get_feature_names()
    words = get_words(vectorizer, doc)
    coefs = {}
    for word in words:
        if word in dic:
            coefs[word] = model.coef_[0, dic.index(word)]
        else:
            coefs[word] = 0
    return dict(sorted(coefs.items(), key=lambda item: item[1], reverse=True))


# returns inverse document frequency term for each word
def get_idf(doc, vectorizer):
    words = get_words(vectorizer, doc)
    idx = []
    vocabulary = vectorizer.vocabulary_
    idf = {}
    for w in words:
        if w in vocabulary:
            idx.append(vocabulary[w])
        else:
            idf[w] = 0
    global_dic = vectorizer.get_feature_names()
    idf_ = vectorizer.idf_
    for i in idx:
        idf[global_dic[i]] = idf_[i]
    return dict(sorted(idf.items(), key=lambda item: item[1], reverse=True))


# rank words of example by lambda_j v_j
def rank_by_coefs(model, example, vectorizer):
    word_coefs = []
    p = vectorizer.build_preprocessor()
    p_doc = p(example)
    t = vectorizer.build_tokenizer()
    words = t(p_doc)
    multiplicities = count_multiplicities(example, vectorizer)  # \m_j
    coefficients = get_coefficients(model, example, vectorizer)  # \lambda_j
    idf = get_idf(example, vectorizer)  # \v_j
    coefs = {w: coefficients[w] * idf[w] for w in words}  # \lambda_jv_j
    coefs = dict(sorted(coefs.items(), key=lambda item: item[1], reverse=True))
    for word in coefs.keys():
        for i in range(multiplicities[word]):
            word_coefs.append(word)
    return word_coefs
