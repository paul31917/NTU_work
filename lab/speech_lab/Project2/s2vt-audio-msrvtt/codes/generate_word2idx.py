import json
from collections import Counter

files = ["./testing_data.json", "./train_val_data.json"]
top_n_frequent = 3000 # Including _PAD and _UNK

# Put all the words in a list
word_list = []
for f in files:
    f = json.load(open(f, 'r'))
    for video in f:
        for sentence in video["caption"]:
            sentence = sentence.rstrip('.').lower().split()
            for word in sentence:
                word_list.append(word)

# Generate the top n word2idx
top_n_common_words = Counter(word_list).most_common(top_n_frequent-2) # For "_PAD" and "_UNK"
word2idx = {}
word2idx["_PAD"] = 0
for i in xrange(1, top_n_frequent-1):
    word2idx[top_n_common_words[i-1][0]] = i
word2idx["_UNK"] = top_n_frequent-1
json.dump(word2idx, open("./word2idx.json", 'w'), indent=4)

