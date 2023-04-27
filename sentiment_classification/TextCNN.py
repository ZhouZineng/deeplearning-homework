"""
Author：ZhouZineng  
Date：2023.04.20
"""

import gensim
import numpy as np
import argparse


def build_word2id(file):
    word2id = {'_PAD_': 0}
    path = ['./data/train.txt', './data/validation.txt']
    print(path)
    for _path in path:
        with open(_path, encoding='utf-8') as f:
            for line in f.readlines():
                sp = line.strip().split()
                for word in sp[1:]:
                    if word not in word2id.keys():
                        word2id[word] = len(word2id)
    with open(file, 'w', encoding='utf-8') as f:
        for w in word2id:
            f.write(w + '\t')
            f.write(str(word2id[w]))
            f.write('\n')


def build_word2vec(fname, word2id, save_to_path=None):
    n_words = max(word2id.values()) + 1
    model = gensim.models.KeyedVectors.load_word2vec_format(fname,
                                                            binary=True)
    word_vecs = np.array(np.random.uniform(-1., 1., [n_words, model.vector_size]))
    for word in word2id.keys():
        try:
            word_vecs[word2id[word]] = model[word]
        except KeyError:
            pass
    if save_to_path:
        with open(save_to_path, 'w', encoding='utf-8') as f:
            for vec in word_vecs:
                vec = [str(w) for w in vec]
                f.write(' '.join(vec))
                f.write('\n')
    return word_vecs


if __name__ == '__main__':
    parsers=argparse.ArgumentParser()
    parsers.add_argument('--word2vec',type=str,default='./data/GoogleNews-vectors-negative300.bin')
    parsers.add_argument('--word2id',type=str,default='./data/word2id.txt')
    parsers.add_argument('--word2vec_save',type=str,default='./data/word2vec.txt')
    args=parsers.parse_args()
    build_word2id(args.word2id)
    build_word2vec(args.word2vec, args.word2id, args.word2vec_save)
