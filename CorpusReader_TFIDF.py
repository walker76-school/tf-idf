from nltk.corpus import brown
from nltk.corpus import state_union
from nltk.corpus import shakespeare
import nltk
import os
from nltk.corpus import PlaintextCorpusReader
from nltk.stem.porter import PorterStemmer
from math import log
from collections import Counter
from threading import Thread
from scipy import spatial


class CorpusReader_TFIDF:

    def __init__(self,
                 corpus,
                 tf="raw",
                 idf="base",
                 stopword=nltk.corpus.stopwords.words('english'),
                 stemmer=PorterStemmer(),
                 ignorecase="yes"):

        self.corpus = corpus
        self.tf_key = tf
        self.idf_key = idf

        if stopword == "none":
            self.stop_words = ()
        elif stopword is None or stopword == nltk.corpus.stopwords.words('english'):
            self.stop_words = set(nltk.corpus.stopwords.words('english'))
        else:
            if not os.path.exists(stopword):
                raise FileExistsError('Invalid stopword file!')
            else:
                reader = PlaintextCorpusReader(str(os.getcwd()), stopword)
                self.stop_words = set(reader.words([stopword, ]))

        self.stemmer = stemmer
        self.ignorecase = ignorecase
        self.dim = set()
        self.tf = {}
        self.idf = {}
        self.tf_idf_res = {}
        self.words_dict = {}

    def fileids(self):
        return self.corpus.fileids()

    def raw(self, fileids):
        if len(fileids) <= 0:
            return self.corpus.raw(self.corpus.fileids())
        else:
            return self.corpus.raw(fileids)

    def words(self, fileids=[]):
        if len(fileids) <= 0:
            ret = []
            for fileid in self.corpus.fileids():
                if fileid in self.words_dict:
                    ret = ret + self.words_dict.get(fileid)
                else:
                    processed = self.process_words(self.corpus.words(fileid))
                    self.words_dict[fileid] = processed
                    ret = ret + processed
            return ret
        elif isinstance(fileids, str):
            if fileids in self.words_dict:
                return self.words_dict.get(fileids)
            else:
                processed = self.process_words(self.corpus.words(fileids))
                self.words_dict[fileids] = processed
                return processed
        else:
            ret = []
            for fileid in fileids:
                if fileid in self.words_dict:
                    ret = ret + self.words_dict.get(fileid)
                else:
                    processed = self.process_words(self.corpus.words(fileid))
                    self.words_dict[fileid] = processed
                    ret = ret + processed
            return ret

    def process_words(self, words_list):
        # Init the words
        if isinstance(words_list, str):
            words = [words_list, ]
        elif len(words_list) <= 0:
            return []
        else:
            words = words_list

        # Ignore case
        if self.ignorecase != "no":
            words = [word.lower() for word in words]

        words = [word for word in words if word not in self.stop_words]

        # Stem all the words if there is a stemmer function
        if self.stemmer is not None:
            words = [self.stemmer.stem(word) for word in words]

        return words

    def open(self, fileid):
        return self.corpus.open(fileid)

    def abspath(self, fileid):
        return self.corpus.abspath(fileid)

    def tf_raw(self):
        print("TF Raw")
        tf = {}

        for fileid in self.fileids():

            word_freq_dict = Counter(self.words(fileid))

            tf[fileid] = word_freq_dict

        return tf

    def tf_lognormalized(self):
        print("TF Log Normalized")
        tf = {}

        for fileid in self.fileids():

            word_freq_dict = Counter(self.words(fileid))

            # This is where we can implement specific algorithms
            for key, value in word_freq_dict.items():
                word_freq_dict[key] = 1 + log(value)

            tf[fileid] = word_freq_dict

        return tf

    def tf_binary(self):
        print("TF Binary")
        tf = {}

        for fileid in self.fileids():

            word_freq_dict = Counter(self.words(fileid))

            tf[fileid] = word_freq_dict

        return tf

    def tf_runner(self):
        print("TF Runner")

        if len(self.tf) > 0:
            return self.tf

        if self.tf_key == "raw":
            self.tf = self.tf_raw()
        elif self.tf_key == "log":
            self.tf = self.tf_lognormalized()
        elif self.tf_key == "binary":
            self.tf = self.tf_binary()
        else:
            raise Exception('Invalid TF key')

        return self.tf

    def idf_base(self):
        print("IDF Base")
        idf = Counter({})
        count = len(self.fileids())

        for fileid in self.fileids():

            sub_idf = Counter({})

            words = self.words(fileid)

            for word in words:
                sub_idf[word] = 1

            # Merge sub_idf into full idf dict
            idf = idf + sub_idf

        # This is where we can implement specific algorithms
        for word in idf:
            idf[word] = log(count / idf[word])

        self.idf = idf
        return idf

    def idf_smooth(self):
        print("IDF Smooth")
        idf = Counter({})
        count = len(self.fileids())

        for fileid in self.fileids():

            sub_idf = Counter({})

            words = self.words(fileid)

            for word in words:
                sub_idf[word] = 1

            # Merge sub_idf into full idf dict
            idf = idf + sub_idf

        # This is where we can implement specific algorithms
        for word in idf:
            idf[word] = log(1 + (count / idf[word]))

        self.idf = idf
        return idf

    def idf_probability(self):
        print("IDF Probability")
        idf = Counter({})
        count = len(self.fileids())

        for fileid in self.fileids():

            sub_idf = Counter({})

            words = self.words(fileid)

            for word in words:
                sub_idf[word] = 1

            # Merge sub_idf into full idf dict
            idf = idf + sub_idf

        # This is where we can implement specific algorithms
        for word in idf:
            ni = idf[word]
            log_var = (count - ni) / ni
            if log_var < 1:
                idf[word] = 0
            else:
                idf[word] = log(log_var)

        self.idf = idf
        return idf

    def idf_runner(self):

        if len(self.idf) > 0:
            return self.idf

        if self.idf_key == "base":
            self.idf = self.idf_base()
        elif self.idf_key == "smooth":
            self.idf = self.idf_smooth()
        elif self.idf_key == "prob":
            self.idf = self.idf_probability()
        else:
            raise Exception('Invalid IDF key')

        return self.idf

    def tf_idf(self, filelist=[]):
        # Init the fileids
        if isinstance(filelist, str):
            fileids = [filelist, ]
        elif len(filelist) <= 0:
            fileids = self.fileids()
        else:
            fileids = filelist

        if len(self.dim) <= 0:
            self.tf_idf_dim()

        res = []
        if len(self.tf_idf_res) <= 0:
            tf_thread = Thread(target=self.tf_runner, args=())
            idf_thread = Thread(target=self.idf_runner, args=())

            if len(self.idf) <= 0:
                idf_thread.start()

            if len(self.tf) <= 0:
                tf_thread.start()

            if idf_thread.is_alive():
                print("IDF is alive")
                idf_thread.join()
                print("IDF is joined")

            if tf_thread.is_alive():
                print("TF is alive")
                tf_thread.join()
                print("TF is joined")

            self_dict = {}
            count = len(self.dim)
            for fileid in self.fileids():
                vec = [0] * count
                index = 0
                tf = self.tf[fileid]
                for word in self.dim:
                    vec[index] = tf.get(word, 0) * self.idf[word]
                    index += 1
                res.append(vec)
                self_dict[fileid] = vec

            self.tf_idf_res = self_dict
        else:
            count = len(self.dim)
            for fileid in fileids:
                vec = [0] * count
                index = 0
                tf = self.tf[fileid]
                for word in self.dim:
                    vec[index] = tf.get(word, 0) * self.idf[word]
                    index += 1
                res.append(vec)

        return res

    def tf_idf_dim(self):

        if len(self.dim) <= 0:
            self.dim = set(self.words())

        return self.dim

    def tf_idf_new(self, words):
        if len(self.tf_idf_res) <= 0:
            self.tf_idf(self.fileids())

        processed_words = self.process_words(words)

        # Calculate term frequencies
        if self.tf_key == "raw":
            new_tf = {}
            for word in processed_words:
                new_tf[word] = new_tf.get(word, 0) + 1
                self.dim.add(word)
        elif self.tf_key == "log":
            new_tf = {}
            for word in processed_words:
                new_tf[word] = new_tf.get(word, 0) + 1

            # This is where we can implement specific algorithms
            for key, value in new_tf.items():
                new_tf[key] = 1 + log(value)
        elif self.tf_key == "binary":
            new_tf = {}
            for word in processed_words:
                if word not in new_tf.keys():
                    new_tf[word] = 1
        else:
            raise Exception('Invalid TF key')

        count = len(self.dim)
        vec = [0] * count
        index = 0

        for word in self.dim:
            vec[index] = new_tf.get(word, 0) * self.idf.get(word, 0)
            index += 1
        return vec

    def cosine_sim(self, fileids):
        if len(fileids) != 2:
            raise Exception('cosine_sim requires two fileids')

        for fileid in fileids:
            if fileid not in self.tf_idf_res:
                self.tf_idf(self.fileids())

        return 1 - spatial.distance.cosine(self.tf_idf_res.get(fileids[0]), self.tf_idf_res.get(fileids[1]))

    def cosine_sim_new(self, words, fileid):
        if fileid not in self.tf_idf_res:
            self.tf_idf(self.fileids())

        new_tf_idf = self.tf_idf_new(words)

        return 1 - spatial.distance.cosine(new_tf_idf, self.tf_idf_res[fileid])


def test(tfidf):
    dim = tfidf.tf_idf_dim()

    count = 0
    for word in dim:
        if count >= 15:
            print()
            break
        print(word, end=" ")
        count += 1

    for id in tfidf.fileids():
        tf_idf_spec = tfidf.tf_idf(id)
        print(id, end=", ")
        for x in range(15):
            print(tf_idf_spec[0][x], end=" ")
        print()

    fileids = tfidf.fileids()
    fileid_len = len(fileids)
    for i in range(fileid_len):
        for j in range(i, fileid_len):
            id = fileids[i]
            sub_id = fileids[j]
            print(id, end=" ")
            print(sub_id, end=" : ")
            cos = tfidf.cosine_sim([id, sub_id])
            print(cos)


if __name__ == "__main__":

    nltk.download("brown")
    nltk.download("state_union")

    print("Brown")
    tfidf = CorpusReader_TFIDF(brown)
    test(tfidf)

    print("State of the Union")
    tfidf = CorpusReader_TFIDF(state_union)
    test(tfidf)

    # normal = 7:17.78
    # download = 7:06.19
    # updated words = 6:13.97
    # swapped threads = 6:11.03


