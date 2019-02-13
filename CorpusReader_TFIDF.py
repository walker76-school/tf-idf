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

        # Setup stop words
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

        # Setup caching self variables
        self.dim = set()
        self.tf = {}
        self.idf = {}
        self.tf_idf_res = {}
        self.words_dict = {}

    # fileids
    #
    # inputs:
    #   none
    # outputs:
    #   list of fileids of the corpus
    #
    def fileids(self):
        return self.corpus.fileids()

    # raw
    #
    # inputs:
    #   fileids - a list or single fileid
    # outputs:
    #   raw text from the files given by the fileids
    #
    def raw(self, fileids):
        if len(fileids) <= 0:
            return self.corpus.raw(self.corpus.fileids())
        else:
            return self.corpus.raw(fileids)

    # words
    #
    # inputs:
    #   fileids - a list or single fileid
    # outputs:
    #   list of words from the fileids
    #
    def words(self, fileids=[]):
        # calculate for all fileids
        if len(fileids) <= 0:
            ret = []
            for fileid in self.corpus.fileids():
                # Retrieve from cache if exists
                if fileid in self.words_dict:
                    ret = ret + self.words_dict.get(fileid)
                else:
                    processed = self.process_words(self.corpus.words(fileid))
                    self.words_dict[fileid] = processed
                    ret = ret + processed
            return ret
        # retrieve words for a single fileid
        elif isinstance(fileids, str):
            # Retrieve from cache if exists
            if fileids in self.words_dict:
                return self.words_dict.get(fileids)
            else:
                processed = self.process_words(self.corpus.words(fileids))
                self.words_dict[fileids] = processed
                return processed
        # retrieve words for the fileids
        else:
            ret = []
            for fileid in fileids:
                # Retrieve from cache if exists
                if fileid in self.words_dict:
                    ret = ret + self.words_dict.get(fileid)
                else:
                    processed = self.process_words(self.corpus.words(fileid))
                    self.words_dict[fileid] = processed
                    ret = ret + processed
            return ret

    # process_words
    #
    # inputs:
    #   words_list - a list of words
    # outputs:
    #   list of words processed by ignorecase, stemmer and stopwords
    #
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

    # open
    #
    # inputs:
    #   fileid - fileid to open a stream to
    # outputs:
    #   stream to the fileid
    #
    def open(self, fileid):
        return self.corpus.open(fileid)

    # abspath
    #
    # inputs:
    #   fileid - fileid
    # outputs:
    #   stream to the fileid
    #
    def abspath(self, fileid):
        return self.corpus.abspath(fileid)

    # tf_raw
    #
    # inputs:
    #   none
    # outputs:
    #   calculated tf using raw method, stored in cache
    #
    def tf_raw(self):
        tf = {}

        # calculates tf raw for each fileid
        for fileid in self.fileids():

            word_freq_dict = Counter(self.words(fileid))

            tf[fileid] = word_freq_dict

        return tf

    # tf_lognormalized
    #
    # inputs:
    #   none
    # outputs:
    #   calculated tf using log normalized method, stored in cache
    #
    def tf_lognormalized(self):
        tf = {}

        # calculates tf normalized for each fileid
        for fileid in self.fileids():

            word_freq_dict = Counter(self.words(fileid))

            # implements normalized method
            for key, value in word_freq_dict.items():
                word_freq_dict[key] = 1 + log(value)

            tf[fileid] = word_freq_dict

        return tf

    # tf_binary
    #
    # inputs:
    #   none
    # outputs:
    #   calculated tf using binary method, stored in cache
    #
    def tf_binary(self):
        tf = {}

        # calculates tf binary for each fileid
        for fileid in self.fileids():

            word_freq_dict = Counter(self.words(fileid))

            # implements binary method
            for key, value in word_freq_dict.items():
                word_freq_dict[key] = 1

            tf[fileid] = word_freq_dict

        return tf

    # tf_runner
    #
    # inputs:
    #   none
    # outputs:
    #   calculated tf
    #
    def tf_runner(self):

        # returns from cache if exists
        if len(self.tf) > 0:
            return self.tf

        # determines what function to use for tf
        if self.tf_key == "raw":
            self.tf = self.tf_raw()
        elif self.tf_key == "log":
            self.tf = self.tf_lognormalized()
        elif self.tf_key == "binary":
            self.tf = self.tf_binary()
        else:
            raise Exception('Invalid TF key')

        return self.tf

    # idf_base
    #
    # inputs:
    #   none
    # outputs:
    #   calculated idf using the base method
    #
    def idf_base(self):
        idf = Counter({})
        count = len(self.fileids())

        # calculate which docs have the term
        for fileid in self.fileids():

            sub_idf = Counter({})

            words = self.words(fileid)

            for word in words:
                sub_idf[word] = 1

            # Merge sub_idf into full idf dict
            idf = idf + sub_idf

        for word in idf:
            idf[word] = log(count / idf[word])

        self.idf = idf
        return idf

    # idf_smooth
    #
    # inputs:
    #   none
    # outputs:
    #   calculated idf using the smoothed method
    #
    def idf_smooth(self):
        idf = Counter({})
        count = len(self.fileids())

        # calculate which docs have the term
        for fileid in self.fileids():

            sub_idf = Counter({})

            words = self.words(fileid)

            for word in words:
                sub_idf[word] = 1

            # Merge sub_idf into full idf dict
            idf = idf + sub_idf

        for word in idf:
            idf[word] = log(1 + (count / idf[word]))

        self.idf = idf
        return idf

    # idf_probability
    #
    # inputs:
    #   none
    # outputs:
    #   calculated idf using the probabilistic method
    #
    def idf_probability(self):
        idf = Counter({})
        count = len(self.fileids())

        # calculate which docs have the term
        for fileid in self.fileids():

            sub_idf = Counter({})

            words = self.words(fileid)

            for word in words:
                sub_idf[word] = 1

            # Merge sub_idf into full idf dict
            idf = idf + sub_idf

        for word in idf:
            ni = idf[word]
            log_var = (count - ni) / ni
            if log_var < 1:
                idf[word] = 0
            else:
                idf[word] = log(log_var)

        self.idf = idf
        return idf

    # idf_runner
    #
    # inputs:
    #   none
    # outputs:
    #   calculated idf
    #
    def idf_runner(self):

        # returns from cache if exists
        if len(self.idf) > 0:
            return self.idf

        # determines what function to use for idf
        if self.idf_key == "base":
            self.idf = self.idf_base()
        elif self.idf_key == "smooth":
            self.idf = self.idf_smooth()
        elif self.idf_key == "prob":
            self.idf = self.idf_probability()
        else:
            raise Exception('Invalid IDF key')

        return self.idf

    # tf_idf
    #
    # inputs:
    #   filelist - list of files to calculate tf_idf for
    # outputs:
    #   calculated tf_idf
    #
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
        # only executes if tf_idf isn't cached
        if len(self.tf_idf_res) <= 0:
            # threads the tf and idf
            tf_thread = Thread(target=self.tf_runner, args=())
            idf_thread = Thread(target=self.idf_runner, args=())

            if len(self.idf) <= 0:
                idf_thread.start()

            if len(self.tf) <= 0:
                tf_thread.start()

            if idf_thread.is_alive():
                idf_thread.join()

            if tf_thread.is_alive():
                tf_thread.join()

            # calculates tf_idf for every fileid in the corpus
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
            # calculates tf_idf for every fileid passed in
            for fileid in fileids:
                vec = [0] * count
                index = 0
                tf = self.tf[fileid]
                for word in self.dim:
                    vec[index] = tf.get(word, 0) * self.idf[word]
                    index += 1
                res.append(vec)

        return res

    # tf_idf_dim
    #
    # inputs:
    #   none
    # outputs:
    #   list of words for the corresponding tf_idf vectors
    #
    def tf_idf_dim(self):

        # calculate tf_idf_dim for cache
        if len(self.dim) <= 0:
            self.dim = set(self.words())

        return self.dim

    # tf_idf_dim
    #
    # inputs:
    #   words - list of words to treat as a document
    # outputs:
    #   tf_idf for "new document"
    #
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

        # allocates vector
        count = len(self.dim)
        vec = [0] * count
        index = 0

        # calculates tf_idf
        for word in self.dim:
            vec[index] = new_tf.get(word, 0) * self.idf.get(word, 0)
            index += 1
        return vec

    # cosine_sim
    #
    # inputs:
    #   fileids - list of two fileids to find the difference between
    # outputs:
    #   the cosine difference between two fileids
    #
    def cosine_sim(self, fileids):
        if len(fileids) != 2:
            raise Exception('cosine_sim requires two fileids')

        # checks that tf_idf exists for fileids
        for fileid in fileids:
            if fileid not in self.tf_idf_res:
                self.tf_idf(self.fileids())

        # returns cosine difference
        return 1 - spatial.distance.cosine(self.tf_idf_res.get(fileids[0]), self.tf_idf_res.get(fileids[1]))

    # cosine_sim_new
    #
    # inputs:
    #   words - list of words to treat as a document
    #   fileid - fileid to find the difference between
    # outputs:
    #   the cosine difference between fileid and the new document
    #
    def cosine_sim_new(self, words, fileid):
        if fileid not in self.tf_idf_res:
            self.tf_idf(self.fileids())

        new_tf_idf = self.tf_idf_new(words)

        return 1 - spatial.distance.cosine(new_tf_idf, self.tf_idf_res[fileid])
