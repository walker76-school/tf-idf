from nltk.corpus import brown
from nltk.corpus import state_union
from nltk.corpus import shakespeare
import nltk
from CorpusReader_TFIDF import CorpusReader_TFIDF


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
