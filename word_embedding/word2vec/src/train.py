import data_prepare as dp
import cbow as cbow
def ancient(self):
    file_name = "../data/poem.txt"
    vocabulary_size = 3000
    embedding_size = 300
    window_size = 2
    num_sampled = 128
    step_num = 300001
    batch_size = 128
    sequence_word = dp.read_data(file_name)
    cbow_ = cbow.CBow(vocabulary_size=vocabulary_size,
              embedding_size=embedding_size,
              window_size=window_size,
              num_sampled=num_sampled,
                words=sequence_word)
    cbow_.create_graph()
    cbow_.train(steps=step_num, batch_size=batch_size, dst_path='../out/word2vec')

def modern(self):
    file_name = "../data/weibo.txt"
    vocabulary_size = 3000
    embedding_size = 300
    window_size = 5
    num_sampled = 128
    step_num = 300001
    batch_size = 128
    sequence_word = dp.read_data(file_name)
    cbow_ = cbow.CBow(vocabulary_size=vocabulary_size,
                      embedding_size=embedding_size,
                      window_size=window_size,
                      num_sampled=num_sampled,
                      words=sequence_word)
    cbow_.create_graph()
    cbow_.train(steps=step_num, batch_size=batch_size, dst_path='../out/word2vec_modern')

if __name__=="""__main__""":
    pass