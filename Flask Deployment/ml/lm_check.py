import corpus
import ml

# path = 'C:/Users/siobh/Desktop/DG Internship/Flask Deployment/corpora/hamlet.txt'
# file = open(path, 'r')
# test_text = file.read()
# file.close()

# tokens3 = corpus.tokenize(test_text)

train_path_list = ['hamlet.txt', 'caesar.txt', 'errors.txt', 'likeit.txt', 'macbeth.txt', 'romeo.txt']
# train_file = open(train_path, 'r')
# train_text = train_file.read()
# train_file.close()
train_text = ''
for fileName in train_path_list:
        file = open('C:/Users/siobh/Desktop/DG Internship/Flask Deployment/corpora/' + fileName, 'r')
        train_text += file.read()
        file.close()

train_tokens = ml.corpus.tokenize(train_text)



lang = ml.LanguageModel(3)
lang.train(train_tokens)

print(lang.find_gram_next('haste'))

print(lang.generate() + ' This was regular')
print(lang.greedy_generate())