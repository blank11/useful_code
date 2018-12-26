# Step1: convert to lower and token phrase
from nltk.tokenize import WordPunctTokenizer
tokenizer = WordPunctTokenizer()

def convert_tok(phrase):
	phrase_tok = []
	for i in range(len(phrase)):
		phrase_tok.append(tokenizer.tokenize(phrase[i].lower()))

	return phrase_tok


# Step2: model
from gensim.models import Word2Vec
from gensim.models.Word2Vec import LineSentence

model = Word2Vec(Sentence, size=, window=5, min_count=5)  # size: 词向量数； window: 窗口大小； min_count: 最小词频

model.get_vector(word)  # 某一词的词向量
model.most_similar(word)  # 与该词最接近的词
model.most_similar(positive=[word1, word2], negative=[word3])

model.vocab.keys()  # 语料库中所有的词


# Step3: phrase vector
def get_phrase_embedding(phrase):
	vector = np.zeros([len(model.vector_size)], dtype='float32')
	phrase = tokenizer.tokenize(phrase.lower())
	num = 0

	for i in range(len(phrase)):
		if phrase[i] in model.vocab.keys():
			vector = vector + model.get_vector(phrase[i])
			num = num + 1

	vector = vector / num

	return vector


# Step4: find k similar sentence with query text
def find_nearest(query, data_vectors, data, k=10):
	"""
	Args:
		query: query text
		data_vectors: training text vectors
		data: training text
		k: most k similar
	Retruns:
		most k similar text
	"""
	query_vector = get_phrase_embedding(query)

	distance = np.zeros([len(data_vectors)], dtype='float32')
	for i in range(len(data_vectors)):
		distance[i] = np.dot(query_vector, data_vectors[i]) / (np.linalg.norm(query_vector) * np.linalg.norm(data_vectors[i]))

	distance_index = np.argsort(-distance)
	most_k_similar = []
	for i in range(k):
		most_k_similar = most_k_similar + data[i]

	return most_k_similar