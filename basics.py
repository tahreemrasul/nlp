### Thie file contains stemming, lemmatization and bag of words techniques
# coding=utf-8
import re
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
# nltk.download()

paragraph = """The son of a livery-stable manager, John Keats received relatively little formal education. His father 
died in 1804, and his mother remarried almost immediately. Throughout his life Keats had close emotional ties to his 
sister, Fanny, and his two brothers, George and Tom. After the breakup of their mother’s second marriage, 
the Keats children lived with their widowed grandmother at Edmonton, Middlesex. John attended a school at Enfield, 
two miles away, that was run by John Clarke, whose son Charles Cowden Clarke did much to encourage Keats’s literary 
aspirations. At school Keats was noted as a pugnacious lad and was decidedly “not literary,” but in 1809 he began to 
read voraciously. After the death of the Keats children’s mother in 1810, their grandmother put the children’s 
affairs into the hands of a guardian, Richard Abbey. At Abbey’s instigation John Keats was apprenticed to a surgeon 
at Edmonton in 1811. He broke off his apprenticeship in 1814 and went to live in London, where he worked as a 
dresser, or junior house surgeon, at Guy’s and St. Thomas’ hospitals. His literary interests had crystallized by this 
time, and after 1817 he devoted himself entirely to poetry. From then until his early death, the story of his life is 
largely the story of the poetry he wrote. """

sentences = nltk.sent_tokenize(paragraph)
print(len(sentences), "sentences were created")
words_ = nltk.word_tokenize(paragraph)
print(len(words_), "words were created")

## stemming
print("\nStemming Output")
stemmer = PorterStemmer()
for i in range(len(sentences)):
    words = nltk.word_tokenize(sentences[i])
    words = [stemmer.stem(word) for word in words if word not in set(stopwords.words('english'))]
    sentences[i] = ' '.join(words)
    print(sentences[i])

## lemmatization
print("\nLemmatization Output")
lemmatizer = WordNetLemmatizer()
sentences_dup = nltk.sent_tokenize(paragraph)
for i in range(len(sentences_dup)):
    words = nltk.word_tokenize(sentences_dup[i])
    words = [lemmatizer.lemmatize(word) for word in words if word not in set(stopwords.words('english'))]
    sentences_dup[i] = ' '.join(words)
    print(sentences_dup[i])


## perform some additional cleanup using re lib
print("\nAdditional Cleanup Output")
stemmer = PorterStemmer()
corpus = []
for i in range(len(sentences)):
    review = re.sub('[^a-zA-Z]', ' ', sentences[i])
    review = review.lower()
    review = review.split()
    review = [stemmer.stem(word) for word in review if word not in set(stopwords.words('english'))]
    sentences[i] = ' '.join(review)
    print(sentences[i])
    corpus.append(review)

corpus = [item for sublist in corpus for item in sublist]

## Creating the bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()
print(X)