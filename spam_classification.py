# coding=utf-8
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

messages = pd.read_csv('spam_classifier_data/SMSSpamCollection', sep='\t',
                      names=['label', 'text'])


## data cleaning and preprocessing
import re
import nltk
#nltk.download()
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
stemmer = PorterStemmer()
corpus = []
for i in range(len(messages)):
    # remove characters or numbers apart from alphabets
    clean = re.sub('[^a-zA-Z]', ' ', messages['text'][i])
    # put everything in lowercase
    clean = clean.lower()
    # split the sentences to a list of words to apply stemming/lemmatization
    clean = clean.split()

    # perform stemming operation
    clean = [stemmer.stem(word) for word in clean if not word in stopwords.words('english')]
    # join the words again to form a string
    clean = ' '.join(clean)
    # append to list
    corpus.append(clean)



## create a bag of words model
from sklearn.feature_extraction.text import CountVectorizer
# restrict maximum words to the most frequent ones using max_features attribute
cv = CountVectorizer(max_features=2500)
X = cv.fit_transform(corpus).toarray()
# create one-hot encoded label values (column:0 = ham, column:1 = spam)
y=pd.get_dummies(messages['label'])
# store back as categories for later use
y_cat = y.stack()
y_cat = pd.Series(pd.Categorical(y_cat[y_cat!=0].index.get_level_values(1)))
# drop the ham column. y=1 for spam, 0 for ham
y=y.iloc[:,1].values

## check and solve for imbalance in data, if any
count_classes = pd.value_counts(messages['label'], sort = True)
plt.figure(figsize=(10,10))
count_classes.plot(kind = 'bar', rot=0)
LABELS = ["Ham", "Spam"]
plt.title("Spam Messages Class Distribution")
plt.xticks(range(2), LABELS)
plt.xlabel("Class")
plt.ylabel("Frequency")
plt.show()
#get shapes of each class
ham = messages[messages['label']=='ham']
spam = messages[messages['label']=='spam']
print("Ham Class Shape: ", ham.shape, "\nSpam Class Shape", spam.shape)
from imblearn.combine import SMOTETomek
smk = SMOTETomek(ratio=0.5)
X_over, y_over = smk.fit_resample(X, y)

## train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

## model training using Naive Bayes Classifier
from sklearn.naive_bayes import MultinomialNB
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
# predict on test data
y_pred = nb_model.predict(X_test)

## model evaluation
# accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of Naive Bayes Classifier using Bag of Words model: ", accuracy)
# confusion matrix
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns
conmat=confusion_matrix(y_test, y_pred)
val=np.mat(conmat)
spam_classes=list(np.unique(y_cat))
df_cm = pd.DataFrame(val, index=spam_classes, columns=spam_classes)
print("Confusion Matrix: \n", df_cm)
# plot confusion matrix as a heatmap
plt.figure(figsize=(10,10))
heat_map_predictions=sns.heatmap(df_cm, annot=True, cmap='Blues', fmt='g')
heat_map_predictions.yaxis.set_ticklabels(heat_map_predictions.yaxis.get_ticklabels(), rotation=0, ha='right')
heat_map_predictions.xaxis.set_ticklabels(heat_map_predictions.xaxis.get_ticklabels(), rotation=45, ha='right')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Spam Messages Bag-of-Words Naive Bayes Classifier Results')
plt.show()

# display the results as percentages of the total number
df_cm = df_cm.astype('float')/df_cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(10,10))
heat_map_predictions=sns.heatmap(df_cm, annot=True, cmap='Blues', fmt='g')
heat_map_predictions.yaxis.set_ticklabels(heat_map_predictions.yaxis.get_ticklabels(), rotation=0, ha='right')
heat_map_predictions.xaxis.set_ticklabels(heat_map_predictions.xaxis.get_ticklabels(), rotation=45, ha='right')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Spam Messages Bag-of-Words Naive Bayes Classifier Results (%)')
plt.show()

