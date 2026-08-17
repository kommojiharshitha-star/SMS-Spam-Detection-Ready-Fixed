
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from preprocess import preprocess_text

df = pd.read_csv('spam.csv', encoding='latin-1')
if 'v1' in df.columns and 'v2' in df.columns:
    df = df[['v1','v2']]
    df.columns = ['label','message']
else:
    df = df[['label','message']]

df['label'] = df['label'].map({'ham':0,'spam':1})
df['processed'] = df['message'].apply(preprocess_text)

vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(df['processed'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print('='*50)
print('SMS SPAM DETECTION MODEL RESULTS')
print('='*50)
print(f'Accuracy : {accuracy_score(y_test, y_pred):.4f}')
print(f'Precision: {precision_score(y_test, y_pred):.4f}')
print(f'Recall   : {recall_score(y_test, y_pred):.4f}')
print(f'F1-score : {f1_score(y_test, y_pred):.4f}')
print()
print('Confusion Matrix')
print(confusion_matrix(y_test, y_pred))
print()
print(classification_report(y_test, y_pred, target_names=['Ham','Spam']))

pickle.dump(model, open('spam_model.pkl','wb'))
pickle.dump(vectorizer, open('vectorizer.pkl','wb'))

print('Model saved successfully.')
