import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
df = pd.read_csv('Iris.csv')

df.head()
print(len(df))
df.isnull().sum() #Vérifier s'il manque des données
df.describe()

sns.pairplot(df, hue='Species')
plt.show()

X = df.iloc[:,:-1].values
y = df.iloc[:,-1].values

tsne = TSNE(n_components = 2, perplexity = 10, learning_rate= 100)
X_embedded = tsne.fit_transform(X)
df_tsne = pd.DataFrame(X_embedded, columns= ['Dim1','Dim2'])
df_tsne['Class'] = y
df_tsne['Class_code'] = df_tsne['Class'].astype('category').cat.codes

plt.figure(figsize=(8,6))
plt.xlabel('Dimension1')
plt.ylabel('Dimension2')
plt.scatter(df_tsne['Dim1'],df_tsne['Dim2'], c = df_tsne['Class_code'], alpha = 0.7)
plt.show()



model = LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, shuffle= True)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_pred, y_test)
mat_conf = confusion_matrix(y_pred,y_test)
print(acc)
print(f"L'accuracy s'élève à {acc}")
print(mat_conf)

map = sns.heatmap(mat_conf, annot= True, xticklabels=model.classes_, yticklabels=model.classes_, cmap='Blues')
plt.show()






