!pip install sentence_transformers

import json
import string
import gzip
import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA

from sentence_transformers import SentenceTransformer

path = 'data/Video_Games_5.json.gz'
def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield json.loads(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

data = getDF(path)
X = data.reviewText[0:1000]
Y = data.overall[0:1000]

countClasses = pd.value_counts(Y, sort=True)
countClasses.plot(kind='bar', rot=0)
plt.title("Rating Class Distribution")
plt.xlabel("Class")
plt.ylabel("Frequency")

def cleaningData(text):
    cleanText = []
    for sent in text:
    	if type(sent) == float:
    		continue
    	words = sent.split()
    	table = str.maketrans("","",string.punctuation)
    	stripped = [w.translate(table) for w in words]
    	words = [word.lower() for word in stripped]
    	sent = " ".join(words)
    	cleanText.append(sent)

    return cleanText
  
cleanText = cleaningData(X)
cleanText[0:5]

def generateEmbedding(text):
	model = SentenceTransformer('bert-base-nli-mean-tokens')
	bertTrain = model.encode(text)
	return bertTrain

embedding = generateEmbedding(cleanText)

from imblearn.combine import SMOTETomek

for i in range(len(Y)):
  Y[i] = int(Y[i])
smk = SMOTETomek(random_state = 42)
text, label = smk.fit_sample(embedding,Y)
countClasses2 = pd.value_counts(label, sort=True)
countClasses2.plot(kind='bar', rot=0)
plt.title("Rating Class Distribution post sampling")
plt.xlabel("Class")
plt.ylabel("Frequency")

def decompose(components, data, labels):
    axes = ['component_1','component_2']
    matrix = PCA(n_components=components).fit_transform(data)
    df_matrix = pd.DataFrame(matrix)
    df_matrix.rename({i:axes[i] for i in range(components)}, axis=1, inplace=True)
    df_matrix['labels'] = labels

    return df_matrix

from sklearn.metrics import silhouette_score
kmeans = [KMeans(n_clusters=k, random_state=42).fit(text) for k in range(3,21)]

silhouetteScores = [silhouette_score(text, model.labels_) for model in kmeans]

print(silhouetteScores)

plt.plot(range(3,21,1), silhouetteScores, label="Silhouette Curve")
plt.xlabel("$k$")
plt.ylabel("Silhouette Score")
plt.grid(linestyle='--')

plt.title("Silhouette curve to predict optimal number of clusters")

k = np.argmax(silhouetteScores) + 2
plt.axvline(x = k+1, linestyle='--', c='green', linewidth=3, label="Optimal number of clusters ({})".format(k))
plt.scatter(k+1,silhouetteScores[k-2], c='red', s=400)
plt.legend(shadow=True)

plt.show()

optimalCluster = k

def clustering_KMeans(embedding, n_clusters):
  clusterModel = KMeans(n_clusters = n_clusters)
  clusterModel.fit(embedding)
  clusterAssignments = clusterModel.labels_

  pca_df = decompose(2,embedding, clusterAssignments)
  sns.scatterplot(x=pca_df.component_1, y=pca_df.component_2, hue = pca_df.labels, palette='Set1')
  plt.show()
  return clusterAssignments

clusterAssignments = clustering_KMeans(text, optimalCluster)

from sklearn.metrics import silhouette_score
heirarchical = [AgglomerativeClustering(n_clusters=k).fit(text) for k in range(3,21)]
silhouetteScores = [silhouette_score(text, model.labels_) for model in heirarchical]

print(silhouetteScores)

plt.plot(range(3,21,1), silhouetteScores, label="Silhouette Curve")
plt.xlabel("$k$")
plt.ylabel("Silhouette Score")
plt.grid(linestyle='--')

plt.title("Silhouette curve to predict optimal number of clusters")

k = np.argmax(silhouetteScores) + 2
plt.axvline(x = k+1, linestyle='--', c='green', linewidth=3, label="Optimal number of clusters ({})".format(k))
plt.scatter(k+1,silhouetteScores[k-2], c='red', s=400)
plt.legend(shadow=True)

plt.show()

optimalCluster = k

def clustering_Agglomerative(embedding, n_clusters):
  clusterModel = AgglomerativeClustering(n_clusters=n_clusters)
  clusterModel.fit(embedding)
  clusterAssignments = clusterModel.labels_

  pca_df = decompose(2,embedding, clusterAssignments)
  sns.scatterplot(x=pca_df.component_1, y=pca_df.component_2, hue = pca_df.labels, palette='Set1')
  plt.show()
  return clusterAssignments

agglomerativeClustering = clustering_Agglomerative(text, optimalCluster)

ctab = pd.crosstab(clusterAssignments, label)
ctab

cluster = [[] for i in range(3)]
for i in range(len(cleanText)):
  cluster[clusterAssignments[i]].append(cleanText[i])

for i in range(3):
  print("Cluster ",i)
  for j in range(5):
    print(j+1,':',cluster[i][j])
  print("\n")

ctab = pd.crosstab(agglomerativeClustering, label)
ctab

cluster = [[] for i in range(3)]
for i in range(len(cleanText)):
  cluster[clusterAssignments[i]].append(cleanText[i])

for i in range(3):
  print("Cluster ",i)
  for j in range(5):
    print(j+1,':',cluster[i][j])
  print("\n")
