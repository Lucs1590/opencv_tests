from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
import utils as ut
import cv2

img = '/home/brito/Documentos/Dev/tcc/img/f1.jpeg'
img = cv2.imread(img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
rect = (430, 196, 800, 310)


img = ut.black_back(img, rect)
img = img.reshape((img.shape[0] * img.shape[1], 3))

# k means determine k
distortions = []
hist_all = []
K = range(1, 10)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(img)
    kmeanModel.fit(img)
    hist_all.append(ut.centroid_histogram(kmeanModel))
    distortions.append(sum(np.min(
        cdist(img, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / img.shape[0])

# Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()
