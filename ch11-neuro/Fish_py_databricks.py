# coding=utf-8

# Copyright 2015 and onwards Sanford Ryza, Uri Laserson, Sean Owen and Joshua Wills
# optimized by Regina_Yang For databricks
#

# See LICENSE file for further information.

# Before run this code, first need import liberary: py4j, showit, 
# Thunder-python
 
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import thunder as td
import pandas as pd

##################
# data exploration
##################

# load some zebrafish brain data
data = td.images.fromexample('fish', engine=sc)

# explore the resulting object
#print(data)
#print(data.values)
#print(data.values.first())
#print(data.values._rdd)
#print(data.values._rdd.first())
#print(data.shape)
#print(data.count())

# plot the raw data
img = data.first()

#plt.imshow(img[0], interpolation='nearest', aspect='equal', cmap='gray')

# plot subsampled data
subsampled = data.subsample((5, 5, 1))

#plt.imshow(subsampled.first()[0], interpolation='nearest', aspect='equal', cmap='gray')

#print(subsampled.shape)

# reshuffle data to series representation
series = data.toseries()
#print(series.shape)
#print(series.index)
#print(series.count())
#print(series.values._rdd.takeSample(False, 1)[0])
#print(series.max().values)

# distributed computation of stats
stddev = series.map(lambda s: s.std())
#print(stddev.values._rdd.take(3))
#print(stddev.shape)

# collecting data locally and repacking it
repacked = stddev.toarray()


#plt.imshow(repacked[:,:,0], interpolation='nearest', cmap='gray', aspect='equal')

#print(type(repacked))
#print(repacked.shape)

# plot some of the time series themselves
img_time = series.center().sample(50).toarray
#img_time=plt.plot(series.center().sample(50).toarray().T)
#display(img_time)

# distributed computatino of custom statistics
series.map(lambda x: x.argmin())


###############################
# Clustering fish brain regions
###############################

import numpy as np
from pyspark.mllib.clustering import KMeans

data.toseries().tobinary('directory', overwrite=True)

series = data.toseries()

normalized = series.normalize(method='mean')
stddevs = (normalized.map(lambda s: s.std()).sample(1000))


img_std = plt.hist(stddevs.values, bins=20)
#print(img_std)

img_filter = plt.plot(normalized.filter(lambda s: s.std() >= 0.1).sample(50).values.T)



# perform k-means on the normalized series
ks = [5, 10, 15, 20, 30, 50, 100, 200]
models = []
for k in ks:
    models.append(KMeans.train(normalized.values._rdd.values(), k))

# define a couple functions to score the clustering quality
def model_error_1(model):
    def series_error(series):
        cluster_id = model.predict(series)
        center = model.centers[cluster_id]
        diff = center - series
        return diff.dot(diff) ** 0.5

    return (normalized.map(series_error).toarray().sum())

def model_error_2(model):
    return model.computeCost(normalized.values._rdd.values())

# compute the error metrics for the different resulting clusterings
errors_1 = np.asarray(map(model_error_1, models))
errors_2 = np.asarray(map(model_error_2, models))
img_er = plt.plot(
    ks, errors_1 / errors_1.sum(), 'k-o',
    ks, errors_2 / errors_2.sum(), 'b:v')

Er1 = errors_1 / errors_1.sum()
#print(Er1)
Er2 = errors_2 / errors_1.sum()
#print(Er2)
# plot the best performing model
model20 = models[3]
img_m20 = plt.plot(np.asarray(model20.centers).T)
#print(img_m20)
#print(np.asarray(model20.centers).T)

# finally, plot each brain region according to its characteristic behavior
from matplotlib.colors import ListedColormap
import mpld3
from mpld3 import plugins

fig, ax = plt.subplots()
by_cluster = normalized.map(lambda s: model20.predict(s)).toarray()

X = by_cluster

im = plt.imshow(by_cluster[0], interpolation='nearest',aspect='equal', cmap='gray')

fig.colorbar(im, ax=ax)

ax.set_title('Cluster membership', size=20)

plugins.connect(fig, plugins.MousePosition(fontsize=14))

displayHTML(mpld3.fig_to_html(fig))
