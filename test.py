import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import folium
%matplotlib inline

df = pd.read_csv('crime.csv',engine = 'python')
df.head()
df.dtypes
df['SHOOTING'].unique()
df.columns.values
df.loc[df.SHOOTING != 'Y', 'SHOOTING'] = 'N'

newdf = df.groupby('YEAR').count()
sns.set(style="darkgrid")
ax1 = sns.countplot(x="YEAR", data=df)
ax2 = sns.countplot(x="MONTH", data=df)
ax3 = sns.countplot(x="DAY_OF_WEEK", data=df)
ax3 = sns.countplot(x="DISTRICT", data=df)

df1 = df[['YEAR','SHOOTING']]
df2 = df1[df1['SHOOTING'] == 'Y']
maxyear = df2.groupby(['YEAR','SHOOTING']).size().idxmax()
maxyear[0]

df3 = df[['DISTRICT','SHOOTING']]
df3 = df3[df3['SHOOTING'] == 'Y']
dist = df3.groupby(['DISTRICT','SHOOTING']).size().idxmax()
dist[0]

df.head()
df['DAY_NIGHT'] = 'NIGHT'
df.loc[(df['HOUR'] >= 6) & (df['HOUR'] <= 18),'DAY_NIGHT'] = 'DAY'

df4 = df[['DAY_NIGHT','INCIDENT_NUMBER']]
hourCrime = df4.groupby('DAY_NIGHT').count().idxmax()
hourCrime[0]

df5 = df[['DAY_NIGHT','OFFENSE_CODE_GROUP']]
df5 = df5[df5['DAY_NIGHT'] == 'DAY']
groupCode = df5.groupby(['DAY_NIGHT','OFFENSE_CODE_GROUP']).size().idxmax()
groupCode[1]

df6 = df[['Lat','Long']]
df6 = df6.dropna()
df6 = df6.loc[(df6['Lat']>40)& (df6['Long']<-60)]
ax = sns.scatterplot(x="Long", y = "Lat", data=df6)

km = KMeans(n_clusters = 2)
km.fit(df6)
y_kmeans = km.predict(df6)

fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111)
scat = ax.scatter(df6['Long'], df6['Lat'], c=y_kmeans, s=50, cmap='viridis')
plt.colorbar(scat)
plt.show()

km = KMeans(n_clusters = 3)
km.fit(df6)
y_kmeans = km.predict(df6)

fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111)
scat = ax.scatter(df6['Long'], df6['Lat'], c=y_kmeans, s=50, cmap='viridis')
plt.colorbar(scat)
plt.show()

km = KMeans(n_clusters = 5)
km.fit(df6)
y_kmeans = km.predict(df6)

fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111)
scat = ax.scatter(df6['Long'], df6['Lat'], c=y_kmeans, s=50, cmap='viridis')
plt.colorbar(scat)
plt.show()

km = KMeans(n_clusters = 10)
km.fit(df6)
y_kmeans = km.predict(df6)

fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111)
scat = ax.scatter(df6['Long'], df6['Lat'], c=y_kmeans, s=50, cmap='viridis')
plt.colorbar(scat)
plt.show()

df7 = df[['Lat','Long','OFFENSE_CODE_GROUP']]
df7 = df7.dropna()


df8 = df7[df7['OFFENSE_CODE_GROUP'] == 'Drug Violation']
len(df8)

from folium.plugins import MarkerCluster

subset_of_df = df8.sample(n=5000)

m = folium.Map(location=[42.3142647,-71.1103692])

mc = MarkerCluster()

for row in subset_of_df.itertuples():
    mc.add_child(folium.Marker(location=[row.Lat,  row.Long]))

m.add_child(mc)
m
