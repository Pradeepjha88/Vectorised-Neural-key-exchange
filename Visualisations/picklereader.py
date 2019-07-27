import pickle
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import numpy as np
with open('TimeSeries4.pickle', 'rb') as f:
    x = pickle.load(f)
#
dat = pd.DataFrame(x)
# x1 = np.linspace(df['k'].min(), df['k'].max(), len(df['k'].unique()))
# y1 = np.linspace(df['n'].min(), df['n'].max(), len(df['n'].unique()))
# x2, y2 = np.meshgrid(x1, y1)
# z1 = griddata((df['k'], df['n']), df['VectorTime'], (x2, y2), method='cubic')
# z2 = griddata((df['k'], df['n']), df['SerialTime'], (x2, y2), method='cubic')
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# surf = ax.plot_surface(x2, y2, z2, rstride=1, cstride=1, linewidth=0, antialiased=False,color="olive", label = "Serial Time")
# surf._facecolors2d=surf._facecolors3d
# surf._edgecolors2d=surf._edgecolors3d
# surf2 = ax.plot_surface(x2, y2, z1, rstride=1, cstride=1, linewidth=0, antialiased=False, color="blue", label = "Vector Time")
# surf2._facecolors2d=surf2._facecolors3d
# surf2._edgecolors2d=surf2._edgecolors3d
dat2 = dat[dat['k']==8]
dat = dat[dat['k'] !=8]
grouped = dat.groupby(['k','n'])
ncols=5
nrows = int(np.ceil(grouped.ngroups/ncols))
dat2.drop_duplicates(subset='l',inplace=True)
dat2.plot(x = 'l', y = "VectorTime", title = "K = 8 and N = 15",)
# fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12,4), sharey=True)
#
# for (key, ax) in zip(grouped.groups.keys(), axes.flatten()):
#     grouped.get_group(key).plot(x = "l",y = "VectorTime",title="K,N = "+str(key), sort_columns = True, ax = ax)
#
# ax.legend()
plt.show()
# #
# VectorTime = list(dat['VectorTime'])
# SerialTime = SerialTime[0:45]
# VectorTime = VectorTime[0:45]
# k = np.asarray(dat['k'])
# n = np.asarray(dat['n'])
#
# kn = np.arange(1,len(SerialTime)+1)
# # X, Y = np.meshgrid(k, n)
# plt.plot(kn, SerialTime, marker='', color='olive', linewidth=2, label = "Serial Time")
# plt.plot(kn, VectorTime, marker='', color='blue', linewidth=2l, label = "Vectorised Time")
# # ax = plt.axes(projection='3d')
# # ax.plot_surface(X, Y, SerialTime, rstride=1, cstride=1,
# #                 cmap='viridis', edgecolor='none')
# plt.xlabel("N")
# plt.ylabel("Time for updates")
# # ax.set_zlabel("Synchronisation Time")
# plt.title('Comparing Times for K = 5')
# plt.legend()
# plt.show()