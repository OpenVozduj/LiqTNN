import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

O_Z6_train = np.load('MAE_O_6train.npy')
B_Z6_train = np.load('MAE_B_6train.npy')
I_Z6_train = np.load('MAE_I_6train.npy')

O_Z7_train = np.load('MAE_O_7train.npy')
B_Z7_train = np.load('MAE_B_7train.npy')
I_Z7_train = np.load('MAE_I_7train.npy')

O_Z8_train = np.load('MAE_O_8train.npy')
B_Z8_train = np.load('MAE_B_8train.npy')
I_Z8_train = np.load('MAE_I_8train.npy')

O_Z9_train = np.load('MAE_O_9train.npy')
B_Z9_train = np.load('MAE_B_9train.npy')
I_Z9_train = np.load('MAE_I_9train.npy')

O_Z10_train = np.load('MAE_O_10train.npy')
B_Z10_train = np.load('MAE_B_10train.npy')
I_Z10_train = np.load('MAE_I_10train.npy')
#%%
font = {'family' : 'Liberation Serif',
        'weight' : 'normal',
        'size'   : 10}
cm=1/2.54
mpl.rc('font', **font)
mpl.rc('axes', linewidth=1)
mpl.rc('lines', lw=2)

fig = plt.figure(2, figsize=(18*cm, 7*cm))
ax = plt.subplot(131)
binsDTrain = np.linspace(0.0, 0.0026, 20)
ax.hist(O_Z6_train, histtype='step', color='red', bins=binsDTrain, label='Z6')
ax.hist(O_Z7_train, histtype='step', color='darkgreen', bins=binsDTrain, label='Z7')
ax.hist(O_Z8_train, histtype='step', color='blue', bins=binsDTrain, label='Z8')
ax.hist(O_Z9_train, histtype='step', color='darkviolet', bins=binsDTrain, label='Z9')
ax.hist(O_Z10_train, histtype='step', color='black', bins=binsDTrain, label='Z10')
ax.set_xlabel('MAE')
ax.set_ylabel('Number of cases')
ax.set_title('Outlets')
ax.legend(loc='best')
ax = plt.subplot(132)
binsMTrain = np.linspace(0.0, 0.0027, 20)
ax.hist(B_Z6_train, histtype='step', color='red', bins=binsMTrain, label='Z6')
ax.hist(B_Z7_train, histtype='step', color='darkgreen', bins=binsMTrain, label='Z7')
ax.hist(B_Z8_train, histtype='step', color='blue', bins=binsMTrain, label='Z8')
ax.hist(B_Z9_train, histtype='step', color='darkviolet', bins=binsMTrain, label='Z9')
ax.hist(B_Z10_train, histtype='step', color='black', bins=binsMTrain, label='Z10')
ax.set_xlabel('MAE')
ax.set_title('Volume control')
ax.legend(loc='best')
ax = plt.subplot(133)
binsYTrain = np.linspace(0.0, 0.0015, 20)
ax.hist(I_Z6_train, histtype='step', color='red', bins=binsYTrain, label='Z6')
ax.hist(I_Z7_train, histtype='step', color='darkgreen', bins=binsYTrain, label='Z7')
ax.hist(I_Z8_train, histtype='step', color='blue', bins=binsYTrain, label='Z8')
ax.hist(I_Z9_train, histtype='step', color='darkviolet', bins=binsYTrain, label='Z9')
ax.hist(I_Z10_train, histtype='step', color='black', bins=binsYTrain, label='Z10')
ax.set_xlabel('MAE')
ax.set_title('Inlet')
ax.legend(loc='best')
fig.tight_layout()
plt.show()

#%%
O_Z6_test = np.load('MAE_O_6test.npy')
B_Z6_test = np.load('MAE_B_6test.npy')
I_Z6_test = np.load('MAE_I_6test.npy')

O_Z7_test = np.load('MAE_O_7test.npy')
B_Z7_test = np.load('MAE_B_7test.npy')
I_Z7_test = np.load('MAE_I_7test.npy')

O_Z8_test = np.load('MAE_O_8test.npy')
B_Z8_test = np.load('MAE_B_8test.npy')
I_Z8_test = np.load('MAE_I_8test.npy')

O_Z9_test = np.load('MAE_O_9test.npy')
B_Z9_test = np.load('MAE_B_9test.npy')
I_Z9_test = np.load('MAE_I_9test.npy')

O_Z10_test = np.load('MAE_O_10test.npy')
B_Z10_test = np.load('MAE_B_10test.npy')
I_Z10_test = np.load('MAE_I_10test.npy')

#%%
fig = plt.figure(3, figsize=(18*cm, 7*cm))
ax = plt.subplot(131)
binsDTrain = np.linspace(0.0, 0.0028, 20)
ax.hist(O_Z6_test, histtype='step', color='red', bins=binsDTrain, label='Z6')
ax.hist(O_Z7_test, histtype='step', color='darkgreen', bins=binsDTrain, label='Z7')
ax.hist(O_Z8_test, histtype='step', color='blue', bins=binsDTrain, label='Z8')
ax.hist(O_Z9_test, histtype='step', color='darkviolet', bins=binsDTrain, label='Z9')
ax.hist(O_Z10_test, histtype='step', color='black', bins=binsDTrain, label='Z10')
ax.set_xlabel('MAE')
ax.set_ylabel('Number of cases')
ax.set_title('Outlets')
ax.legend(loc='best')
ax = plt.subplot(132)
binsMTrain = np.linspace(0.0, 0.0048, 20)
ax.hist(B_Z6_test, histtype='step', color='red', bins=binsMTrain, label='Z6')
ax.hist(B_Z7_test, histtype='step', color='darkgreen', bins=binsMTrain, label='Z7')
ax.hist(B_Z8_test, histtype='step', color='blue', bins=binsMTrain, label='Z8')
ax.hist(B_Z9_test, histtype='step', color='darkviolet', bins=binsMTrain, label='Z9')
ax.hist(B_Z10_test, histtype='step', color='black', bins=binsMTrain, label='Z10')
ax.set_xlabel('MAE')
ax.set_title('Volume control')
ax.legend(loc='best')
ax = plt.subplot(133)
binsYTrain = np.linspace(0.0, 0.0026, 20)
ax.hist(I_Z6_test, histtype='step', color='red', bins=binsYTrain, label='Z6')
ax.hist(I_Z7_test, histtype='step', color='darkgreen', bins=binsYTrain, label='Z7')
ax.hist(I_Z8_test, histtype='step', color='blue', bins=binsYTrain, label='Z8')
ax.hist(I_Z9_test, histtype='step', color='darkviolet', bins=binsYTrain, label='Z9')
ax.hist(I_Z10_test, histtype='step', color='black', bins=binsYTrain, label='Z10')
ax.set_xlabel('MAE')
ax.set_title('Inlet')
ax.legend(loc='best')
fig.tight_layout()
plt.show()
