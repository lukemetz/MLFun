import pandas as pd
import numpy as np
import pylab as plt

clean = pd.read_csv("clean.csv")
norm = pd.read_csv("norm.csv")
norm2 = pd.read_csv("norm2.csv")

train = clean['train_cost'].dropna()
test = clean['test_cost'].dropna()

plt.plot(train, 'r')
plt.plot(test, 'r')

train = norm['train_cost'].dropna()
test = norm['test_cost'].dropna()
plt.plot(train, 'k')
plt.plot(test, 'k')

train = norm2['train_cost'].dropna()
test = norm2['test_cost'].dropna()
plt.plot(train, 'b')
plt.plot(test, 'b')

plt.show()

