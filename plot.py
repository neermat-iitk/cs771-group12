import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


plt.figure()
df = pd.read_csv('cache/lambda_2.csv')
plt.plot(df['Lambda'], df['avg'], 'b')
plt.plot(df['Lambda'], df['avg'], 'b*')
plt.xlabel('$\lambda$')
plt.ylabel('nDCG@20')
plt.grid()
plt.savefig('lambda.png', bbox_inches='tight')

plt.figure()
df = pd.read_csv('cache/k.csv')
plt.plot(df['k'], df['avg'], 'b')
plt.plot(df['k'], df['avg'], 'b*')
plt.xlabel('$k$')
plt.ylabel('nDCG@20')
plt.grid()
plt.savefig('k.png', bbox_inches='tight')

plt.figure()
df = pd.read_csv('cache/alpha.csv')
plt.plot(df['alpha'], df['avg'], 'b')
plt.plot(df['alpha'], df['avg'], 'b*')
plt.xlabel('$\\alpha$')
plt.ylabel('nDCG@20')
plt.grid()
plt.savefig('alpha.png', bbox_inches='tight')
