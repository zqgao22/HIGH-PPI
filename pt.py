import numpy as np

# x1 = np.array([0.5303,0.5170,0.5006,0.4824,0.4875,0.5061,0.4946,0.4940])
# x2 = np.array([0.4988,0.4902,0.5294,0.5492,0.5462,0.5155,0.5319,0.5267])

x1 = np.array([0.1998,0.1868,0.1813,0.1766,0.1911,0.1808,0.1809,0.1935])
x2 = np.array([0.1332,0.1300,0.1379,0.1421,0.1147,0.1194,0.1403,0.1525])

# x1 = np.array([0.537,0.570,0.581,0.604,0.620,0.602,0.657,0.739,0.717])
# x2 = np.array([0.505,0.526,0.545,0.555,0.563,0.575,0.621,0.655,0.665])

# x1 = np.array([0.192,0.266,0.306,0.348,0.368,0.422,0.440,0.565,0.698])
# x2 = np.array([0.171,0.180,0.199,0.213,0.236,0.262,0.362,0.599,0.641])

n = 100000
t_0 = np.abs(np.mean(x1)-np.mean(x2))
x_all = np.concatenate((x1,x2))
c = 0.99
t = np.zeros(n)
for i in range(n):
    x_all_i = x_all
    np.random.shuffle(x_all_i)
    x1_new = x_all_i[:8]
    x2_new = x_all_i[-8:]
    t[i] = np.abs(np.mean(x1_new) - np.mean(x2_new))
p_v = np.sum(t>t_0)/n
if p_v<1-c:
    print('decision=significant')
else:
    print('decision=not significant')
print('1-c:',1-c)
print('p_v:',p_v)

