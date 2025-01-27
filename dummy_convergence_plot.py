import numpy as np
import matplotlib.pyplot as plt
import math

x = []
y = []
cap = 1000
for i in range(1,cap + 1):
    x.append(i)
    y.append(math.log(i/(cap)))
    print(y)


plt.plot(x,y)

plt.show()
