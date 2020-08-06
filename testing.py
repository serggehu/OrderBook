import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


fig = plt.figure()

position = np.arange(6) + .5 

plt.tick_params(axis = 'x', colors = '#072b57')
plt.tick_params(axis = 'y', colors = '#072b57')

speeds = [1, 2, 3, 4, 1, 2]
heights = [0, 0, 0, 0, 0, 0]
rects = plt.bar(position, heights, align = 'center', color = '#b8ff5c') 
plt.xticks(position, ('A', 'B', 'C', 'D', 'E', 'F'))

plt.xlabel('X Axis', color = '#072b57')
plt.ylabel('Y Axis', color = '#072b57')
plt.title('My Chart', color = '#072b57')

plt.ylim((0,100))
plt.xlim((0,6))

plt.grid(True)



rs = [r for r in rects]

def init():
    return rs

def animate(i):
    global rs, heights
    if all(map(lambda x: x==100, heights)):
        heights = [0, 0, 0, 0, 0, 0]
    else:
        heights = [min(h+s,100) for h,s in zip(heights,speeds)]
    for h,r in zip(heights,rs):
        r.set_height(h)
        print("r ", dir(r))
    return rs

anim = animation.FuncAnimation(fig, animate, init_func=init,frames=200, interval=20, blit=True)

plt.show()