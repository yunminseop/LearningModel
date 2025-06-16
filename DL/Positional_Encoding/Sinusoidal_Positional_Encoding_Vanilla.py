import numpy as np
import matplotlib.pyplot as plt



graph = []
Pos = np.linspace(0, 15, 100)
xticks = np.linspace(0, 15, 16)

for i in range(513):
    if i % 2 == 0:
        graph.append(np.sin(Pos / (10000**(i/512))))
    else:
        graph.append(np.cos(Pos / (10000**(i/512))))

graph = np.array(graph)

# 그래프 그리기
plt.figure(figsize=(18, 9))
cmap = plt.cm.get_cmap('hsv', 19)
colors = [cmap(i) for i in range(16)]

    
for each_pos, y in enumerate(graph):
    pos = each_pos%16
    for i in y:
        if i % 2 == 0:
            if np.arcsin(i)*(10000**(each_pos/512)).is_integer():
                plt.plot(Pos, y, alpha=0.5, linewidth=0.7, color=colors[each_pos%len(colors)])
                print("True")
            else:
                plt.plot(Pos, y, alpha=0.3, linewidth=0.1, color=colors[each_pos%len(colors)])
        else:
            if np.arccos(i)*(10000**(each_pos/512)).is_integer():
                plt.plot(Pos, y, alpha=0.5, linewidth=0.7, color=colors[each_pos%len(colors)])
            else:
                plt.plot(Pos, y, alpha=0.3, linewidth=0.1, color=colors[each_pos%len(colors)])
    
plt.title('Sinusoidal Positional Encoding (Pos=[0,15])')
plt.xlabel('Pos')
plt.ylabel('PE')
plt.xticks(xticks)
plt.legend()
plt.grid(True)
plt.show()