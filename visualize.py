import pandas as pd
import matplotlib.pyplot as plt

depths = [1,3,5,7,9]
processors = [i for i in range(1, 11)]
time = []

for depth in depths:
    with open(f'result_depth_{depth}', 'r') as f:
        lines = f.readlines()
        cnt = 0
        for line in lines:
            if cnt % 2:
                time.append(float(line.strip()))
            cnt += 1
        plt.plot(processors, time, label=f'depth={depth}')
        plt.xlabel('Number of processors')
        plt.ylabel('Average Time taken')
        plt.title('Average Time taken vs. Number of Processors')
        plt.legend()
        plt.grid(True)
        time = []
plt.show()
# save the plot


