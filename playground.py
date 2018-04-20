# -*- coding: utf-8 -*-
import plot
path = r"C:\Users\w00406273\Desktop\result.txt"
pre = list()
y_test = list()
diff = list()
with open(path, 'r') as f:
    for line in f.readlines():
        ss = line.split("\t")
        pre.append(float(ss[0]))
        y_test.append(float(ss[1]))
        diff.append(float(ss[2]))

# plot.histgram_demo(y_test)

bins = list()
cnts = list()
for i in xrange(15):
    bins.append(0)
    cnts.append(0)
for i in range(len(pre)):
    idx = int(y_test[i] // 10)
    bins[idx] += diff[i]
    cnts[idx] += 1
for i in xrange(10):
    print("%d-%d"%(i*10, (i+1)*10 ),cnts[i], bins[i]/cnts[i])