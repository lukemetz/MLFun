#bunch of plots to get a sense of RNN performance
import matplotlib
from matplotlib import pyplot as plt

#First test, running on a 2 unit hidden layer of rnn, varrying seqlength.

#1 -- 6.7840
#10 8.39
#seqlen 50 -- 14.5 seconds
#seqlen 100 -- 22.56
#150 28.62
#200 -- 34.38

x = [1, 10, 50, 100, 150, 200]
y = [6.784, 8.39, 14.5, 22.56, 28.62, 34.38]

if False:
    plt.plot(x, y, '-o')
    plt.plot([350], [59], '-o')
    plt.xlabel("seq length")
    plt.ylabel("time (sec)")
    plt.show()

# y-intercept of around 6.5, meaning each new sequence lenght adds like 0.07 seconds,
# The other junk is is small relative.
# Long tailed with median ish at 200, mean: 350
# this means i can plot, well looks right on the graph

# now to look at hidden dims @seqlen 100
#dim -- time

# 4 - 22.56
# 16 - 22.13 - 22.14
# 64 - 26.0 -- 22.44
# 128 - 26.9 25.3
# 256  38.58, 32.68
# 512 - 53.972
# 700 - 72.9
# 1028  -102.706
x, y = zip(*[
    (4, 22.56),
    (16, 22.14),
    (64, 22.4),
    (90, 24.19),
    (128, 25.3),
    (256, 32.68),
    (512, 53.68),
    (700, 72.9),
    (1028, 102.7),
    ])

plt.plot(x, y, '-o')
plt.plot([350], [59], '-o')
plt.xlabel("seq length")
plt.ylabel("time (sec)")
plt.show()

# So it looks like I should just never use RNN's of size < 128. They all perform the same.
# At the cost of 7 seconds, it looks like 256 is a good number. I think all of my rnns will be at this size based or 128 based on this tradeoff.

