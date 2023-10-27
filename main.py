from train import train
from test import test
from plotter import plot


model, loss, acc = train()
plot(loss, acc)
test()