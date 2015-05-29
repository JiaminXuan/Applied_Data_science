from pybrain.datasets.supervised import SupervisedDataSet 
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
import cv2
 
def loadImage(path):
    im = cv2.imread(path)
    return flatten(im)
 
def flatten(x):
    result = []
    for el in x:
        if hasattr(el, "__iter__") and not isinstance(el, basestring):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result
 
if __name__ == "__main__":
 
    t = loadImage('pic/a.png')
    

    net = buildNetwork(len(t), .03*len(t), 1)
    ds = SupervisedDataSet(len(t), 1)
    ds.addSample(loadImage('pic/a.png'),(1,))
    ds.addSample(loadImage('pic/d.png'),(1,))
    ds.addSample(loadImage('pic/e.png'),(1,))
    ds.addSample(loadImage('pic/1.png'),(0,))
    ds.addSample(loadImage('pic/b.png'),(1,))
    ds.addSample(loadImage('pic/2.png'),(0,))
 
    trainer = BackpropTrainer(net, ds)
    error = 10
    iteration = 0
    while error > 0.0001: 
        error = trainer.train()
        iteration += 1
        print "Iteration: {0} Error {1}".format(iteration, error)
 
    print "\nResult: ", net.activate(loadImage('pic/a.png'))
    print "\nResult: ", net.activate(loadImage('pic/1.png'))
    print "\nResult: ", net.activate(loadImage('pic/b.png'))
    print "\nResult: ", net.activate(loadImage('pic/2.png'))
    print "\nResult: ", net.activate(loadImage('pic/3.png'))
