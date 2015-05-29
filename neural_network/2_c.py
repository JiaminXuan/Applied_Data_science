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
 
    t = loadImage('p/face_2copy.png')
    

    net = buildNetwork(len(t), .03*len(t), 1)
    ds = SupervisedDataSet(len(t), 1)
    ds.addSample(loadImage('pic/1.png'),(2,))
    ds.addSample(loadImage('pic/a.png'),(1,))
    ds.addSample(loadImage('pic/b.png'),(1,))
    ds.addSample(loadImage('pic/c.png'),(1,))
    ds.addSample(loadImage('pic/d.png'),(1,))
    ds.addSample(loadImage('pic/e.png'),(1,))
    ds.addSample(loadImage('pic/f.png'),(1,))
    ds.addSample(loadImage('pic/g.png'),(1,))
    ds.addSample(loadImage('pic/h.png'),(1,))
    ds.addSample(loadImage('pic/2.png'),(2,))
 
    trainer = BackpropTrainer(net, ds)
    error = 10
    iteration = 0
    while error > 0.001: 
        error = trainer.train()
        iteration += 1
        print "Iteration: {0} Error {1}".format(iteration, error)
 
    print "\nResult: ", net.activate(loadImage('pic/21.png'))
    print "\nResult: ", net.activate(loadImage('pic/22.png'))
    print "\nResult: ", net.activate(loadImage('pic/23.png'))
    print "\nResult: ", net.activate(loadImage('pic/24.png'))
    print "\nResult: ", net.activate(loadImage('pic/25.png'))
    print "\nResult: ", net.activate(loadImage('pic/26.png'))