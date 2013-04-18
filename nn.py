#!/usr/bin/python2


import math
import numpy
import matplotlib.pyplot
import logging

#numpy.random.seed(0)

def plot(inputs, outputs, actual):
    f = matplotlib.pyplot.figure()
    a = f.add_subplot(111)
    a.plot(inputs, actual, 'b-')
    a.plot(inputs, outputs, 'r.')
    matplotlib.pyplot.draw()

def sigmoid(x):
    return math.tanh(x)

def dsigmoid(y):
    return 1.0 - y**2

def random_matrix(x,y,min,max):
    range = max - min
    m = numpy.array(range * numpy.random.random_sample((x,y)) - max)
    m.resize((x,y))
    return m


class NN:
    def __init__(self, ninputs, nhidden, noutputs, regression = False):
        
        # include +1 for bias on Inputs (or NOR case won't work)
        self.ni = ninputs + 1
        self.nh = nhidden
        self.no = noutputs
        
        self.ai = numpy.ones(self.ni)
        self.ah = numpy.ones(self.nh)
        self.ao = numpy.ones(self.no)
        
        self.wi = random_matrix(self.ni, self.nh, -1, 1)
        self.wo = random_matrix(self.nh, self.no, -1, 1)
        
        self.ci = numpy.zeros((self.ni, self.nh))
        self.co = numpy.zeros((self.nh, self.no))
        
        self.regression = regression
        
    
    def feed_forward(self, inputs):
        
        if len(inputs) != (self.ni-1):
            raise ValueError('wrong number of inputs: received {0}, expected {1}.'.format(len(inputs), self.ni))
        
        for i in range(self.ni-1):
            self.ai[i] = inputs[i]
        
        for h in range(self.nh):
            total = 0.0
            for i in range(self.ni):
                total += self.ai[i] * self.wi[i][h]
            self.ah[h] = sigmoid(total)
        
        for o in range(self.no):
            total = 0.0
            for h in range(self.nh):
                total += self.ah[h] * self.wo[h][o]
            self.ao[o] = total if self.regression else sigmoid(total)
        
        return self.ao[:]
        
    
    def back_propogate(self, targets, rate, momentum):
        
        output_deltas, hidden_deltas = self._calculate_weight_deltas(targets)
        self._update_weights(output_deltas, hidden_deltas, rate, momentum)
        return self.calculate_error(targets)
        
    
    def calculate_error(self, targets):
        
        if len(targets) != self.no:
            raise ValueError('wrong number of target values: received {0}, expected {1}.'.format(len(targets), self.no))
        
        error = 0.0
        for o in range(self.no):
            error += 0.5*((targets[o]-self.ao[o])**2)
        return error
        
    
    def _update_weights(self, output_deltas, hidden_deltas, rate, momentum):
        
        for h in range(self.nh):
            for o in range(self.no):
                change = output_deltas[o] * self.ah[h]
                self.wo[h][o] += rate * change + momentum * self.co[h][o]
                self.co[h][o] = change
        
        for i in range(self.ni):
            for h in range(self.nh):
                change = hidden_deltas[h] * self.ai[i]
                self.wi[i][h] += rate * change + momentum * self.ci[i][h]
                self.ci[i][h] = change
        
    
    def _calculate_weight_deltas(self, targets):
        
        if len(targets) != self.no:
            raise ValueError('wrong number of target values: received {0}, expected {1}.'.format(len(targets), self.no))
        
        output_deltas = numpy.zeros(self.no)
        hidden_deltas = numpy.zeros(self.nh)
        
        for o in range(self.no):
            output_deltas[o] = targets[o] - self.ao[o] if self.regression else dsigmoid(self.ao[o]) * (targets[o] - self.ao[o])
        
        for h in range(self.nh):
            error = 0.0
            for o in range(self.no):
                error += output_deltas[o]*self.wo[h][o]
            hidden_deltas[h] = dsigmoid(self.ah[h]) * error
        
        return (output_deltas, hidden_deltas)
        
    
    def get_weights(self):
        return [self.wi, self.wo]
        
    
    def train_bp(self, patterns, max_iterations=1000, error_threshold=0.001, rate=0.5, momentum=0.1):
        
        logfrmt = 'i= %{0}d, error = %-7f'.format(str(len(str(max_iterations))))
        i = -1
        for i in xrange(0,max_iterations+1):
            error = 0.0
            for p in patterns:
                self.feed_forward(p[0])
                error += self.back_propogate(p[1], rate, momentum)
            
            if error < error_threshold:
                break
            
            if (i % 100 == 0):
                logging.info(logfrmt, i, error)
            
        return (i,error)
        
    

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Neural Network Simulator")
    parser.add_argument('--loglevel',   default='WARNING',      help="set logging level (default: warning).",   choices=['INFO','WARNING','CRITICAL'])
    parser.add_argument('--logfile',    default='/dev/stderr',  help="file to log to (default: /dev/stderr).")
    parser.add_argument('--hidden',     default='2',            help="number of hidden neurons (default: 2).",  type=int)
    parser.add_argument('--patterns',   default='n',            help="patterns specified (default: n).")
    parser.add_argument('--weights',    default='n',            help="display weights when done (default: n).")
    parser.add_argument('method',       default='bp',           help="training method for network.",            choices=['bp', 'ga', 'pso'])
    args = parser.parse_args()
    args = vars(args)
    
    loglevel =  args['loglevel']
    logfile =   args['logfile']
    nh =        int(args['hidden'])
    method =    args['method']
    pattern =   args['patterns']
    weights =   args['weights']
    
    numeric_level = getattr(logging, loglevel, None)
    logging.basicConfig(filename=logfile, level=numeric_level, format='%(asctime)s %(levelname)s:\t%(message)s', datefmt='%H:%M:%S')
    
    ni = -1
    no = -1
    patterns = []
    op = "unspecified"
    
    if 'n' != pattern:
        ni = int(raw_input("Number of inputs: "))
        no = int(raw_input("Number of outputs: "))
        print("Enter patterns as '{0} = {1}' where i's are input values and o's are output values".format(" ".join(["i"]*ni), " ".join(["o"]*no)))
        i = 1
        while True:
            line = raw_input("%2d: " % (i))
            if not line:
                break
            inputs, outputs = line.split('=')
            patterns.append([map(float, inputs.strip().split(" ")), map(float, outputs.strip().split(" "))])
            i += 1
        op = raw_input("Operator name: ")
    else:
        ni = 2
        no = 1
        patterns = numpy.array([[[0,0],[0]], [[0,1],[1]], [[1,0],[1]], [[1,1],[0]]])
        op = "xor"

    
    net = NN(ni,nh,no,regression=False)
    
    if 'bp' == method:
        i, error = net.train_bp(patterns, max_iterations=10000, error_threshold=0.01, rate=0.5, momentum=0.25)
        print("Network trained to error of %7f in %d epochs." % (error,i))
    else:
        raise NotImplementedError("No other methods ready yet :-/")
    
    for p in patterns:
        r = net.feed_forward(p[0])
        print(u"%7f %s %7f = %+7f \u2248 %d." % (p[0][0], op, p[0][1], r, round(r),))
        
    if 'n' != weights:
        inputs, outputs = net.get_weights()
        print("Inputs to Hidden:\n%s" % inputs)
        print("Hidden to Outputs:\n%s" % outputs)
    

if __name__ == '__main__':
    main()
