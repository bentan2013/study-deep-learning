# https://blog.csdn.net/weixin_34268579/article/details/92848782 
 
# ----------
#
# In this exercise, you will create a network of perceptrons that can represent
# the XOR function, using a network structure like those shown in the previous
# quizzes.
#
# You will need to do two things:
# First, create a network of perceptrons with the correct weights
# Second, define a procedure EvalNetwork() which takes in a list of inputs and
# outputs the value of this network.
#
# ----------
 
import numpy as np
 
class Perceptron:
    """
    This class models an artificial neuron with step activation function.
    """
 
    def __init__(self, weights = np.array([1]), threshold = 0):
        """
        Initialize weights and threshold based on input arguments. Note that no
        type-checking is being performed here for simplicity.
        """
        self.weights = weights
        self.threshold = threshold
 
 
    def activate(self, values):
        """
        Takes in @param values, a list of numbers equal to length of weights.
        @return the output of a threshold perceptron with given inputs based on
        perceptron weights and threshold.
        """
               
        # First calculate the strength with which the perceptron fires
        strength = np.dot(values,self.weights)
        
        # Then return 0 or 1 depending on strength compared to threshold  
        return int(strength >= self.threshold)#this row changed by myself
        
 
            
# Part 1: Set up the perceptron network
Network = [
    
    # input layer, declare input layer perceptrons here
    [ Perceptron([1,0],1),Perceptron([1,1],2),Perceptron([0,1],1) ], \
    # output node, declare output layer perceptron here
    [ Perceptron([1, -2, 1],   1) ]
 
]
# Part 2: Define a procedure to compute the output of the network, given inputs
def EvalNetwork(inputValues, Network):
    """
    Takes in @param inputValues, a list of input values, and @param Network
    that specifies a perceptron network. @return the output of the Network for
    the given set of inputs.
    """
    
    # MY MAIN CODE HERE
 
    # Be sure your output value is a single number
   
    #Method1 :
    return Network[1][0].activate([p.activate(inputValues) for p in Network[0]])
    # p is an instance of Perceptron.
    # inner brackets -->input layer
    # Network[1][0] -->Perceptron([1, -2, 1],   1)  -- Only one element
    
    #Method2 :
    # OutputValue = inputValues
    # for layer in Network:
    #     OutputValue = map(lambda p:p.activate(OutputValue), layer)
    # return OutputValue 
    ## but warning:this method return a list ,not a single number
    ## to review Python Grammar?
 
def test():
    """
    A few tests to make sure that the perceptron class performs as expected.
    """
    print ("0 XOR 0 = 0?:", EvalNetwork(np.array([0,0]), Network))
    print ("0 XOR 1 = 1?:", EvalNetwork(np.array([0,1]), Network))
    print ("1 XOR 0 = 1?:", EvalNetwork(np.array([1,0]), Network))
    print ("1 XOR 1 = 0?:", EvalNetwork(np.array([1,1]), Network))
 
if __name__ == "__main__":
    test()
 