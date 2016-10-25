#include "Neuron.h"
#include <iostream>
#include <random>
#include <chrono>
#include <math.h>
#include <cmath>
#include <vector>

using namespace std;

Neuron::Neuron(unsigned numOutputs, unsigned myIndex)
{
    for (unsigned c=0;c<numOutputs;++c){
        m_outputWeights.push_back(Connection());
        m_outputWeights.back().weight=randomWeight();
    }
    m_myIndex=myIndex;
}

void Neuron::setOutputVal(double val)
{
     m_outputVal=val;
}

double Neuron::getOutputVal(void)
{
    return m_outputVal;
}

void Neuron::feedForward(Layer &prevLayer)
{
    double sum=0.0;
    // Sum the previous Layer's outputs (now the new inputs)
    //include the bias node from previous layer
    for (unsigned n=0;n<prevLayer.size();++n){
        sum+=prevLayer[n].getOutputVal() *
            prevLayer[n].m_outputWeights[m_myIndex].weight;
    }
    m_outputVal=Neuron::activationFunction(sum);
}

double Neuron::activationFunction(double x)
{
    // tanh - output range [-1.0..1.0]
    return tanh(x);
}
double Neuron::activationFunctionDerivative(double x)
{
    // tanh derivative
    return 1.0 - x * x;
}

double Neuron::randomWeight(void)
{
    return rand() / double(RAND_MAX);
}

void Neuron::calcOutputGradients(double targetVal)
{
    double delta=targetVal-m_outputVal;
    m_gradient=delta*Neuron::activationFunctionDerivative(m_outputVal);
}

void Neuron::calcHiddenGradients(Layer &nextLayer)
{
    double dow=sumDOW(nextLayer);
    m_gradient=dow*Neuron::activationFunctionDerivative(m_outputVal);
}

double Neuron::sumDOW(Layer &nextLayer)
{
    double sum=0.0;
    //Sum of our contributions to the errors at the nodes we feed

    for(unsigned n=0;n<nextLayer.size()-1;++n){
        sum+=m_outputWeights[n].weight*nextLayer[n].m_gradient;
    }
    return sum;
}

void Neuron::updateInputWeights(Layer &prevLayer)
{
    // The weights to be updated are in the Connection container
    // in the neurons in the preceding layer

    for (unsigned n = 0; n < prevLayer.size(); ++n) {
        Neuron &neuron = prevLayer[n];
        double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;

        double newDeltaWeight =
                // Individual input, magnified by the gradient and train rate:
                eta
                * neuron.getOutputVal()
                * m_gradient
                // Also add momentum = a fraction of the previous delta weight;
                + alpha
                * oldDeltaWeight;

        neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
        neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
    }
}

double Neuron::eta=0.15; //overall net learning rate [0.0,1.0]
double Neuron::alpha=0.5; // momentum, multiplier of last deltaWeight [0.0..n]

