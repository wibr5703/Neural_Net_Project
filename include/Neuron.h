#ifndef NEURON_H
#define NEURON_H

#include<vector>
using namespace std;

class Neuron;
typedef vector<Neuron> Layer;

struct Connection
{
    double weight;
    double deltaWeight;
};

class Neuron
{
private:
    static double activationFunction(double x);
    static double activationFunctionDerivative(double x);
    double sumDOW(Layer &nextLayer);
    static double randomWeight(void);
    double m_outputVal;
    vector<Connection> m_outputWeights;
    unsigned m_myIndex;
    double m_gradient;
    static double eta; //[0.0,1.0] overall net training rate
    static double alpha; //[0.0..n] multiplier of last weight change (momentum)

public:
    Neuron(unsigned numOutputs, unsigned myIndex);
    void feedForward(Layer &prevLayer);
    void setOutputVal(double val);
    double getOutputVal(void);
    void calcOutputGradients(double targetVal);
    void calcHiddenGradients(Layer &nextLayer);
    void updateInputWeights(Layer &prevLayer);
};

#endif // NEURON_H
