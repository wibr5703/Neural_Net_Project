#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include<vector>
using namespace std;

class Neuron;
typedef vector<Neuron> Layer;

class Neural_Network
{
private:
    vector<Layer> m_layers; //m_layers[layerNum][neuronNum]
    double m_error;
    double m_recentAverageSmoothingError;
public:
    double m_recentAverageError;
    Neural_Network(vector<unsigned> & topology); //constructor
    void feedForward( vector<double> & inputVals);
    void backProp(vector<double> & targetVal);
    void getResults(vector<double> & resultVal);
};

#endif // NEURAL_NETWORK_H
