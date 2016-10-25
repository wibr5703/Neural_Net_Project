#include "Neural_Network.h"
#include <iostream>
#include <cassert>
#include "Neuron.h"
#include <vector>
#include <math.h>
using namespace std;

class Neuron;
typedef vector<Neuron> Layer;

Neural_Network::Neural_Network(vector<unsigned> &topology)
{
    unsigned numLayers=topology.size();
    for(unsigned layerNum=0;layerNum<numLayers;++layerNum){
        m_layers.push_back(Layer());
        unsigned numOutputs=layerNum==topology.size()-1 ? 0 : topology[layerNum +1];
        //we have made a new layer, now fill it with neurons
        //add bias neuron in the layer
        for(unsigned neuronNum=0;neuronNum<=topology[layerNum];++neuronNum){
            m_layers.back().push_back(Neuron(numOutputs,neuronNum));
            cout<<"Made a neuron"<<endl; //test
        }
        // force the bias node's output value to 1.0
        m_layers.back().back().setOutputVal(1.0);
    }
}

void Neural_Network::feedForward(vector<double> &inputVals)
{
    assert(inputVals.size()==m_layers[0].size()-1);
    //Assign (latch) the input values into the input neurons
    for (unsigned i=0;i<inputVals.size();++i){
        m_layers[0][i].setOutputVal(inputVals[i]);
    }
    //forward propagate
    for (unsigned layerNum=1;layerNum<m_layers.size();++layerNum){
        Layer &prevLayer=m_layers[layerNum-1];
        for (unsigned n=0; n<m_layers[layerNum].size()-1;++n){
            m_layers[layerNum][n].feedForward(prevLayer);
        }
    }

}
void Neural_Network::backProp(vector<double> &targetVal)
{
    // Calculate overall net error (RMS="root mean square error")
    Layer &outputLayer=m_layers.back();
    m_error=0.0;

    for(unsigned n=0;n<outputLayer.size()-1;++n){
        double delta=targetVal[n]-outputLayer[n].getOutputVal(); //takes the value stored in the output Neuron and finds the difference with the expected output
        m_error+=delta*delta;
    }
    m_error/=outputLayer.size()-1; //get average error squared
    m_error=sqrt(m_error); //RMS

    //Implement a recent average measurement:

    m_recentAverageError =
        (m_recentAverageError*m_recentAverageSmoothingError+m_error)
        /(m_recentAverageSmoothingError + 1.0);
    // Calculate output layer gradients

    for(unsigned n=0; n<outputLayer.size()-1;++n){
        outputLayer[n].calcOutputGradients(targetVal[n]);

    }

    // Calculate gradients on hidden layers

    for(unsigned layerNum=m_layers.size()-2;layerNum>0;--layerNum) { //starts with right-most hidden layer
        Layer &hiddenLayer=m_layers[layerNum];
        Layer &nextLayer=m_layers[layerNum+1];

        for(unsigned n=0;n<hiddenLayer.size();++n){
            hiddenLayer[n].calcHiddenGradients(nextLayer);
        }
    }

    // For all layers from outputs to first hidden layer,
    //update connection weights

    for (unsigned layerNum=m_layers.size()-1;layerNum>0;--layerNum){
        Layer &layer=m_layers[layerNum];
        Layer &prevLayer=m_layers[layerNum-1];

        for (unsigned n=0;n<layer.size()-1;++n){
           layer[n].updateInputWeights(prevLayer);
        }
    }
}
void Neural_Network::getResults(vector<double> &resultVal)
{
    resultVal.clear();  //clears the output layer so the results
                        //of the preceding layer can update it

    for (unsigned n=0;n<m_layers.back().size()-1;++n){
        resultVal.push_back(m_layers.back()[n].getOutputVal());
    }

}


