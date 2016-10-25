#include <iostream>
#include <fstream>
#include <vector>
#include <Neural_Network.h>
#include <Neuron.h>
#include <random>

using namespace std;

class Neuron;
typedef vector<Neuron> Layer;

double randomWeight(void)
{
    return rand() / double(RAND_MAX);
}

int main(){
    vector<unsigned> topology;
    topology.push_back(2);
    topology.push_back(4);
    topology.push_back(1);
    Neural_Network myNet(topology); // created a neuralNet object called myNet
                                    //input layer(2 neurons), 1 hidden layer(4 neurons),
                                    //output layer (1 neuron)

    //produce a training set
    ofstream inputDataCreation;
    inputDataCreation.open("/home/user/Desktop/inputData.txt");
    string a="0,0:0";
    string b="0,1:1";
    string c="1,0:1";
    string d="1,1:0";
    for(int i=0;i<=1000;i++){
        int temp=(int)(randomWeight()*4);
        if((int)temp==0)
            inputDataCreation<<a<<endl;
        if((int)temp==1)
            inputDataCreation<<b<<endl;
        if((int)temp==2)
            inputDataCreation<<c<<endl;
        if((int)temp==3)
            inputDataCreation<<d<<endl;
    }
    inputDataCreation.close();

    //input training set into neural net
    ifstream inputData;
    inputData.open("/home/user/Desktop/inputData.txt");
    if(inputData.is_open()){
        vector<double> inputVals;
        vector<double> targetVal;
        string line, i1, i2,t;
        while(getline(inputData,line)){
        getline(inputData,i1,',');
        getline(inputData,i2,':');
        getline(inputData,t);
        inputVals.push_back(stoi(i1));
        inputVals.push_back(stoi(i2));
        targetVal.push_back(stoi(t));
        myNet.feedForward(inputVals);
        myNet.backProp(targetVal);
        vector<double> resultVal;
        myNet.getResults(resultVal);
        cout<<"Inputs: "<<i1<<", "<<i2<<endl;
        cout<<"Output: "<<resultVal[0]<<endl;
        cout<<"Target: "<<t<<endl;
        cout<<"Net recent average error: "<<myNet.m_recentAverageError<<endl;
        }
    }
    else{
        cout<<"File did not open successfully.";
    }
    inputData.close();
    return 0;
}
