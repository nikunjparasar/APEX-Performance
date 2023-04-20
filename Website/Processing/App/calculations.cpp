#include <iostream>

#include "VehicleParameters.hpp"
#include "TireParameters.hpp"
using namespace std;

int main(int argc, char* argv[]) {
    
    VehicleParameters v;
    TireParameters t;

    

    cout << v.vectorized_to_string();
    cout << endl;
    cout << t.vectorized_to_string();
    
    cout << "THEORETICAL LAPTIME ON TRACK: ";

    return 0;
}

double calculate_time(VehicleParameters v, TireParameters t, ){
    v.getPeakPower();
    v.getMass();
    v.getXInertia();
    v.getYInertia();
    v.getZInertia();
    v.getWheelbase();
    v.getCOMFrontAxle();
    v.getCOMRearAxle();
    v.getCOMheight();
    v.getFrontalArea();
    v.getRollDistribution();
    v.getFWtoCenter();
    v.getRWtoCenter();
    v.getWheelRadius();
    v.getDiffFriction();   
    return 0.0;
}

    