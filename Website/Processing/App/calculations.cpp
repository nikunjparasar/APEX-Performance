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



    return 0;
}

    