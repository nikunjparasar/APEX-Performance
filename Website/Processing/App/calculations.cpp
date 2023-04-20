#include <iostream>
#include <fstream>

#include "VehicleParameters.hpp"
#include "TireParameters.hpp"
using namespace std;

int main(int argc, char* argv[]) {
    
    VehicleParameters v;
    TireParameters t;

    ofstream outfile("../public/output.txt", ios::trunc);

    outfile << v.vectorized_to_string() << endl;
    outfile << t.vectorized_to_string();

    // Close the output file
    outfile.close();

    return 0;
}


    