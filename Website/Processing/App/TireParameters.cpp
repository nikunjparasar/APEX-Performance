#include <iostream>
#include <vector>

using namespace std;

class TireParameters{

    //TODO:   CHANGE ALL OF THE NAMES TO MATCH TIRE PARAMETERS
    /*

                  Tire Parameters
        Symbol          Description                                     Default Value
        ------------------------------------------------------------------
        Fz1      |      Reference load 1                        |       2000 N         
        Fz2      |      Reference load 2                        |       6000 N
        mux1     |      Peak longitudinal frict coef @ load 1   |       1.75
        mux2     |      Peak longitudinal frict coef @ load 2   |       1.40
        kappa1   |      Slip coef for the frict peak @ load 1   |       0.11
        kappa2   |      Slip coef for the frict peak @ load 2   |       0.10
        mu1      |      Peak lat friction coefficient @ load 1  |       1.80
        mu2      |      Peak lat friction coefficient @ load 2  |       1.45
        alpha1   |      Slip angle for the frict peak @ load 1  |       9 deg
        alpha2   |      Slip angle for the frict peak @ load 2  |       8 deg
        Qx       |      Longitudinal shape factor               |       1.9
        Qy       |      Lateral shape factor                    |       1.9    */


               
                const double RLOAD_1_DEFAULT = 2000;
                const double RLOAD_2_DEFAULT = 6000;
                const double PEAK_L1_FRICTION_DEFAULT = 1.75;
                const double PEAK_L2_FRICTION_DEFAULT = 1.4;
                const double SLIP_L1_COEF_DEFAULT = 0.11;
                const double SLIP_L2_COEF_DEFAULT = 0.10;
                const double PEAK_LAT_FRICTION_L1_DEFAULT = 1.8;
                const double PEAK_LAT_FRICTION_L2_DEFAULT = 1.45;
                const double SLIP_ANGLE_L1_DEFAULT = 9;
                const double SLIP_ANGLE_L2_DEFAULT = 8;
                const double LONG_SHAPE_FACTOR_DEFAULT = 1.9;
                const double LAT_SHAPE_FACTOR_DEFAULT = 1.9;

                pair<double, const pair<string, string>> Fz1 = {RLOAD_1_DEFAULT, {"Reference Load 1", "N"}};
                pair<double, const pair<string, string>> Fz2 = {RLOAD_2_DEFAULT, {"Reference Load 2", "N"}};
                pair<double, const pair<string, string>> mux1 = {PEAK_L1_FRICTION_DEFAULT, {"Peak longitudinal frict coef @ load 1", ""}};
                pair<double, const pair<string, string>> mux2 = {PEAK_L2_FRICTION_DEFAULT, {"Peak longitudinal frict coef @ load 2", ""}};
                pair<double, const pair<string, string>> kappa1 = {SLIP_L1_COEF_DEFAULT, {"Slip coef for the frict peak @ load 1 ", ""}};
                pair<double, const pair<string, string>> kappa2 = {SLIP_L2_COEF_DEFAULT, {"Slip coef for the frict peak @ load 2", ""}};
                pair<double, const pair<string, string>> mu1 = {PEAK_LAT_FRICTION_L1_DEFAULT, {"Peak lat friction coefficient @ load 1", ""}};
                pair<double, const pair<string, string>> mu2 = {PEAK_LAT_FRICTION_L2_DEFAULT, {"Peak lat friction coefficient @ load 2 ", ""}};
                pair<double, const pair<string, string>> alpha1 = {SLIP_ANGLE_L1_DEFAULT, {"Slip angle for the frict peak @ load 1", "deg"}};
                pair<double, const pair<string, string>> alpha2 = {SLIP_ANGLE_L2_DEFAULT, {"Slip angle for the frict peak @ load 2","deg"}};
                pair<double, const pair<string, string>> Qx = {LONG_SHAPE_FACTOR_DEFAULT, {"Longitudinal shape factor", ""}};
                pair<double, const pair<string, string>> Qy = {LAT_SHAPE_FACTOR_DEFAULT, {"Lateral shape factor" , ""}}; 
                
                vector<pair<double, const pair<string, string>>> vectorized_parameters = {Fz1, Fz2, mux1, mux2, kappa1, kappa2, mu1, mu2, 
                                                                                                alpha1, alpha2, Qx, Qy};
        public:

                string vectorized_to_string(){
                        string result = "";
                        for(auto it = vectorized_parameters.begin(); it != vectorized_parameters.end(); it++){
                                result.append(it->second.first);
                                result.append(": ");
                                result.append(to_string(it->first));
                                result.append(" ");
                                result.append(it->second.second);
                                result.append("\n");
                        }
                        return result;
                }
                void restoreDefaults(){
                       Fz1.first = RLOAD_1_DEFAULT;
                       Fz2.first = RLOAD_2_DEFAULT;
                       mux1.first = PEAK_L1_FRICTION_DEFAULT;
                       mux2.first = PEAK_L2_FRICTION_DEFAULT;
                       kappa1.first = SLIP_L1_COEF_DEFAULT;
                       kappa2.first = SLIP_L2_COEF_DEFAULT;
                       mu1.first = PEAK_LAT_FRICTION_L1_DEFAULT;
                       mu2.first = PEAK_LAT_FRICTION_L2_DEFAULT;
                       alpha1.first = SLIP_ANGLE_L1_DEFAULT;
                       alpha2.first = SLIP_ANGLE_L2_DEFAULT;
                       Qx.first = LONG_SHAPE_FACTOR_DEFAULT;
                       Qy.first = LAT_SHAPE_FACTOR_DEFAULT;
                }
                
                double getReferenceLoad1(){return Fz1.first;}
                double getReferenceLoad2(){return Fz2.first;}
                double getPeakLongFriction1(){return mux1.first;}
                double getPeakLongFriction2(){return mux2.first;}
                double getSlipCoef1(){return kappa1.first;}
                double getSlipCoef2(){return kappa2.first;}
                double getPeakLatFriction1(){return mu1.first;}
                double getPeakLatFriction2(){return mu2.first;}
                double getSlipAngle1(){return alpha1.first;}
                double getSlipAngle2(){return alpha2.first;}
                double getLongShapeFactor(){return Qx.first;}
                double getLatShapeFactor(){return Qy.first;}
              
                void setReferenceLoad1(double value){Fz1.first = value;}
                void setReferenceLoad2(double value){Fz2.first = value;}
                void setPeakLongFriction1(double value){mux1.first = value;}
                void setPeakLongFriction2(double value){mux2.first = value;}
                void setSlipCoef1(double value){kappa1.first = value;}
                void setSlipCoef2(double value){kappa2.first = value;}
                void setPeakLatFriction1(double value){mu1.first = value;}
                void setPeakLatFriction2(double value){mu2.first = value;}
                void setSlipAngle1(double value){alpha1.first = value;}
                void setSlipAngle2(double value){alpha2.first = value;}
                void setLongShapeFactor(double value){Qx.first = value;}
                void setLatShapeFactor(double value){Qy.first = value;}


};


