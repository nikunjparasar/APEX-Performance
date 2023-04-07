#ifndef VEHICLEPARAMETERS_HPP
#define VEHICLEPARAMETERS_HPP

#include <iostream>
#include <vector>

using namespace std;

class VehicleParameters {
        private:

                static constexpr double POWER_MAX_DEFAULT = 560;
                static constexpr double MASS_DEFAULT = 660;
                static constexpr double INERTIA_X_DEFAULT = 112.5;
                static constexpr double INERTIA_Y_DEFAULT = 450;
                static constexpr double INERTIA_Z_DEFAULT = 450;
                static constexpr double WHEELBASE_DEFAULT = 3.4;
                static constexpr double COM_FRONT_AXLE_DEFAULT = 1.8;
                static constexpr double COM_REAR_AXLE_DEFAULT = WHEELBASE_DEFAULT - COM_FRONT_AXLE_DEFAULT;
                static constexpr double COM_HEIGHT_DEFAULT = 0.3;
                static constexpr double FRONTAL_AREA_DEFAULT = 1.5;
                static constexpr double ROLL_MOMENT_DEFAULT = 0.5;
                static constexpr double FW_TO_CENTER_DEFAULT = 0.73;
                static constexpr double RW_TO_CENTER_DEFAULT = 0.73;
                static constexpr double WHEEL_RADIUS_DEFAULT = 0.33;
                static constexpr double DIFFERENTIAL_FRICTION_DEFAULT = 10.47;

                pair<double, const pair<string, string>> P_max = {POWER_MAX_DEFAULT, {"Peak Engine Power", "kW"}};
                pair<double, const pair<string, string>> M = {MASS_DEFAULT, {"Vehicle Mass", "kg"}};
                pair<double, const pair<string, string>> I_x = {INERTIA_X_DEFAULT, {"Moment of Inertia about the x-axis", "kg/m^2"}};
                pair<double, const pair<string, string>> I_y = {INERTIA_Y_DEFAULT, {"Moment of Inertia about the y-axis", "kg/m^2"}};
                pair<double, const pair<string, string>> I_z = {INERTIA_Z_DEFAULT, {"Moment of Inertai about the z-axis", "kg/m^2"}};
                pair<double, const pair<string, string>> W = {WHEELBASE_DEFAULT, {"Wheelbase", "m"}};
                pair<double, const pair<string, string>> A = {COM_FRONT_AXLE_DEFAULT, {"Distance of mass center from rear axle", "m"}};
                pair<double, const pair<string, string>> B = {W.first-A.first, {"Distance of mass center from rear axle", "m"}};
                pair<double, const pair<string, string>> H = {COM_HEIGHT_DEFAULT, {"Center of mass height", "m"}};
                pair<double, const pair<string, string>> Ar = {FRONTAL_AREA_DEFAULT, {"Frontal Area","m^2"}};
                pair<double, const pair<string, string>> D_roll = {ROLL_MOMENT_DEFAULT, {"Roll moment distribution (fraction @ front axle)", ""}};
                pair<double, const pair<string, string>> w_f = {FW_TO_CENTER_DEFAULT, {"Front Wheel to car centerline distance" , "m"}}; 
                pair<double, const pair<string, string>> w_r = {RW_TO_CENTER_DEFAULT, {"Rear Wheel to car centerline distance" , "m"}};;
                pair<double, const pair<string, string>> R = {WHEEL_RADIUS_DEFAULT, {"Wheel Radius", "m"}};
                pair<double, const pair<string, string>> k_d = {DIFFERENTIAL_FRICTION_DEFAULT, {"Differential Friction coefficient", "Nm s/rad"}};

                vector<pair<double, const pair<string, string>>> vectorized_parameters = {P_max, M, I_x, I_y, I_y, I_z, W,
                                                                                        W, A, B, H, Ar, D_roll, w_f, w_r, R, k_d};
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
                        P_max.first = POWER_MAX_DEFAULT;
                        M .first= MASS_DEFAULT;
                        I_x.first = INERTIA_X_DEFAULT;
                        I_y.first = INERTIA_Y_DEFAULT;
                        I_z.first = INERTIA_Z_DEFAULT;
                        W.first = WHEELBASE_DEFAULT;
                        A.first = COM_FRONT_AXLE_DEFAULT;
                        B.first = COM_REAR_AXLE_DEFAULT;
                        H.first = COM_HEIGHT_DEFAULT;
                        Ar.first = FRONTAL_AREA_DEFAULT;
                        D_roll.first = ROLL_MOMENT_DEFAULT;
                        w_f.first = FW_TO_CENTER_DEFAULT;
                        w_r.first = RW_TO_CENTER_DEFAULT;
                        R.first = WHEEL_RADIUS_DEFAULT;
                        k_d.first = DIFFERENTIAL_FRICTION_DEFAULT;
                }
                
                double getPeakPower(){return P_max.first;}
                double getMass(){return M.first;}
                double getXInertia(){return I_x.first;}
                double getYInertia(){return I_y.first;}
                double getZInertia(){return I_z.first;}
                double getWheelbase(){return W.first;}
                double getCOMFrontAxle(){return A.first;}
                double getCOMRearAxle(){return B.first;}
                double getCOMheight(){return H.first;}
                double getFrontalArea(){return Ar.first;}
                double getRollDistribution(){return D_roll.first;}
                double getFWtoCenter(){return w_f.first;}
                double getRWtoCenter(){return w_r.first;}
                double getWheelRadius(){return R.first;}
                double getDiffFriction(){return k_d.first;}

                void setPeakPower(double value){P_max.first = value;}
                void setMass(double value){M.first = value;}
                void setXInertia(double value){I_x.first = value;}
                void setYInertia(double value){I_y.first = value;}
                void setZInertia(double value){I_z.first = value;}
                void setWheelbase(double value){W.first = value;}
                void setCOMFrontAxle(double value){A.first = value;}
                void setCOMRearAxle(double value){B.first = value;}
                void setCOMheight(double value){H.first = value;}
                void setFrontalArea(double value){Ar.first = value;}
                void setRollDistribution(double value){D_roll.first = value;}
                void setFWtoCenter(double value){w_f.first = value;}
                void setRWtoCenter(double value){w_r.first = value;}
                void setWheelRadius(double value){R.first = value;}
                void setDiffFriction(double value){k_d.first = value;}


};

#endif