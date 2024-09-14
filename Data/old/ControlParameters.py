class VehicleParameters:
    POWER_MAX_DEFAULT = 560
    MASS_DEFAULT = 660
    INERTIA_X_DEFAULT = 112.5
    INERTIA_Y_DEFAULT = 450
    INERTIA_Z_DEFAULT = 450
    WHEELBASE_DEFAULT = 3.4
    COM_FRONT_AXLE_DEFAULT = 1.8
    COM_REAR_AXLE_DEFAULT = WHEELBASE_DEFAULT - COM_FRONT_AXLE_DEFAULT
    COM_HEIGHT_DEFAULT = 0.3
    FRONTAL_AREA_DEFAULT = 1.5
    ROLL_MOMENT_DEFAULT = 0.5
    FW_TO_CENTER_DEFAULT = 0.73
    RW_TO_CENTER_DEFAULT = 0.73
    WHEEL_RADIUS_DEFAULT = 0.33
    DIFFERENTIAL_FRICTION_DEFAULT = 10.47

    def __init__(self):
        self.parameters = {
            "Peak Engine Power": [self.POWER_MAX_DEFAULT, "kW"],
            "Vehicle Mass": [self.MASS_DEFAULT, "kg"],
            "Moment of Inertia about the x-axis": [self.INERTIA_X_DEFAULT, "kg/m^2"],
            "Moment of Inertia about the y-axis": [self.INERTIA_Y_DEFAULT, "kg/m^2"],
            "Moment of Inertia about the z-axis": [self.INERTIA_Z_DEFAULT, "kg/m^2"],
            "Wheelbase": [self.WHEELBASE_DEFAULT, "m"],
            "Distance of mass center from rear axle": [self.COM_FRONT_AXLE_DEFAULT, "m"],
            "Center of mass height": [self.COM_HEIGHT_DEFAULT, "m"],
            "Frontal Area": [self.FRONTAL_AREA_DEFAULT, "m^2"],
            "Roll moment distribution (fraction @ front axle)": [self.ROLL_MOMENT_DEFAULT, ""],
            "Front Wheel to car centerline distance": [self.FW_TO_CENTER_DEFAULT, "m"],
            "Rear Wheel to car centerline distance": [self.RW_TO_CENTER_DEFAULT, "m"],
            "Wheel Radius": [self.WHEEL_RADIUS_DEFAULT, "m"],
            "Differential Friction coefficient": [self.DIFFERENTIAL_FRICTION_DEFAULT, "Nm s/rad"],
        }

    def vectorized_to_string(self):
        lines = ["+---------------------------------------------------------------------------------------------------+",
                 "|                                       VEHICLE PARAMETERS                                          |",
                 "+---------------------------------------------------------------------------------------------------+",
                 "|                     Parameter Name                      |          Value            |     Units   |",
                 "+---------------------------------------------------------+---------------------------+-------------+"]
        for name, (value, unit) in self.parameters.items():
            lines.append(f"| {name:<55} | {value:>25} | {unit:<11} |")
        lines.append("+---------------------------------------------------------+---------------------------+-------------+")
        return '\n'.join(lines)

    def restore_defaults(self):
        for name in self.parameters:
            self.parameters[name][0] = getattr(self, f"{name.upper().replace(' ', '_').replace('@', '').replace('-', '_')}_DEFAULT")


    def getPeakPower(self):
        return self.parameters["Peak Engine Power"][0]
    def getMass(self):
        return self.parameters["Vehicle Mass"][0]   
    def getInertiaX(self):
        return self.parameters["Moment of Inertia about the x-axis"][0]
    def getInertiaY(self):
        return self.parameters["Moment of Inertia about the y-axis"][0]
    def getInertiaZ(self):
        return self.parameters["Moment of Inertia about the z-axis"][0]
    def getWheelbase(self):
        return self.parameters["Wheelbase"][0]
    def getCOMFrontAxle(self):
        return self.parameters["Distance of mass center from rear axle"][0]
    def getCOMHeight(self):
        return self.parameters["Center of mass height"][0]
    def getFrontalArea(self):
        return self.parameters["Frontal Area"][0]
    def getRollMoment(self):
        return self.parameters["Roll moment distribution (fraction @ front axle)"][0]
    def getFWToCenter(self):
        return self.parameters["Front Wheel to car centerline distance"][0]
    def getRWToCenter(self):
        return self.parameters["Rear Wheel to car centerline distance"][0]
    def getWheelRadius(self):
        return self.parameters["Wheel Radius"][0]
    def getDifferentialFriction(self):
        return self.parameters["Differential Friction coefficient"][0]
    
    def setPeakPower(self, value):
        self.parameters["Peak Engine Power"][0] = value
    def setMass(self, value):
        self.parameters["Vehicle Mass"][0] = value
    def setInertiaX(self, value):
        self.parameters["Moment of Inertia about the x-axis"][0] = value
    def setInertiaY(self, value):
        self.parameters["Moment of Inertia about the y-axis"][0] = value
    def setInertiaZ(self, value):
        self.parameters["Moment of Inertia about the z-axis"][0] = value
    def setWheelbase(self, value):
        self.parameters["Wheelbase"][0] = value
    def setCOMFrontAxle(self, value):
        self.parameters["Distance of mass center from rear axle"][0] = value
    def setCOMHeight(self, value):
        self.parameters["Center of mass height"][0] = value
    def setFrontalArea(self, value):
        self.parameters["Frontal Area"][0] = value
    def setRollMoment(self, value):
        self.parameters["Roll moment distribution (fraction @ front axle)"][0] = value
    def setFWToCenter(self, value):
        self.parameters["Front Wheel to car centerline distance"][0] = value
    def setRWToCenter(self, value):
        self.parameters["Rear Wheel to car centerline distance"][0] = value
    def setWheelRadius(self, value):
        self.parameters["Wheel Radius"][0] = value
    def setDifferentialFriction(self, value):
        self.parameters["Differential Friction coefficient"][0] = value
        
        
        
        
        
        
        
class TireParameters:
    RLOAD_1_DEFAULT = 2000
    RLOAD_2_DEFAULT = 6000
    PEAK_L1_FRICTION_DEFAULT = 1.75
    PEAK_L2_FRICTION_DEFAULT = 1.4
    SLIP_L1_COEF_DEFAULT = 0.11
    SLIP_L2_COEF_DEFAULT = 0.10
    PEAK_LAT_FRICTION_L1_DEFAULT = 1.8
    PEAK_LAT_FRICTION_L2_DEFAULT = 1.45
    SLIP_ANGLE_L1_DEFAULT = 9
    SLIP_ANGLE_L2_DEFAULT = 8
    LONG_SHAPE_FACTOR_DEFAULT = 1.9
    LAT_SHAPE_FACTOR_DEFAULT = 1.9

    def __init__(self):
        self.parameters = {
            "Reference Load 1": [self.RLOAD_1_DEFAULT, "N"],
            "Reference Load 2": [self.RLOAD_2_DEFAULT, "N"],
            "Peak longitudinal friction coefficient @ load 1": [self.PEAK_L1_FRICTION_DEFAULT, ""],
            "Peak longitudinal friction coefficient @ load 2": [self.PEAK_L2_FRICTION_DEFAULT, ""],
            "Slip coefficient for the friction peak @ load 1": [self.SLIP_L1_COEF_DEFAULT, ""],
            "Slip coefficient for the friction peak @ load 2": [self.SLIP_L2_COEF_DEFAULT, ""],
            "Peak lateral friction coefficient @ load 1": [self.PEAK_LAT_FRICTION_L1_DEFAULT, ""],
            "Peak lateral friction coefficient @ load 2": [self.PEAK_LAT_FRICTION_L2_DEFAULT, ""],
            "Slip angle for the friction peak @ load 1": [self.SLIP_ANGLE_L1_DEFAULT, "deg"],
            "Slip angle for the friction peak @ load 2": [self.SLIP_ANGLE_L2_DEFAULT, "deg"],
            "Longitudinal shape factor": [self.LONG_SHAPE_FACTOR_DEFAULT, ""],
            "Lateral shape factor": [self.LAT_SHAPE_FACTOR_DEFAULT, ""],
        }

    def vectorized_to_string(self):
        lines = ["+---------------------------------------------------------------------------------------------------+",
                 "|                                       TIRE PARAMETERS                                             |",
                 "+---------------------------------------------------------------------------------------------------+",
                 "|                     Parameter Name                      |          Value            |     Units   |",
                 "+---------------------------------------------------------+---------------------------+-------------+"]
        for name, (value, unit) in self.parameters.items():
            lines.append(f"| {name:<55} | {value:>25} | {unit:<11} |")
        lines.append("+---------------------------------------------------------+---------------------------+-------------+")
        return '\n'.join(lines)

    def restore_defaults(self):
        for name in self.parameters:
            default_attr_name = name.upper().replace(' ', '_').replace('@', '').replace('-', '_') + '_DEFAULT'
            self.parameters[name][0] = getattr(self, default_attr_name)

    def getReferenceLoad1(self):
        return self.parameters["Reference Load 1"][0]
    def getReferenceLoad2(self):
        return self.parameters["Reference Load 2"][0]
    def getPeakLongitudinalFriction1(self):
        return self.parameters["Peak longitudinal friction coefficient @ load 1"][0]
    def getPeakLongitudinalFriction2(self):
        return self.parameters["Peak longitudinal friction coefficient @ load 2"][0]
    def getSlipCoefficient1(self):
        return self.parameters["Slip coefficient for the friction peak @ load 1"][0]
    def getSlipCoefficient2(self):
        return self.parameters["Slip coefficient for the friction peak @ load 2"][0]
    def getPeakLateralFriction1(self):
        return self.parameters["Peak lateral friction coefficient @ load 1"][0]
    def getPeakLateralFriction2(self):
        return self.parameters["Peak lateral friction coefficient @ load 2"][0]
    def getSlipAngle1(self):
        return self.parameters["Slip angle for the friction peak @ load 1"][0]
    def getSlipAngle2(self):
        return self.parameters["Slip angle for the friction peak @ load 2"][0]
    def getLongitudinalShapeFactor(self):
        return self.parameters["Longitudinal shape factor"][0]
    def getLateralShapeFactor(self):
        return self.parameters["Lateral shape factor"][0]
    
    def setReferenceLoad1(self, value):
        self.parameters["Reference Load 1"][0] = value
    def setReferenceLoad2(self, value):
        self.parameters["Reference Load 2"][0] = value
    def setPeakLongitudinalFriction1(self, value):
        self.parameters["Peak longitudinal friction coefficient @ load 1"][0] = value
    def setPeakLongitudinalFriction2(self, value):
        self.parameters["Peak longitudinal friction coefficient @ load 2"][0] = value
    def setSlipCoefficient1(self, value):
        self.parameters["Slip coefficient for the friction peak @ load 1"][0] = value
    def setSlipCoefficient2(self, value):
        self.parameters["Slip coefficient for the friction peak @ load 2"][0] = value
    def setPeakLateralFriction1(self, value):
        self.parameters["Peak lateral friction coefficient @ load 1"][0] = value
    def setPeakLateralFriction2(self, value):
        self.parameters["Peak lateral friction coefficient @ load 2"][0] = value
    def setSlipAngle1(self, value):
        self.parameters["Slip angle for the friction peak @ load 1"][0] = value
    def setSlipAngle2(self, value):
        self.parameters["Slip angle for the friction peak @ load 2"][0] = value
    def setLongitudinalShapeFactor(self, value):
        self.parameters["Longitudinal shape factor"][0] = value
    def setLateralShapeFactor(self, value):
        self.parameters["Lateral shape factor"][0] = value
        
        