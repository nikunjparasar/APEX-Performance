import numpy as np

# Define vehicle parameters
mass = 700  # kg
Cd = 0.3  # drag coefficient
A = 2.0  # frontal area (m^2)
rho = 1.2  # air density (kg/m^3)
mu = 1.0  # coefficient of friction
g = 9.81  # gravitational acceleration (m/s^2)
v_max = 30  # maximum velocity (m/s)

# Define simulation parameters
t_step = 0.1  # time step (s)
t_final = 100  # final time (s)

# Define objective function
def objective(x, y):
    # Calculate the length of the racing line
    n_points = len(x)
    dx = np.diff(x)
    dy = np.diff(y)
    d = np.sqrt(dx**2 + dy**2)
    L = np.sum(d)
    
    # Calculate the lap time
    v = np.sqrt(np.diff(x)**2 + np.diff(y)**2) / t_step  # calculate velocity
    v = np.append(v, v[-1])  # add last velocity to match size of x and y
    v = np.minimum(v, v_max)  # limit velocity to maximum value
    F_gravity = mass * g * np.sin(np.arctan2(dy, dx))  # gravitational force
    F_friction = mass * g * mu * np.cos(np.arctan2(dy, dx))  # frictional force
    F_aero = 0.5 * Cd * A * rho * v**2  # aerodynamic drag force
    F_total = F_gravity + F_friction + F_aero  # total force
    a = F_total / mass  # acceleration
    t = np.zeros(n_points)
    for i in range(1, n_points):
        t[i] = t[i-1] + np.sqrt((dx[i-1]/v[i-1])**2 + (dy[i-1]/v[i-1])**2 + (2*a[i-1]*d[i-1])/v[i-1]**2)
    lap_time = t[-1]
    
    return lap_time

# Define function to simulate vehicle dynamics
def simulate_vehicle(x, y):
    # Set initial conditions
    v = np.zeros(len(x))
    x_vehicle = x[0]
    y_vehicle = y[0]
    theta = np.arctan2(y[1]-y[0], x[1]-x[0])
    v[0] = 1  # initial velocity (m/s)

    # Simulate vehicle dynamics
    for i in range(1, len(x)):
        # Calculate distance to next point
        d = np.sqrt((x[i]-x_vehicle)**2 + (y[i]-y_vehicle)**2)

        # Update velocity and position
        v_prev = v[i-1]
        a_gravity = mass * g * np.sin(theta)
        a_friction = mass * g * mu * np.cos(theta)
        a_aero = 0.5 * Cd * A * rho * v_prev**2 / mass
        a = a_gravity + a_friction + a_aero
        v[i] = v_prev + a * t_step
        v[i] = min(v[i], v_max)  # limit velocity to maximum value
        dx = v[i] * np.cos(theta) * t_step
       
