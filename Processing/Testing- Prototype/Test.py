import numpy as np
import matplotlib.pyplot as plt

# Load track data from CSV file
data = np.genfromtxt('/Users/nikkparasar/Documents/Personal Projects/apexperformance/Processing/silverstone.csv', delimiter=',')

# Extract data columns
x_m = data[:, 0]
y_m = data[:, 1]
w_tr_right_m = data[:, 2]
w_tr_left_m = data[:, 3]

# Calculate points that are the width away from the center line
theta = np.arctan2(np.gradient(y_m), np.gradient(x_m))
x_tr_right_m = x_m + w_tr_right_m * np.sin(theta)
y_tr_right_m = y_m - w_tr_right_m * np.cos(theta)
x_tr_left_m = x_m - w_tr_left_m * np.sin(theta)
y_tr_left_m = y_m + w_tr_left_m * np.cos(theta)

# Plot center line and track boundaries
# plt.plot(x_m, y_m, 'k--', label='Center line')
plt.plot(x_tr_right_m, y_tr_right_m, 'gray', label='Right track boundary')
plt.plot(x_tr_left_m, y_tr_left_m, 'gray', label='Left track boundary')

# Add labels and legend
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.legend()

# Show plot
plt.show()