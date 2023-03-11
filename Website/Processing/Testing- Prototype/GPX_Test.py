import gpxpy
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Parse the GPX file
gpx_file = open("/Users/nikkparasar/Documents/Personal Projects/apexperformance/Processing/Silverstone_The_British_F1_GP_Circuit.gpx", "r")
gpx = gpxpy.parse(gpx_file)

# Extract the latitude, longitude, and elevation data
latitudes = []
longitudes = []
elevations = []
for track in gpx.tracks:
    for segment in track.segments:
        for point in segment.points:
            latitudes.append(point.latitude)
            longitudes.append(point.longitude)
            elevations.append(point.elevation)

# Convert the latitude, longitude, and elevation data to Cartesian coordinates
x = []
y = []
z = []
for lat, lon, elev in zip(latitudes, longitudes, elevations):
    r = elev + 6371000
    x.append(r * math.cos(math.radians(lat)) * math.cos(math.radians(lon)))
    y.append(r * math.cos(math.radians(lat)) * math.sin(math.radians(lon)))
    z.append(r * math.sin(math.radians(lat)))

# Normalize the elevation values to be between 0 and 1
elevations_norm = (np.array(elevations) - min(elevations)) / (max(elevations) - min(elevations))

# Create a colormap for the elevations
cmap = plt.get_cmap("plasma")

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Plot the racetrack surface
surf = ax.plot_trisurf(x, y, z, cmap=cmap, linewidth=3)

# Set the limits of the axes
xmin, xmax = min(x), max(x)
ymin, ymax = min(y), max(y)
zmin, zmax = min(z), max(z)
ax.set_xlim3d(xmin, xmax)
ax.set_ylim3d(ymin, ymax)
ax.set_zlim3d(zmin, zmax)

# Set the labels of the axes
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

# Add a colorbar
fig.colorbar(surf)

# Show the plot
plt.show()




