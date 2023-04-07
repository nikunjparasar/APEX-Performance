# Makefile for vehicle and tire model processing

# Compiler options
CXX = g++
CXXFLAGS = -std=c++17 -g -Wall 

# Files
SRC = calculations.cpp
HDRS = VehicleParameters.hpp TireParameters.hpp
OBJ = $(SRC:.cpp=.o)

# Target
TARGET = calculations

# Build rules
$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) $(OBJ) -o $(TARGET)

$(OBJ): $(HDRS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Phony targets
.PHONY: clean

clean:
	rm -f $(OBJ) $(TARGET)
