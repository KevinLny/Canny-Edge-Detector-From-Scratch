# Compiler settings
CXX = g++
CXXFLAGS = -Wall -O3 $(shell pkg-config --cflags opencv4)
LDFLAGS = $(shell pkg-config --libs opencv4)

# Target executable name
TARGET = canny_proj

# Source files
SRC = src/main.cpp

# Default build rule
all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(SRC) -o $(TARGET) $(CXXFLAGS) $(LDFLAGS)

# Clean rule to remove the executable
clean:
	rm -f $(TARGET)

.PHONY: all clean