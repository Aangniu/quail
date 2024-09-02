// Gmsh 2.2 Geometry File for Unstructured 2D Triangular Mesh with Meshed Embedded Line

// Set mesh size parameter (controls general mesh density)
lc = 0.5; // Typical element size

// Parameters for the rectangular box
Lx = 10; // Length of the rectangle in the x-direction
Ly = 5;  // Height of the rectangle in the y-direction

// Define the points for the rectangle
Point(1) = {0, 0, 0, lc};   // Bottom-left corner
Point(2) = {Lx, 0, 0, lc};  // Bottom-right corner
Point(3) = {Lx, Ly, 0, lc}; // Top-right corner
Point(4) = {0, Ly, 0, lc};  // Top-left corner

// Define the points for the horizontal line
y_line = 2.5; // y-coordinate for the horizontal line
x_edge = 0.5;
Point(5) = {x_edge, y_line, 0, lc};  // Left end of the horizontal line
Point(6) = {Lx-x_edge, y_line, 0, lc}; // Right end of the horizontal line

// Define lines for the rectangle
Line(1) = {1, 2}; // Bottom edge
Line(2) = {2, 3}; // Right edge
Line(3) = {3, 4}; // Top edge
Line(4) = {4, 1}; // Left edge

// Define the horizontal line as a line segment
Line(5) = {5, 6}; // Horizontal embedded line

// Define line loops and surface
Line Loop(1) = {1, 2, 3, 4}; // Outer rectangle loop
Plane Surface(1) = {1};      // Rectangle surface

// Embed the line in the surface to ensure meshing
Curve{5} In Surface{1};

// Define physical surface
Physical Surface("SimulationDomain") = {1};

// Define physical groups for boundaries
Physical Line("y1") = {1};
Physical Line("x2") = {2};
Physical Line("y2") = {3};
Physical Line("x1") = {4};

// Define physical group for the horizontal line to ensure it is meshed
Physical Line("FaultInterface") = {5};

// Mesh the surface with 2D elements
Mesh 2;
Mesh.MshFileVersion = 2.2;