// Gmsh 2.2 Geometry File for Unstructured 2D Triangular Mesh with Meshed Embedded Line

// Set mesh size parameter (controls general mesh density)
lc = 0.5; // Typical element size

// Parameters for the rectangular box
Lx = 10; // Length of the rectangle in the x-direction
Ly = 10;  // Height of the rectangle in the y-direction

// Define the points for the rectangle
Point(1) = {-Lx/2, -Ly/2, 0, lc};   // Bottom-left corner
Point(2) = {Lx/2, -Ly/2, 0, lc};  // Bottom-right corner
Point(3) = {Lx/2, Ly/2, 0, lc}; // Top-right corner
Point(4) = {-Lx/2, Ly/2, 0, lc};  // Top-left corner

// Define lines for the rectangle
Line(1) = {1, 2}; // Bottom edge
Line(2) = {2, 3}; // Right edge
Line(3) = {3, 4}; // Top edge
Line(4) = {4, 1}; // Left edge

// Define line loops and surface
Line Loop(1) = {1, 2, 3, 4}; // Outer rectangle loop
Plane Surface(1) = {1};      // Rectangle surface

// Define physical surface
Physical Surface("MeshInterior") = {1};

// Define physical groups for boundaries
Physical Line("y1") = {1};
Physical Line("x2") = {2};
Physical Line("y2") = {3};
Physical Line("x1") = {4};

// Coarsening from this point:
Point(10) = {-0.1, -0.1, 0, lc};
// Define a distance field for mesh coarsening from a specific point
Field[1] = Distance;
Field[1].NodesList = {10};  // Use Point 1 (-0.1, -0.1) to control distance, or specify another point

// Threshold distance for mesh refinement
Field[2] = Threshold;
Field[2].IField = 1;
Field[2].LcMin = 0.05; // Minimum characteristic length (finer mesh near the point)
Field[2].LcMax = lc; // Maximum characteristic length (coarser mesh farther away)
Field[2].DistMin = 0.0;  // Distance at which LcMin is applied
Field[2].DistMax = 5.0;  // Distance at which LcMax is applied

// Use this field to set the mesh size
Background Field = 2;


// Mesh the surface with 2D elements
Mesh 2;
Mesh.MshFileVersion = 2.2;