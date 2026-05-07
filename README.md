tongue_3D.npy is the voxel data obtained via CT/MRI imaging of the frog-tongue. Each voxel has a tag associated with air, tissue, arteries, or veins. 

c_dom.npy a reference 3D NumPy matrix that stores the counter number for each tissue voxel in the tongue_3D.npy domain.

arteries_element_database.csv and veins_element_database.csv 
The CSV files consist of the connectivity information of the segmented vessels of arterial and venous tree. The columns represent Segment Index, starting Node , ending Node , Radius, and Length, respectively.

arteries_outlet_coordinate.csv and veins_outlet_coordinate.csv
The CSV files consist of the (x,y,z) coordinates of the arterial and venous outlets. These are voxel coordinate indices and are corrected using the respective voxel dimensions in the code wherever required. The columns are the Outlet node index, x-coordinate, y-coordinate, and z-coordinate, respectively. 


to obtain the pressure map inside the flow domain, run the codes in the following order. Before running, create three new directories named new_method_flow_rcd, constants and nbrhd_matrices in the same directory where you keep the provided .py, .npy, and .csv files.
1. run neighbourhood_matrixUpdated.py. check if the nbrhd_matrices has been poppulated with the generated .npy files.
2. run 02_C_calculatorUpdated.py .  check if the constants has been populated with the generated .npy files.
3. run arterial_compartment_flow_equation_read_write.py, Arterial_Venal_Other_Equations_read_write.py and venal_compartment_flow_equations_read_write.py in any order. check if the new_method_flow_rcd has been populated with the generated .csv files.
4. finally run Flow_load_and_solve_new_method_csv.py. the solution will be saved in new_method_flow_rcd/"your e value" directory.

if you wish to play around with parameters, change them in Flow_solver_parameter_file_loader.py.

The solver may require preconditioner matrix. You can solve the system for a very small e value (preferably 1 or 2 voxel length) and use that solution as the preconditioner. 

use vessels_plot.py for plotting the pressure map
