import numpy as np
import scipy.interpolate as interp
import matplotlib as mpl
import matplotlib.pyplot as plt
from discretize.utils import mkvc, refine_tree_xyz, active_from_xyz
from SimPEG.utils import plot2Ddata

def pf_rs(
        pf_data, 
        del_new,
        bounds = None
        ):
    """
    Resamples gravity and magnetic data (.xyz) to a new sampling interval.
    
    Parameters
    ----------
    pf_data : np.ndarray
        Gravity or magnetic data with columns for X, Y, Z, and M-Component Data.
    del_new : float
        New sampling interval in meters.
    
    Returns
    ----------
    pf_final : np.ndarray
        Resampled gravity data with columns for X, Y, Z, and M-Component Data.
    nx_new : int
        New number of cells in the x dimension.
    ny_new : int
        New number of cells in the y dimension.
    """
    
    # Extract Grid information
    x_loc = pf_data[:,0]
    y_loc = pf_data[:,1]
    topo = pf_data[:,2]
    grav = pf_data[:,3:]

    # New Grid Parameters (These new additions allow for non-sorted arrays)
    if bounds == None:
        max_x=np.max(x_loc)
        min_x=np.min(x_loc)
        max_y=np.max(y_loc)
        min_y=np.min(y_loc)
    else:
        max_x = bounds[1]
        min_x = bounds[0]
        max_y = bounds[3]
        min_y = bounds[2]
    
    
    nx_new = int((max_x-min_x)/del_new) + 1 # New number of cells in x
    ny_new = int((max_y-min_y)/del_new) + 1 # New number of cells in y

    x_new = np.linspace(min_x, max_x, nx_new)
    y_new = np.linspace(min_y, max_y, ny_new)
    
    # Defining Mesh Grid
    x_new_grid, y_new_grid = np.meshgrid(x_new, y_new)
    
    # Defining topography interpolator
    interp_topo = interp.LinearNDInterpolator((x_loc, y_loc), topo)
    topo_new = interp_topo(x_new_grid, y_new_grid)

    # Defining gravity interpolators
    for i in range(np.shape(grav)[1]):
        interp_pf = interp.LinearNDInterpolator((x_loc, y_loc), grav[:,i])
        pf_new = interp_pf(x_new_grid, y_new_grid)
        if i == 0:
            pf_new_out = mkvc(pf_new)
        else:
            pf_new_out = np.c_[pf_new_out, mkvc(pf_new)]
    
    # Vectorizing the new grid
    x_new_out, y_new_out, z_new_out = mkvc(x_new_grid), mkvc(y_new_grid), mkvc(topo_new)
    pf_final = np.c_[x_new_out, y_new_out, z_new_out, pf_new_out]
    
    return [pf_final, nx_new, ny_new]


def topo(
        model, 
        nx, 
        ny, 
        x, 
        y, 
        z, 
        bound
        ):
    """
    Creates a topography mesh to separate active cells from the block model.
    
    Parameters
    ----------
    model : np.ndarray
        Block Model.
    nx, ny : floats
        Number of cells in x and y dimension.
    x, y, z : np.ndarray (Can be empty arrays)
        Known Topography values
    bound : np.ndarray (1x4)
        Values of topography at corners
        Format - bound = [x[0]y[0], x[0]y[-1], x[-1]y[0], x[-1]y[-1]]
    
    Returns
    ----------
    topo_xyz : np.ndarray (Nx3)
        Topography mesh (x, y, z) triplets in meters.
    x_topo, y_topo, z_topo : np.ndarray (Nx1)
        Topography values in x, y and z direction over the mesh.
    """
    
    # Calculate the boundary values in x, y and z directions
    x_bound = [model[0,0], model[0,0], model[-1,0], model[-1,0]]
    y_bound = [model[0,1], model[-1,1], model[0,1], model[-1,1]]
    z_bound = bound
    
    # Append the boundary values to the topography arrays
    x = np.append(x, x_bound)
    y = np.append(y, y_bound)
    z = np.append(z, z_bound)
    
    # Define an interpolator using CloughTocher2DInterpolator
    interpolator = interp.CloughTocher2DInterpolator(np.array([x,y]).T, z)
    
    # Define a mesh grid for the topography
    x_topo, y_topo = np.meshgrid(
        np.linspace(model[0,0], model[-1,0], nx), np.linspace(model[0,1], model[-1,1], ny)
        )
    z_topo = interpolator(x_topo, y_topo)
    x_topo, y_topo, z_topo = mkvc(x_topo), mkvc(y_topo), mkvc(z_topo)
    topo_xyz = np.c_[x_topo, y_topo, z_topo]
    
    return [topo_xyz, x_topo, y_topo, z_topo]

def plot_model_slice(
        mesh,
        ind_active, 
        model, 
        normal, 
        ind_plot_arr, 
        clim, 
        set=0, 
        sec_loc=True, 
        which_prop='Den', 
        gdlines=True, 
        cmap='bwr',
        save_plt=True, 
        path_to_output='.',
        name='Model'
        ):
    """
    Plots a slice of a given model in the specified direction and location.

    Parameters
    ----------
    mesh : SimPEG TensorMesh
        The mesh object.
    ind_active : numpy.ndarray
        Boolean array indicating which cells in the mesh are active.
    model : numpy.ndarray
        The model to be plotted.
    normal : str
        The direction of the slice. Must be one of 'X', 'Y', or 'Z'.
    ind_plot : int
        The index of the slice along the specified direction.
    which_prop : str
        Chooses between Density and Susceptibility to change title & colorbar. Must be one of 'Den' (default) or 'Sus'.
    cmap : str, optional
        The name of the colormap to use. Default is 'bwr'.
    save_plt : Bool
        Saves the plot as a pdf. Defaults to True.

    Returns
    -------
    None

    """
    
    # Create a subplot
    fig = plt.figure(figsize=(6, 6))
    ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    # Get the appropriate location of the slice

    loc_plot_x = mesh.cell_centers_x[ind_plot_arr[0]]
    loc_plot_y = mesh.cell_centers_y[ind_plot_arr[1]]
    loc_plot_z = mesh.cell_centers_z[ind_plot_arr[2]]

    if normal == 'X':
        loc_plot = loc_plot_x
        ind_plot = ind_plot_arr[0]
        xlabel = r"y (m)"
        ylabel = r"z (m)"
    elif normal == 'Y':
        loc_plot = loc_plot_y
        ind_plot = ind_plot_arr[1]
        xlabel = r"x (m)"
        ylabel = r"z (m)"
    elif normal == 'Z':
        loc_plot = loc_plot_z
        ind_plot = ind_plot_arr[2]
        xlabel = r"x (m)"
        ylabel = r"y (m)"
    else:
        print(f"Error: Invalid direction '{normal}'.")
        return
    
    formatted_loc_plot = "{:.2E}".format(loc_plot)

    # Grid Opts
    if gdlines == False:
        lwidth = 0
    else:
        lwidth = 0.0001

    # Plot the slice
    mesh.plot_slice(
        model,  # Data to plot
        normal=normal,  # Direction of slice
        ax=ax1,  # Subplot to use
        ind=ind_plot,  # Index of slice
        grid=gdlines,  # Plot gridlines
        grid_opts={"edgecolor": 'face', "linewidth":lwidth},  # Gridline options
        clim=(clim[0], clim[1]),  # Colorbar limits
        pcolor_opts={"cmap": cmap},  # Plotting options
    )

    if which_prop == 'Den':
        # title = r"\textbf{Inverted }$\boldsymbol{\Delta\rho}$\textbf{ Model}" + "\n" + r"Slice at {0} = {1} m".format(normal,formatted_loc_plot)
        title = r"Inverted Den Model" + "\n" + r"Slice at {0} = {1} m".format(normal,formatted_loc_plot)
        # cbar_unit = r"$\bf{g.cm^{-3}}$"
        cbar_unit = r"g/cc"
    elif which_prop == 'Sus':
        title = r"\textbf{Inverted }$\boldsymbol{\Delta\chi}$\textbf{ Model}" + "\n" + r"Slice at {0} = {1} m".format(normal,formatted_loc_plot)
        cbar_unit = r"$\bf{SI}$"
    elif which_prop == 'QGM':
        # title = r"\textbf{Inverted }$\boldsymbol{\Delta\rho}$\textbf{ Model}" + "\n" + r"Slice at {0} = {1} m".format(normal,formatted_loc_plot)
        title = r"Quasi-Geological Model" + "\n" + r"Slice at {0} = {1} m".format(normal,formatted_loc_plot)
        # cbar_unit = r"$\bf{g.cm^{-3}}$"
        cbar_unit = r"Rock Unit"
    else:
        print(f"Error: Invalid model property '{which_prop}'.")
        return

    # Set the subplot title and axis labels
    ax1.set_title(title)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.set_aspect("equal", 'box')

    if normal == "X" or normal == "Y":
        ax1.set_ylim([np.nanmin(mesh.cell_centers_z), np.nanmax(mesh.cell_centers[ind_active][:,2])])
        ax1.set_aspect("equal", 'box')
    
    ax1.ticklabel_format(axis="both", style="scientific", scilimits=(0, 0))

    # Adding locations of depth sections
    if sec_loc == True:
        if normal == "Z":
            plt.axhline(loc_plot_y, color="k")
            plt.axvline(loc_plot_x, color="k")
        if normal == "X" or normal == "Y":
            plt.axhline(loc_plot_z, color="k")

    # Create a colorbar subplot
    ax2 = fig.add_axes([ax1.get_position().x1+0.02,ax1.get_position().y0,0.02,ax1.get_position().height])

    # Create a colorbar with limits that match the data
    norm = mpl.colors.Normalize(vmin=clim[0], vmax=clim[1])
    cbar = mpl.colorbar.ColorbarBase(
        ax2, norm=norm, orientation="vertical", cmap=mpl.cm.get_cmap(cmap)
    )

    # Set the label for the colorbar and disable scientific notation
    cbar.set_label(cbar_unit, size=14)
    cbar.formatter.set_powerlimits((0,0))

    # Save the plot as a pdf
    if save_plt == True:
        plt.savefig(
            path_to_output + "/" + str(set) + "_" + name + "_" + which_prop + "_" + normal + "_" + str(formatted_loc_plot) + ".pdf"
            )

    # Display the plot
    plt.show()

def plot_2D_data(
        data, 
        clim, 
        which_data="Grav",
        comp="xx", 
        cmap="inferno",
        path_to_output='.',
        name='Data'
        ):
    """
    Plot 2D data as contour plot with a colorbar.

    Parameters
    ----------
    data : np.ndarray
        Format should be (x, y, z, dat).
    which_data : str
        Checks to see if the data is Bouguer Anomaly or TMI. Must be one of, "Topo", "Grav", "Mag", "gMisfit", "FTG".
    cmap : str or Colormap
        Name of the colormap or a colormap instance.

    Returns
    -------
    None

    """

    # Create a figure with a size of 6x6 inches
    fig = plt.figure(figsize=(6, 6))

    # Create an axes object with specific position and size on the figure
    ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    # Plot the 2D gravity data as contours and save the plot object
    pltDataGrav = plot2Ddata(
        data[:,[0,1]],
        data[:,3],
        nx=200,
        ny=200,
        ax=ax1,
        contourOpts={"cmap": cmap},
        clim=(clim[0], clim[1]),
        ncontour=500,
    )

    if which_data == "Grav":
        title = 'Gravity Anomaly'
        cbar_unit = 'mGal'
    elif which_data == "gMisfit":
        title = 'Gravity Misfit'
        cbar_unit = 'mGal'
    elif which_data == "Topo":
        title = 'Topography'
        cbar_unit = 'm'
    elif which_data == "Mag":
        title = 'Total Magnetic Intensity'
        cbar_unit = 'nT'
    elif which_data == "FTG":
        title = r'FTG ($G_{' + comp + '}$)'
        cbar_unit = 'eotvos'
    else:
        print("Error: Invalid data_type")
        return

    # Set the plot title, axis labels, and aspect ratio of the axes object
    ax1.set_title(title)
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    ax1.set_aspect('equal', 'box')

    # Set the tick labels of both x and y axes to scientific notation
    ax1.ticklabel_format(axis='both', style='scientific', scilimits=(0, 0))

    # Set the edge colors of the contours to match the face color
    for c in pltDataGrav[0].collections:
        c.set_edgecolor('face')

    # Create a new axes object for the colorbar and set its position and size
    ax2 = fig.add_axes([ax1.get_position().x1+0.02,ax1.get_position().y0,0.02,ax1.get_position().height])

    # Normalize the colorbar to the minimum and maximum of the gravity data values
    norm = mpl.colors.Normalize(vmin=clim[0], vmax=clim[1])

    # Create a colorbar with the specified orientation, colormap, and normalization
    cbar = mpl.colorbar.ColorbarBase(
        ax2, norm=norm, orientation='vertical', cmap=mpl.cm.get_cmap(cmap)
    )

    # Set the label of the colorbar to indicate the unit of the gravity data
    cbar.set_label(cbar_unit, size=14)

    # Save the plot as a pdf
    plt.savefig(path_to_output + "/" + name + "_" + which_data + ".pdf")

    # Display the plot
    plt.show()