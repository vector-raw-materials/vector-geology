import gempy as gp
import numpy as np


def optimize_nuggets_for_group(geo_model: gp.data.GeoModel, structural_group: gp.data.StructuralGroup,
                               plot_evaluation: bool = False, plot_result: bool = False) -> None:
    temp_structural_frame = gp.data.StructuralFrame(
        structural_groups=[structural_group],
        color_gen=gp.data.ColorsGenerator()
    )
    
    
    previous_structural_frame = geo_model.structural_frame
    
    
    geo_model.structural_frame = temp_structural_frame
    
    gp.API.compute_API.optimize_and_compute(
        geo_model=geo_model,
        engine_config=gp.data.GemPyEngineConfig(
            backend=gp.data.AvailableBackends.PYTORCH,
        ),
        max_epochs=100,
        convergence_criteria=1e5
    )

    nugget_effect = geo_model.taped_interpolation_input.surface_points.nugget_effect_scalar.detach().numpy()
    np.save(f"nuggets_{structural_group.name}", nugget_effect)
    
    if plot_evaluation:
        import matplotlib.pyplot as plt
        
        plt.hist(nugget_effect, bins=50, color='black', alpha=0.7, log=True)
        plt.xlabel('Eigenvalue')
        plt.ylabel('Frequency')
        plt.title('Histogram of Eigenvalues (nugget-grad)')
        plt.show()
        
    if plot_result:
        import gempy_viewer as gpv
        import pyvista as pv
        
        gempy_vista = gpv.plot_3d(
            model=geo_model,
            show=False,
            kwargs_plot_structured_grid={'opacity': 0.3}
        )

        # Create a point cloud mesh
        surface_points_xyz = geo_model.surface_points.df[['X', 'Y', 'Z']].to_numpy()
    
        point_cloud = pv.PolyData(surface_points_xyz[0:])
        point_cloud['values'] = nugget_effect

        gempy_vista.p.add_mesh(
            point_cloud,
            scalars='values',
            cmap='inferno',
            point_size=25,
        )

        gempy_vista.p.show()
    
    geo_model.structural_frame = previous_structural_frame
    return nugget_effect

