import taichi as ti
import numpy as np
import trimesh
import pyvista as pv
from sim import MpmLagSim

def test_lag_mpm():
    ti.init(arch=ti.vulkan)
    dt = 1e-4
    sim = MpmLagSim(origin=np.asarray([-0.5,] * 3), dt=dt)
    box_mesh = trimesh.load_mesh('./data/plasticbag1.obj')
    sim.set_soft(box_mesh)
    sim.init_sim()
    sim.gravity = 10
    sim.bending_p = 1

    while not sim.window.is_pressed(ti.GUI.ESCAPE):
        sim.substep()
        sim.update_scene()
        sim.show()
        if sim.window.is_pressed(ti.GUI.SPACE):
            pl = pv.Plotter()
            reader = pv.get_reader('./vtk/plasticbag1.vtk')
            mesh = reader.read()
            mesh.points = sim.x_soft.to_numpy()
            pl.add_mesh(mesh, show_edges=True, color='white',opacity = 0.5,lighting = False)
            pl.show()

if __name__ == '__main__':
    test_lag_mpm()