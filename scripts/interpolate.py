#!/usr/bin/env python3

import h5py
import sys
from os import path
import numpy as np
from dedalus import public as de
from dedalus.tools import post
from glob import glob
import shutil

import utils
import filtering

def main(data_dir):
    with h5py.File(path.join(data_dir, 'state.h5'), mode='r') as state_file:

        dims = state_file['tasks']['u'].dims
        num_dims = len(dims) - 1

        u = np.array(state_file['tasks']['u'])
        v = np.array(state_file['tasks']['v']) if num_dims == 3 else None
        w = np.array(state_file['tasks']['w'])

        t = np.array(dims[0]["sim_time"])
        x = np.array(dims[1][0])
        y = np.array(dims[2][0]) if num_dims == 3 else None
        z = np.array(dims[3 if num_dims == 3 else 2][0])

        u_interp = np.zeros(u.shape)
        v_interp = np.zeros(v.shape) if v is not None else None
        w_interp = np.zeros(w.shape)

        print("Interpolating velocity fields")
        for i in range(u.shape[0]):
            if i % 10 == 0: print(f"{(100 * i) // u.shape[0]}%")
            u_interp[i] = filtering.interp_to_fourier(u[i], axis=-1)
            if v_interp is not None and v is not None:
                v_interp[i] = filtering.interp_to_fourier(v[i], axis=-1)
            w_interp[i] = filtering.interp_to_fourier(w[i], axis=-1)

        print("Saving to interp.h5")
        xbasis = de.Fourier('x', len(x), interval=(x[0], x[-1]))
        ybasis = de.Fourier('y', len(y), interval=(y[0], y[-1])) if y is not None else None
        zbasis = de.Fourier('z', len(z), interval=(z[0], z[-1]))
        domain = de.Domain((xbasis, ybasis, zbasis) if ybasis is not None else (xbasis, zbasis), grid_dtype=np.float64)
        problem = de.IVP(domain,variables=['u', 'v', 'w'])
        problem.add_equation("dt(u) = 0")
        problem.add_equation("dt(v) = 0")
        problem.add_equation("dt(w) = 0")
        solver = problem.build_solver(de.timesteppers.RK443)
        handler = solver.evaluator.add_file_handler(path.join(data_dir, "interp"), mode='overwrite')
        handler.add_system(solver.state, layout='g')
        handler.last_wall_div = handler.last_sim_div = handler.last_iter_div = 0
        last_t = 0
        for i in range(len(t)):
            current_t = t[i]
            dt = current_t - last_t
            solver.step(dt)
            last_t = t[i]
            solver.state['u']['g'] = u_interp[i]
            if v_interp is not None: solver.state['v']['g'] = v_interp[i]
            solver.state['w']['g'] = w_interp[i]
            solver.evaluator.evaluate_handlers((handler,), world_time=0, wall_time=0, sim_time=solver.sim_time, timestep=dt, iteration=i)

        interp_dir = path.join(data_dir, "interp")
        post.merge_process_files(interp_dir, cleanup=True)
        set_paths = glob(path.join(interp_dir, '*.h5'))
        post.merge_sets(path.join(data_dir, 'interp.h5'), set_paths, cleanup=True)
        shutil.rmtree(interp_dir)

        print("Done")

if __name__ == '__main__':

    if len(sys.argv) < 2:
        print("Please provide one argument: The file path to the directory to read the analysis from.")
        exit(1)
    data_dir = sys.argv[1]

    main(data_dir)
    # test(data_dir)
