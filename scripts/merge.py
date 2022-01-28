#!/usr/bin/env python3

from dedalus.tools import post
from os import path
import sys
import shutil
from glob import glob
import time

def merge(data_dir):

    print('Merging files...')
    t0 = time.time()

    analysis_dir = path.join(data_dir, "analysis")
    post.merge_process_files(analysis_dir, cleanup=True)
    set_paths = glob(path.join(analysis_dir, '*.h5'))
    post.merge_sets(path.join(data_dir, 'analysis_new.h5'), set_paths, cleanup=True)
    shutil.move(path.join(data_dir, 'analysis_new.h5'), path.join(data_dir, 'analysis.h5'))

    state_dir = path.join(data_dir, "state")
    post.merge_process_files(state_dir, cleanup=True)
    set_paths = glob(path.join(state_dir, '*.h5'))
    post.merge_sets(path.join(data_dir, 'state_new.h5'), set_paths, cleanup=True)
    shutil.move(path.join(data_dir, 'state_new.h5'), path.join(data_dir, 'state.h5'))

    shutil.rmtree(path.join(data_dir, "analysis"))
    shutil.rmtree(path.join(data_dir, "state"))

    print(f'Finished merging files, took {time.time() - t0} seconds')

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Please provide one argument: The file path to the directory to save the analysis in.")
        exit(1)
    data_dir = sys.argv[1]
    merge(data_dir)
