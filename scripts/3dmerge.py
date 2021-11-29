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

    post.merge_process_files(path.join(data_dir, "snapshots"), cleanup=True)
    set_paths = glob(path.join(data_dir, "snapshots", '*.h5'))
    post.merge_sets(path.join(data_dir, "snapshots", "snapshots_new.h5"), set_paths, cleanup=True)
    shutil.move(path.join(data_dir, 'snapshots', 'snapshots_new.h5'), path.join(data_dir, 'snapshots', 'snapshots.h5'))

    post.merge_process_files(data_dir, cleanup=True)
    set_paths = glob(path.join(data_dir, '*.h5'))
    post.merge_sets(path.join(data_dir, 'analysis_new.h5'), set_paths, cleanup=True)
    shutil.move(path.join(data_dir, 'analysis_new.h5'), path.join(data_dir, 'analysis.h5'))

    print(f'Finished merging files, took {time.time() - t0} seconds')

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Please provide one argument: The file path to the directory to save the analysis in.")
        exit(1)
    data_dir = sys.argv[1]
    merge(data_dir)
