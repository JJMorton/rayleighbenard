#!/usr/bin/env python3

from dedalus.tools import post
from os import path
import sys
import shutil
from glob import glob
import time

import logging
logger = logging.getLogger(__name__)

def merge(data_dir):

    logger.info('Merging files...')
    t0 = time.time()

    analysis_dir = path.join(data_dir, "analysis")
    analysis_file = path.join(data_dir, 'analysis.h5')
    post.merge_process_files(analysis_dir, cleanup=False)
    set_paths = glob(path.join(analysis_dir, 'analysis_s*.h5'))
    # If analysis.h5 already exists, merge it with all the new data
    if path.exists(analysis_file):
        old_file = path.join(data_dir, 'analysis_old.h5')
        shutil.move(analysis_file, old_file)
        set_paths.append(old_file)
    post.merge_sets(path.join(data_dir, 'analysis.h5'), set_paths, cleanup=True)

    state_dir = path.join(data_dir, "state")
    state_file = path.join(data_dir, 'state.h5')
    post.merge_process_files(state_dir, cleanup=False)
    set_paths = glob(path.join(state_dir, 'state_s*.h5'))
    # If state.h5 already exists, merge it with all the new data
    if path.exists(state_file):
        old_file = path.join(data_dir, 'state_old.h5')
        shutil.move(state_file, old_file)
        set_paths.append(old_file)
    post.merge_sets(path.join(data_dir, 'state.h5'), set_paths, cleanup=True)
    
    vel_dir = path.join(data_dir, "vel")
    vel_file = path.join(data_dir, 'vel.h5')
    post.merge_process_files(vel_dir, cleanup=False)
    set_paths = glob(path.join(vel_dir, 'vel_s*.h5'))
    # If vel.h5 already exists, merge it with all the new data
    if path.exists(vel_file):
        old_file = path.join(data_dir, 'vel_old.h5')
        shutil.move(vel_file, old_file)
        set_paths.append(old_file)
    post.merge_sets(path.join(data_dir, 'vel.h5'), set_paths, cleanup=True)

    logger.info('Finished merging files, took {} seconds'.format(time.time() - t0))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Please provide one argument: The file path to the directory to save the analysis in.")
        exit(1)
    data_dir = sys.argv[1]
    merge(data_dir)
