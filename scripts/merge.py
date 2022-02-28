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
    post.merge_process_files(analysis_dir, cleanup=True)
    set_paths = glob(path.join(analysis_dir, 'analysis_s*.h5'))
    post.merge_sets(path.join(data_dir, 'analysis.h5'), set_paths, cleanup=True)

    state_dir = path.join(data_dir, "state")
    post.merge_process_files(state_dir, cleanup=True)
    set_paths = glob(path.join(state_dir, 'state_s*.h5'))
    post.merge_sets(path.join(data_dir, 'state.h5'), set_paths, cleanup=True)

    logger.info('Finished merging files, took {} seconds'.format(time.time() - t0))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Please provide one argument: The file path to the directory to save the analysis in.")
        exit(1)
    data_dir = sys.argv[1]
    merge(data_dir)
