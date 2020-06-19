import os
import argparse
#from multiprocessing import ThreadPool
import shutil
import time
from itertools import zip_longest


def copy_jpg_files(args):
    """Retrieve files in folder and write them to new folder"""
    in_parent_folder = args.in_dir
    out_parent_folder = args.out_dir
    # walk through each folder and retrieve valid jpeg images
    # and save in out_parent_folder
    folders = [folder for folder in os.listdir(
        in_parent_folder) if os.path.isdir(os.path.join(in_parent_folder, folder))]
    #new_path = os.path.join(out_parent_folder, folder)
    #if not os.path.exists(new_path):
    #    os.makedirs(new_path)
    for folder in folders:
        start = time.time()
        print(f'>>> Copying files from {folder} folder ...', end='')
        if folder == '0samples':  # copy entire 0samples cos it has no issues
            #new_path = os.path.join(out_parent_folder, folder)
            #os.makedirs(new_path)
            #shutil.copytree(os.path.join(in_parent_folder,folder),new_path)
            continue
        else:
            files = [file for file in os.listdir(
                os.path.join(in_parent_folder, folder)) if os.path.isfile(os.path.join(in_parent_folder, folder, file)) and file.endswith('.jpg')]
            #print(f'>>> Total files {sum(files)}')
            try:
                for file in files:
                    shutil.copy2(os.path.join(
                        in_parent_folder, folder, file), out_parent_folder)
            except Exception as error:
                print(f'>>> An error occurred: {error}')
        print(f'>>> Copying {folder} folder done in {time.time()-start:.2f}')


if __name__ == '__main__':
    # get a list of the subdirs and process them using processes
    parser = argparse.ArgumentParser(
        description='a program to move jpg files to a new folder for annotation')
    parser.add_argument('--in_dir', type=str, default=os.path.abspath('data/LogosInTheWild-v2/clean_data/voc_format'),
                        help='path to source folder to copy the JPG files')
    parser.add_argument('--out_dir', type=str, default=os.path.abspath('data/litw_annotations'),
                        help='path to the destination folder for annotation')
    args = parser.parse_args()
    copy_jpg_files(args)
