import os

def abs_path(rel_path):
    dir_path = os.path.dirname(os.path.realpath(__file__))  # Get the directory of the script
    abs_file_path = os.path.join(dir_path, rel_path)
    return abs_file_path