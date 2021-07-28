import os
import zipfile
import tarfile

from glob import glob
from pathlib import Path

import pandas as pd


def get_dir_dataframe(path, pattern="*.csv"):
    """
    This methods reads all files in the given directory. Reads the file and return a Pandas DataFrame
    concatenating all the files into a single df
    :return: Pandas DataFrame
    """
    files = sorted(glob(f"{path}/{pattern}"))
    df = pd.concat((pd.read_csv(file) for file in files), index=False)

    return df


def unzip_file(zipfile_path, dest_dir=None):
    """
    Unzip the given file
    :param zipfile_path: Zip file path
    :param dest_dir: Destination directory to keep unzip files, takes below values
            None - to extract in the current directory pass None
            same - to extract in the directory where zip file is present else give destination directory.
            <directory - any directory path, if not exist will be created
    """
    # Check if file exists
    assert os.path.isfile(zipfile_path), f'{zipfile_path} - Zip file does not exists'
    if dest_dir is not None and dest_dir != 'same':
        assert os.path.isdir(dest_dir), f'Invalid destination directory : {dest_dir}'

    cwd = os.getcwd()

    zipfile_path = Path(zipfile_path).resolve()
    zipfile_dir = zipfile_path.parent

    # check the destination directory
    if dest_dir is None:
        dest_dir = cwd
    elif dest_dir == 'same':
        dest_dir = zipfile_dir
    else:
        dest_dir = Path(dest_dir).resolve()
        # Create the destination dir, if not exists
        if not os.path.isdir(dest_dir):
            os.mkdir(dest_dir)

    try:
        # Unzip the file
        zip_ref = zipfile.ZipFile(zipfile_path, 'r')
        zip_content_files = [os.path.join(zipfile_dir, v.filename) for v in zip_ref.filelist]
        # Extract all files
        zip_ref.extractall(dest_dir)
        zip_ref.close()
        print(f'{zipfile_path} extracted to directory : {dest_dir}')
    except Exception as e:
        print(e)


def load_from_json(json_file_path):
    import json
    with open(json_file_path) as json_file:
        json_data = json.load(json_file)
    return json_data


def extract_tar(tarfile_path, destn_dir=None):
    assert os.path.exists(tarfile_path), 'Tar file does not exists : '+tarfile_path
    tar = tarfile.open(tarfile_path)
    assert tar.is_tarfile(tarfile_path), 'Invalid tar file : '+tarfile_path
    # create the destination directory
    if not os.path.isdir(destn_dir):
        os.mkdir(destn_dir)
        print('created directory : ' + destn_dir)
    # extract the contents
    tar.extractall(destn_dir)
    print('Successfully extracted file to : ' + destn_dir)
    
