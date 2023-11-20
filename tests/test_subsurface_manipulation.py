import os
from dotenv import dotenv_values

config = dotenv_values()
path = config.get("PATH_TO_STONEPARK_Subsurface")


def find_paired_files(directory):
    # Store the base names of the files with .le and .json extensions
    le_files = set()
    json_files = set()

    # Scan the directory for files
    for filename in os.listdir(directory):
        base, ext = os.path.splitext(filename)
        if ext == '.le':
            le_files.add(base)
        elif ext == '.json':
            json_files.add(base)

    # Find the intersection of both sets to get the common base names
    common_bases = le_files.intersection(json_files)

    # Create a list of tuples with the paired file names
    paired_files = [(f"{base}.le", f"{base}.json") for base in common_bases]

    return paired_files


def test_merge_files():
    paired_files = find_paired_files(path)
    print(paired_files)
    from subsurface.reader.from_binary import binary_file_to_base_structs
    data_array = binary_file_to_base_structs(
        le_file_path=path + "/" + paired_files[0][0],
        json_file_path=path + "/" + paired_files[0][1]
    )
    print(data_array)