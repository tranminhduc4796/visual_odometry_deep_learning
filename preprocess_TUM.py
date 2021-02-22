import numpy as np
from spatialmath.base import q2r


def tum2kitti(txt_file_path):
    """
    Convert data from TUM format to KITTI format
    """
    # read txt_file
    with open(txt_file_path, 'r') as f:
        lines = f.readlines()

    # write to new txt_file
    with open('new_format.txt', 'w') as f:
        for i in lines[3:]:
            i = i.replace('\n', '').split(' ')
            t = np.array(i[1:4]).reshape(3, 1)
            R = np.asarray(q2r(i[4:]))

            Rt_flatten = np.hstack((R, t)).flatten()
            kitti_fmt = np.insert(Rt_flatten, 0, i[0])
            f.write("%s\n" % " ".join(kitti_fmt))
