# THIS UTILS SECTION IS IMPLEMENTED BY Sora Ryu, another graduate student at UMASS AMHERST

import glob
import numpy as np
import pickle
from scipy import ndimage as ndimage
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from numpy.lib.npyio import NpzFile
import cv2 as cv
import open3d as o3d
import os
import sys
import plotly.graph_objects as go
from mpl_toolkits import mplot3d
# import pypotree
import re
import torch

# UNCOMMENT THE FOLLOWING IF YOU ARE USING GOOGLE COLAB

# from google.colab import drive
# drive.mount('/content/drive')
# !mkdir dataset_dhg1428
# !wget http://www-rech.telecom-lille.fr/DHGdataset/DHG2016.zip
# !unzip -u DHG2016.zip -d dataset_dhg1428

# Farthest Point Sampling
# A greedy algorithm that samples from a point cloud data iteratively.
# Starts from a random single sample of point, and at each iteration, it samples from the rest points
# that is the farthest from the set of sampled points.
'''
Implementation of the publication
Y. Eldar, M. Lindenbaum, M. Porat and Y. Y. Zeevi, "The farthest point strategy for progressive image sampling,"
Proceedings of the 12th IAPR International Conference on Pattern Recognition, Vol. 2 - Conference B: Computer Vision & Image Processing.
(Cat. No.94CH3440-5), 1994, pp. 93-97 vol.3, doi: 10.1109/ICPR.1994.577129.
'''

# ------------- Point Cloud sampling with FPS Algorithm ----------------
# fixed the number of point cloud to generate = 600
# fixed frame size = 20
'''
Input: Depth frame of each RGB-D video sequence
Output: (# of Sequences, # of Frames, # of Points, # of Point Cloud Dimension)
        = (1398, 20, 600, 3)

Needed to fix # of points and # of frames, to generate tensor.
'''


def generate_hand_pcd_with_fps(root):
    sequence_dir = root + '/gesture_*/finger_2/subject_*/essai_*'
    # finger_1: one finger used
    # finger_2: whole hand used => only using this case

    sequence_filenames = sorted(glob.glob(sequence_dir), key=lambda var: [int(x) if x.isdigit() else x for x in
                                                                          re.findall(r'[^0-9]|[0-9]+', var)])
    data, labels = np.empty((0, 20, 600, 3), dtype=np.float32), []

    for sequence_path in sequence_filenames:  # For each video sequence

        pattern_depth = sequence_path + '/depth_*.png'
        depth_filenames = sorted(glob.glob(pattern_depth), key=lambda var: [int(x) if x.isdigit() else x for x in
                                                                            re.findall(r'[^0-9]|[0-9]+', var)])
        per_sequence = np.empty((0, 600, 3), dtype=np.float32)

        for depth_path in depth_filenames:  # For every frame within the sequence

            print(depth_path)
            depth_img = cv.imread(depth_path, cv.IMREAD_ANYDEPTH)

            # Hand Segmentation based on the depth threshold.
            # As the face and torso have been also shown as well, we need to segment the hand as ROI.
            hand_depth = depth_img.copy()
            hand_depth[hand_depth > 680] = 0

            if len(hand_depth[hand_depth > 0]) < 2000:  # Didn't segment hand completely.. Do not create point clouds.
                continue

            path = '/'.join(depth_path.split('/')[:-1])
            filename = 'hand_' + depth_path.split('/')[-1]
            cv.imwrite(os.path.join(path, filename), hand_depth)

            # Create 3D Point Cloud from the hand depth image.
            depth_raw = o3d.io.read_image(os.path.join(path,
                                                       filename))  # Need to save hand_depth file and upload the image file. Cannot directly use hand_depth...
            pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_raw,
                                                                  o3d.camera.PinholeCameraIntrinsic(
                                                                      o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault),
                                                                  np.identity(4), depth_scale=1000.0,
                                                                  depth_trunc=1000.0)

            # Flip it, otherwise the pointcloud will be upside down
            pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

            # Uniformly downsample point cloud by collecting every n-th points.
            uni_down_pcd = pcd.uniform_down_sample(every_k_points=10)

            # Removes points that are further away from their neighbors compared to the average for the point cloud
            cl, _ = uni_down_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=0.2)

            if len(cl.points) < 600:
                continue

            # ---- Farthest Point Sampling ----
            sampled_fps = fps(np.asarray(cl.points), 600)
            sampled_fps = np.expand_dims(sampled_fps, axis=0)
            per_sequence = np.concatenate((per_sequence, sampled_fps))

        if per_sequence.size != 0 and per_sequence.shape[0] >= 20:
            idx = np.random.choice(per_sequence.shape[0], size=20, replace=False)
            per_sequence_sampled = per_sequence[np.sort(idx)]
            per_sequence_sampled = np.expand_dims(per_sequence_sampled, axis=0)
            data = np.concatenate((data, per_sequence_sampled))
            labels.append(int(sequence_path.split('/')[-4].split('_')[1]))

    return torch.from_numpy(data), torch.tensor(labels)


# ----------- Point Cloud Random Sampling ---------------
# fixed the number of point cloud to generate = 600
# fixed frame size = 20
'''
Input: Depth frame of each RGB-D video sequence
Output: (# of Sequences, # of Frames, # of Points, # of Point Cloud Dimension)
        = (1398, 20, 600, 3)

Needed to fix # of points and # of frames, to generate tensor.
'''


def generate_hand_pcd(root):
    sequence_dir = root + '/gesture_*/finger_2/subject_*/essai_*'
    # finger_1: one finger used
    # finger_2: whole hand used => only using this case

    sequence_filenames = sorted(glob.glob(sequence_dir), key=lambda var: [int(x) if x.isdigit() else x for x in
                                                                          re.findall(r'[^0-9]|[0-9]+', var)])
    data, labels = np.empty((0, 20, 600, 3), dtype=np.float32), []

    for sequence_path in sequence_filenames:  # For each video sequence

        pattern_depth = sequence_path + '/depth_*.png'
        depth_filenames = sorted(glob.glob(pattern_depth), key=lambda var: [int(x) if x.isdigit() else x for x in
                                                                            re.findall(r'[^0-9]|[0-9]+', var)])
        per_sequence = np.empty((0, 600, 3), dtype=np.float32)

        for depth_path in depth_filenames:  # For every frame within the sequence

            print(depth_path)
            depth_img = cv.imread(depth_path, cv.IMREAD_ANYDEPTH)

            # Hand Segmentation based on the depth threshold.
            # As the face and torso have been also shown as well, we need to segment the hand as ROI.
            hand_depth = depth_img.copy()
            hand_depth[hand_depth > 680] = 0

            if len(hand_depth[hand_depth > 0]) < 2000:  # Didn't segment hand completely.. Do not create point clouds.
                continue

            path = '/'.join(depth_path.split('/')[:-1])
            filename = 'hand_' + depth_path.split('/')[-1]
            cv.imwrite(os.path.join(path, filename), hand_depth)

            # Create 3D Point Cloud from the hand depth image.
            depth_raw = o3d.io.read_image(os.path.join(path,
                                                       filename))  # Need to save hand_depth file and upload the image file. Cannot directly use hand_depth...
            pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_raw,
                                                                  o3d.camera.PinholeCameraIntrinsic(
                                                                      o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault),
                                                                  np.identity(4), depth_scale=1000.0,
                                                                  depth_trunc=1000.0)

            # Flip it, otherwise the pointcloud will be upside down
            pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

            # Uniformly downsample point cloud by collecting every n-th points.
            uni_down_pcd = pcd.uniform_down_sample(every_k_points=20)

            # Removes points that are further away from their neighbors compared to the average for the point cloud
            cl, _ = uni_down_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=0.3)

            if len(cl.points) < 200:
                continue

            # ----- Random Sampling -----
            if len(cl.points) > 600:
                idx = np.random.choice(len(cl.points), size=600, replace=False)  # Pick 600 point clouds
            else:
                idx = np.random.choice(len(cl.points), size=600, replace=True)  # Pick 600 point clouds

            rand = np.asarray(cl.points)[idx].reshape(600, 3)
            rand = np.expand_dims(rand, axis=0)
            per_sequence = np.concatenate((per_sequence, rand))

        if per_sequence.size != 0 and per_sequence.shape[0] >= 20:
            idx = np.random.choice(per_sequence.shape[0], size=20, replace=False)
            per_sequence_sampled = per_sequence[np.sort(idx)]
            per_sequence_sampled = np.expand_dims(per_sequence_sampled, axis=0)
            data = np.concatenate((data, per_sequence_sampled))
            labels.append(int(sequence_path.split('/')[-4].split('_')[1]))

    return torch.from_numpy(data), torch.tensor(labels)


data, labels = generate_hand_pcd(root='/content/dataset_dhg1428')


def distance(a, b):
    return np.linalg.norm(a - b, ord=2, axis=2)


def fps(pcd, n_samples):
    n_pts, dim = pcd.shape[0], pcd.shape[1]
    remaining_pts = np.copy(pcd)

    # Randomly pick a start point
    sampled_pts = np.zeros((n_samples, 1, dim), dtype=np.float32)
    sampled_pts[0] = remaining_pts[np.random.randint(n_pts - 1)]

    for idx in range(1, n_samples):
        distances = distance(remaining_pts, sampled_pts[:idx]).T
        min_distances = np.min(distances, axis=1, keepdims=True)
        sampled_pts[idx] = remaining_pts[np.argmax(min_distances)]

    return np.squeeze(sampled_pts, axis=1)


def write_data(data, filepath):
    with open(filepath, 'wb') as output_file:
        pickle.dump(data, output_file)


def load_data(filepath='./dhg_data.pckl'):
    file = open(filepath, 'rb')
    data = pickle.load(file, encoding='latin1')
    file.close()
    return data['x_train'], data['x_test'], data['y_train'], data['y_test']


# Split the dataset into train and test sets with 70:30 ratio:
# x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.30)

# Save the dataset
#data = {
#    'x_train': x_train,
#    'x_test': x_test,
#    'y_train': y_train,
#    'y_test': y_test
#}
#write_data(data, filepath='dhg_data.pckl')


