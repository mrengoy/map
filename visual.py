from math import atan, sqrt
import math
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random as rand


def get_data(center: Tuple[int, int] = (662700, 8170720), dims: Tuple[int, int] = (20000, 20000), point_limit: int = 2000):
    '''
    Gets data from csv. Defaults to lokeslottet.
    pointlimit = -1, means no limit
    Returns (terrain_x, terrain_y, terrain_z)
        '''
    print('get_data start')
    df = pd.read_csv('./data/MohnsRidge_Merged.csv')
    filter_data = True
    #point_limit = 2000
    x_min, x_max = center[0] - dims[0]/2,  center[0] + dims[0]/2  # den toppen
    y_min, y_max = center[1] - dims[1]/2, center[1] + dims[1]/2  # den toppen
    #print(x_min, x_max, y_min, y_max)
    # print(df.shape)
    df.columns = ['y', 'x', 'pz', 'z']
    if filter_data:
        df = df[
            (x_min < df['x']) &
            (x_max > df['x']) &
            (y_min < df['y']) &
            (y_max > df['y'])
        ]

    if df.shape[0] > point_limit and point_limit != -1:
        df = df.sample(n=point_limit)
    # print(df.shape)

    terrain_x = df.iloc[:, 1].to_numpy()
    terrain_y = df.iloc[:, 0].to_numpy()
    terrain_z = df.iloc[:, 3].to_numpy()
    print('get_data end')
    return (terrain_x, terrain_y, terrain_z)


def plot_surface(terrain):
    (terrain_x, terrain_y, terrain_z) = terrain
    print('plot_surface start')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_trisurf(terrain_x, terrain_y, terrain_z, cmap=plt.cm.jet)
    smallest_x = np.amin(a=terrain_x)
    smallest_y = np.amin(a=terrain_y)
    ax.scatter3D(smallest_x, smallest_y, 0, color='#00000000')
    fig.colorbar(surf)
    print('plot_surface end')
    plt.show()


def scatter3d(terrain, c=[]):
    (x, y, z) = terrain
    # Creating figure
    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection="3d")
    smallest_x = np.amin(a=x)
    smallest_y = np.amin(a=y)
    ax.scatter3D(smallest_x, smallest_y, 0, color='#00000000')
    # Creating plot
    if len(c) == 0:
        ax.scatter3D(x, y, z, c=z, cmap=plt.cm.jet)
    else:
        ax.scatter3D(x, y, z, c=c, cmap=plt.cm.jet)
    plt.title("simple 3D scatter plot")
    # show plot
    plt.show()




def parish_terrain(terrain, center, dims, bucket_grid=(100, 100)):
    '''
    Currently just avg
    Derivative. Best if dims is div by 2 and center is integer
    '''
    bucket_dims = np.array(bucket_grid)  # Size of buckets grid
    bucket_size = dims // bucket_dims
    buckets_sum = np.zeros(shape=bucket_dims)
    buckets_count = np.zeros(shape=bucket_dims, dtype=np.int32)
    p_o = np.array([center[0] - dims[0]/2, center[1] - dims[1]/2])
    (terrain_x, terrain_y, terrain_z) = terrain
    for p_a in zip(terrain_x, terrain_y, terrain_z):
        p_r = p_a[:2] - p_o
        bucket_index = p_r // bucket_size
        (x, y) = bucket_index.astype(np.int32)
        buckets_sum[y][x] += p_a[2]
        buckets_count[y][x] += 1
    buckets = buckets_sum / buckets_count
    return buckets

def distance_between_points(a,b):
    return sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)



if __name__ == '__main__':
    termals = {
        'lokeslottet' : [662700, 8170720],
        'mohns' : [538436, 8079212], # Update
        'aegir' : [451086, 8027891],
        #'knipovitsj_b' : [659052, 8308587],
        #'knipovitsj_m' : [627466, 8528009],
        #'knipovitsj_t' : [609421, 8621352],
    }
    center = (termals['aegir'][0] - 66_000, termals['aegir'][1] + 50_000)
    dims = (675_000, 450_000) # Adjust to covert dataset
    #center = (termals['aegir'][0], termals['aegir'][1])
    #dims = (100_000, 100_000)
    terrain = get_data(center=center, dims=dims, point_limit=-1)
    bucket_dims = (1000, 1000)
    distance_between_buckets = (dims[0] // bucket_dims[0], dims[1] // bucket_dims[1])
    avg_buckets = parish_terrain(
        terrain, center=center, dims=dims, bucket_grid=bucket_dims)
    
    [grad_y, grad_x] = np.gradient(avg_buckets)
    grad = np.sqrt(grad_x**2 + grad_y**2)
    [_, grad2_x] = np.gradient(grad_x)
    [grad2_y, _] = np.gradient(grad_y)
    grad2 = np.sqrt(grad2_x**2 + grad2_y**2)
    t_x = []
    t_y = []
    t_z = []

    height = dims[1]
    width = dims[0]
    origin_x = center[0] - dims[0]/2
    origin_y = center[1] - dims[1]/2
    def bucket_cord_to_correct_cord(x, y):
        return (origin_x + width*x/bucket_dims[0], origin_y + height*y/bucket_dims[1])

    print(avg_buckets.shape)
    z_scale_factor = 1/(math.sqrt(distance_between_buckets[0]*distance_between_buckets[1]))
    # for (y, row) in enumerate(avg_buckets):
    bucket_cords_x = []
    bucket_cords_y = []
    bucket_cords_z = []

    for (y, row) in enumerate(grad):
        for (x, z) in enumerate(row):
            if not (np.isnan(x) or np.isnan(y) or np.isnan(z)):
                bucket_cords_x.append(x)
                bucket_cords_y.append(y)
                bucket_cords_z.append(avg_buckets[y][x])
                correct_cords = bucket_cord_to_correct_cord(x, y)
                t_x.append(correct_cords[0])
                t_y.append(correct_cords[1])    
                t_z.append(z*z_scale_factor)

    t_xnp = np.array(t_x)
    t_ynp = np.array(t_y)
    t_znp = np.array(t_z) 

    min_z = np.nanmin(t_znp)
    max_z = np.nanmax(t_znp)
    range_z = max_z - min_z

    def color_add(a, b):
        '''
        a on top of b
        '''
        a_s = a[3]
        b_s = (1 - a[3])*b[3]
        return (
            a[0]*a_s + b[0]*b_s,
            a[1]*a_s + b[1]*b_s,
            a[2]*a_s + b[2]*b_s,
            a[3] + b[3] - a[3]*b[3]
        )


    max_incline = atan(20/180*math.pi)
    min_incline = 0
    print('min max z')
    print(min(t_znp), max(t_znp))
    print('min max incline')
    print(min_incline, max_incline)

    #
    def f(x, y, z, bx, by, bz):
        z_percent = 0.5 + ((z - min_z)/range_z)*0.5
        base_color = (z_percent, z_percent, z_percent, 1)
        distance_to_center = min(
            distance_between_points((x,y), termals['lokeslottet']),
            distance_between_points((x,y), termals['mohns']),
            distance_between_points((x,y), termals['aegir']),
        )
        if distance_to_center < 1_000_000:
            if bz > -1600 and min_incline < z < max_incline:
                return color_add((0,1,0,0.3), base_color)
            else:
                return base_color
        else: 
             return color_add((1,0,0,0.3), base_color)

    score = np.array(
        list(map(f, t_xnp, t_ynp, t_znp, bucket_cords_x, bucket_cords_y, bucket_cords_z))
    )

    print(termals.values())
    #scatter3d((np.array(t_x),np.array(t_y),np.array(t_z)), c=np.ndarray.flatten(grad))
    print(center[0] - dims[0]/2, termals['lokeslottet'][0], center[0] + dims[0]/2)
    print(center[1] - dims[1]/2, termals['lokeslottet'][1], center[1] + dims[1]/2)
    #scatter2d((t_xnp, t_ynp, t_znp), c=score, points_of_interest=points_of_interest)

    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes()
    # Background
    #ax.scatter(t_xnp, t_ynp, c=t_znp, cmap=plt.cm.gray)
    ax.scatter(t_xnp, t_ynp, c=score, s=10, marker='s')
    
    min_x = center[0] - dims[0]/2
    max_x = center[0] + dims[0]/2
    min_y = center[1] - dims[1]/2
    max_y = center[1] + dims[1]/2
    for p in termals.values():
        if((min_x < p[0] < max_x) and min_y < p[1] < max_y):
            print(f'added {p}')
            ax.scatter(p[0], p[1], color='lime')
    transparent = '#00000000'
    ax.scatter(min_x, min_y, color=transparent)
    ax.scatter(max_x, max_y, color=transparent)
    plt.title("simple 2D scatter plot")
    plt.show()
