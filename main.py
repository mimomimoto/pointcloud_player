import asyncio
import base64
from fastapi import APIRouter, Body, FastAPI, Request, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import numpy as np
from typing import Any, Dict, List, Union
from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder
import open3d as o3d
import matplotlib.pyplot as plt
import pandas as pd
from natsort import natsorted
import glob
import colorsys
import time
from fastapi.responses import HTMLResponse, JSONResponse

templates = Jinja2Templates(directory="templates")
stored_data: Dict[str, Any] = {"pointcloud": {}, "log": "", "clearLog": False}

color_sep = 7

h_list = [0]

for i in range(1, color_sep + 1):
    tmp_list = []
    for j in h_list:
        tmp_list.append(j + 1 / (2 ** i))
    h_list += tmp_list

color_list = []
index1 = []
for i in range(len(h_list)):
    rgb = colorsys.hsv_to_rgb(h_list[i], 1, 1)
    rgb_255 = [0, 0, 0]
    for j in range(3):
        rgb_255[j] = rgb[j]
    color_list.append(rgb_255)
    index1.append(i)


columns1 =["x", "y", "pred_x", "pred_y", "height", "pcl_feature", "update"]
gallary_df = pd.DataFrame(index=index1, columns=columns1)

class PointCloudData(BaseModel):
    name: str
    points: str
    colors: str


class PointCloudDataArray(BaseModel):
    array: List[PointCloudData]
    clear: bool

def _encode(s: bytes) -> str:
    return base64.b64encode(s).decode("utf-8")


def _decode(s: str) -> bytes:
    return base64.b64decode(s.encode("utf-8"))
app = FastAPI()
app.mount(
    '/templates/static', 
    StaticFiles(directory="templates/static"), 
    name='static'
    )

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request
    })

@app.post("/get_config_data")
async def get_config_data(config: Dict[str, Any] = Body(...)):
    ROUGH_DBSCAN_EPS = float(config['ROUGH_DBSCAN_EPS'])
    ROUGH_DBSCAN_MIN_POINTS = int(config['ROUGH_DBSCAN_MIN_POINTS'])

    CLUSTER_HEIGHT_THERESH = float(config['CLUSTER_HEIGHT_THERESH'])
    CLUSTER_POINTS_THERESH = int(config['CLUSTER_POINTS_THERESH'])
    CLUSTER_BOX_SIZE_THERESH = float(config['CLUSTER_BOX_SIZE_THERESH'])

    STRICT_DBSCAN_EPS = float(config['STRICT_DBSCAN_EPS'])
    STRICT_DBSCAN_MIN_POINTS_DIV = float(config['STRICT_DBSCAN_MIN_POINTS_DIV'])
    
    CURRENT_POS_WEIGHT = float(config['CURRENT_POS_WEIGHT'])
    PRED_POS_WEIGHT = float(config['PRED_POS_WEIGHT'])
    TRACKING_MIN_DISTANCE = float(config['TRACKING_MIN_DISTANCE'])
    GALLARY_RETENTION_PERIOD = float(config['GALLARY_RETENTION_PERIOD'])
    
    DIR_PATH = config['directoryName']
    data_path_list = natsorted(glob.glob(f'./data/{DIR_PATH}/*.pcd'))
    BASE_FILE_PATH = config['basePathName']
    SOURCE_PCD = o3d.io.read_point_cloud(f'./base/{BASE_FILE_PATH}')
    
    def divide_cluster(pcd, pcd_arrays, thr_points_num):
        global index1
        device = o3d.core.Device("CPU:0")
        cpu_device = o3d.core.Device("CPU:0")
        dtype = o3d.core.float64
        x_y_df = pd.DataFrame(pcd.points,
                    columns = ["x","y","z"],)
        x_y_df["z"] = 0
        
        z_df = pd.DataFrame(pcd.points,
                    columns = ["x","y","z"],)
        
        x_y_pcd = o3d.geometry.PointCloud()
        x_y_pcd.points = o3d.utility.Vector3dVector(x_y_df.to_numpy())
        thresh_min_points = int(len(x_y_df.index)/STRICT_DBSCAN_MIN_POINTS_DIV)
        
        tmp_x_y_pcd = o3d.t.geometry.PointCloud(device)
        tmp_x_y_pcd.point.positions = o3d.core.Tensor(np.asarray(x_y_pcd.points), dtype, device)
        
        with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
            labels_tensor = tmp_x_y_pcd.cluster_dbscan(eps=STRICT_DBSCAN_EPS, min_points=thresh_min_points, print_progress=False)
            labels_tensor_cpu = labels_tensor.to(cpu_device)
            labels = np.array(labels_tensor_cpu.cpu().numpy())
            
        tmp_cluster_pcd = o3d.geometry.PointCloud()
        
        min_cluster_points = 0

        if labels.size != 0: 
            for i in range(labels.max() + 1):
                pc_indices = np.where(labels == i)[0]
                xyz = np.asarray(z_df.to_numpy())[pc_indices, :]
                if xyz.T[2].max() > CLUSTER_HEIGHT_THERESH:
                    if len(xyz) >= CLUSTER_POINTS_THERESH:
                        tmp_pcd = o3d.geometry.PointCloud()
                        tmp_pcd.points = o3d.utility.Vector3dVector(xyz)
                        
                        plane_xyz = np.asarray(x_y_df.to_numpy())[pc_indices, :]
                        plane_pcd = o3d.geometry.PointCloud()
                        plane_pcd.points = o3d.utility.Vector3dVector(plane_xyz)
                        if min_cluster_points == 0:
                            min_cluster_points = len(tmp_pcd.points)
                        elif min_cluster_points > len(tmp_pcd.points):
                            min_cluster_points = len(tmp_pcd.points)
                        pcd_arrays.append(xyz)
                        
                        tmp_cluster_pcd += tmp_pcd
            
            dists = pcd.compute_point_cloud_distance(tmp_cluster_pcd)
            dists = np.asarray(dists)
            ind = np.where(dists > 0.03 )[0]
            noise_pcd = pcd.select_by_index(ind)
            if len(noise_pcd.points) >= min_cluster_points:
                pcd_arrays = divide_cluster(noise_pcd, pcd_arrays, min_cluster_points)
        return pcd_arrays

    def cluster(path):
        cpu_device = o3d.core.Device("CPU:0")
        device = o3d.core.Device("CPU:0")
        dtype = o3d.core.float64

        target = o3d.io.read_point_cloud(path)

        dists = target.compute_point_cloud_distance(SOURCE_PCD)
        dists = np.asarray(dists)
        ind = np.where(dists > 0.1)[0]
        tmp_object_pcd = target.select_by_index(ind)
        cl, ind = tmp_object_pcd.remove_radius_outlier(nb_points=10, radius=0.5)
        tmp_object_pcd = tmp_object_pcd.select_by_index(ind)
        object_pcd = o3d.t.geometry.PointCloud(device)
        object_pcd.point.positions = o3d.core.Tensor(np.asarray(tmp_object_pcd.points), dtype, device)

        with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
            labels_tensor = object_pcd.cluster_dbscan(eps=ROUGH_DBSCAN_EPS, min_points=ROUGH_DBSCAN_MIN_POINTS, print_progress=False)
            labels_tensor_cpu = labels_tensor.to(cpu_device)
            labels = np.array(labels_tensor_cpu.cpu().numpy())


        cluster_pcd = o3d.geometry.PointCloud()
        
        pcd_arrays = []
        
        if labels.size != 0: 
            for i in range(labels.max() + 1):
                pc_indices = np.where(labels == i)[0]

                if pc_indices.size > 0:
                    xyz = np.asarray(tmp_object_pcd.points)[pc_indices, :]
                    
                    if xyz.T[2].max() > CLUSTER_HEIGHT_THERESH:
                        if len(xyz) >= CLUSTER_POINTS_THERESH:
                            pcd = o3d.geometry.PointCloud()
                            pcd.points = o3d.utility.Vector3dVector(xyz)
                            bounding_box = pcd.get_oriented_bounding_box()
                            size_bounding_box = bounding_box.get_max_bound() - bounding_box.get_min_bound()
                            ts_size = size_bounding_box[0] * size_bounding_box[1]
                            if ts_size >= CLUSTER_BOX_SIZE_THERESH:
                                pcd_arrays = divide_cluster(pcd, pcd_arrays, 0)
                            else:
                                pcd_arrays.append(xyz)
                            cluster_pcd += pcd

        return pcd_arrays


    def preprocess_point_cloud(pcd, voxel_size):

        radius_normal = voxel_size * 2
        # print(":: Estimate normal with search radius %.3f." % radius_normal)
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=50))

        radius_feature = voxel_size * 5
        # print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        return pcd_fpfh
    
    def extract_feature(pcd):
        feature_list = []
        axis_aligned_bounding_box = pcd.get_axis_aligned_bounding_box()
        x_y = axis_aligned_bounding_box.get_center()[0:2]
        x = x_y[0]
        y = x_y[1]
        z = axis_aligned_bounding_box.get_max_bound()[2]
        pcd_fpfh = preprocess_point_cloud(pcd, 0.05)
        fpfh_array = pcd_fpfh.data
        fpfh_mean = np.mean(fpfh_array, axis = 1)
        feature_list.extend([x, y, z])
        feature_list.extend(fpfh_mean.tolist())
        return x, y, z, fpfh_mean

    def first_regist(pcd_arrays, feature_arrays):
        global gallary_df
        global index1
        global columns1
        gallary_df = pd.DataFrame(index=index1, columns=columns1)
        cluster_pcd = o3d.geometry.PointCloud()
        for i in range(len(feature_arrays)):
            x, y, z, fpfh_mean = feature_arrays[i]
            id = gallary_df[gallary_df.isna().any(axis=1)].iloc[0].name
            print("register data:", id)
            gallary_df.loc[id] = [x, y, x, y, z, fpfh_mean, time.time()]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pcd_arrays[i])
            pcd.paint_uniform_color(color_list[index1.index(id)])
            cluster_pcd += pcd
        return cluster_pcd
    
    def normal_regist(pcd_arrays, feature_arrays):
        global gallary_df
        global index1
        current_val = CURRENT_POS_WEIGHT
        pred_val = PRED_POS_WEIGHT
        
        gallary_df_copy = gallary_df.copy()
        gallary_df_copy = gallary_df_copy.dropna(how="all", axis=0)
        gallary_df_copy_id_list = gallary_df_copy.index.values

        gallary_data = gallary_df_copy.to_numpy()
        
        if len(gallary_data) == 0:
            return first_regist(pcd_arrays, feature_arrays)

        else:
            exist_gallary_df = pd.DataFrame(gallary_df_copy, columns=gallary_df_copy_id_list)
            objective_array = []
            print(len(gallary_data))
            for i in range(len(feature_arrays)):
                tmp = []
                for j in range(len(gallary_data)):
                    objective = (((gallary_data[j][0] - feature_arrays[i][0]) ** 2 + (gallary_data[j][1] - feature_arrays[i][1]) ** 2 + (gallary_data[j][4] - feature_arrays[i][2]) ** 2) ** 0.5) * current_val + (((gallary_data[j][2] - feature_arrays[i][0]) ** 2 + (gallary_data[j][3] - feature_arrays[i][1]) ** 2 + (gallary_data[j][4] - feature_arrays[i][2]) ** 2) ** 0.5) * pred_val
                    tmp.append(objective)
                objective_array.append(tmp)
                
            objective_df = pd.DataFrame(objective_array, columns=gallary_df_copy_id_list)
            
            cluster_pcd = o3d.geometry.PointCloud()
            for i in range(len(objective_df)):
                try:
                    min_value, min_row, min_column, objective_df = match_id(objective_df)
                    print(min_value, min_row, min_column)
                    if min_value > TRACKING_MIN_DISTANCE:
                        break
                    else:
                        id = min_column
                        x, y, z, fpfh_mean = feature_arrays[min_row]
                        gallary_df.loc[id] = [x, y, 2 * x - gallary_df.loc[id]["x"], 2 * y - gallary_df.loc[id]["y"], z, fpfh_mean, time.time()]
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(pcd_arrays[min_row])
                        pcd.paint_uniform_color(color_list[index1.index(id)])
                        cluster_pcd += pcd
                except:
                    break
            
            if len(objective_df) > 0:
                new_id_list = objective_df.index.values
                tmp_gallary_data = gallary_df.to_numpy()
                for i in new_id_list:
                    x, y, z, fpfh_mean = feature_arrays[i]
                    
                    distances = np.sum((tmp_gallary_data[:, :2] - np.array([x, y]))**2, axis=1) ** 0.5
                    min_distance_gallary = np.nanmin(distances)
                    print('caulurate distance')

                    if min_distance_gallary <= 0.5:
                        print('ignore the cluster')
                        print(min_distance_gallary)
                        continue
                    
                    id = gallary_df[gallary_df.isna().any(axis=1)].iloc[0].name
                    gallary_df.loc[id] = [x, y, x, y, z, fpfh_mean, time.time()]
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(pcd_arrays[i])
                    pcd.paint_uniform_color(color_list[index1.index(id)])
                    cluster_pcd += pcd
            
            diffuse_gallary_id_list = objective_df.columns.values

            for i in diffuse_gallary_id_list:
                if time.time() - gallary_df['update'][i] >= GALLARY_RETENTION_PERIOD:
                    gallary_df.iloc[i] = np.nan
                    
                    print('delete missing data')

            return cluster_pcd
    
    def match_id(df):
        min_value = df.min().min()
        min_column = df.min().idxmin()
        min_row = df[min_column].idxmin()
        if min_value <= 1:
            df = df.drop(index=min_row, columns=min_column)
        return min_value, min_row, min_column, df
    
    cluster_data_points = []
    cluster_data_colors = []
    cluster_data_num = []

    for path in data_path_list:
        pcd_numpy = cluster(path)
        
        feature_arrays = []
        cluster_pcd = o3d.geometry.PointCloud()
        
        
        
        if len(pcd_numpy) != 0:
            for human_pcd_data in pcd_numpy:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(human_pcd_data)
                feature_arrays.append(extract_feature(pcd))


            cluster_pcd = normal_regist(list(pcd_numpy), feature_arrays)

            launch_flag = 1
            cluster_pcd = cluster_pcd.voxel_down_sample(voxel_size=0.1)

        else:
            launch_flag = 0
            extract_time = 0
            regist_time = 0
        
        
        tmp_cluster_points = np.asarray(cluster_pcd.points, dtype='float32')
        tmp_cluster_colors = np.asarray(cluster_pcd.colors, dtype='float32')
        
        if len(tmp_cluster_points) == 0:
            cluster_data_points.append([])
            cluster_data_colors.append([])
            cluster_data_num.append(0)
            continue
        
        cluster_data_points.append(np.concatenate(tmp_cluster_points, axis=0))
        cluster_data_colors.append(np.concatenate(tmp_cluster_colors, axis=0))
        size_cluster_data_points = len(np.concatenate(tmp_cluster_points, axis=0))
        cluster_data_num.append(size_cluster_data_points)
    
    
    cluster_data_points = np.concatenate(cluster_data_points, axis=0)
    cluster_data_colors = np.concatenate(cluster_data_colors, axis=0)
    
    cluster_data_points = np.asarray(cluster_data_points, dtype='float32')
    cluster_data_colors = np.asarray(cluster_data_colors, dtype='float32')
    cluster_data_num = np.asarray(cluster_data_num, dtype='int32')
    
    
    send_data_points = _encode(cluster_data_points.tobytes("C"))
    send_data_colors = _encode(cluster_data_colors.tobytes("C"))
    send_data_num = _encode(cluster_data_num.tobytes("C"))

    SOURCE_PCD.paint_uniform_color([1, 1, 1])

    base_pcd_np = np.asarray(SOURCE_PCD.points, dtype='float32')
    base_colors_np = np.asarray(SOURCE_PCD.colors, dtype='float32')
    send_base_points = _encode(np.concatenate(base_pcd_np, axis=0).tobytes("C"))
    send_base_colors = _encode(np.concatenate(base_colors_np, axis=0).tobytes("C"))


    return JSONResponse({"status": "success", "data": config, "cluster_data_points": send_data_points, "cluster_data_colors": send_data_colors, "cluster_data_num": send_data_num, "base_data_points": send_base_points, "base_data_colors": send_base_colors})
