import cv2
import os
import requests
import numpy as np
import os.path
from tqdm import tqdm
from datetime import datetime

TILE_SIZE = 256
file_dir = os.path.dirname(__file__)
default_prefs = {
    'url': 'https://khms0.google.com/kh/v=932?x={x}&y={y}&z={z}',
    'dir': os.path.join(file_dir, 'images'),
    'region_ul': '',
    'region_br': '',
    'zoom': '',
    'headers': {
        'cache-control': 'max-age=0',
        'sec-ch-ua': '" Not A;Brand";v="99", "Chromium";v="99", "Google Chrome";v="99"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
        'sec-fetch-dest': 'document',
        'sec-fetch-mode': 'navigate',
        'sec-fetch-site': 'none',
        'sec-fetch-user': '?1',
        'upgrade-insecure-requests': '1',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.82 Safari/537.36'
    }
}

def download_tile(url, headers):
    response = requests.get(url, headers=headers)
    arr =  np.asarray(bytearray(response.content), dtype=np.uint8)
    return cv2.imdecode(arr, -1)

# wxl write，使用全局x,y及zoom获得GPS 
def get_real_lonlat(x, y, scale):
    lon_rel = (x/TILE_SIZE/scale-0.5)*360
    temp = np.exp((0.5-y/TILE_SIZE/scale)*4*np.pi)
    lat_rel = np.arcsin((temp-1)/(temp+1))/np.pi*180
    return lat_rel, lon_rel

# Mercator projection 
# https://developers.google.com/maps/documentation/javascript/examples/map-coordinates
def project_with_scale(lat, lon, scale):
    siny = np.sin(lat * np.pi / 180)
    siny = min(max(siny, -0.9999), 0.9999)
    x = scale * (0.5 + lon / 360)
    y = scale * (0.5 - np.log((1 + siny) / (1 - siny)) / (4 * np.pi))
    return x, y


# (lat1, lon1) is the upper left corner of the region
# (lat2, lon2) is the bottom right corner of the region
# url should be a string with {x}, {y} and {z} in place of the tile coordinates and zoom
def generate_image(lat1: float, lon1: float, lat2: float, lon2: float, zoom: int, url: str, headers: str, save_dir: str):
    zoom = int(zoom)
    scale = 1 << zoom

    # Finding the pixel and tile coordinates of the region
    ul_proj_x, ul_proj_y = project_with_scale(lat1, lon1, scale)
    br_proj_x, br_proj_y = project_with_scale(lat2, lon2, scale)

    ul_pixel_x = int(ul_proj_x * TILE_SIZE)
    ul_pixel_y = int(ul_proj_y * TILE_SIZE)
    br_pixel_x = int(br_proj_x * TILE_SIZE)
    br_pixel_y = int(br_proj_y * TILE_SIZE)

    ul_tile_x = int(ul_proj_x)
    ul_tile_y = int(ul_proj_y)
    br_tile_x = int(br_proj_x)
    br_tile_y = int(br_proj_y)

    # Creating the output image
    img_w = abs(ul_pixel_x - br_pixel_x)
    img_h = br_pixel_y - ul_pixel_y
    img = np.ndarray((img_h, img_w, 3), np.uint8)

    lat1_rel,lon1_rel = get_real_lonlat(ul_pixel_x,ul_pixel_y,scale)
    lat2_rel,lon2_rel = get_real_lonlat(br_pixel_x,br_pixel_y,scale)
    print(lat1_rel,lon1_rel, lat2_rel,lon2_rel)

    for i in tqdm(range(ul_tile_y, br_tile_y + 1)):
        for j in range(ul_tile_x, br_tile_x + 1):
            tile = download_tile(url.format(x=j, y=i, z=zoom), headers)

            # Finding the coordinates of the new tile relative to the output image
            ul_rel_x = j * TILE_SIZE - ul_pixel_x
            ul_rel_y = i * TILE_SIZE - ul_pixel_y
            br_rel_x = ul_rel_x + TILE_SIZE
            br_rel_y = ul_rel_y + TILE_SIZE

            # Defining the part of the otuput image where the tile will be placed
            i_x_l = max(0, ul_rel_x)
            i_x_r = min(img_w + 1, br_rel_x)
            i_y_l = max(0, ul_rel_y)
            i_y_r = min(img_h + 1, br_rel_y)

            # Defining how the tile will be cropped in case it is a border tile
            t_x_l = max(0, -ul_rel_x)
            t_x_r = TILE_SIZE + min(0, img_w - br_rel_x)
            t_y_l = max(0, -ul_rel_y)
            t_y_r = TILE_SIZE + min(0, img_h - br_rel_y)

            # Placing the tile
            img[i_y_l:i_y_r, i_x_l:i_x_r] = tile[t_y_l:t_y_r, t_x_l:t_x_r]

    # Saving the image
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    cv2.imwrite(os.path.join(save_dir, f'img_{timestamp}.png'), img)

#  "region_ul": "39.74818627986782, -86.2691883721032", 
#   "region_br": "39.7466638708528, -86.26720831322838",


# (lat1, lon1) is the upper left corner of the region
# (lat2, lon2) is the bottom right corner of the region
# url should be a string with {x}, {y} and {z} in place of the tile coordinates and zoom
def generate_image_return(lat1: float, lon1: float, lat2: float, lon2: float, zoom: int, url: str, headers: str):
    zoom = int(zoom)
    scale = 1 << zoom

    # Finding the pixel and tile coordinates of the region
    ul_proj_x, ul_proj_y = project_with_scale(lat1, lon1, scale)
    br_proj_x, br_proj_y = project_with_scale(lat2, lon2, scale)

    ul_pixel_x = int(ul_proj_x * TILE_SIZE)
    ul_pixel_y = int(ul_proj_y * TILE_SIZE)
    br_pixel_x = int(br_proj_x * TILE_SIZE)
    br_pixel_y = int(br_proj_y * TILE_SIZE)

    ul_tile_x = int(ul_proj_x)
    ul_tile_y = int(ul_proj_y)
    br_tile_x = int(br_proj_x)
    br_tile_y = int(br_proj_y)

    ul_pixel_x = int(ul_tile_x * TILE_SIZE)
    ul_pixel_y = int(ul_tile_y * TILE_SIZE)
    br_pixel_x = int(br_tile_x * TILE_SIZE) + TILE_SIZE
    br_pixel_y = int(br_tile_y * TILE_SIZE) + TILE_SIZE

    # Creating the output image
    img_w = abs(ul_pixel_x - br_pixel_x)
    img_h = br_pixel_y - ul_pixel_y
    img = np.ndarray((img_h, img_w, 3), np.uint8)

    lat1_rel,lon1_rel = get_real_lonlat(ul_pixel_x,ul_pixel_y,scale)
    lat2_rel,lon2_rel = get_real_lonlat(br_pixel_x,br_pixel_y,scale)
    print(lat1_rel,lon1_rel, lat2_rel,lon2_rel)

    for i in tqdm(range(ul_tile_y, br_tile_y + 1)):
        for j in range(ul_tile_x, br_tile_x + 1):
            tile = download_tile(url.format(x=j, y=i, z=zoom), headers)

            # Finding the coordinates of the new tile relative to the output image
            ul_rel_x = j * TILE_SIZE - ul_pixel_x
            ul_rel_y = i * TILE_SIZE - ul_pixel_y
            br_rel_x = ul_rel_x + TILE_SIZE
            br_rel_y = ul_rel_y + TILE_SIZE

            # Defining the part of the otuput image where the tile will be placed
            i_x_l = max(0, ul_rel_x)
            i_x_r = min(img_w + 1, br_rel_x)
            i_y_l = max(0, ul_rel_y)
            i_y_r = min(img_h + 1, br_rel_y)

            # Defining how the tile will be cropped in case it is a border tile
            t_x_l = max(0, -ul_rel_x)
            t_x_r = TILE_SIZE + min(0, img_w - br_rel_x)
            t_y_l = max(0, -ul_rel_y)
            t_y_r = TILE_SIZE + min(0, img_h - br_rel_y)

            # Placing the tile
            img[i_y_l:i_y_r, i_x_l:i_x_r] = tile[t_y_l:t_y_r, t_x_l:t_x_r]

    # Saving the image
    return img, [lat1_rel,lon1_rel, lat2_rel,lon2_rel]
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    cv2.imwrite(os.path.join(save_dir, f'img_{timestamp}.png'), img)

#  "region_ul": "39.74818627986782, -86.2691883721032", 
#   "region_br": "39.7466638708528, -86.26720831322838",

# 根据当前卫星图GPS及像素坐标获得某像素的GPS
def get_loc_lonlat(lat1, lon1, u, v, zoom):
    scale = 1<<zoom
    ul_proj_x, ul_proj_y = project_with_scale(lat1, lon1, scale)
    ul_tile_x = int(ul_proj_x)
    ul_tile_y = int(ul_proj_y)

    ul_pixel_x = int(ul_tile_x * TILE_SIZE)
    ul_pixel_y = int(ul_tile_y * TILE_SIZE)

    lat1_rel,lon1_rel = get_real_lonlat(ul_pixel_x + u,ul_pixel_y+v,scale)
    return lat1_rel,lon1_rel

# 根据某像素位置的GPS获得其在某张卫星图上的像素坐标
def get_rel_loc(lat1, lon1, lat, lon, zoom):
    scale = 1<<zoom
    ul_proj_x, ul_proj_y = project_with_scale(lat1, lon1, scale)
    ul_tile_x = int(ul_proj_x)
    ul_tile_y = int(ul_proj_y)

    ul_pixel_x = int(ul_tile_x * TILE_SIZE)
    ul_pixel_y = int(ul_tile_y * TILE_SIZE)

    br_proj_x, br_proj_y = project_with_scale(lat, lon, scale)

    br_pixel_x = int(br_proj_x * TILE_SIZE)
    br_pixel_y = int(br_proj_y * TILE_SIZE)

    return br_pixel_x-ul_pixel_x,br_pixel_y-ul_pixel_y
