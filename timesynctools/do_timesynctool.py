import os
import geopandas as gpd
from multiprocessing import Pool
import gdal
from os import listdir

#from timesynctools.force import extract_chips, pixel_grid
#from timesynctools import utils
#from timesynctools.force import extract_chips, pixel_grid, plot_file

from timesync_force import extract_chips, pixel_grid, extract_plot
from utils import coord_snap, plot_file, read_list

if __name__ == "__main__":

    # chip_polygon = r'l:\Brandsat\timesync\test_chip2.gpkg'
    # dst_pixelsize = 30.0
    # ws = 255
    # coordinate = (4526308, 3312467)
    # coordinate = (4539015,3300038)

    cores = 10
    project_id = 9004
    dst_ref = pixel_grid(r'X:\\dc\\deu\\ard\\X0068_Y0041\\20000813_LEVEL2_LND05_BOA.tif')
    path = r'p:\timesync'
    tsync_path = r'A:\timesync'
    force_path = r'x:\dc\deu\ard'
    gis_path = r'p:\timesync\bb\gis'

    # create plot file and boundary file from shapefile
    plotFile = plot_file(r'p:\timesync\bb\gis\tsync_wind_2_2.gpkg', project_id=project_id,
                               idfield='id', dst_ref=dst_ref, refIsCenter=False)

    cores = 10
    points = gpd.read_file(r'p:\timesync\bb\gis\tsync_wind_2_2_tsync.gpkg', layer='center')
    force_root = r'x:\dc\deu\ard'
    tile_grid = r'p:\timesync\bb\gis\datacube-grid_DEU_10km.gpkg'
    chip_path = r'A:\timesync\tschips'
    spectra_path = r'A:\timesync\spectra'


    pool = Pool(processes=cores)
    for i, v in points.iterrows():
        coord = (v.geometry.x, v.geometry.y)
        point_id = v.get("plotid")
    plotFile = plot_file(os.path.join(gis_path, 'tsync_wind_2_2.gpkg'), project_id=project_id,
                         idfield='id', dst_ref=dst_ref, refIsCenter=False, overwrite = True)

        pool.apply_async(extract_chips, (force_root, chip_path, spectra_path),
                         kwds={'project_id': project_id,
                               'coordinate': coord,
                               'point_id': point_id,
                               'tile_grid': tile_grid,
                               'dst_ref': dst_ref,
                               'tc':True, 'b743':True, 'b432':False})
    points = gpd.read_file(os.path.join(gis_path, 'tsync_wind_2_2_tsync.gpkg'), layer='center')
    tile_grid_file = os.path.join(gis_path, 'datacube-grid_DEU_10km.gpkg')

    pool.close()
    pool.join()
    extract_chips(points, force_path=force_path, tsync_path=tsync_path, project_id=project_id, dst_ref=dst_ref,
                  tile_grid_file=tile_grid_file, cores=cores)
