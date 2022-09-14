import os
import numpy as np
import fnmatch
import geopandas as gpd
from multiprocessing import Pool
from osgeo import gdal_array
from itertools import groupby
from datetime import date
from shapely.geometry import Polygon
from math import floor
from imageio import imwrite as imsave
from osgeo import gdal, ogr


def coord_snap(coord, ref_coord, pixel_size=30.0, refIsCenter=False, outputCenter=True):

    # returns coordinate snapped to the center coordinate of the pixel
    if refIsCenter:
        ref_coord_corner = (ref_coord[0] - (pixel_size/2.0), ref_coord[1] + (pixel_size/2.0))
    else:
        ref_coord_corner = ref_coord

    fx = floor((coord[0] - ref_coord_corner[0]) / pixel_size)
    fy = floor((ref_coord_corner[1] - coord[1]) / pixel_size)

    x = ref_coord_corner[0] + fx * pixel_size
    y = ref_coord_corner[1] - fy * pixel_size

    if outputCenter:
        x += pixel_size/2.0
        y -= pixel_size/2.0

    return x, y


def which_tiles(tile_grid, coordinate, w, chip_polygon=None):
    tiles = gpd.read_file(tile_grid)

    x, y = coordinate

    # w is half width
    x_point_list = [x - w, x - w, x + w, x + w, x - w]
    y_point_list = [y - w, y + w, y + w, y - w, y - w]

    polygon_geom = Polygon(zip(x_point_list, y_point_list))
    polygon = gpd.GeoDataFrame(index=[0], crs=tiles.crs, geometry=[polygon_geom])

    if chip_polygon is not None:
        polygon.to_file(chip_polygon, layer='chip', driver="GPKG")

    inter = gpd.overlay(tiles, polygon, how='intersection')

    return list(inter["Tile_ID"])


def file_search(path, pattern='*', directory=False, full_names=True):

    if directory:
        rlist = [dirpath for dirpath, dirnames, files in os.walk(path)
                 if fnmatch.fnmatch(os.path.basename(dirpath), pattern)]
    else:
        rlist = [os.path.join(dirpath, f)
                 for dirpath, dirnames, files in os.walk(path)
                 for f in fnmatch.filter(files, pattern)]

    if not full_names:
        rlist = [os.path.basename(x) for x in rlist]

    return rlist


def bytescale(data, cmin=None, cmax=None, high=255, low=0):
    """
    Byte scales an array (image).

    Byte scaling means converting the input image to uint8 dtype and scaling
    the range to ``(low, high)`` (default 0-255).
    If the input image already has dtype uint8, no scaling is done.

    Parameters
    ----------
    data : ndarray
        PIL image data array.
    cmin : scalar, optional
        Bias scaling of small values. Default is ``data.min()``.
    cmax : scalar, optional
        Bias scaling of large values. Default is ``data.max()``.
    high : scalar, optional
        Scale max value to `high`.  Default is 255.
    low : scalar, optional
        Scale min value to `low`.  Default is 0.

    Returns
    -------
    img_array : uint8 ndarray
        The byte-scaled array.

    Examples
    --------
    >>> img = array([[ 91.06794177,   3.39058326,  84.4221549 ],
                     [ 73.88003259,  80.91433048,   4.88878881],
                     [ 51.53875334,  34.45808177,  27.5873488 ]])
    >>> bytescale(img)
    array([[255,   0, 236],
           [205, 225,   4],
           [140,  90,  70]], dtype=uint8)
    >>> bytescale(img, high=200, low=100)
    array([[200, 100, 192],
           [180, 188, 102],
           [155, 135, 128]], dtype=uint8)
    >>> bytescale(img, cmin=0, cmax=255)
    array([[91,  3, 84],
           [74, 81,  5],
           [52, 34, 28]], dtype=uint8)

    """
    if data.dtype == np.uint8:
        return data

    if high < low:
        raise ValueError("`high` should be larger than `low`.")

    if cmin is None:
        cmin = data.min()
    if cmax is None:
        cmax = data.max()

    cscale = cmax - cmin
    if cscale < 0:
        raise ValueError("`cmax` should be larger than `cmin`.")
    elif cscale == 0:
        cscale = 1

    scale = float(high - low) / cscale
    bytedata = (data * 1.0 - cmin) * scale + 0.4999
    bytedata[bytedata > high] = high
    bytedata[bytedata < 0] = 0
    return np.cast[np.uint8](bytedata) + np.cast[np.uint8](low)


def plot_file(pts_file, dst_ref=None, dst_pixelsize=30.0, idfield='plotid', refIsCenter=False,
              project_id=1, overwrite=False):

    out_box_file = pts_file.replace(os.path.splitext(pts_file)[1], '_tsync.gpkg')
    out_csv_file = pts_file.replace(os.path.splitext(pts_file)[1], '_tsync.csv')

    if os.path.isfile(out_box_file) and overwrite is False:
        print("%s exists. Use overwrite keyword." % out_box_file)
        return out_csv_file

    if dst_ref is None:
        dst_ref = (0.0, 0.0)

    f_out = open(out_csv_file, "w")
    f_out.writelines('project_id,plotid,x,y\n')

    # open shapefiles
    pts_dst = ogr.Open(pts_file)

    # create the output mem layers
    mem_driver = ogr.GetDriverByName('Memory')
    mem_pts_dst = mem_driver.CreateDataSource('inpts')
    out_box_dst = mem_driver.CreateDataSource('outbox')
    out_pts_dst = mem_driver.CreateDataSource('outpts')

    # pts_lyr = pts_dst.GetLayer()
    pts_lyr = mem_pts_dst.CopyLayer(pts_dst.GetLayer(0), 'plotid')
    pts_srs = pts_lyr.GetSpatialRef()

    out_box_lyr = out_box_dst.CreateLayer("plot boundary", geom_type=ogr.wkbPolygon, srs=pts_srs)
    out_pts_lyr = out_pts_dst.CreateLayer("plot center", geom_type=ogr.wkbPoint, srs=pts_srs)

    # add fields
    out_pts_lyr.CreateField(ogr.FieldDefn('project_id', ogr.OFTInteger))
    out_pts_lyr.CreateField(ogr.FieldDefn('plotid', ogr.OFTInteger))
    out_pts_lyr.CreateField(ogr.FieldDefn('xraw', ogr.OFTInteger))
    out_pts_lyr.CreateField(ogr.FieldDefn('yraw', ogr.OFTInteger))
    out_pts_lyr.CreateField(ogr.FieldDefn('x', ogr.OFTInteger))
    out_pts_lyr.CreateField(ogr.FieldDefn('y', ogr.OFTInteger))

    out_box_lyr.CreateField(ogr.FieldDefn('project_id', ogr.OFTInteger))
    out_box_lyr.CreateField(ogr.FieldDefn('plotid', ogr.OFTInteger))

    # get the output layer's feature definition
    out_box_LayerDefn = out_box_lyr.GetLayerDefn()
    out_pts_LayerDefn = out_pts_lyr.GetLayerDefn()

    k = dst_pixelsize / 2.0

    # loop through the input features
    #pts_feat = pts_lyr.GetNextFeature()
    nones = []
    for pts_feat in pts_lyr:

        loc_feat = pts_feat.Clone()
        loc_geom = loc_feat.GetGeometryRef()

        plotid = loc_feat.GetField(idfield)

        if plotid % 100 == 0:
            print('Outlining %s' % plotid)

        # get the input geometry
        pts_geom = pts_feat.GetGeometryRef()

        random_x = pts_geom.GetX()
        random_y = pts_geom.GetY()

        x, y = coord_snap([pts_geom.GetX(), pts_geom.GetY()], dst_ref, 30.0, refIsCenter=refIsCenter)

        # create center geometry
        out_pts_geom = ogr.Geometry(ogr.wkbPoint)
        out_pts_geom.AddPoint(x, y)

        # Create center point feature
        out_pts_feat = ogr.Feature(out_pts_LayerDefn)
        # set the geometry and attribute
        out_pts_feat.SetGeometry(out_pts_geom)
        out_pts_feat.SetField('project_id', project_id)
        out_pts_feat.SetField('plotid', pts_feat.GetField(idfield))
        out_pts_feat.SetField('x', x)
        out_pts_feat.SetField('y', y)
        out_pts_feat.SetField('xraw', random_x)
        out_pts_feat.SetField('yraw', random_y)
        # add the feature to the shapefile
        out_pts_lyr.CreateFeature(out_pts_feat)

        # Create boundary geometry
        boundary = ogr.Geometry(ogr.wkbLinearRing)
        boundary.AddPoint(x - k, y + k)
        boundary.AddPoint(x + k, y + k)
        boundary.AddPoint(x + k, y - k)
        boundary.AddPoint(x - k, y - k)
        boundary.AddPoint(x - k, y + k)
        poly_geom = ogr.Geometry(ogr.wkbPolygon)
        poly_geom.AddGeometry(boundary)

        # Create boundary feature
        out_box_feat = ogr.Feature(out_box_LayerDefn)
        # set the geometry and attribute
        out_box_feat.SetGeometry(poly_geom)
        out_box_feat.SetField('project_id', project_id)
        out_box_feat.SetField('plotid', pts_feat.GetField(idfield))
        out_box_lyr.CreateFeature(out_box_feat)

        # write to csv file
        line = '%s,%s,%s,%s\n' % (project_id, pts_feat.GetField(idfield), x, y)
        f_out.writelines(line)


        # destroy the features and get the next input feature
        out_box_feat = None
        out_pts_feat = None
        #pts_feat = pts_lyr.GetNextFeature()

    f_out.close()
    # create the output layer
    driver = ogr.GetDriverByName('GPKG')

    if os.path.exists(out_box_file):
        driver.DeleteDataSource(out_box_file)

    shp_box_dst = driver.CreateDataSource(out_box_file)
    shp_box_dst.CopyLayer(out_box_dst.GetLayer('plot boundary'), 'boundary')
    shp_box_dst.CopyLayer(out_pts_dst.GetLayer('plot center'), 'center')

    return out_csv_file


def readImage(file_name, subset=None, band=None, expand=True):

    src_ds = gdal.Open(file_name)
    src_band = src_ds.GetRasterBand(1)

    if band is None:
        bands = list(range(1, src_ds.RasterCount + 1))
    else:
        if type(band) is not list:
            bands = list(band)
        else:
            bands = band

    gt = src_ds.GetGeoTransform()
    gtInverse = gdal.InvGeoTransform(gt)

    (o_ulx, o_uly), (o_lrx, o_lry) = subset

    # dst_ref = gt[0] % gt[1], gt[3] % -gt[5]
    # o_ulx, o_uly = coord_snap((o_ulx, o_uly), dst_ref, gt[1], refIsCenter=False)
    # o_lrx, o_lry = coord_snap((o_lrx, o_lry), dst_ref, gt[1], refIsCenter=False)

    ulx, uly = gdal.ApplyGeoTransform(gtInverse, o_ulx, o_uly)
    lrx, lry = gdal.ApplyGeoTransform(gtInverse, o_lrx, o_lry)

    if ulx >= src_ds.RasterXSize:
        print("Subset does not match image")
        return

    if uly >= src_ds.RasterYSize:
        print("Subset does not match image")
        return

    if lrx <= 0:
        print("Subset does not match image")
        return

    if lry <= 0:
        print("Subset does not match image")
        return

    # crop file coordinates to match image dimensions
    a_ulx = max(ulx, 0)
    a_uly = max(uly, 0)
    a_lrx = min(lrx, src_ds.RasterXSize)
    a_lry = min(lry, src_ds.RasterYSize)

    o_ulx, o_uly = gdal.ApplyGeoTransform(gt, a_ulx, a_uly)

    # calc file coordinates for numpy array
    b_ulx = round(a_ulx - ulx)
    b_uly = round(a_uly - uly)
    nx = b_ulx + round(a_lrx - a_ulx)
    ny = b_uly + round(a_lry - a_uly)

    # Now, we create an in-memory raster
    mem_drv = gdal.GetDriverByName('MEM')

    # The size of the raster is given the new projection and pixel spacing
    # Using the values we calculated above. Also, setting it to store one band
    if expand:
        out_y_dim = round(lry - uly)
        out_x_dim = round(lrx - ulx)
    else:
        out_y_dim = round(a_lry - a_uly)
        out_x_dim = round(a_lrx - a_ulx)

    dst_ds = mem_drv.Create('', out_x_dim, out_y_dim, len(bands), src_band.DataType)

    # Set the geotransform and projection
    dst_gt = (o_ulx, gt[1], gt[2], o_uly, gt[4], gt[5])
    dst_ds.SetGeoTransform(dst_gt)
    dst_ds.SetProjection(src_ds.GetProjectionRef())

    for b in bands:

        i_band = src_ds.GetRasterBand(b)

        arr = i_band.ReadAsArray(round(a_ulx), round(a_uly), round(a_lrx - a_ulx), round(a_lry - a_uly))

        if expand:
            out_arr = np.full((round(lry - uly), round(lrx - ulx)),
                              fill_value=src_band.GetNoDataValue(),
                              dtype=gdal_array.GDALTypeCodeToNumericTypeCode(src_band.DataType))
            out_arr[b_uly:ny, b_ulx:nx] = arr
        else:
            out_arr = arr

        o_band = dst_ds.GetRasterBand(b)
        o_band.WriteArray(out_arr)
        o_band.SetNoDataValue(i_band.GetNoDataValue())
        o_band = None

    return dst_ds


def pixel_grid(file_name):

    src_ds = gdal.Open(file_name)
    gt = src_ds.GetGeoTransform()

    return gt[0] % gt[1], gt[3] % -gt[5]


def thumbnail(imchip, rgb='bgw', out_png_file=None, nodata_value=-9999):

    valid = np.max(imchip != nodata_value, 2)

    if rgb == 'bgw':

        imgstretch = np.array([[  604, 5592, 1],
                               [   49, 3147, 1],
                               [-2245,  843, 1]])
        cmin = imgstretch[:, 0] * imgstretch[:, 2]
        cmax = imgstretch[:, 1] * imgstretch[:, 2]

        tccoeff = np.array([[0.2043, 0.4158, 0.5524, 0.5741, 0.3124, 0.2303],
                            [-0.1603, -0.2819, -0.4934, 0.7940, -0.0002, -0.1446],
                            [0.0315, 0.2021, 0.3102, 0.1594, -0.6806, -0.6109]], dtype=np.float32)
        tccoeff = np.transpose(tccoeff, [1, 0])

        chip_long = imchip.reshape(imchip.shape[0] * imchip.shape[1], imchip.shape[2])
        tcchip = np.dot(chip_long, tccoeff).reshape((imchip.shape[0], imchip.shape[1], 3))

        bytarr = np.zeros((imchip.shape[0], imchip.shape[1], 3), dtype=np.uint8)
        bytarr[:, :, 0] = bytescale(tcchip[:, :, 0], cmin=cmin[0], cmax=cmax[0]) * valid
        bytarr[:, :, 1] = bytescale(tcchip[:, :, 1], cmin=cmin[1], cmax=cmax[1]) * valid
        bytarr[:, :, 2] = bytescale(tcchip[:, :, 2], cmin=cmin[2], cmax=cmax[2]) * valid
        imsave(out_png_file, bytarr)

    elif rgb == '743':

        imgstretch = np.array([[   0,    0, 1],
                               [  50, 1150, 1],
                               [-300, 2500, 1],
                               [ 151, 4951, 1],
                               [   0,    0, 1],
                               [-904, 3696, 1]])
        cmin = imgstretch[:, 0] * imgstretch[:, 2]
        cmax = imgstretch[:, 1] * imgstretch[:, 2]
        l = [5, 3, 2]

        bytarr = np.zeros((imchip.shape[0], imchip.shape[1], 3), dtype=np.uint8)
        bytarr[:, :, 0] = bytescale(imchip[:, :, l[0]], cmin=cmin[l[0]], cmax=cmax[l[0]])
        bytarr[:, :, 1] = bytescale(imchip[:, :, l[1]], cmin=cmin[l[1]], cmax=cmax[l[1]])
        bytarr[:, :, 2] = bytescale(imchip[:, :, l[2]], cmin=cmin[l[2]], cmax=cmax[l[2]])
        imsave(out_png_file, bytarr)

    elif rgb == '432':

        imgstretch = np.array([[   0,    0, 1],
                               [  50, 1150, 1],
                               [-300, 2500, 1],
                               [ 151, 4951, 1],
                               [   0,    0, 1],
                               [-904, 3696, 1]])
        l = [3, 2, 1]
        cmin = imgstretch[:, 0] * imgstretch[:, 2]
        cmax = imgstretch[:, 1] * imgstretch[:, 2]

        bytarr = np.zeros((imchip.shape[0], imchip.shape[1], 3), dtype=np.uint8)
        bytarr[:, :, 0] = bytescale(imchip[:, :, l[0]], cmin=cmin[l[0]], cmax=cmax[l[0]])
        bytarr[:, :, 1] = bytescale(imchip[:, :, l[1]], cmin=cmin[l[1]], cmax=cmax[l[1]])
        bytarr[:, :, 2] = bytescale(imchip[:, :, l[2]], cmin=cmin[l[2]], cmax=cmax[l[2]])
        imsave(out_png_file, bytarr)
    else:
        print('Unknown rgb combination.')
        return
"""
extract_plot(coord, force_path=force_path, tsync_path=tsync_path, project_id=project_id, point_id=point_id,
                         tile_grid=tile_grid, dst_ref=dst_ref, tc=True, b743=True, b432=False)
coordinate = coord
"""
def extract_plot(coordinate, force_path=None, tsync_path=None,
                 project_id=None, point_id=None, tile_grid=None, thumbnail_size=255,
                 dst_ref=None, pixel_size=30.0, tc=True, b743=True, b432=False):

    chip_path = os.path.join(tsync_path, 'tschips', 'prj_%s' % project_id)
    spectra_path = os.path.join(tsync_path, 'spectra', 'prj_%s' % project_id)

    if not os.path.exists(chip_path):
        os.mkdir(chip_path)

    if not os.path.exists(spectra_path):
        os.mkdir(spectra_path)

    if tc:
        if not os.path.exists(os.path.join(chip_path, 'tc')):
            os.mkdir(os.path.join(chip_path, 'tc'))
        chip_path_tc = os.path.join(chip_path, 'tc', 'plot_%s' % point_id)
        if not os.path.exists(chip_path_tc):
            os.mkdir(chip_path_tc)

    if b743:
        if not os.path.exists(os.path.join(chip_path, 'b743')):
            os.mkdir(os.path.join(chip_path, 'b743'))
        chip_path_743 = os.path.join(chip_path, 'b743', 'plot_%s' % point_id)
        if not os.path.exists(chip_path_743):
            os.mkdir(chip_path_743)

    if b432:
        if not os.path.exists(os.path.join(chip_path, 'b432')):
            os.mkdir(os.path.join(chip_path, 'b432'))
        chip_path_432 = os.path.join(chip_path, 'b432', 'plot_%s' % point_id)
        if not os.path.exists(chip_path_432):
            os.mkdir(chip_path_432)

    spectra_file = os.path.join(spectra_path, "plot_%s_spectra.csv" % point_id)
    f_out = open(spectra_file, "w")

    if dst_ref is None:
        x, y = coordinate
    else:
        x, y = coord_snap(coordinate, dst_ref, pixel_size, refIsCenter=False)

    w = thumbnail_size * pixel_size / 2.0  # halfwidth
    tiles = which_tiles(tile_grid, (x, y), w=w)

    boa_files = []
    for tile in tiles:
        boa_files += ((tile, fn) for fn in file_search(os.path.join(force_path, tile), '*LND*BOA.tif', full_names=False))

    # group unique images across tiles
    groups = []
    uniquekeys = []
    boa_files = sorted(boa_files, key=lambda tup: tup[1])
    for k, g in groupby(boa_files, lambda tup: tup[1]):
        img_files = [os.path.join(force_path, x[0], x[1]) for x in g]
        # groups.append(list(g))  # Store group iterator as a list
        # uniquekeys.append(k)

        # img_files = ['n:\\dc\\deu\\ard\\X0068_Y0041\\20000813_LEVEL2_LND05_BOA.tif',
        #              'n:\\dc\\deu\\ard\\X0069_Y0041\\20000813_LEVEL2_LND05_BOA.tif',
        #              'n:\\dc\\deu\\ard\\X0068_Y0042\\20000813_LEVEL2_LND05_BOA.tif',
        #              'n:\\dc\\deu\\ard\\X0069_Y0042\\20000813_LEVEL2_LND05_BOA.tif']

        # construct file name for png thumbnail
        im_year = os.path.basename(img_files[0])[0:4]
        im_month = os.path.basename(img_files[0])[4:6]
        im_day = os.path.basename(img_files[0])[6:8]
        im_doy = date(int(im_year), int(im_month), int(im_day)).timetuple().tm_yday
        im_sensor = os.path.basename(img_files[0])[16:21]

        # read first image
        img_ds = readImage(img_files[0], subset=((x - w, y + w), (x + w, y - w)), expand=True)
        img_arr = img_ds.ReadAsArray()
        qai_ds = readImage(img_files[0].replace("BOA", "QAI"), subset=((x - w, y + w), (x + w, y - w)), expand=True)
        qai_arr = qai_ds.ReadAsArray()

        # add other images if needed
        for i in range(1, len(img_files)):
            img_arr = np.maximum(img_arr, readImage(img_files[i], subset=((x - w, y + w), (x + w, y - w)),
                                                    expand=True).ReadAsArray())
            qai_file = img_files[i].replace("BOA", "QAI")
            qai_arr = np.maximum(qai_arr, readImage(qai_file, subset=((x - w, y + w), (x + w, y - w)),
                                                    expand=True).ReadAsArray())

        # out_png_file = r'l:\Brandsat\timesync\test_chip.png'
        # out_file = r'l:\Brandsat\timesync\test_chip.tif'
        # writeRaster(img_arr, out_file, src_ds=img_ds, nodata=img_ds.GetRasterBand(1).GetNoDataValue())
        # out_qai_file = r'l:\Brandsat\timesync\test_chip_qai.tif'
        # writeRaster(qai_arr, out_qai_file, src_ds=qai_ds)
        #
        # img_file = img_files[1]
        # chp_ds = readImage(img_file, subset=((x-w, y+w), (x+w, y-w)))
        # arr = chp_ds.ReadAsArray()
        # import matplotlib.pyplot as plt
        # plt.imshow(arr[0,:,:])
        # plt.show()

        spec_value = img_arr[:, round((thumbnail_size - 1) / 2), round((thumbnail_size - 1) / 2)]
        qai_value = qai_arr[round((thumbnail_size - 1) / 2), round((thumbnail_size - 1) / 2)]

        if not np.any(spec_value != img_ds.GetRasterBand(1).GetNoDataValue()):
            continue

        # print(os.path.basename(img_files[0]))

        spectra = ",".join([str(i) for i in spec_value])
        f_out.writelines('%s,%s,%s,%s,%s,%s\n' % (im_sensor, point_id, im_year, im_doy, spectra, qai_value))

        out_png_base = 'plot_%s_%s_%s.png' % (point_id, im_year, "{:03d}".format(im_doy))
        if tc:
            thumbnail(img_arr.transpose(1, 2, 0), rgb='bgw', out_png_file=os.path.join(chip_path_tc, out_png_base))
        if b743:
            thumbnail(img_arr.transpose(1, 2, 0), rgb='743', out_png_file=os.path.join(chip_path_743, out_png_base))
        if b432:
            thumbnail(img_arr.transpose(1, 2, 0), rgb='432', out_png_file=os.path.join(chip_path_432, out_png_base))


    f_out.close()
    print("Finished chips for plot: %s" % point_id)


def extract_chips(points, force_path=None, tsync_path=None, project_id=None, dst_ref=None,
                  tile_grid_file=None, cores=1):

    if cores == 0:
        print("Skipping extract chips, because cores=1.")
    elif cores == 1:
        for i, v in points.iterrows():
            print(i)
            coord = (v.geometry.x, v.geometry.y)
            point_id = v.get("plotid")
            extract_plot(coord, force_path=force_path, tsync_path=tsync_path, project_id=project_id, point_id=point_id,
                         tile_grid=tile_grid, dst_ref=dst_ref, tc=True, b743=True, b432=False)
    else:
        pool = Pool(processes=cores)
        for i, v in points.iterrows():
            print(i)
            coord = (v.geometry.x, v.geometry.y)
            point_id = v.get("plotid")

            pool.apply_async(extract_plot, (coord, ),
                             kwds={'force_path': force_path,
                                   'tsync_path': tsync_path,
                                   'project_id': project_id,
                                   'point_id': point_id,
                                   'tile_grid': tile_grid,
                                   'dst_ref': dst_ref,
                                   'tc': True, 'b743': True, 'b432': False})

        pool.close()
        pool.join()

"""
if __name__ == "__main__":

    cores = 10
    project_id = 9003
    dst_ref = pixel_grid(r'n:\\dc\\deu\\ard\\X0068_Y0041\\20000813_LEVEL2_LND05_BOA.tif')
    tsync_path = r'l:\Brandsat\timesync'
    force_path = r'n:\dc\deu\ard'
    gis_path = r'p:\Brandsat\timesync\bb\gis'

    points = gpd.read_file(os.path.join(gis_path, 'tsync_wind_1_force_tsync.gpkg'), layer='center')
    tile_grid_file = os.path.join(gis_path, 'datacube-grid_DEU_10km.gpkg')

    extract_chips(points, force_path=force_path, tsync_path=tsync_path, project_id=project_id, dst_ref=dst_ref,
                  tile_grid_file=tile_grid_file, cores=cores)
"""
