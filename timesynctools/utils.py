import os
from osgeo import ogr
from math import floor


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


def read_list(text_file):
    with open(text_file, 'rU') as f:
        content = f.readlines()
        content = [line.rstrip('\n') for line in content]
    return content

