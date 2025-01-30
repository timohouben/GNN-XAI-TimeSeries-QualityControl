import pyproj as proj



def transform_coordinates(x, y, epsg_in, epsg_out):
    """
    Transform coordinates from one coordinate system to another.
    :param x: x coordinate
    :param y: y coordinate
    :param epsg_in: epsg code of input coordinate system
    :param epsg_out: epsg code of output coordinate system
    :return: x, y
    """
    # define the projection
    inProj = proj.Proj(init='epsg:' + str(epsg_in))
    outProj = proj.Proj(init='epsg:' + str(epsg_out))
    # transform the coordinates
    x2, y2 = proj.transform(inProj, outProj, x, y)
    return x2, y2