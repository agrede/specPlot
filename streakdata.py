import numpy as np
from configparser import ConfigParser
import re
import pint

UREG = pint.UnitRegistry()

def rdbytes(x, signed=True):
    return int.from_bytes(x, byteorder='little', signed=signed)


def readCalScale(f, config, axis, ureg=UREG):
    rgxspos = re.compile(r"\"#(\d+),(\d+)\".*")
    units = ureg.parse_expression(
        config['Scaling']["scaling{:s}unit".format(
            axis.lower())][1:-1]).u
    if m := rgxspos.match(
            config['Scaling']["scaling{:s}scalingfile".format(
                axis.lower())]):
        f.seek(int(m.group(1)))
        return np.frombuffer(
            f.read(int(m.group(2))*4), dtype=np.float32)*units
    else:
        return None


def readStreak(pth, ureg=UREG):
    """
    Read streak camera file
    Returns:
    numpy.ndarray: wavelengths
    numpy.ndarray: times
    numpy.ndarray: counts
    """
    rgxsplt = re.compile(r",(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)")
    ftypes = [np.int8, None, np.int16, np.int32]
    with open(pth, 'rb') as f:
        if f.read(2) != b'IM':
            return None, None, None, None
        comment_length = rdbytes(f.read(2))
        width = rdbytes(f.read(2))
        height = rdbytes(f.read(2))
        xoffset = rdbytes(f.read(2))
        yoffset = rdbytes(f.read(2))
        ftype = ftypes[rdbytes(f.read(2))]
        f.seek(64)
        cnfstr = "\n".join(
            rgxsplt.split(f.read(comment_length).decode('utf-8')))
        config = ConfigParser()
        config.read_string(cnfstr)
        if ftype is not None:
            values = np.frombuffer(
                f.read(width*height*np.dtype(ftype).itemsize),
                dtype=ftype).reshape((height, width))
        else:
            values = None
        scaleX = readCalScale(f, config, "x", ureg)
        scaleY = readCalScale(f, config, "y", ureg)
        if config['Acquisition']['zaxisunit'].lower() == "count":
            values = values*ureg.count
    return scaleX, scaleY, values, config
