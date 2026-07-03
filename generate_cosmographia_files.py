#!/usr/bin/env python3
import xml.etree.ElementTree as ET
from astropy.time import Time
from astropy.io import fits
import json


def xml_to_xyzv(xml_path, out_path):
    root = ET.parse(xml_path).getroot()
    with open(out_path, "w") as f:
        for sv in root.findall("./body/segment/data/stateVector"):
            tdb = Time(sv.find("EPOCH").text.rstrip("Z"), format='isot', scale='utc').tdb.jd
            x, y, z   = sv.find("X").text,     sv.find("Y").text, sv.find("Z").text
            vx, vy, vz = sv.find("X_DOT").text, sv.find("Y_DOT").text, sv.find("Z_DOT").text
            f.write(f"{tdb:.8f}  {x}  {y}  {z}  {vx}  {vy}  {vz}\n")


def fits_to_q(fits_path, out_path):
    with fits.open(fits_path) as hdul:
        d = hdul['SCI_RAW_HkAsy30767'].data
    with open(out_path, "w") as f:
        for row in d:
            tdb = Time(float(row['MJD_TIME']), format='mjd', scale='utc').tdb.jd
            w =  float(row['PSE_quaternion_scal'])
            x = -float(row['PSE_quaternion_x'])
            y = -float(row['PSE_quaternion_y'])
            z = -float(row['PSE_quaternion_z'])
            f.write(f"{tdb:.8f}  {w:.10f}  {x:.10f}  {y:.10f}  {z:.10f}\n")


def write_json(out_path, start_utc, end_utc):
    catalog = {
        "version": "1.0",
        "name": "CHEOPS",
        "items": [{
            "name": "CHEOPS",
            "class": "spacecraft",
            "startTime": start_utc,
            "endTime": end_utc,
            "center": "Earth",
            "trajectoryFrame": "EquatorJ2000",
            "trajectory": {"type": "InterpolatedStates", "source": "cheops_trajectory.xyzv"},
            #"bodyFrame":  {"type": "TwoVector", "source": "cheops_attitude.q"},
            "rotationModel": {"type": "Interpolated", "source": "cheops_attitude.q"},
            "geometry":   {"type": "Mesh", "size": 0.5, "source": "CHEOPS_3D.3ds"},
            "label": {"color": [0.8, 0.9, 1.0]},
            "bodyAxes": {"visible": True, "size": 0.3},
            "trajectoryPlot": {"duration": "3 h","color": [0.4, 0.8, 1.0], "linewidth": 1.5, "fade": 0.4}
        }]
    }
    with open(out_path, "w") as f:
        json.dump(catalog, f, indent=4)

# run 
xml_to_xyzv(
    "../temp_data/CH_FDS_ORBRES_OPER_20250725074855_20250723074613_20250725000000.xml",
    "cheops_trajectory.xyzv"
)

fits_to_q(
    "../temp_data/CH_PR350112_TG001901_TU2025-07-24T00-23-00_SCI_RAW_HkAsy30767_V0300.fits",
    "cheops_attitude.q"
)

write_json(
    "cheops.json",
    start_utc="2025-07-23 07:46:13",
    end_utc="2025-07-25 00:00:00"
)
