#!/usr/bin/env python3
"""
v0.0: whereischeops_v0.0.py computes CHEOPS position and orientation for a given UTC or JD timestamp, or for a time range.
"""
import warnings
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from astropy.time import Time
from astropy.io import fits
from astropy.coordinates import get_sun, get_body
import astropy.units as u
import argparse
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from pathlib import Path
from datetime import datetime
import os
import glob
from scipy.spatial.transform import Rotation
from typing import Any

from astropy.utils.exceptions import AstropyWarning
warnings.filterwarnings("ignore", category=AstropyWarning)
####################################################################

### CHANGE YOUR DIRECTORY HERE
DATA_DIR = os.path.expanduser("~/Downloads/Academics/MSc/SPRING2026/APL_II/code/temp_data/")

# Spacecraft body frame axes (unit vectors in satellite body frame)
BODY_AXES = {
    "LoS":           np.array([1.0, 0.0, 0.0]), #, X_SAT  # telescope line of sight = +X (slightly different but for now keep like this)
    "anti_nadir":    np.array([0.0, 0.0, 1.0]),   # +Z
    "radiators":     np.array([np.cos(np.radians(30)), 0, np.cos(np.radians(60))]), 
    "STOH_L":         np.array([0.0,  np.cos(np.radians(45)), np.sin(np.radians(45))]), # left
    "STOH_R":         np.array([0.0, -np.cos(np.radians(45)), np.sin(np.radians(45))]), # right
    "solar_panel_l": np.array([-np.cos(np.radians(67)),np.cos(np.radians(23)),0]), # left
    "solar_panel_r": np.array([-np.cos(np.radians(67)),-np.cos(np.radians(23)),0]), # right
    #"test": np.array([np.cos(np.radians(40)), 0, np.cos(np.radians(50))]), # add a test vector 

}   

R_EARTH_KM = 6371.0  
JITTER_THRESHOLD_ARCSEC = 2.5
_COLORS = plt.cm.Paired.colors
_STYLES = ['-', '-', '-', '-']                       

PLOT_DESCRIPTIONS = {
    "roll_angle": "Roll angle vs time",
    "attitude":   "RA, Dec, and Roll angle vs time",
    "sun":        "Angle from all spacecraft axes to Sun",
    "moon":       "Angle from all spacecraft axes to Moon",
    "earth":      "Angle from all spacecraft axes to Earth (center and limb)",
    "all":        "4-panel overview: roll + Sun/Moon/Earth for key axes",
}

####################################################################
# Time conversion helpers

def utc_to_jd(utc_string: str) -> float:
    """Convert UTC ISO string to Julian Date."""
    utc_string = str(utc_string).rstrip("Z")
    return float(Time(utc_string, format='isot', scale='utc').jd)


def jd_to_utc(jd: float) -> str:
    """Convert Julian Date to UTC ISO string."""
    return str(Time(jd, format='jd', scale='utc').isot) + "Z"

def _xml_date_to_jd(s: str) -> float:
    """
    Parse a compact date string from an OEM XML filename to JD
    XML filenames: YYYYMMDDTHHMMSS
    """
    dt = datetime.strptime(s.replace("T", ""), "%Y%m%d%H%M%S")
    return float(Time(dt.isoformat(), format='isot', scale='utc').jd)


####################################################################
# LOADING functions

def _load_xml(filepath: str):
    """
    Parse a single OEM XML orbit file
    Returns (jd, pos) where pos is a Nx3 array [km, km/s]

    """
    root = ET.parse(filepath).getroot()
    state_vectors = root.findall("./body/segment/data/stateVector")
    if not state_vectors:
        raise ValueError(f"No <stateVector> elements found in {filepath}")

    pos_tags = ["X", "Y", "Z"]

    epochs, positions = [], []
    for sv in state_vectors:
        epochs.append(utc_to_jd(sv.find("EPOCH").text))
        positions.append([float(sv.find(t).text) for t in pos_tags])

    jd  = np.array(epochs)
    pos = np.array(positions)
    return jd, pos

def load_data(data_dir: str, jd_start: float, jd_end: float) -> dict:
    """
    Load all orbit, attitude, and quaternion data files in data_dir that cover the interval [jd_start, jd_end]

    - For a single-timestamp query, pass jd_start == jd_end
    - When multiple XML files cover the same window the most recently created one is used
    - All attitude and quaternion FITS files in data_dir are loaded and concatenated. 
    - The internal JD timestamps are used directly for interpolation and nearest-sample lookup
    - Visit metadata (target name, program/visit ID, coordinates) is read from the attitude FITS header

    Input:
        data_dir : str
            Directory containing all CHEOPS data files
        jd_start, jd_end : float
            Query interval in Julian Date

    Returns:
        dict with keys:
            'orbit'    : {'jd', 'pos' (Nx3 km EME2000)}
            'attitude' : {'jd', 'utc', 'ra', 'dec', 'roll'}
            'quat'     : {'jd', 'quat' (Nx4 [x,y,z,w] scipy convention)}
            'meta'     : {'program_id', 'visit_id', 'target', 'ra_targ', 'dec_targ'}
    """
    # Orbit XML
    # Filename: CH_FDS_ORBRES_OPER_<creation>_<start>_<end>_<version>.xml
    xml_candidates = glob.glob(os.path.join(data_dir, "CH_FDS_ORBRES_OPER_*.xml"))
    xml_by_window = {}                
    for fp in xml_candidates:
        parts = Path(fp).stem.split("_")
        try:
            creation_str  = parts[4]
            start_str     = parts[5]
            end_str       = parts[6]
            file_jd_start = _xml_date_to_jd(start_str)
            file_jd_end   = _xml_date_to_jd(end_str)
        except (IndexError, ValueError):
            continue
        if file_jd_end < jd_start or file_jd_start > jd_end:
            continue                    
        key = (start_str, end_str)
        if key not in xml_by_window or creation_str > xml_by_window[key][0]:
            xml_by_window[key] = (creation_str, fp)

    if not xml_by_window:
        raise FileNotFoundError(
            f"No XML orbit files in {data_dir!r} cover "
            f"[{jd_to_utc(jd_start)}, {jd_to_utc(jd_end)}]."
        )

    all_jd_o, all_pos = [], [] 
    for creation_str, fp in sorted(xml_by_window.values()):
        jd_o, pos = _load_xml(fp)
        all_jd_o.append(jd_o)
        all_pos.append(pos)

    jd_orbit  = np.concatenate(all_jd_o)
    pos_orbit = np.concatenate(all_pos, axis=0)
    s         = np.argsort(jd_orbit)
    jd_orbit  = jd_orbit[s]
    pos_orbit = pos_orbit[s]

    # Attitude SCI_RAW_Attitude FITS
    att_candidates = glob.glob(os.path.join(data_dir, "CH_*_SCI_RAW_Attitude_*.fits"))
    att_utc, att_ra, att_dec, att_roll = [], [], [], []
    meta_src_fp = None

    for fp in sorted(att_candidates):
        with fits.open(fp) as hdul:
            d = hdul['SCI_RAW_Attitude'].data
            att_utc.extend([t.strip() for t in d['UTC_TIME']])
            att_ra.extend(d['SC_RA'])
            att_dec.extend(d['SC_DEC'])
            att_roll.extend(d['SC_ROLL_ANGLE'])
        if meta_src_fp is None:
            meta_src_fp = fp

    if not att_utc:
        raise FileNotFoundError(
            f"No attitude FITS files in {data_dir!r} cover "
            f"[{jd_to_utc(jd_start)}, {jd_to_utc(jd_end)}]."
        )

    att_utc_arr = np.array(att_utc)
    att_jd_arr  = Time(np.char.rstrip(att_utc_arr, 'Z'), format='isot', scale='utc').jd
    s           = np.argsort(att_jd_arr)
    att_jd_arr  = att_jd_arr[s]
    att_utc_arr = att_utc_arr[s]
    att_ra_arr  = np.array(att_ra)[s]
    att_dec_arr = np.array(att_dec)[s]
    att_roll_arr= np.array(att_roll)[s]

    # Quaternions SCI_RAW_HkAsy30767 FITS 
    quat_candidates = glob.glob(os.path.join(data_dir, "CH_*_SCI_RAW_HkAsy30767_*.fits"))
    all_jd_q, all_quat = [], []

    for fp in sorted(quat_candidates):
        with fits.open(fp) as hdul:
            d = hdul['SCI_RAW_HkAsy30767'].data
            all_jd_q.append(d['MJD_TIME'] + 2400000.5)          
            all_quat.append(np.column_stack([
                d['PSE_quaternion_x'],
                d['PSE_quaternion_y'],
                d['PSE_quaternion_z'],
                d['PSE_quaternion_scal'],
            ]))

    if not all_jd_q:
        raise FileNotFoundError(
            f"No quaternion FITS files in {data_dir!r} cover "
            f"[{jd_to_utc(jd_start)}, {jd_to_utc(jd_end)}]."
        )

    jd_quat  = np.concatenate(all_jd_q)
    quat_arr = np.concatenate(all_quat, axis=0)
    s        = np.argsort(jd_quat)
    jd_quat  = jd_quat[s]
    quat_arr = quat_arr[s]

    # Visit metadata (attitude FITS filename + primary header)
    meta = {"program_id": None, "visit_id": None, "target": None,
            "ra_targ": None, "dec_targ": None}
    if meta_src_fp:
        parts = Path(meta_src_fp).stem.split("_")
        if len(parts) > 1 and parts[1].startswith("PR"):
            meta["program_id"] = parts[1][2:] 
        if len(parts) > 2 and parts[2].startswith("TG"):
            meta["visit_id"]   = parts[2][2:]
        with fits.open(meta_src_fp) as hdul:
            hdr = hdul[0].header
            meta["target"]   = hdr.get("OBJECT",   hdr.get("TARGNAME",  None))
            meta["ra_targ"]  = hdr.get("RA_TARG",  hdr.get("RA_OBJ",    None))
            meta["dec_targ"] = hdr.get("DEC_TARG", hdr.get("DEC_OBJ",   None))

    return {
        "orbit":    {"jd": jd_orbit,   "pos": pos_orbit},
        "attitude": {"jd": att_jd_arr, "utc": att_utc_arr,
                     "ra": att_ra_arr, "dec": att_dec_arr, "roll": att_roll_arr},
        "quat":     {"jd": jd_quat,    "quat": quat_arr},
        "meta":     meta,
    }

####################################################################
# Physics helpers

def nearest_rotations(jd_q: np.ndarray, quat: np.ndarray,
                       jd_query: np.ndarray) -> Rotation:
    """
    For each element of jd_query, find the quaternion in jd_q (sorted) whose timestamp
    is closest, and returns the corresponding scipy Rotation stack (length = len(jd_query))

    Parameters:
        jd_q: 1-D array of quaternion timestamps
        quat: Nx4 array of quaternions [x, y, z, w] (scipy convention)
        jd_query: 1D array of query timestamps
    """ 
    idx = np.searchsorted(jd_q, jd_query)
    idx = np.clip(idx, 1, len(jd_q) - 1)
    left = idx - 1
    use_left = (jd_query - jd_q[left]) < (jd_q[idx] - jd_query)
    return Rotation.from_quat(quat[np.where(use_left, left, idx)])

def angle_to_target(rotations: Rotation, body_vec: np.ndarray,
                    target_vecs: np.ndarray) -> np.ndarray:
    """
    Rotates body_vec from the satellite body frame to the inertial frame using each rotation in the stack, 
    then computes the angle to the corresponding row of target_vecs via the dot product and arccos. 
    Returns an array of angles in the inertial reference frame in degrees

    Parameters:
        rotations: scipy Rotation stack of length N
        body_vec: (3,) unit vector in the satellite body frame
        target_vecs: (N, 3) unit vectors in the inertial frame
    """
    v = rotations.apply(body_vec) 
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return np.degrees(np.arccos(
        np.einsum('ij,ij->i', v, target_vecs)
    ))

####################################################################
# Pointing jitter helpers

def load_pointing_jitter(data_dir: str, jd_start: float, jd_end: float):
    """
    Load pointing-error data from PIP_REP_MultiParameters.fits

    extract rows where NAME == 'pointing error', and filters to samples that
    fall within [jd_start, jd_end]

    Returns:
        dict  {'jd': ndarray, 'val': ndarray [arcsec]}
        None  if no matching file or no data in the requested window.
    """
    pattern = os.path.join(data_dir, "CH_*_PIP_REP_MultiParameters*_activity_*.fits")
    files   = sorted(glob.glob(pattern))
    if not files:
        return None

    all_jd, all_val = [], []
    for fp in files:
        with fits.open(fp) as hdul:
            d    = hdul['PIP_REP_MultiParameters'].data
            mask = np.array([n.strip() == 'pointing error' for n in d['NAME']])
            if not mask.any():
                continue
            sub      = d[mask]
            jd_arr   = sub['MJD_TIME'].astype(float) + 2400000.5
            in_range = (jd_arr >= jd_start) & (jd_arr <= jd_end)
            if not in_range.any():
                continue
            all_jd.append(jd_arr[in_range])
            all_val.append(sub['VALUE'].astype(float)[in_range])

    if not all_jd:
        return None

    jd  = np.concatenate(all_jd)
    val = np.concatenate(all_val)
    s   = np.argsort(jd)
    return {'jd': jd[s], 'val': val[s]}


def _jitter_intervals_dt(jd_arr, val_arr):
    """
    Return a list of (t_start, t_end) pandas Timestamps for contiguous runs
    where val_arr > JITTER_THRESHOLD_ARCSEC (2.5 arcsec)
    """
    mask = val_arr >= JITTER_THRESHOLD_ARCSEC
    if not mask.any():
        return []

    spike_jd  = jd_arr[mask]
    spike_iso = Time(spike_jd, format='jd', scale='utc').isot
    spike_dt  = pd.to_datetime(spike_iso)

    intervals, i0 = [], 0
    for i in range(1, len(spike_jd)):
        if (spike_jd[i] - spike_jd[i - 1]) * 86400 > 10: 
            intervals.append((spike_dt[i0], spike_dt[i - 1]))
            i0 = i
    intervals.append((spike_dt[i0], spike_dt[-1]))
    return intervals

####################################################################
# Plot helpers

def _fmt_xaxis(axs):
    """Apply HH:MM UTC formatter to the bottom subplot."""
    axs[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    axs[-1].set_xlabel("Time [UTC]")


def _make_legend_clickable(fig, ax):
    """
    Attach a pick handler so clicking a legend line toggles its data line.
    """
    leg = ax.get_legend()
    if leg is None:
        return
    lined = {}
    for legline, origline in zip(leg.get_lines(), ax.get_lines()):
        legline.set_picker(True)
        legline.set_pickradius(6)
        lined[id(legline)] = (legline, origline)

    def on_pick(event):
        key = id(event.artist)
        if key not in lined:
            return
        legline, origline = lined[key]
        visible = not origline.get_visible()
        origline.set_visible(visible)
        legline.set_alpha(1.0 if visible else 0.15)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('pick_event', on_pick)


####################################################################
# Plot functions

def plot_orbit(jd: np.ndarray, pos: np.ndarray, t_query: float, pos_query: np.ndarray):
    """3D orbit trajectory with the query point highlighted."""
    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection='3d')
    ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], "b-", lw=1, label="Trajectory")
    ax.scatter(*pos_query, color="red", s=80, zorder=5, label=f"Query: {jd_to_utc(t_query)}")
    ax.set_xlabel("X [km]")
    ax.set_ylabel("Y [km]")
    ax.set_zlabel("Z [km]")
    ax.set_title("CHEOPS Orbit")
    ax.legend()
    plt.tight_layout()
    plt.show()

def plot_single_angle(t_array, angle_array, title, ylabel):
    """Single-panel angle vs time (used for roll_angle)."""
    t_dt = pd.to_datetime(t_array.isot)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t_dt, angle_array, lw=1.2, color='steelblue')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    _fmt_xaxis([ax])
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()

def plot_attitude_range(t_array, jd_array, attitude):
    """RA, Dec, and Roll angle vs time (3 panels)."""
    s = np.argsort(attitude["jd"])
    ra_r   = np.interp(jd_array, attitude["jd"][s], attitude["ra"][s])
    dec_r  = np.interp(jd_array, attitude["jd"][s], attitude["dec"][s])
    roll_r = np.interp(jd_array, attitude["jd"][s], attitude["roll"][s])

    t_dt = pd.to_datetime(t_array.isot)
    fig, axs = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(f"CHEOPS Attitude ({t_dt[0].strftime('%Y-%m-%d')})", fontsize=12)
    for ax, data, label in zip(axs, [ra_r, dec_r, roll_r],
                               ["RA [deg]", "Dec [deg]", "Roll [deg]"]):
        ax.plot(t_dt, data, lw=1, color='steelblue')
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    _fmt_xaxis(axs)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()

def plot_body_angles(t_array, angles_for_target, target_name):
    """
    Plot angle from every spacecraft axis to one celestial body vs time.

    Parameters:
        t_array: astropy Time array
        angles_for_target: {axis_name: angle_array [deg]}
        target_name: 'Sun' or 'Moon'
    """
    t_dt = pd.to_datetime(t_array.isot)
    fig, ax = plt.subplots(figsize=(13, 5))
    for i, (name, ang) in enumerate(angles_for_target.items()):
        ax.plot(t_dt, ang,
                color=_COLORS[i % len(_COLORS)],
                ls=_STYLES[i % len(_STYLES)],
                lw=1.2, label=name)
    ax.set_ylabel(f"Angle to {target_name} [deg]")
    ax.set_title(f"All CHEOPS axes w.r.t {target_name} ({t_dt[0].strftime('%Y-%m-%d')})")
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    _fmt_xaxis([ax])
    _make_legend_clickable(fig, ax)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()


def plot_sun_with_jitter(t_array, angles_for_target, jitter):
    """
    Two-panel Sun plot with pointing-error overlay

    Top panel: all spacecraft axes Sun, with red axvspan zones where 
    pointing error exceeds JITTER_THRESHOLD_ARCSEC
    Bottom panel: pointing error time series with red fill above the threshold
    """
    t_dt    = pd.to_datetime(t_array.isot)
    jit_t   = Time(jitter['jd'], format='jd', scale='utc')
    jit_dt  = pd.to_datetime(jit_t.isot)

    intervals_dt = _jitter_intervals_dt(jitter['jd'], jitter['val'])

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(13, 8), sharex=True)
    fig.suptitle(f"All CHEOPS axes w.r.t Sun ({t_dt[0].strftime('%Y-%m-%d')})", fontsize=12)

    for i, (name, ang) in enumerate(angles_for_target.items()):
        ax_top.plot(t_dt, ang,
                    color=_COLORS[i % len(_COLORS)],
                    #marker='.',
                    ls=_STYLES[i % len(_STYLES)],
                    lw=1.2, label=name)

    for t0, t1 in intervals_dt:
        ax_top.axvspan(t0, t1, color='red', alpha=0.25, linewidth=0)

    ax_top.set_ylabel("Angle to Sun [deg]")
    ax_top.grid(True, alpha=0.3)

    spike_patch = Patch(color='red', alpha=0.5,
                        label=f'Pointing error > {JITTER_THRESHOLD_ARCSEC}"')
    ax_top.legend(loc='upper right', fontsize=8, ncol=2,
                  handles=list(ax_top.lines) + [spike_patch])
    _make_legend_clickable(fig, ax_top)  

    # Bottom (pointing error)
    ax_bot.plot(jit_dt, jitter['val'], color='black', lw=0.5, alpha=0.8,
                label='Pointing error')
    ax_bot.axhline(JITTER_THRESHOLD_ARCSEC, color='red', lw=1.5, ls='--',
                   label=f'{JITTER_THRESHOLD_ARCSEC}" threshold')
    ax_bot.fill_between(jit_dt, jitter['val'], JITTER_THRESHOLD_ARCSEC,
                        where=(jitter['val'] > JITTER_THRESHOLD_ARCSEC),
                        color='red', alpha=0.5)
    ax_bot.set_ylabel('Pointing error ["]')
    ax_bot.legend(loc='upper right', fontsize=8)
    ax_bot.grid(True, alpha=0.3)
    _make_legend_clickable(fig, ax_bot)

    _fmt_xaxis([ax_top, ax_bot])
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()

def plot_earth_angles(t_array, center_angles):
    """
    Plot angles to Earth center (top) and Earth limb (bottom) for all axes.
    """
    t_dt = pd.to_datetime(t_array.isot)
    fig, axs = plt.subplots(figsize=(12, 4))
    fig.suptitle(f"All CHEOPS axes w.r.t Earth ({t_dt[0].strftime('%Y-%m-%d')})", fontsize=12)

    for i, (name, ang) in enumerate(center_angles.items()):
        axs.plot(t_dt, ang,
                    color=_COLORS[i % len(_COLORS)],
                    #marker='.',
                    ls=_STYLES[i % len(_STYLES)],
                    lw=1.2, label=name)
    axs.set_ylabel("Angle to Earth center [deg]")
    axs.legend(loc='upper right', fontsize=8, ncol=2)
    axs.grid(True, alpha=0.3)
    _make_legend_clickable(fig, axs)

    # for i, (name, ang) in enumerate(limb_angles.items()):
    #     axs[1].plot(t_dt, ang,
    #                 color=_COLORS[i % len(_COLORS)],
    #                 marker='.',
    #                 ls=_STYLES[i % len(_STYLES)],
    #                 lw=1.5, label=name)
    # axs[1].axhline(0, color='black', lw=0.8, ls='--', alpha=0.6)
    # axs[1].set_ylabel("Angle to Earth limb [deg]")
    # axs[1].legend(loc='upper right', fontsize=8, ncol=2)
    # axs[1].grid(True, alpha=0.3)
    # _make_legend_clickable(fig, axs[1])

    _fmt_xaxis([axs])
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()


def plot_all_angles(t_array, roll_array, angles):
    """4-panel overview: roll + Sun/Moon/Earth-center angles for key axes"""
    KEY_AXES = ["LoS", "anti_nadir", "STOH_L", "STOH_R", "solar_panel_l", "solar_panel_r"]
    t_dt = pd.to_datetime(t_array.isot)

    fig, axs = plt.subplots(4, 1, figsize=(13, 14), sharex=True)
    fig.suptitle(f"CHEOPS Component Orientations ({t_dt[0].strftime('%Y-%m-%d')})", fontsize=13)

    axs[0].plot(t_dt, roll_array, color='black', lw=1.2)
    axs[0].set_ylabel("Roll angle [deg]")
    axs[0].grid(True, alpha=0.3)

    for ax, target, key in zip(axs[1:], ("Sun", "Moon", "Earth center"), ("sun", "moon", "earth")):
        for i, name in enumerate(KEY_AXES):
            if name in angles:
                ax.plot(t_dt, angles[name][key],
                        color=_COLORS[i % len(_COLORS)],
                        ls=_STYLES[i % len(_STYLES)],
                        lw=1.2, label=name)
        ax.set_ylabel(f"Angle to {target} [deg]")
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        _make_legend_clickable(fig, ax)

    _fmt_xaxis(axs)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()

    ####################################################################
# Metadata display

def print_range_metadata(data, jd_start, jd_end):
    """Print visit metadata and the --plot argument menu."""
    meta    = data["meta"]
    att_jd  = data["attitude"]["jd"]
    quat_jd = data["quat"]["jd"]
    W = 60
    print(f"\n{'='*W}")
    print(f"  Visit Metadata")
    print(f"{'='*W}")
    if meta.get("program_id"):
        print(f"  Program ID  : {meta['program_id']}")
    if meta.get("visit_id"):
        print(f"  Visit ID    : {meta['visit_id']}")
    if meta.get("target"):
        print(f"  Target      : {meta['target']}")
    if meta.get("ra_targ") is not None:
        print(f"  RA (target) : {meta['ra_targ']:.4f} deg")
    if meta.get("dec_targ") is not None:
        print(f"  Dec (target): {meta['dec_targ']:.4f} deg")
    print(f"  Start UTC   : {jd_to_utc(jd_start)}")
    print(f"  End UTC     : {jd_to_utc(jd_end)}")
    print(f"  Duration    : {(jd_end - jd_start) * 24:.2f} hours")

    att_n  = int(np.sum((att_jd  >= jd_start) & (att_jd  <= jd_end)))
    quat_n = int(np.sum((quat_jd >= jd_start) & (quat_jd <= jd_end)))
    print(f"\n  Data in range:")
    print(f"    Attitude samples   : {att_n}")
    print(f"    Quaternion samples : {quat_n}")

    print(f"\n{'='*W}")
    print(f"  Available --plot arguments:")
    print(f"{'='*W}")
    for key, desc in PLOT_DESCRIPTIONS.items():
        print(f"  {key:<22}  {desc}")
    print()

####################################################################
# Main

def main():
    parser = argparse.ArgumentParser(
        description="Query CHEOPS position and orientation")

    # Single-point mode
    time_group = parser.add_mutually_exclusive_group(required=False)
    time_group.add_argument(
        "--utc", metavar="DATETIME",
        help='Single query time in UTC, e.g. "2026-02-09T08:00:00"')
    time_group.add_argument(
        "--jd", type=float, metavar="JD",
        help="Single query time as Julian Date")

    # Time range mode
    parser.add_argument(
        "--timerange", action="store_true",
        help="Enable time range mode (follow up with --start and --end 'YYYY-MM-DDTHH:MM:SS' in UTC)")
    parser.add_argument(
        "--start", metavar="UTC",
        help='Range start in UTC, e.g. "2026-02-08T06:00:00"')
    parser.add_argument(
        "--end", metavar="UTC",
        help='Range end in UTC, e.g. "2026-02-08T22:00:00"')

    parser.add_argument(
        "--plot", nargs='?', const='all', default=None, metavar="WHAT",
        help="(single-point) Show 3D orbit. "
             "(range) What to plot — run --timerange without --plot to see the full list.")
    parser.add_argument(
        "--orientation", action="store_true",
        help="(single-point) Print orientation angles to Sun, Moon, Earth")

    args = parser.parse_args()

    # Validate 
    if args.timerange:
        if not (args.start and args.end):
            parser.error("--timerange requires both --start and --end")
        if args.plot and args.plot not in PLOT_DESCRIPTIONS:
            valid = ", ".join(PLOT_DESCRIPTIONS)
            parser.error(f"Unknown --plot value '{args.plot}'. Valid options: {valid}")
    elif not (args.utc or args.jd):
        parser.error("Provide --utc / --jd for a single-point query, "
                     "or --timerange --start ... --end ... for a time range.")
        
    # TIME RANGE MODE
    if args.timerange:
        jd_start = utc_to_jd(args.start)
        jd_end   = utc_to_jd(args.end)
        step_jd  = 60 / 86400.0
        jd_array = np.arange(jd_start, jd_end + step_jd / 2, step_jd)

        print("\nLoading data files...")
        data = load_data(DATA_DIR, jd_start, jd_end)
        print_range_metadata(data, jd_start, jd_end)

        if args.plot is None:
            return   # user needs to re-run with --plot

        t_array  = Time(jd_array, format='jd', scale='utc')
        attitude = data["attitude"]
        quat_d   = data["quat"]

        if args.plot == "roll_angle":
            s      = np.argsort(attitude["jd"])
            roll_r = np.interp(jd_array, attitude["jd"][s], attitude["roll"][s])
            plot_single_angle(t_array, roll_r, "Roll angle", "Roll [deg]")
            return

        if args.plot == "attitude":
            plot_attitude_range(t_array, jd_array, attitude)
            return

        t_array   = Time(jd_array, format='jd', scale='utc')
        attitude  = data["attitude"]
        rotations = nearest_rotations(data["quat"]["jd"], data["quat"]["quat"], jd_array)

        need_sun   = args.plot in ("sun",   "all")
        need_moon  = args.plot in ("moon",  "all")
        need_earth = args.plot in ("earth", "all")

        sun_vecs = moon_vecs = earth_vecs = None
        earth_angular_radius = None

        if need_sun:
            print("Computing Sun positions...")
            sun_xyz  = get_sun(t_array).cartesian.xyz.value
            sun_vecs = (sun_xyz / np.linalg.norm(sun_xyz, axis=0)).T   

        if need_moon:
            print("Computing Moon positions...")
            moon_xyz  = get_body('moon', t_array).cartesian.xyz.value
            moon_vecs = (moon_xyz / np.linalg.norm(moon_xyz, axis=0)).T 

        if need_earth:
            print("Computing orbit positions for Earth angles...")
            orbit    = data["orbit"]
            pos_r    = np.column_stack([
                np.interp(jd_array, orbit["jd"], orbit["pos"][:, i]) for i in range(3)
            ])
            pos_norm  = np.linalg.norm(pos_r, axis=1)
            earth_vecs = -pos_r / pos_norm[:, np.newaxis]      
            earth_angular_radius = np.degrees(np.arcsin(R_EARTH_KM / pos_norm))  

        # --plot WHAT  
        if args.plot == "sun":
            sun_angles = {name: angle_to_target(rotations, BODY_AXES[name], sun_vecs)
                          for name in BODY_AXES}
            jitter = load_pointing_jitter(DATA_DIR, jd_start, jd_end)
            if jitter is not None:
                plot_sun_with_jitter(t_array, sun_angles, jitter)
            else:
                plot_body_angles(t_array, sun_angles, "Sun")

        elif args.plot == "moon":
            plot_body_angles(
                t_array,
                {name: angle_to_target(rotations, BODY_AXES[name], moon_vecs)
                 for name in BODY_AXES},
                "Moon",
            )

        elif args.plot == "earth":
            center = {name: angle_to_target(rotations, BODY_AXES[name], earth_vecs)
                      for name in BODY_AXES}
            limb   = {name: center[name] - earth_angular_radius for name in BODY_AXES}
            plot_earth_angles(t_array, center)

        elif args.plot == "all":
            KEY_AXES = ["LoS", "anti_nadir", "STOH1", "STOH2"]
            roll_r = np.interp(jd_array, attitude["jd"], attitude["roll"])
            angles_dict = {name: {
                "sun":   angle_to_target(rotations, BODY_AXES[name], sun_vecs),
                "moon":  angle_to_target(rotations, BODY_AXES[name], moon_vecs),
                "earth": angle_to_target(rotations, BODY_AXES[name], earth_vecs),
            } for name in KEY_AXES}
            plot_all_angles(t_array, roll_r, angles_dict)

        return

    # SINGLE-POINT MODE
    if args.utc:
        t_jd = utc_to_jd(args.utc)
        t_utc = args.utc
    else:
        t_jd  = args.jd
        t_utc = jd_to_utc(t_jd)

    print("\nLoading data files...")
    data = load_data(DATA_DIR, t_jd, t_jd)

    orbit = data["orbit"]
    xyz   = np.array([np.interp(t_jd, orbit["jd"], orbit["pos"][:, i]) for i in range(3)])

    print(f"\nQuery time:")
    print(f"  UTC : {t_utc}")
    print(f"\nInterpolated position:")
    print(f"  X = {xyz[0]:>15.4f} km")
    print(f"  Y = {xyz[1]:>15.4f} km")
    print(f"  Z = {xyz[2]:>15.4f} km")

    att     = data["attitude"]
    att_idx = np.argmin(np.abs(att["jd"] - t_jd))
    print(f"\nAttitude (nearest sample):")
    print(f"  UTC: {att['utc'][att_idx]}")
    print(f"  RA: {att['ra'][att_idx]:.6f} deg")
    print(f"  Dec: {att['dec'][att_idx]:.6f} deg")
    print(f"  Roll Angle: {att['roll'][att_idx]:.6f} deg")

    # Orientation (--orientation)
    if args.orientation:
        t_ap   = Time(t_utc, format='isot', scale='utc')
        r      = nearest_rotations(data["quat"]["jd"], data["quat"]["quat"],
                                   np.array([t_jd]))

        sun_v  = get_sun(t_ap).cartesian.xyz.value.flatten()
        sun_v /= np.linalg.norm(sun_v)
        moon_v  = get_body('moon', t_ap).cartesian.xyz.value.flatten()
        moon_v /= np.linalg.norm(moon_v)
        earth_v = -xyz / np.linalg.norm(xyz)

        print(f"\n{'='*50}")
        print(f"  {'Axis':<13} {'Sun':>9} {'Moon':>9} {'Earth':>9}  [deg]")
        print(f"  {'-'*46}")
        for name, body_vec in BODY_AXES.items():
            s_ang = angle_to_target(r, body_vec, sun_v[np.newaxis])[0]
            m_ang = angle_to_target(r, body_vec, moon_v[np.newaxis])[0]
            e_ang = angle_to_target(r, body_vec, earth_v[np.newaxis])[0]
            print(f"  {name:<13} {s_ang:>9.2f} {m_ang:>9.2f} {e_ang:>9.2f}")

    # 3D orbit plot (--plot)
    if args.plot is not None:
        plot_orbit(orbit["jd"], orbit["pos"], t_jd, xyz)


if __name__ == "__main__":
    main()

        