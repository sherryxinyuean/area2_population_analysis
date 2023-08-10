import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import time
import uuid
import scipy.signal as signal
import multiprocessing

from datetime import datetime
from dateutil.tz import tzlocal, gettz
from pathlib import Path
from collections import defaultdict
from typing import Optional

from pynwb import NWBFile, NWBHDF5IO, TimeSeries, ProcessingModule
from pynwb.file import Subject
from pynwb.device import Device
from pynwb.misc import Units
from pynwb.ecephys import ElectrodeGroup
from pynwb.behavior import BehavioralTimeSeries, Position, SpatialSeries


### Helper functions

def calculate_onset(
    data: np.ndarray,
    timestamps: np.ndarray,
    trial_info: pd.DataFrame,
    start_field='go_cue',
    end_field='stop_time',
    min_ds=1,
    s_thresh=5,
    peak_offset=200, # ms
    start_offset=200, # ms
    peak_divisor=10,
    ignored_trials=None
):
    """Calculates movement onset, inspired by the 'peak' method in 
    Matt Perich's MATLAB implementation here: 
    https://github.com/mattperich/TrialData/blob/master/Tools/getMoveOnsetAndPeak.m
    
    Parameters
    ----------
    data : np.ndarray
        1d array containing speed
    timestamps : np.ndarray
        1d array containing timestamps, in seconds, corresponding to `data`
    trial_info : pd.DataFrame
        DataFrame containing trial info
    start_field: str
        The field name of the start of the window to consider.
    end_field: str
        The field name of the end of the window to consider.
    min_ds: float
        The minimum diff (speed) to find movement onset, by default 1.9
    s_thresh: float
        Speed threshold (secondary method if first fails)
    peak_offset: float
        The number of ms after start_field to find max speed
    start_offset: float,
        The number of ms after start_field to find movement onset
    peak_divisor: float
        The number to divide the peak by to find the threshold.
    ignored_trials: pd.Series
        The trials to ignore for this calculation.
    """
    # Ignore trials that don't have the required fields
    if ignored_trials is None:
        ignored_trials == pd.Series(False, index=trial_info.index)
    ignored_trials = (
        trial_info[start_field].isnull()
        | trial_info[end_field].isnull()
        | ignored_trials
    )
    ti = trial_info[~ignored_trials]
    onset_list = []
    for index, row in ti.iterrows():
        onset = np.nan
        # mask into specific trial
        start_time = row[start_field]
        end_time = row[end_field]
        move_start = start_time + start_offset * 0.001
        peak_start = start_time + peak_offset * 0.001
        start_time = min(start_time, move_start, peak_start)
        trial_timestamps = timestamps[(timestamps >= start_time) & (timestamps < end_time)]
        trial_data = data[(timestamps >= start_time) & (timestamps < end_time)]
        if np.any(np.isnan(trial_data)):
            onset_list.append(onset)
            continue
        # make masks for valid peaks/onsets
        valid_move = trial_timestamps >= move_start
        valid_peak = trial_timestamps >= peak_start
        # get acceleration and jerk
        dm = np.diff(trial_data, prepend=[trial_data[0]])
        ddm = np.diff(dm, prepend=[dm[0]])
        # get peak accels by descending zero crossings of jerk
        peaks = (ddm > 0) & (np.pad(ddm, (0,1))[1:] < 0)
        accel_peaks = peaks & valid_peak & (dm > min_ds) & valid_move
        # if peaks found, find onset
        if accel_peaks.sum() > 0:
            # find first peak and compute threshold
            first_peak = np.nonzero(accel_peaks)[0][0]
            peak_accel = dm[first_peak]
            threshold = peak_accel / peak_divisor
            below_threshold = (dm < threshold) & valid_move & (np.arange(len(dm)) < first_peak)
            if below_threshold.sum() > 0:
                thresh_crossing = np.max(np.nonzero(below_threshold[::-1])[0])
                onset = trial_timestamps[thresh_crossing]
        if np.isnan(onset):
            above_threshold = (trial_data > s_thresh) & valid_move
            if above_threshold.sum() > 0:
                thresh_crossing = np.min(np.nonzero(above_threshold)[0])
                onset = trial_timestamps[thresh_crossing]
        onset_list.append(onset)
    onset_series = pd.Series(onset_list, index=ti.index)
    return onset_series


def get_pair_xcorr(
    spikes,
    max_points=None,
):
    """Calculate the cross-correlations between channels.
    Cross-correlation is computed as the sum of the element-wise
    product of the channels, normalized by product of their
    deviation from 0.

    Parameters
    ----------
    spikes : np.ndarray
        Time x Neuron array of spiking activity
    max_points : int, optional
        The number of points to use when calculating the correlation,
        taken from the beginning of the data, by default None.
    """

    if max_points is not None:
        np_data = spikes[:max_points]
    else:
        np_data = spikes

    n_dim = np_data.shape[1]
    pairs = [(i, k) for i in range(n_dim) for k in range(i)]

    def xcorr_func(args):
        i, k = args
        c = np.sum(np_data[:, i] * np_data[:, k]).astype(np.float32)
        if c == 0:
            return 0.0
        # normalize
        c /= np.sqrt(np.sum(np_data[:, i] ** 2) * np.sum(np_data[:, k] ** 2))
        return c

    corr_list = parmap(xcorr_func, pairs)

    return pairs, corr_list


def drop_neurons_by_xcorr(
    pairs: list, # list[tuple[int, int]],
    xcorrs: list, # list[float],
    threshold: float,
):
    """Choose channels to drop based on cross-correlation threshold

    Parameters
    ----------
    pairs : list of tuple of int
        List of pairs of channel indices, indicating which pair of
        channels corresponds to each cross-correlation value
    xcorrs : list of float
        List of cross-correlation values
    """
    # convert to numpy to start
    pairs = np.array(pairs)
    xcorrs = np.array(xcorrs)
    # sort and isolate above threshold pairs
    xcorr_sorted = np.sort(xcorrs)[::-1]
    above_threshold = xcorr_sorted > threshold
    pairs_bad = pairs[np.argsort(xcorrs)][::-1,:][above_threshold]
    xcorr_bad = xcorr_sorted[above_threshold]
    # remove channels until no above threshold pairs
    dropped_channels =  []
    while len(pairs_bad) > 0:
        # get indices for neurons in pair
        n1 = pairs_bad[0][0]
        n2 = pairs_bad[0][1]
        # get corr for all the other pairs with each neuron
        n1_pairs = np.any(pairs_bad == n1, axis=1)
        n2_pairs = np.any(pairs_bad == n2, axis=1)
        cnt1 = np.sum(n1_pairs)
        cnt2 = np.sum(n2_pairs)
        # determine which channel has more pairs
        if cnt1 > cnt2:
            chan_drop = n1
            drop_mask = n1_pairs
        elif cnt1 < cnt2:
            chan_drop = n2
            drop_mask = n2_pairs
        else:
            # if equal, remove channel with higher correlations
            if np.mean(xcorr_bad[n1_pairs]) > np.mean(xcorr_bad[n2_pairs]):
                chan_drop = n1
                drop_mask = n1_pairs
            else:
                chan_drop = n2
                drop_mask = n2_pairs
        dropped_channels.append(chan_drop)
        pairs_bad = pairs_bad[~drop_mask]
        xcorr_bad = xcorr_bad[~drop_mask]
    return dropped_channels


def fun(f, q_in, q_out):
    """wrapper for function for `parmap`"""
    while True:
        i, x = q_in.get()
        if i is None:
            break
        q_out.put((i, f(x)))


def parmap(f, X, nprocs=multiprocessing.cpu_count()):
    """equivalent to Pool.map but works with functions inside Class methods"""
    q_in = multiprocessing.Queue(1)
    q_out = multiprocessing.Queue()

    proc = [
        multiprocessing.Process(target=fun, args=(f, q_in, q_out))
        for _ in range(nprocs)
    ]
    for p in proc:
        p.daemon = True
        p.start()

    sent = [q_in.put((i, x)) for i, x in enumerate(X)]
    [q_in.put((None, None)) for _ in range(nprocs)]
    res = [q_out.get() for _ in range(len(sent))]

    [p.join() for p in proc]

    return [x for i, x in sorted(res)]


### Main conversion function

def area2_to_nwb(
    file_path: Path,
    save_path: Path,
    splits: dict = {}, # dict[str, float] = {},
    heldout_unit_frac: Optional[float] = None,
    stratified: bool = True,
    drop_neurons: bool = True,
    fr_threshold: Optional[float] = None,
    corr_threshold: Optional[float] = None,
    seed: Optional[int] = None,
    verbose: bool = True,
):
    """Convert a dataset from Raeed's Area2 data to NWB

    Parameters
    ----------
    file_path : Path
        path fo original data file
    save_path : Path
        path to save NWB file
    splits : dict, default: {}
        An optional dict mapping trial split names to their
        percentage of the total number of successful trials.
        For example, to create an 80/20 train/val split,
        set `splits={'train': 0.8, 'val': 0.2}`. The default of {}
        does not create any trial splits.
    heldout_unit_frac : float, optional
        If you want to designate a certain subset of neurons
        as held-out, as in NLB, then specify the fraction
        of the total neurons here. The default of None
        does not hold out any neurons
    stratified : bool, default: True
        If `splits` is provided, this argument indicates whether
        trial splits should be created with stratified sampling,
        i.e. sampling separately for each condition so that
        conditions are more evenly represented across splits.
    drop_neurons : bool, default: True
        Whether sorted units should be dropped for either firing rate
        or cross-correlation, default True.
    fr_threshold : float, optional
        If `drop_neurons` is True, then `fr_threshold` is the minimum
        firing rate in Hz for a neuron to be included in the NWB file.
        If no threshold is provided, but `drop_neurons` is True, a
        histogram ofunit firing rates will be plotted and saved,
        and the user will be prompted to choose a firing rate threshold
    corr_threshold : float, optional
        If `drop_neurons` is True, then `corr_threshold` is the maximum
        cross-correlation for a pair of neurons not to be flagged for
        removal. For pairs that are above the threshold, the unit that is
        in more pairs that exceed the threshold is removed first. This
        process continues until there are no pairs above the threshold.
    seed : int, optional
        A seed for the numpy random number generator, useful if you
        want consistent trial splits and held-out unit selections.
    verbose : bool, default: True
        Whether to print out a bunch of messages or not
    """
    #################
    ###   SETUP   ###
    #################

    # Initialize RNG
    rng = np.random.default_rng(seed)
    
    # Load the HDF5 file
    if verbose:
        print(f"Reading from {file_path}")
    h5file = h5py.File(file_path, 'r')

    # Fetch the data
    ds = h5file['trial_data']
    bin_width = ds['bin_size'][0,0]
    rate = float(round(1. / bin_width))
    assert ('tgtDir' in ds.keys()) or ('targetDir' in ds.keys()), "No target direction field found"
    target_dir_key = 'tgtDir' if ('tgtDir' in ds.keys()) else 'targetDir'

    # Convert text fields to strings
    name = ''.join(chr(val) for val in ds['monkey'][:].squeeze())
    # name = ''.join(chr(ds['monkey'][:].squeeze()))
    if 'date' in ds.keys():
        date = ''.join(chr(val) for val in ds['date'][:].squeeze())
        fmt = "%m/%d/%Y %H:%M:%S.%f"
    elif 'date_time' in ds.keys():
        date = ''.join(chr(val) for val in ds['date_time'][:].squeeze())
        fmt = "%Y/%m/%d %H:%M:%S.%f"
    else:
        raise AssertionError("Unable to find date for dataset")
    has_muscle_data = 'muscle_names' in ds.keys()
    if has_muscle_data:
        muscle_names = [''.join(chr(val) for val in h5file[ds['muscle_names'][i,0]][:].squeeze()).strip('\"\'') for i in range(ds['muscle_names'].shape[0])]
    has_joint_data = 'joint_names' in ds.keys()
    if has_joint_data:
        joint_names = [''.join(chr(val) for val in h5file[ds['joint_names'][i,0]][:].squeeze()).strip('\"\'') for i in range(ds['joint_names'].shape[0])]

    # Create start time
    start_time = datetime.strptime(date, fmt).replace(tzinfo=gettz('America/New_York'))

    # Create nwbfile
    description = (
        f'Data from monkey {name} performing center-out active-passive task, '
        f'recorded on {start_time.strftime("%Y/%m/%d")}.'
    )
    nwbfile = NWBFile(
        session_description=description,
        session_id=f'{name}_{start_time.strftime("%Y%m%d")}',
        identifier=str(uuid.uuid1()),
        session_start_time=start_time,
        experimenter="Raeed H. Chowdhury",
        lab="Miller",
        institution="Northwestern University",
        keywords=['somatosensory cortex', 'reaching', 'macaque'],
        experiment_description="Center out reaching task with perturbations during center hold period before reach",
        related_publications="https://doi.org/10.7554/eLife.48198",
    )

    # Subject info
    nwbfile.subject = Subject(
        # description=f'A macaque named {name}', 
        subject_id=f'{name}',
        sex='M',
        # age=?,
        species="Macaca mulatta"
    )

    ###########################
    ###   TRIAL SELECTION   ###
    ###########################

    # Identify successful trials and store their conditions
    good_trials = []
    cond_dict = defaultdict(list)
    for i in range(len(ds['trialID'])):
        if chr(ds['result'][i][0]) == "R": # skip if unsuccessful
            isbump = (ds['ctrHoldBump'][i][0] == 1)
            cond = (
                isbump, 
                ds['bumpDir'][i][0] if isbump else ds[target_dir_key][i][0]
            )
            if np.isnan(cond[1]): # skip if no bump/target direction
                continue
            good_trials.append(i)
            cond_dict[cond].append(i)
    if verbose:
        print(f"Found {len(good_trials)} successful trials out of {len(ds['trialID'])} total")
    
    # Divide up trials into splits if desired
    if len(splits) > 1:
        trial_split_idxs = {key: [] for key in splits.keys()}
        if stratified:
            trial_lists = list(cond_dict.values())
        else:
            trial_lists = [good_trials]
        for trial_list in trial_lists:
            shuffled_trials = rng.permutation(trial_list).tolist()
            split_indices = np.array([0] + np.cumsum(list(splits.values())).tolist())
            split_indices /= split_indices[-1]
            split_indices *= len(shuffled_trials)
            split_indices = split_indices.round().astype(int)
            assert split_indices[-1] == len(shuffled_trials)
            assert len(split_indices) == (len(splits) + 1)
            for i, key in enumerate(splits.keys()):
                split_start = split_indices[i]
                split_end = split_indices[i+1]
                trial_split_idxs[key] += shuffled_trials[split_start:split_end]
    elif len(splits) == 1:
        trial_split_idxs = {list(splits.keys())[0]: good_trials}

    ######################
    ###   ADD TRIALS   ###
    ######################

    # Trial info cols
    nwbfile.add_trial_column('result', "Result of the trial, either 'R' (reward), 'A' (abort), 'I' (incomplete) or 'F' (fail)")
    nwbfile.add_trial_column('ctr_hold', 'Required hold time on center before reach, in seconds')
    nwbfile.add_trial_column('ctr_hold_bump', 'Whether there was bump during center hold')
    nwbfile.add_trial_column('bump_dir', 'Angle (in degrees) of bump direction, if there was one. 0 degrees is directly to the right, and 90 degrees is directly upward')
    nwbfile.add_trial_column('target_on_time', 'Time of target presentation')
    nwbfile.add_trial_column('target_dir', 'Direction of target, in degrees. 0 degrees is directly to the right, and 90 degrees is directly upward')
    nwbfile.add_trial_column('go_cue_time', 'Time of go cue delivery')
    nwbfile.add_trial_column('bump_time', 'Time of bump delivery, if there was one')
    # nwbfile.add_trial_column('move_onset_time', 'Time of move onset around bump for passive trials and after go cue for active trials')
    nwbfile.add_trial_column('cond_dir', 'Direction of bump for passive trials and target direction for active trials, for convenience when filtering trials')
    if len(splits) > 0:
        nwbfile.add_trial_column('split', 'Trial split that the trial belongs to. Can be "train", "val", "test", or "none"')

    # idx_goCueTime, idx_tgtOnTime, and idx_bumpTime don't occur every trial,
    # but the trials they correspond to aren't listed, so you need to check
    # each trial to see if it happens that trial
    def check_between(trial_idx, field):
        between = (ds['idx_startTime'][trial_idx][0] < ds[field]) & (ds[field] < ds['idx_endTime'][trial_idx][0])
        if between.sum() == 1:
            return ds[field][between][0]
        elif between.sum() == 0:
            return np.nan
        else:
            raise AssertionError(f'Multiple occurences of {field} during trial {trial_idx}')
    
    if verbose:
        print(f"Writing trial information")
    # Add trial info
    for i in range(len(ds['trialID'])):
        # initialize dict for trial info
        trial_info = {}
        # get trial id
        trial_id = ds['trialID'][i][0]
        # get split if available
        if len(splits) > 0:
            split = [key for key, val in trial_split_idxs.items() if i in val]
            assert len(split) <= 1, f"Trial {i} found in multiple splits???"
            trial_info['split'] = split[0] if len(split) == 1 else 'none'
        # get trial event times
        go_cue = check_between(i, 'idx_goCueTime')
        tgt_time = check_between(i, 'idx_tgtOnTime')
        bump_time = check_between(i, 'idx_bumpTime')
        # convert to seconds, add to trial info
        trial_info['start_time'] = round(ds['idx_startTime'][i][0] * bin_width, 6)
        trial_info['stop_time'] = round(ds['idx_endTime'][i][0] * bin_width, 6)
        trial_info['go_cue_time'] = round(go_cue * bin_width, 6)
        trial_info['target_on_time'] = round(tgt_time * bin_width, 6)
        trial_info['bump_time'] = round(bump_time * bin_width, 6)
        # add other trial info
        trial_info['result'] = chr(ds['result'][i][0])
        trial_info['ctr_hold'] = ds['ctrHold'][i][0]
        trial_info['ctr_hold_bump'] = (ds['ctrHoldBump'][i][0] == 1)
        trial_info['target_dir'] = ds[target_dir_key][i][0]
        trial_info['bump_dir'] = ds['bumpDir'][i][0]
        trial_info['cond_dir'] = (ds['bumpDir'][i][0]) if (ds['ctrHoldBump'][i][0] == 1) else (ds[target_dir_key][i][0])
        # create trial entry
        nwbfile.add_trial(**trial_info)

    #############################
    ###   BEHAVIORAL TRACES   ###
    #############################
    
    # Add behavior timeseries
    if verbose:
        print(
            "Writing behavioral data, including hand pos/vel/force, " + 
            ("muscle len/vel, " if has_muscle_data else "") + 
            ("and joint ang/vel " if has_joint_data else "")
        )
    assert ds['pos'].shape[0] == 2
    pos = SpatialSeries(
        name='hand_pos',
        data=ds['pos'][:].T + np.array([[0, 35]]),
        rate=rate,
        starting_time=0.0,
        conversion=0.01,
        description=(
            'Hand position in Cartesian coordinates, in cm. '
            'First column is x coordinate, second column is y coordinate. '
            'Sampled at 1 kHz with required anti-alias filtering. '
            'Further low-pass filtered at 100 Hz cutoff. '
            'Resampled to correct bin size and aligned to rest of data'
        ),
        comments='columns=[x,y]',
        reference_frame='(0, 0) is screen center'
    )

    assert ds['vel'].shape[0] == 2
    vel = TimeSeries(
        name='hand_vel',
        data=ds['vel'][:].T,
        rate=rate,
        starting_time=0.0,
        conversion=0.01,
        description=(
            'Hand velocity, in cm/s. First column is x direction, second column is y direction. '
            'Sampled at 1 kHz with required anti-alias filtering. '
            'Further low-pass filtered at 100 Hz cutoff. '
            'Resampled to correct bin size and aligned to rest of data'
        ),
        comments='columns=[x,y]',
        unit='m/s',
    )

    assert ds['force'].shape[0] == 6
    force = TimeSeries(
        name='force',
        data=ds['force'][:].T,
        rate=rate,
        starting_time=0.0,
        description=(
            'Interface forces and torques between the hand and the manipulandum handle, Newtons or Newton-meters. '
            'First three columns are forces in X, Y, and Z directions, respectively, and last three columns are X, Y, and Z moments, respectively. '
            'Sampled at 1 kHz with required anti-alias filtering. Resampled to correct bin size and aligned to rest of data'
        ),
        comments='columns=[x,y,z,xmo,ymo,zmo]',
        unit='Newton',
    )
    
    data_list = [pos, vel, force]

    # add muscle timeseries if available
    if has_muscle_data:
        assert ds['muscle_len'].shape[0] == len(muscle_names)
        muslen = TimeSeries(
            name='muscle_len',
            data=ds['muscle_len'][:].T,
            rate=rate,
            starting_time=0.0,
            description='Length of various monkey arm muscles in m, calculated from motion tracking data',
            comments=f'columns={muscle_names}',
            unit='m',
        )

        assert ds['muscle_vel'].shape[0] == len(muscle_names)
        musvel = TimeSeries(
            name='muscle_vel',
            data=ds['muscle_vel'][:].T,
            rate=rate,
            starting_time=0.0,
            description='Velocity of various monkey arm muscles in m/s, calculated from motion tracking data',
            comments=f'columns={muscle_names}',
            unit='m/s',
        )
        
        data_list += [muslen, musvel]

    # add joint timeseries if available
    if has_joint_data:
        assert ds['joint_ang'].shape[0] == len(joint_names)
        jang = TimeSeries(
            name='joint_ang',
            data=ds['joint_ang'][:].T,
            rate=rate,
            starting_time=0.0,
            description='Angle of monkey arm joints in degrees, calculated from motion tracking data',
            comments=f'columns={joint_names}',
            unit='degrees',
        )

        assert ds['joint_vel'].shape[0] == len(joint_names)
        jvel = TimeSeries(
            name='joint_vel',
            data=ds['joint_vel'][:].T,
            rate=rate,
            starting_time=0.0,
            description='Velocity of monkey arm joints in degrees/s, calculated from motion tracking data',
            comments=f'columns={joint_names}',
            unit='degrees/s',
        )

        data_list += [jang, jvel]

    # Create behavior processing module
    behavior_module = nwbfile.create_processing_module(
        name='behavior',
        description='Processed behavioral data'
    )
    behavior_module.add(data_list)

    ##########################
    ###   MOVEMENT ONSET   ###
    ##########################

    speed = np.sqrt(np.sum(ds['vel'][:].T**2, axis=1)) # np.linalg.norm(ds['vel'][:], axis=0),
    trial_info_df = nwbfile.trials.to_dataframe()
    act_move_onset = calculate_onset(
        data=speed,
        timestamps=np.round(np.arange(ds['vel'].shape[1]) * bin_width, 6),
        trial_info=trial_info_df,
        start_field='go_cue_time',
        end_field='stop_time',
        min_ds=1,
        s_thresh=5,
        peak_offset=200, # ms
        start_offset=200, # ms
        peak_divisor=10,
        ignored_trials=(trial_info_df.ctr_hold_bump)
    )
    pas_move_onset = calculate_onset(
        data=speed,
        timestamps=np.round(np.arange(ds['vel'].shape[1]) * bin_width, 6),
        trial_info=trial_info_df,
        start_field='bump_time',
        end_field='go_cue_time',
        min_ds=1,
        s_thresh=5,
        peak_offset=-50, # ms
        start_offset=-50, # ms
        peak_divisor=10,
        ignored_trials=(~trial_info_df.ctr_hold_bump)
    )
    trial_info_df['move_onset'] = pd.concat([act_move_onset, pas_move_onset], axis=0)
    nwbfile.add_trial_column(
        'move_onset_time', 
        'Time of move onset around bump for passive trials and after go cue for active trials',
        data=trial_info_df['move_onset'].to_numpy(),
    )
    if verbose:
        print(
            f"Computed movement onset time for {(~trial_info_df['move_onset'].isna()).sum()} "
            f"out of {len(trial_info_df)} trials"
        )

    ############################
    ###   ECEPHYS METADATA   ###
    ############################

    device = nwbfile.create_device(
        name='electrode_array',
        description="96-electrode Utah array",
        manufacturer="Blackrock Microsystems"
    )

    electrode_group = nwbfile.create_electrode_group(
        name='electrode_group',
        description='Electrodes in an implanted Utah array',
        location='Primary somatosensory cortex, Brodmann area 2',
        device=device
    )

    elecinds = {}
    for i, count in enumerate(np.unique(ds['S1_unit_guide'][0, :])):
        nwbfile.add_electrode(
            id=int(i),
            x=np.nan,
            y=np.nan,
            z=np.nan,
            imp=np.nan,
            location='S1 Area 2',
            group=electrode_group,
            filtering='Anti-alias and high-pass filter (unknown cutoff)'
        )
        elecinds[count] = i

    nwbfile.units = Units(
        name='units',
        # description="Sampled at 30 kHz with anti-alias " \
        #     "filtering and a high pass filter to remove LFP. " \
        #     "Thresholds detected online at ~-4 x Signal RMS and 1.6 ms snippets surrounding " \
        #     "crossing saved to disk. Snippets were sorted offline into putative single units. " \
        #     "Spikes were counted within each bin and aligned to rest of data.",
    ) # resolution=bin_width)

    #############################
    ###   CHANNEL SELECTION   ###
    #############################

    spike_arr = ds['S1_spikes'][()].T
    # assert not np.any(spike_arr > 1) # not required, but currently a hard-coded assumption
    # Units with unit number 0 in S1_unit_guide are leftover spikes that do not belong to a single unit
    # channel_idxs = np.arange(ds['S1_unit_guide'].shape[1])[ds['S1_unit_guide'][1, :] != 0]
    channel_idxs = np.arange(ds['S1_unit_guide'].shape[1])

    # drop neurons based on firing rate
    neur_fr = spike_arr[:, channel_idxs].sum(axis=0) / spike_arr[:, channel_idxs].shape[0] / bin_width
    if drop_neurons and fr_threshold is None:
        plt.hist(neur_fr, bins=15)
        plt.savefig('firing_rate_distribution.png')
        plt.close()
        fr_threshold = input(
            f"Please select a firing rate threshold in Hz to drop all channels with "
            "firing rates below the threshold. The distribution of firing rates is "
            "shown in `firing_rate_distribution.png`. Enter the threshold here: "
        )
        os.remove('firing_rate_distribution.png')
        try:
            fr_threshold = float(fr_threshold)
        except:
            raise ValueError(f"{fr_threshold} cannot be converted to a float")
    if drop_neurons:
        fr_dropped_channels = channel_idxs[np.nonzero(neur_fr < fr_threshold)[0]].tolist()
        if verbose:
            print(f"Dropping {len(fr_dropped_channels)} channels with firing rate below {fr_threshold} Hz")
        channel_idxs = channel_idxs[(neur_fr >= fr_threshold)]

    # drop neurons based on cross-correlation
    # pairs, xcorrs = get_pair_xcorr(spike_arr[:, channel_idxs], max_points=None)
    # if drop_neurons and corr_threshold is None:
    #     plt.hist(xcorrs, bins=200)
    #     plt.yscale('log')
    #     plt.savefig('xcorr_distribution.png')
    #     plt.close()
    #     corr_threshold = input(
    #         f"Please select a cross correlation threshold to drop channels until all "
    #         "pairs have cross-correlation below the threshold. The distribution of "
    #         "cross-correlations is shown in `xcorr_distribution.png`. Enter the " 
    #         "threshold here: "
    #     )
    #     os.remove('xcorr_distribution.png')
    #     try:
    #         corr_threshold = float(corr_threshold)
    #     except:
    #         raise ValueError(f"{corr_threshold} cannot be converted to a float")
    # if drop_neurons:
    #     corr_dropped_channels = drop_neurons_by_xcorr(
    #         pairs,
    #         xcorrs,
    #         corr_threshold,
    #     )
    #     if verbose:
    #         print(
    #             f"Dropping {len(corr_dropped_channels)} channels for cross-correlation "
    #             f"with threshold of {corr_threshold}"
    #         )
    #     corr_dropped_channels = [channel_idxs[i] for i in corr_dropped_channels]
    #     channel_idxs = channel_idxs[~np.isin(channel_idxs, corr_dropped_channels)]
    
    # if verbose:
    #     print(f"Writing {len(channel_idxs)} channels")

    #####################
    ###   ADD UNITS   ###
    #####################

    if heldout_unit_frac is not None and heldout_unit_frac > 0:
        nwbfile.add_unit_column('heldout', 'Whether the unit is held out in the test data (True) or not (False) in the Neural Latents Benchmark')
        heldout_idxs = rng.choice(channel_idxs, round(heldout_unit_frac*len(channel_idxs)), replace=False)
    else:
        heldout_idxs = None

    stime, neur = np.nonzero(spike_arr)
    for i in channel_idxs:
        # if ds['S1_unit_guide'][1, i] == 0:
            # print("Warning: unit indexing may be off. Unsorted unit still included after attempting exclusion.")
            # continue
        spike_times = (stime[neur == i] * bin_width).round(6)
        electrode = elecinds[ds['S1_unit_guide'][0, i]]
        unit_info = {
            'id': int(round(ds['S1_unit_guide'][0,i]*100 + ds['S1_unit_guide'][1,i])),
            'spike_times': spike_times,
            'electrodes': [electrode],
            'obs_intervals': np.array([[0.0, round(spike_arr.shape[0] * bin_width, 6)]]),
        }
        if heldout_idxs is not None:
            unit_info['heldout'] = (i in heldout_idxs)
        nwbfile.add_unit(**unit_info)

    # Write nwbfile
    if verbose:
        print(f"Saving output to {save_path}")
    with NWBHDF5IO(save_path, 'w') as io:
        io.write(nwbfile)

if __name__ == "__main__":
    file_path = Path('/Users/sherryan/area2_population_analysis/s1-kinematics/raeed/Chips_20170913_COactpas_TD.mat')

    
    # file_path = Path('/snel/share/share/data/raeed_s1/Han_20171116_COactpas_TD-2.mat')
    # file_path = Path('/home/fpei2/lvm/data/orig/Han_20171204_COactpas_TD_1ms.mat')
    # file_path = Path('/home/fpei2/lvm/data/orig/Han_20171207_COactpas_TD_1ms.mat')

    save_path = Path('/Users/sherryan/area2_population_analysis/s1-kinematics/actpas_NWB/Chips_20170913_COactpas_TD.nwb')

    area2_to_nwb(
        file_path=file_path,
        save_path=save_path,
        splits={'train': 1},
        heldout_unit_frac=None,
        stratified=False,
        drop_neurons=True,
        fr_threshold=0.1,
        corr_threshold=None,
        seed=0,
        verbose=True,
    )
