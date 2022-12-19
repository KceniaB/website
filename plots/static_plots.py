# -------------------------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------------------------

import matplotlib.pyplot as plt
from matplotlib.image import NonUniformImage
from matplotlib.lines import Line2D
from pathlib import Path
import numpy as np
import pandas as pd
import copy
from collections import OrderedDict
import seaborn as sns
import yaml
from scipy.stats import zscore

from brainbox.task.trials import find_trial_ids
from brainbox.task.passive import get_stim_aligned_activity
from brainbox.population.decode import xcorr
from brainbox.processing import bincount2D
from brainbox.behavior.wheel import velocity
from brainbox.ephys_plots import plot_brain_regions
from brainbox.plot_base import arrange_channels2banks, ProbePlot
from brainbox.behavior.training import plot_psychometric, plot_reaction_time, plot_reaction_time_over_trials
from ibllib.plots import Density
from ibllib.atlas.regions import BrainRegions
from iblutil.util import Bunch
import time

import one.alf.io as alfio


# -------------------------------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / 'static/data'
CACHE_DIR = ROOT_DIR / 'static/cache'
BRAIN_REGIONS = BrainRegions() 


# -------------------------------------------------------------------------------------------------
# Loading functions
# -------------------------------------------------------------------------------------------------

def load_clusters(pid):
    clusters = alfio.load_object(DATA_DIR.joinpath(pid), object='clusters')
    return clusters


def load_channels(pid):
    channels = alfio.load_object(DATA_DIR.joinpath(pid), object='channels')
    return channels


def load_spikes(pid):
    spikes = alfio.load_object(DATA_DIR.joinpath(pid), object='spikes')
    return spikes


def load_trials(eid):
    trials = alfio.load_object(DATA_DIR.joinpath(eid), object='trials')
    return trials


def load_cluster_waveforms(pid):
    wfs = np.load(DATA_DIR.joinpath(pid, 'clusters.waveforms.npy'))
    wf_chns = np.load(DATA_DIR.joinpath(
        pid, 'clusters.waveformsChannels.npy'))

    return wfs, wf_chns


def load_rms(pid):
    rms = np.load(DATA_DIR.joinpath(pid, '_iblqc_ephysChannels.apRMS.npy'))
    return rms[1, :]


def load_raw_data(pid):
    raw_data = np.load(DATA_DIR.joinpath(pid, 'raw_ephys_data.npy'))
    with open(DATA_DIR.joinpath(pid, 'raw_ephys_info.yaml'), 'r') as fp:
        raw_info = yaml.safe_load(fp)

    return raw_info, raw_data


def load_camera(eid, camera):
    camera = alfio.load_object(DATA_DIR.joinpath(eid), object=f'{camera}Camera')
    return camera


def load_licks(eid):
    licks = np.load(DATA_DIR.joinpath(eid, 'licks.times.npy'))
    return licks


def load_wheel(eid):
    wheel = alfio.load_object(DATA_DIR.joinpath(eid), object='wheel')
    return wheel


# -------------------------------------------------------------------------------------------------
# Filtering
# -------------------------------------------------------------------------------------------------


def _filter(obj, idx):
    obj = Bunch(copy.deepcopy(obj))
    for key in obj.keys():
        obj[key] = obj[key][idx]

    return obj


def filter_spikes_by_good_clusters(spikes):
    return _filter(spikes, spikes.good)


def filter_clusters_by_good_clusters(clusters):
    idx = np.where(clusters.label == 1)[0]
    return _filter(clusters, idx)


def filter_spikes_by_cluster_idx(spikes, cluster_idx):
    idx = np.where(spikes.clusters == cluster_idx)[0]

    return _filter(spikes, idx)


def filter_spikes_by_trial(spikes, tstart, tend):
    idx = np.where(np.bitwise_and(
        spikes.times >= tstart, spikes.times <= tend))[0]
    return _filter(spikes, idx)


def filter_spikes_by_sample(spikes, sample):
    for key in spikes.keys():
        spikes[key] = spikes[key][::sample]

    return spikes


def filter_trials_by_trial_idx(trials, trial_idx):
    return _filter(trials, trial_idx)


def filter_clusters_by_cluster_idx(clusters, cluster_idx):
    try:
        idx = np.where(clusters.cluster_id == cluster_idx)[0][0]
    except IndexError:
        return None
    return _filter(clusters, idx)


def filter_wfs_by_cluster_idx(waveforms, waveform_channels, clusters, cluster_idx):
    idx = np.where(clusters.cluster_id == cluster_idx)[0][0]
    return waveforms[idx], waveform_channels[idx]


def filter_features_by_pid(features, pid, column):
    feat = features[features['pid'] == pid]
    values = feat[column].values
    if len(values) == 0:
        return np.full(384, np.nan)
    else:
        return feat[column].values


# -------------------------------------------------------------------------------------------------
# Processing functions
# -------------------------------------------------------------------------------------------------

def bin_spikes(spike_times, align_times, pre_time, post_time, bin_size, weights=None):

    n_bins_pre = int(np.ceil(pre_time / bin_size))
    n_bins_post = int(np.ceil(post_time / bin_size))
    n_bins = n_bins_pre + n_bins_post
    tscale = np.arange(-n_bins_pre, n_bins_post + 1) * bin_size
    ts = np.repeat(align_times[:, np.newaxis], tscale.size, axis=1) + tscale
    epoch_idxs = np.searchsorted(spike_times, np.c_[ts[:, 0], ts[:, -1]])
    bins = np.zeros(shape=(align_times.shape[0], n_bins))

    for i, (ep, t) in enumerate(zip(epoch_idxs, ts)):
        xind = (
            np.floor((spike_times[ep[0]:ep[1]] - t[0]) / bin_size)).astype(np.int64)
        w = weights[ep[0]:ep[1]] if weights is not None else None
        r = np.bincount(xind, minlength=tscale.shape[0], weights=w)
        if w is not None:
            r_norm = np.bincount(xind, minlength=tscale.shape[0])
            r_norm[r_norm == 0] = 1
            r = r / r_norm
        bins[i, :] = r[:-1]

    tscale = (tscale[:-1] + tscale[1:]) / 2

    return bins, tscale


# Taken from phy
def _compute_histogram(
        data, x_max=None, x_min=None, n_bins=None, normalize=True, ignore_zeros=False):
    """Compute the histogram of an array."""
    assert x_min <= x_max
    assert n_bins >= 0
    n_bins = _clip(n_bins, 2, 1000000)
    bins = np.linspace(float(x_min), float(x_max), int(n_bins))
    if ignore_zeros:
        data = data[data != 0]
    histogram, _ = np.histogram(data, bins=bins)
    if not normalize:  # pragma: no cover
        return histogram, bins
    # Normalize by the integral of the histogram.
    hist_sum = histogram.sum() * (bins[1] - bins[0])
    return histogram / (hist_sum or 1.), bins


def _clip(x, a, b):
    return max(a, min(b, x))


# -------------------------------------------------------------------------------------------------
# Styling functions
# -------------------------------------------------------------------------------------------------

def set_figure_style(fig, margin_inches=0.8):
    x_inches, y_inches = fig.figure.get_size_inches()
    fig.subplots_adjust(margin_inches / x_inches, margin_inches / y_inches)
    return fig


def set_axis_style(ax, fontsize=12, **kwargs):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set_xlabel(kwargs.get('xlabel', None), fontsize=fontsize)
    ax.set_ylabel(kwargs.get('ylabel', None), fontsize=fontsize)
    ax.set_title(kwargs.get('title', None), fontsize=fontsize)

    return ax


def remove_spines(ax, spines=('left', 'right', 'top', 'bottom')):
    for sp in spines:
        ax.spines[sp].set_visible(False)

    return ax


def remove_frame(ax):

    ax.set_frame_on(False)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    return ax


# -------------------------------------------------------------------------------------------------
# Plotting functions
# -------------------------------------------------------------------------------------------------

class DataLoader:

    # Loading functions
    # ---------------------------------------------------------------------------------------------

    def __init__(self):
        self.session_df = pd.read_parquet(DATA_DIR.joinpath('session.table.pqt'))
        self.session_df = self.session_df.set_index('pid')

        # # Channels and brain acronyms.
        # self.channels_df = pd.read_parquet(DATA_DIR.joinpath('channels.pqt'))

        # load in the waveform tables
        self.features = pd.read_parquet(DATA_DIR.joinpath('raw_ephys_features.pqt'))
        self.features = self.features.reset_index()

    def session_init(self, pid):
        assert self.session_df is not None
        if pid not in self.session_df.index:
            raise ValueError(f"session {pid} is not in session.table.pqt")

        self.pid = pid
        self.load_session_data(pid)
        # self.get_session_details()
        self.compute_session_raster()

    def load_session_data(self, pid):
        """
        Load in the data associated with selected pid
        :param pid:
        :return:
        """
        self.session_info = self.session_df[self.session_df.index == pid].to_dict(orient='records')[0]
        self.eid = self.session_info['eid']
        self.spikes = filter_spikes_by_good_clusters(load_spikes(pid))
        self.trials = load_trials(self.eid)
        self.trial_intervals, self.trial_idx = self.compute_trial_intervals()
        self.clusters = load_clusters(pid)
        self.clusters_good = filter_clusters_by_good_clusters(self.clusters)
        self.cluster_wfs, self.cluster_wf_chns = load_cluster_waveforms(pid)
        self.channels = load_channels(pid)
        self.rms_chns = load_rms(pid)

        # # self.brain_regions is a string with the list of brain regions in a given session
        # self.brain_regions = self.channels_df.groupby("pid")["acronym"].unique().\
        #     transform(lambda x: ', '.join(sorted(x)))
        # self.brain_regions = self.brain_regions[self.brain_regions.index == pid][0]

        self.rms_ap = filter_features_by_pid(self.features, pid, 'rms_ap')
        self.lfp = filter_features_by_pid(self.features, pid, 'psd_delta')

        self.depth_lim = [0, 4000]
        self.amp_lim = [-10, 800]

    def get_session_details(self):
        """
        Get dict of metadata for session
        :return:
        """
        # TODO this isn't ordered after the request sent (on js side)
        details = OrderedDict()
        details['ID'] = self.pid
        details['Subject'] = self.session_info['subject']
        details['Lab'] = self.session_info['lab']
        details['DOB'] = self.session_info['dob']
        details['Recording date'] = self.session_info['date']
        details['Recording length'] = f'{int(np.max(self.spikes.times) / 60)} minutes'
        details['Probe name'] = self.session_info['probe']
        details['Probe type'] = self.session_info['probe_model']
        details['N trials'] = f'{self.trials.stimOn_times.size}'
        details['N spikes'] = f'{self.spikes.clusters.size}'
        details['N clusters'] = f'{self.clusters_good.cluster_id.size} good, {self.clusters.cluster_id.size} overall'
        details['eid'] = self.session_info['eid']
        details['pid'] = self.pid

        details['dset_bwm'] = self.session_info['2022_Q4_IBL_et_al_BWM']
        details['dset_rs'] = self.session_info['2022_Q2_IBL_et_al_RepeatedSite']

        # Sort by cluster depth.
        idx = np.argsort(self.clusters_good.depths)[::-1]

        # Internal fields used by the frontend.
        details['_trial_ids'] = [int(_) for _ in self.trial_idx]

        # Trial intervals.
        details['_trial_onsets'] = [float(_) if not np.isnan(_) else None for _ in self.trial_intervals[:, 0]]
        details['_trial_offsets'] = [float(_) if not np.isnan(_) else None for _ in self.trial_intervals[:, 1]]

        details['_cluster_ids'] = [int(_) for _ in self.clusters_good.cluster_id[idx]]
        details['_acronyms'] = self.clusters_good.acronym[idx].tolist()
        # details['_brain_regions'] = self.brain_regions
        # details['_brain_regions'] = sorted(set(details['_acronyms']))
        details['_colors'] = BRAIN_REGIONS.get(self.clusters_good.atlas_id[idx]).rgb.tolist()

        regions = sorted(set(BRAIN_REGIONS.get(self.clusters_good.atlas_id[idx]).name))
        regions_acronyms = sorted(set(BRAIN_REGIONS.get(self.clusters_good.atlas_id[idx]).acronym))
        regions += regions_acronyms
        regions = [_.lower() for _ in regions]
        regions = ', '.join(regions)

        details['_regions'] = regions
        details['_duration'] = np.max(self.spikes.times)

        return details

    def get_trial_details(self, trial_idx):
        trials = filter_trials_by_trial_idx(self.trials, trial_idx)

        def _get_block_probability(pLeft):
            if pLeft == 0.5:
                return 'neutral'
            elif pLeft == 0.8:
                return 'left'
            elif pLeft == 0.2:
                return 'right'

        details = OrderedDict()
        details['Trial #'] = trial_idx
        details['Contrast'] = np.nanmean([trials.contrastLeft, trials.contrastRight]) * 100
        details['Stim side'] = 'left' if np.isnan(trials.contrastRight) else 'right'
        details['Block proba'] = _get_block_probability(trials.probabilityLeft)
        details['Resp type'] = 'correct' if trials.feedbackType == 1 else 'incorrect'
        details['Resp time'] = f'{np.round((trials.feedback_times - trials.stimOn_times) * 1e3, 0)} ms'

        return details

    def get_cluster_details(self, cluster_idx):
        cluster = filter_clusters_by_cluster_idx(self.clusters, cluster_idx)
        if not cluster:
            return

        details = {
            'Cluster #': cluster_idx,
            'Brain region': BRAIN_REGIONS.id2acronym(cluster.atlas_id, mapping='Beryl')[0],
            'N spikes': len(filter_spikes_by_cluster_idx(self.spikes, cluster_idx)['times']),
            'Overall firing rate': f'{np.round(cluster["firing_rate"], 2)} Hz',
            'Max amplitude': f'{np.round(cluster["amp_max"] * 1e6, 2)} uV'
        }

        return details

    def compute_trial_intervals(self):
        # Find the nan trials and remove these
        nan_idx = np.bitwise_or(np.isnan(self.trials['stimOn_times']), np.isnan(self.trials['feedback_times']))
        trial_no = np.arange(len(self.trials['stimOn_times']))

        t0 = np.nanmax(np.c_[self.trials['stimOn_times'] - 1, self.trials['intervals'][:, 0]], axis=1)
        t1 = np.nanmin(np.c_[self.trials['feedback_times'] + 1.5, self.trials['intervals'][:, 1]], axis=1)

        # For the first trial limit t0 to be 0.18s before stimOn, to be consistent with the videos
        t0[0] = self.trials['stimOn_times'][0] - 0.18

        t0[nan_idx] = None
        t1[nan_idx] = None

        return np.c_[t0, t1], trial_no[~nan_idx]

    # Plotting functions
    # ---------------------------------------------------------------------------------------------

    def compute_session_raster(self, t_bin=0.1, d_bin=10):
        """
        Compute raster across whole duration of session
        :param t_bin:
        :param d_bin:
        :return:
        """
        kp_idx = ~np.isnan(self.spikes.depths)

        self.session_raster, self.t_vals, self.d_vals = \
            bincount2D(self.spikes.times[kp_idx], self.spikes.depths[kp_idx], t_bin, d_bin, ylim=[0, 3840])

        self.session_raster = self.session_raster / t_bin

    def get_brain_regions(self, restrict_labels=True, mapping='Beryl'):
        atlas_ids = BRAIN_REGIONS.id2id(self.channels['brainLocationIds_ccf_2017'], mapping=mapping)
        regions, region_labels, region_colours = \
            plot_brain_regions(channel_ids=atlas_ids, channel_depths=self.channels.localCoordinates[:, 1],
                               brain_regions=BRAIN_REGIONS, display=False)
        if restrict_labels:
            # Remove any void or root region labels and those that are less than 60 um
            reg_idx = np.where(~np.bitwise_or(np.isin(region_labels[:, 1], ['void', 'root']),
                                              (regions[:, 1] - regions[:, 0]) < 150))[0]
            region_labels = region_labels[reg_idx, :]

        return regions, region_labels, region_colours

    def plot_brain_regions(self, ax=None, restrict_labels=True):

        fig = ax.get_figure()

        regions, region_labels, region_colours = self.get_brain_regions(restrict_labels=restrict_labels)
        for reg, col in zip(regions, region_colours):
            height = np.abs(reg[1] - reg[0])
            color = col / 255
            ax.bar(x=0.5, height=height, width=1, color=color, bottom=reg[0], edgecolor='w')

        ax.yaxis.tick_right()
        ax.set_yticks(region_labels[:, 0].astype(int))
        ax.yaxis.set_tick_params(labelsize=10)
        ax.set_yticklabels(region_labels[:, 1])
        ax.set_ylim(*self.depth_lim)

        ax.get_xaxis().set_visible(False)

        remove_spines(ax)

        return fig

    def plot_ap_rms(self, ax=None, ax_cbar=None):

        if ax is None:
            fig, axs = plt.subplots(2, 1, figsize=(9, 6), gridspec_kw={'height_ratios': [1, 10]})
            ax = axs[1]
            ax_cbar = axs[0]
        else:
            fig = ax.get_figure()

        # TODO use actual data
        data_bank, x_bank, y_bank = arrange_channels2banks(self.rms_ap * 1e6, self.channels.localCoordinates, depth=None,
                                                           pad=True, x_offset=1)
        data = ProbePlot(data_bank, x=x_bank, y=y_bank, cmap='plasma')
        data.set_labels(ylabel='Depth (um)', clabel=f'AP rms (uV)')
        clim = np.nanquantile(np.concatenate([np.squeeze(np.ravel(d)) for d in data_bank]).ravel(), [0.1, 0.9])
        data.set_clim(clim)

        data = data.convert2dict()
        for (x, y, dat) in zip(data['data']['x'], data['data']['y'], data['data']['c']):
            im = NonUniformImage(ax, interpolation='nearest', cmap=data['cmap'])
            im.set_clim(data['clim'][0], data['clim'][1])
            im.set_data(x, y, dat.T)
            ax.images.append(im)

        ax.set_xlim(data['xlim'][0], data['xlim'][1])
        ax.set_ylim(*self.depth_lim)
        ax.set_xlabel(data['labels']['xlabel'])
        ax.set_ylabel(data['labels']['ylabel'])
        ax.set_title(data['labels']['title'])

        ax.get_xaxis().set_visible(False)
        remove_spines(ax, spines=['right', 'top', 'bottom'])

        cbar = fig.colorbar(im, orientation="horizontal", ax=ax_cbar)
        cbar.set_label(data['labels']['clabel'])
        remove_frame(ax_cbar)

        return fig

    def plot_lfp_spectrum(self, ax=None, ax_cbar=None):

        if ax is None:
            fig, axs = plt.subplots(2, 1, figsize=(9, 6), gridspec_kw={'height_ratios': [1, 10]})
            ax = axs[1]
            ax_cbar = axs[0]
        else:
            fig = ax.get_figure()

        data_bank, x_bank, y_bank = arrange_channels2banks(self.lfp, self.channels.localCoordinates, depth=None,
                                                           pad=True, x_offset=1)
        data = ProbePlot(data_bank, x=x_bank, y=y_bank, cmap='viridis')
        data.set_labels(ylabel='Depth (um)', clabel=f'LFP power (dB)')
        clim = np.nanquantile(np.concatenate([np.squeeze(np.ravel(d)) for d in data_bank]).ravel(),
                              [0.1, 0.9])
        data.set_clim(clim)

        data = data.convert2dict()
        for (x, y, dat) in zip(data['data']['x'], data['data']['y'], data['data']['c']):
            im = NonUniformImage(ax, interpolation='nearest', cmap=data['cmap'])
            im.set_clim(data['clim'][0], data['clim'][1])
            im.set_data(x, y, dat.T)
            ax.images.append(im)

        ax.set_xlim(data['xlim'][0], data['xlim'][1])
        ax.set_ylim(*self.depth_lim)
        ax.set_xlabel(data['labels']['xlabel'])
        ax.set_ylabel(data['labels']['ylabel'])
        ax.set_title(data['labels']['title'])

        ax.get_xaxis().set_visible(False)
        remove_spines(ax, spines=['right', 'top', 'bottom'])

        cbar = fig.colorbar(im, orientation="horizontal", ax=ax_cbar)
        cbar.set_label(data['labels']['clabel'])
        remove_frame(ax_cbar)

        return fig

    def plot_raw_data(self, axs=None, raster=True):

        if axs is None:
            fig, axs = plt.subplots(1, 4, figsize=(9, 6))
        else:
            fig = axs[0].get_figure()

        info, raw_ephys = load_raw_data(self.pid)
        spikes = load_spikes(self.pid)

        times = info['t']
        ts = info['t_offset']
        te = info['t_offset'] + info['t_display']
        fs = info['fs']

        vmin = -0.000050
        vmax = 0.000050
        cmap = 'Greys'

        if raster:
            ax0 = axs[0]
            self.plot_session_raster(ax=ax0)
            ax0.set_ylim(20, 3840)
            ax0.vlines(times, *ax0.get_ylim(), color='k', ls='--')

        for iT, time in enumerate(times):
            spike_idx = slice(*np.searchsorted(spikes['samples'], [int((time + ts) * fs), int((time + te) * fs)]))
            spike_channels = self.clusters['channels'][spikes['clusters'][spike_idx]]
            spike_times = (spikes['samples'][spike_idx] / fs - (time + ts)) * 1000
            spike_labels = self.clusters['label'][spikes['clusters'][spike_idx]]

            ax = axs[iT + 1] if raster else axs[iT]

            _ = Density(-raw_ephys[:, :, iT], fs=fs, taxis=1, ax=ax, vmin=vmin, vmax=vmax, cmap=cmap)
            ax.scatter(spike_times[spike_labels != 1], spike_channels[spike_labels != 1], c='r', alpha=0.8, s=3)
            ax.scatter(spike_times[spike_labels == 1], spike_channels[spike_labels == 1], c='g', alpha=0.8, s=3)
            ax.set_title(f'T = {time} s')
            if not raster and iT != 0:
                ax.get_yaxis().set_visible(False)
            else:
                ax.images[0].set_extent([0, 50, 20, 3840])
                ax.set_ylim(20, 3840)
                ax.set_ylabel('Depth (um)')

        return fig

    def plot_session_raster(self, cluster_idx=None, trial_idx=None, ax=None, xlabel='Time (s)'):

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(9, 6))
        else:
            fig = ax.get_figure()

        ax.imshow(self.session_raster,
                  extent=np.r_[np.min(self.t_vals), np.max(self.t_vals), np.min(self.d_vals), np.max(self.d_vals)],
                  aspect='auto', origin='lower', vmax=50, cmap='binary')

        ax.set_xlim(0, np.max(self.t_vals))
        ax.set_ylim(*self.depth_lim)
        set_axis_style(ax, xlabel=xlabel, ylabel='Depth (um)')

        if cluster_idx is not None:
            # TODO Allen colours
            spikes = filter_spikes_by_cluster_idx(self.spikes, cluster_idx)
            ax.scatter(spikes.times, spikes.depths, s=spikes.sizes,
                       facecolors='none', edgecolors='r')

        if trial_idx is not None:
            trials = filter_trials_by_trial_idx(self.trials, trial_idx)
            ax.axvline(trials['intervals'][0], *ax.get_ylim(), c='k', ls='--')
            ax.text(trials['intervals'][0], 1.01, f'Trial {trial_idx}', c='k', rotation=45,
                    rotation_mode='anchor', ha='left', transform=ax.get_xaxis_transform())

        return fig


    def plot_wheel_trace(self, trial_idx, ax=None): 
        import brainbox.behavior.wheel as wh 
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(9, 6))
        else:
            fig = ax.get_figure()

        wheel = load_wheel(self.eid)
        speed = velocity(wheel.timestamps, wheel.position)

        trials = filter_trials_by_trial_idx(self.trials, trial_idx)
        t0 = self.trial_intervals[trial_idx, 0]
        t1 = self.trial_intervals[trial_idx, 1]

        goCue = self.trials.goCue_times[trial_idx]
        responses = self.trials.response_times[trial_idx]
        firstMovement = self.trials.firstMovement_times[trial_idx]

        # Cut up the wheel vectors
        pos, t = wh.interpolate_position(wheel.timestamps, wheel.position)
        traces = wh.traces_by_trial(t, pos, start=[t0], end=[t1])[0]

        ax.plot(traces[0], traces[1], 'k-') 
        ax.axvline(x=goCue, color='g', label='go cue', linestyle=':')
        ax.axvline(x=responses, color='r', label='response', linestyle=':')
        ax.axvline(x=firstMovement, color='b', label='1st movement', linestyle=':') 
        ax.legend() 
        ax.set_title('Trial #'+str(trial_idx))

        # Add labels
        ax.set_xlabel('time / sec')
        ax.set_ylabel('position / rad')

        return ax






    def plot_trial_raster(self, trial_idx, cluster_idx=None, ax=None, xlabel='Time (s)', ylabel='Depth (um)'):

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(9, 6))
        else:
            fig = ax.get_figure()

        trials = filter_trials_by_trial_idx(self.trials, trial_idx)
        t0 = self.trial_intervals[trial_idx, 0]
        t1 = self.trial_intervals[trial_idx, 1]

        spikes = filter_spikes_by_trial(self.spikes, t0, t1)

        t_bin = 0.005
        d_bin = 5
        kp_idx = ~np.isnan(spikes.depths)

        raster, t_vals, d_vals = bincount2D(spikes.times[kp_idx], spikes.depths[kp_idx], t_bin, d_bin, ylim=[0, 3840])
        raster = raster / t_bin

        ax.imshow(raster, extent=np.r_[np.min(t_vals), np.max(t_vals), np.min(d_vals), np.max(d_vals)],
                  aspect='auto', origin='lower', vmax=50, cmap='binary')

        ax.set_ylim(*self.depth_lim)
        ax.set_xlim(np.min(spikes.times), np.max(spikes.times))

        if cluster_idx is not None:
            spikes = filter_spikes_by_cluster_idx(spikes, cluster_idx)
            ax.scatter(spikes.times, spikes.depths, s=spikes.sizes, c='r')

        self.add_trial_events_to_raster(ax, trials)
        set_axis_style(ax, xlabel=xlabel, ylabel=ylabel)

        return fig

    def plot_psychometric_curve(self, ax=None, ax_legend=None):

        if ax is None:
            fig, axs = plt.subplots(1, 2, figsize=(6, 6), gridspec_kw={'width_ratios': [3, 1]})
            ax = axs[0]
            ax_legend = axs[1]
        else:
            fig = ax.get_figure()

        plot_psychometric(self.trials, ax=ax)
        set_axis_style(ax, xlabel='Contrasts', ylabel='Probability Choosing Right')

        ax.get_legend().remove()

        legend_elements = [Line2D([0], [0], color='w', lw=0, label='20 % of trials on left side'),
                           Line2D([0], [0], color='w', lw=0, label='equal % of trials on both sides'),
                           Line2D([0], [0], color='w', lw=0, label='80 % of trials on left side'),
                           Line2D([0], [0], color='w', marker='o', markerfacecolor='k', label='data', markersize=10),
                           Line2D([0], [0], color='k', lw=2, label='model fit')]

        leg = ax_legend.legend(handles=legend_elements, loc=4, frameon=False)
        remove_frame(ax_legend)
        cmap = sns.diverging_palette(20, 220, n=3, center="dark")

        for text, hand in zip(leg.get_texts(), leg.legendHandles):
            if '80 %' in text.get_text():
                text.set_color(cmap[2])
                text.set_position((-40, 0))
                hand.set_visible(False)
            elif '20 %' in text.get_text():
                text.set_color(cmap[0])
                text.set_position((-40, 0))
            elif 'equal %' in text.get_text():
                text.set_color(cmap[1])
                text.set_position((-40, 0))

        return fig

    def plot_chronometric_curve(self, ax=None, ax_legend=None):

        if ax is None:
            fig, axs = plt.subplots(1, 2, figsize=(6, 6), gridspec_kw={'width_rations': [3, 1]})
            ax = axs[0]
            ax_legend = axs[1]
        else:
            fig = ax.get_figure()

        plot_reaction_time(self.trials, ax=ax)
        set_axis_style(ax, xlabel='Contrasts', ylabel='Reaction time (s)')
        leg = ax.get_legend()
        h = leg.legendHandles
        l = [str(x._text) for x in leg.texts]
        ax.get_legend().remove()
        if ax_legend is not None:
            ax_legend.legend(handles=h, labels=l, frameon=False, loc=7)
            remove_frame(ax_legend)

        return fig

    def plot_reaction_time(self, ax=None):

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        else:
            fig = ax.get_figure()

        plot_reaction_time_over_trials(self.trials, ax=ax)
        set_axis_style(ax, xlabel='Trial number', ylabel='Reaction time (s)')

        return fig

    def add_trial_events_to_raster(self, ax, trials):

        events = ['goCue_times', 'firstMovement_times', 'feedback_times']
        colors = ['b', 'g', 'r']
        labels = ['Go Cue', 'First Move', 'Feedback']
        trans = ax.get_xaxis_transform()

        for e, c, l in zip(events, colors, labels):
            ax.axvline(trials[e], *ax.get_ylim(), c=c)
            ax.text(trials[e], 1.01, l, c=c, rotation=45,
                    rotation_mode='anchor', ha='left', transform=trans)

        return ax

    def plot_good_bad_clusters(self, ax=None, ax_legend=None, xlabel='Amplitude (uV)', ylabel='Depth (um)'):
        if ax is None:
            fig, axs = plt.subplots(2, 1, figsize=(4, 6), gridspec_kw={'height_ratios': [1, 8]})
            ax = axs[1]
            ax_legend = axs[0]
        else:
            fig = ax.get_figure()

        mua = ax.scatter(self.clusters.amps * 1e6, self.clusters.depths, c='r')
        good = ax.scatter(self.clusters_good.amps * 1e6, self.clusters_good.depths, c='g')

        ax.set_ylim(*self.depth_lim)
        ax.set_xlim(*self.amp_lim)
        set_axis_style(ax, xlabel=xlabel, ylabel=ylabel)

        ax_legend.legend(handles=[mua, good], labels=['mua', 'good'], frameon=False, bbox_to_anchor=(0.8, 0.2))
        remove_frame(ax_legend)

        return fig

    def plot_spikes_amp_vs_depth_vs_firing_rate(self, cluster_idx=None, ax=None, ax_cbar=None,
                                                xlabel='Amplitude (uV)', ylabel='Depth (um)'):

        if ax is None:
            fig, axs = plt.subplots(2, 1, figsize=(4, 6), gridspec_kw={'height_ratios': [1, 8]})
            ax = axs[1]
            ax_cbar = axs[0]
        else:
            fig = ax.get_figure()

        scat = ax.scatter(self.clusters_good.amps * 1e6, self.clusters_good.depths, c=self.clusters_good.firing_rate, cmap='hot',
                          edgecolors='grey')
        if cluster_idx is not None:
            clusters = filter_clusters_by_cluster_idx(self.clusters_good, cluster_idx)
            if clusters is not None:
                ax.scatter(clusters.amps * 1e6, clusters.depths, c=clusters.firing_rate, cmap='hot', edgecolors='grey',
                           linewidths=2, s=80)
        ax.set_ylim(*self.depth_lim)
        ax.set_xlim(*self.amp_lim)
        set_axis_style(ax, xlabel=xlabel, ylabel=ylabel)

        cbar = fig.colorbar(scat, ax=ax_cbar, orientation="horizontal")
        cbar.set_label('Firing Rate (Hz)')
        remove_frame(ax_cbar)

        return fig

    def plot_spikes_amp_vs_depth(self, cluster_idx, ax=None, xlabel='Amplitude (uV)', ylabel=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(4, 6))
        else:
            fig = ax.get_figure()

        col = (BRAIN_REGIONS.get(self.clusters_good.atlas_id).rgb / 255).tolist()
        scat = ax.scatter(self.clusters_good.amps * 1e6, self.clusters_good.depths, c=col, edgecolors='grey')
        clusters = filter_clusters_by_cluster_idx(self.clusters_good, cluster_idx)
        col_clus = (BRAIN_REGIONS.get(clusters.atlas_id).rgb / 255).tolist()
        if clusters is not None:
            ax.scatter(clusters.amps * 1e6, clusters.depths, c=col_clus, edgecolors='black',
                       linewidths=2, s=80)

        _, region_labels, _ = self.get_brain_regions()
        ax.set_yticks(region_labels[:, 0].astype(int))
        ax.yaxis.set_tick_params(labelsize=10)
        ax.set_yticklabels(region_labels[:, 1])

        ax.set_ylim(*self.depth_lim)
        ax.set_xlim(*self.amp_lim)
        set_axis_style(ax, xlabel=xlabel, ylabel=ylabel)

        return fig

    def plot_spikes_fr_vs_depth(self, cluster_idx, ax=None, xlabel='Firing Rate Hz', ylabel='Depth (um)'):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(4, 6))
        else:
            fig = ax.get_figure()

        ax.scatter(self.clusters_good.firing_rate, self.clusters_good.depths,
                   facecolors='none', edgecolors='grey')
        clusters = filter_clusters_by_cluster_idx(self.clusters_good, cluster_idx)
        if clusters is not None:
            ax.scatter(clusters.firing_rate, clusters.depths, c='r')
        ax.set_ylim(*self.depth_lim)
        ax.set_xlim(*self.amp_lim)
        set_axis_style(ax, xlabel=xlabel, ylabel=ylabel)

        return fig

    def plot_event_aligned_activity(self, axs=None, ax_cbar=None, ylabel='Depth (um)'):

        if axs is None:
            fig, axs = plt.subplots(1, 4, figsize=(4, 6), gridspec_kw={'width_ratios': [4, 4, 4, 1]})
            ax_cbar = axs[-1]
        else:
            fig = axs[0].get_figure()

        stim_events = {'Stim On': self.trials['stimOn_times'],
                       'First Move': self.trials['firstMovement_times'],
                       'Feedback': self.trials['feedback_times']}
        kp_idx = ~np.isnan(self.spikes.depths)
        pre_stim = 0.4
        post_stim = 1
        data = get_stim_aligned_activity(stim_events, self.spikes.times[kp_idx], self.spikes.depths[kp_idx], pre_stim=pre_stim,
                                         post_stim=post_stim, y_lim=[0, 3840])

        for i, (key, d) in enumerate(data.items()):
            im = axs[i].imshow(d, aspect='auto', extent=np.r_[-1 * pre_stim, post_stim, 0, 3840], cmap='bwr', vmax=10, vmin=-10,
                               origin='lower')
            if i == 0:
                set_axis_style(axs[i], xlabel=f'T from {key} (s)', ylabel=ylabel)
            else:
                axs[i].set_yticklabels([])
                set_axis_style(axs[i], xlabel=f'T from {key} (s)')

            axs[i].set_ylim(*self.depth_lim)
            axs[i].set_xlim(-1 * pre_stim, post_stim)
            axs[i].axvline(0, *axs[i].get_ylim(), c='k', ls='--', zorder=10)

        cbar = fig.colorbar(im, ax=ax_cbar, orientation="horizontal", fraction=0.8)
        cbar.set_label('Firing Rate (Z-score)')
        remove_frame(ax_cbar)

        return fig

    def plot_dlc_feature_raster(self, camera, feature, axs=None, xlabel='T from Stim On (s)', ylabel0='Speed (px/s)',
                                ylabel1='Sorted Trial Number', title=None, zscore_flag=False, norm=False):

        camera = load_camera(self.eid, camera)
        feature = camera.computedFeatures[feature]

        if zscore_flag:
            feature = zscore(feature, nan_policy='omit')

        trial_idx, dividers = find_trial_ids(self.trials, sort='choice')
        fig, axs = self.single_cluster_raster(camera.times, self.trials['stimOn_times'], trial_idx, dividers, ['b', 'r'],
                                              ['correct', 'incorrect'], weights=feature, axs=axs, fr=False, norm=norm)

        set_axis_style(axs[1], xlabel=xlabel, ylabel=ylabel1)
        set_axis_style(axs[0], ylabel=ylabel0, title=title)

        return fig

    def plot_lick_raster(self, axs=None, xlabel='T from Feedback (s)', ylabel0='Licks (count)',
                         ylabel1='Sorted Trial Number', title=None):

        licks = load_licks(self.eid)

        trial_idx, dividers = find_trial_ids(self.trials, sort='choice')
        fig, axs = self.single_cluster_raster(licks, self.trials['stimOn_times'], trial_idx, dividers, ['b', 'r'],
                                              ['correct', 'incorrect'], axs=axs, fr=False)

        set_axis_style(axs[1], xlabel=xlabel, ylabel=ylabel1)
        set_axis_style(axs[0], ylabel=ylabel0, title=title)

        return fig

    def plot_wheel_raster(self, axs=None, xlabel='T from First Move (s)', ylabel0='Wheel velocity (rad/s)',
                          ylabel1='Sorted Trial Number', title=None):

        wheel = load_wheel(self.eid)
        speed = velocity(wheel.timestamps, wheel.position)

        trial_idx, dividers = find_trial_ids(self.trials, sort='side')
        fig, axs = self.single_cluster_raster(
            wheel.timestamps, self.trials['firstMovement_times'], trial_idx, dividers, ['g', 'y'], ['left', 'right'],
            weights=speed, fr=False, axs=axs)

        set_axis_style(axs[1], xlabel=xlabel, ylabel=ylabel1)
        set_axis_style(axs[0], ylabel=ylabel0, title=title)

        return fig

    def plot_left_right_single_cluster_raster(self, cluster_idx, axs=None, xlabel='T from First Move (s)',
                                              ylabel0='Firing Rate (Hz)', ylabel1='Sorted Trial Number'):

        spikes = filter_spikes_by_cluster_idx(self.spikes, cluster_idx)
        trial_idx, dividers = find_trial_ids(self.trials, sort='side')
        fig, axs = self.single_cluster_raster(
            spikes.times, self.trials['firstMovement_times'], trial_idx, dividers, ['g', 'y'], ['left', 'right'], axs=axs)

        set_axis_style(axs[1], xlabel=xlabel, ylabel=ylabel1)
        set_axis_style(axs[0], ylabel=ylabel0)

        return fig

    def plot_correct_incorrect_single_cluster_raster(self, cluster_idx, axs=None, xlabel='T from Feedback (s)',
                                                     ylabel0='Firing Rate (Hz)', ylabel1='Sorted Trial Number'):

        spikes = filter_spikes_by_cluster_idx(self.spikes, cluster_idx)
        trial_idx, dividers = find_trial_ids(self.trials, sort='choice')
        fig, axs = self.single_cluster_raster(spikes.times, self.trials['feedback_times'], trial_idx, dividers, ['b', 'r'],
                                              ['correct', 'incorrect'], axs=axs)

        set_axis_style(axs[1], xlabel=xlabel, ylabel=ylabel1)
        set_axis_style(axs[0], ylabel=ylabel0)

        return fig

    def plot_block_single_cluster_raster(self, cluster_idx, axs=None, xlabel='T from Stim On (s)',
                                         ylabel0='Firing Rate (Hz)', ylabel1='Sorted Trial Number'):

        spikes = filter_spikes_by_cluster_idx(self.spikes, cluster_idx)
        trial_idx = np.arange(len(self.trials['probabilityLeft']))
        dividers = np.where(np.diff(self.trials['probabilityLeft']) != 0)[0]

        blocks = self.trials['probabilityLeft'][np.r_[0, dividers + 1]]
        cmap = sns.diverging_palette(20, 220, n=3, center="dark")
        colours = np.full((blocks.shape[0], 3), np.array([*cmap[0]]))
        colours[np.where(blocks == 0.5)] = np.array([*cmap[1]])
        colours[np.where(blocks == 0.8)] = np.array([*cmap[2]])

        fig, axs = self.single_cluster_raster(spikes.times, self.trials['stimOn_times'], trial_idx, list(dividers), colours,
                                              blocks, axs=axs)

        set_axis_style(axs[1], xlabel=xlabel, ylabel=ylabel1)
        set_axis_style(axs[0], ylabel=ylabel0)

        return fig

    def plot_contrast_single_cluster_raster(self, cluster_idx, axs=None, xlabel='T from Stim On (s)',
                                            ylabel0='Firing Rate (Hz)', ylabel1='Sorted Trial Number'):

        spikes = filter_spikes_by_cluster_idx(self.spikes, cluster_idx)
        contrasts = np.nanmean(np.c_[self.trials.contrastLeft, self.trials.contrastRight], axis=1)
        trial_idx = np.argsort(contrasts)
        dividers = list(np.where(np.diff(np.sort(contrasts)) != 0)[0])
        labels = [str(_ * 100) for _ in np.unique(contrasts)]
        colors = ['0.9', '0.7', '0.5', '0.3', '0.0']
        fig, axs = self.single_cluster_raster(
            spikes.times, self.trials['stimOn_times'], trial_idx, dividers, colors, labels, axs=axs)

        set_axis_style(axs[1], xlabel=xlabel, ylabel=ylabel1)
        set_axis_style(axs[0], ylabel=ylabel0)

        return fig

    def single_cluster_raster(self, spike_times, events, trial_idx, dividers, colors, labels, weights=None, fr=True, norm=False,
                              axs=None):

        pre_time = 0.4
        post_time = 1
        raster_bin = 0.01
        psth_bin = 0.05
        raster, t_raster = bin_spikes(
            spike_times, events, pre_time=pre_time, post_time=post_time, bin_size=raster_bin, weights=weights)
        psth, t_psth = bin_spikes(
            spike_times, events, pre_time=pre_time, post_time=post_time, bin_size=psth_bin, weights=weights)

        if fr:
            psth = psth / psth_bin

        if norm:
            psth = psth - np.repeat(psth[:, 0][:, np.newaxis], psth.shape[1], axis=1)
            raster = raster - np.repeat(raster[:, 0][:, np.newaxis], raster.shape[1], axis=1)

        dividers = [0] + dividers + [len(trial_idx)]
        if axs is None:
            fig, axs = plt.subplots(2, 1, figsize=(4, 6), gridspec_kw={'height_ratios': [1, 3], 'hspace': 0}, sharex=True)
        else:
            fig = axs[0].get_figure()

        label, lidx = np.unique(labels, return_index=True)
        label_pos = []
        for lab, lid in zip(label, lidx):
            idx = np.where(np.array(labels) == lab)[0]
            for iD in range(len(idx)):
                if iD == 0:
                    t_ids = trial_idx[dividers[idx[iD]] + 1:dividers[idx[iD] + 1] + 1]
                    t_ints = dividers[idx[iD] + 1] - dividers[idx[iD]]
                else:
                    t_ids = np.r_[t_ids, trial_idx[dividers[idx[iD]] + 1:dividers[idx[iD] + 1] + 1]]
                    t_ints = np.r_[t_ints, dividers[idx[iD] + 1] - dividers[idx[iD]]]

            psth_div = np.nanmean(psth[t_ids], axis=0)
            std_div = np.nanstd(psth[t_ids], axis=0) / np.sqrt(len(t_ids))

            axs[0].fill_between(t_psth, psth_div - std_div,
                                psth_div + std_div, alpha=0.4, color=colors[lid])
            axs[0].plot(t_psth, psth_div, alpha=1, color=colors[lid])

            lab_max = idx[np.argmax(t_ints)]
            label_pos.append((dividers[lab_max + 1] - dividers[lab_max]) / 2 + dividers[lab_max])

        axs[1].imshow(raster[trial_idx], cmap='binary', origin='lower',
                      extent=[np.min(t_raster), np.max(t_raster), 0, len(trial_idx)], aspect='auto')

        width = raster_bin * 4
        for iD in range(len(dividers) - 1):
            axs[1].fill_between([post_time + raster_bin / 2, post_time + raster_bin / 2 + width],
                                [dividers[iD + 1], dividers[iD + 1]], [dividers[iD], dividers[iD]], color=colors[iD])

        axs[1].set_xlim([-1 * pre_time, post_time + raster_bin / 2 + width])
        secax = axs[1].secondary_yaxis('right')

        secax.set_yticks(label_pos)
        secax.set_yticklabels(label, rotation=90,
                              rotation_mode='anchor', ha='center')
        for ic, c in enumerate(np.array(colors)[lidx]):
            secax.get_yticklabels()[ic].set_color(c)

        remove_spines(axs[1], spines=['right', 'top'])
        remove_spines(axs[0], spines=['right', 'top'])

        axs[0].axvline(0, *axs[0].get_ylim(), c='k', ls='--', zorder=10)  # TODO this doesn't always work
        axs[1].axvline(0, *axs[1].get_ylim(), c='k', ls='--', zorder=10)

        return fig, axs

    def plot_cluster_waveforms(self, cluster_idx, ax=None):

        wfs, wf_chns = filter_wfs_by_cluster_idx(
            self.cluster_wfs, self.cluster_wf_chns, self.clusters, cluster_idx)

        clusters = filter_clusters_by_cluster_idx(self.clusters, cluster_idx)
        amp_norm = self.rms_chns[clusters.channels]
        # Divide all channels by the noise on the max channel
        wfs_norm = wfs / amp_norm

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        else:
            fig = ax.get_figure()

        wf_chn_pos = np.c_[self.channels['localCoordinates'][:, 0]
                           [wf_chns], self.channels['localCoordinates'][:, 1][wf_chns]]

        x_max = np.max(wf_chn_pos[:, 0])
        x_min = np.min(wf_chn_pos[:, 0])

        y_max = np.max(wf_chn_pos[:, 1])
        y_min = np.min(wf_chn_pos[:, 1])

        n_xpos = np.unique(wf_chn_pos[:, 0]).size
        n_ypos = np.unique(wf_chn_pos[:, 1]).size
        time_len = wfs.shape[0]

        # Taken from dj code
        # https://github.com/int-brain-lab/IBL-pipeline/blob/master/ibl_pipeline/plotting/ephys_plotting.py#L252
        # time (in ms) * x_scale (in um/ms) + x_start(coord[0]) = position
        # the waveform takes 0.9 of each x interval between adjacent channels
        dt = 1 / 30
        x_scale = (x_max - x_min) / (n_xpos - 1) * 0.9 / (dt * time_len)

        wf_peak = np.max(np.abs(wfs)) * 1e6
        if wf_peak < 100:
            wf_peak = 100

        wf_peak_norm = np.max(np.abs(wfs_norm))

        # peak waveform takes 2 times of each y interval between adjacent channels
        y_scale = (y_max - y_min) / (n_ypos - 1) / wf_peak * 1.5
        y_scale_norm = (y_max - y_min) / (n_ypos - 1) / wf_peak_norm * 1.5

        time = np.arange(time_len) * dt

        for wf, wf_norm, pos in zip(wfs.T * 1e6, wfs_norm.T, wf_chn_pos):
            ax.plot(time * x_scale + pos[0], wf * y_scale + pos[1], color='grey')

        # Scale bar
        x0 = x_min - 6
        y0 = y_min - 15
        ax.text(x0, y0 - 10, '1ms')
        ax.text(x0 - 6, y0, '100uV', rotation='vertical')
        ax.plot([x0, x0 + x_scale], [y0, y0], color='grey')
        ax.plot([x0, x0], [y0, y0 + y_scale * 100], color='grey')

        y0 = y0 + y_scale * 100 + 20
        ax.text(x0 - 6, y0, '5 x noise', rotation='vertical')
        ax.plot([x0, x0], [y0, y0 + y_scale_norm * 5], color='grey')

        ax.set_xlim([x_min - 10, x_max + 15])
        ax.set_ylim([y_min - 30, y_max + 30])

        # add in distance between electrodes in y axis
        ax.plot([x0, x0], [y_max - 40, y_max], color='black')
        ax.plot([x0 - 2, x0 + 2], [y_max - 40, y_max - 40], color='black')
        ax.plot([x0 - 2, x0 + 2], [y_max, y_max], color='black')
        ax.text(x0 - 6, y_max - 35, '40 um', rotation='vertical')

        remove_frame(ax)

        return fig

    def plot_channel_probe_location(self, cluster_idx, ax=None):

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        else:
            fig = ax.get_figure()

        _, wf_chns = filter_wfs_by_cluster_idx(
            self.cluster_wfs, self.cluster_wf_chns, self.clusters, cluster_idx)

        probe = np.ones((len(np.unique(self.channels.localCoordinates[:, 1])),
                         len(np.unique(self.channels.localCoordinates[:, 0])))) * 10
        x0 = np.min(self.channels.localCoordinates[:, 0])
        xdiff = np.min(np.abs(np.diff(self.channels.localCoordinates[:, 0])))

        y0 = np.min(self.channels.localCoordinates[:, 1])
        yvals = np.abs(np.diff(self.channels.localCoordinates[:, 1]))
        ydiff = np.min(yvals[yvals > 0])

        coords_x = ((self.channels.localCoordinates[wf_chns, 0] - x0) / xdiff).astype(int)
        coords_y = ((self.channels.localCoordinates[wf_chns, 1] - y0) / ydiff).astype(int)

        for x in np.unique(coords_x):
            idx = np.where(coords_x == x)[0]
            probe[coords_y[idx], x] = 100

        ax.imshow(probe, extent=[0, 160, np.min(self.channels.localCoordinates[:, 1]),
                                 np.max(self.channels.localCoordinates[:, 1])], origin='lower', cmap='binary', vmin=0, vmax=100)
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.spines['top'].set(alpha=0.2)
        ax.spines['bottom'].set(alpha=0.2)
        ax.spines['left'].set(alpha=0.2)
        ax.spines['right'].set(alpha=0.2)

        return fig

    def plot_autocorrelogram(self, cluster_idx, ax=None, xlabel='Time from spike (ms)', ylabel='Spike count'):

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        else:
            fig = ax.get_figure()

        spikes = filter_spikes_by_cluster_idx(self.spikes, cluster_idx)

        x_corr = xcorr(spikes.times, spikes.clusters, 1 / 1e3, 50 / 1e3)
        corr = x_corr[0, 0, :]
        # m_corr = np.max(corr)
        # if m_corr == 0:
        #     m_corr = 1
        # corr = corr / m_corr  # normalise
        ax.bar(np.arange(corr.size) - 50 / 2, height=corr, width=0.8, color='grey')
        set_axis_style(ax, xlabel=xlabel, ylabel=ylabel)

        return fig

    def plot_inter_spike_interval(self, cluster_idx, ax=None, xlabel='Inter spike interval (ms)', ylabel='No. of intervals'):

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        else:
            fig = ax.get_figure()

        spikes = filter_spikes_by_cluster_idx(self.spikes, cluster_idx)

        isi, bins = _compute_histogram(np.diff(spikes.times), 0.01, 0, 50)
        bins = bins * 1e3
        # m_isi = np.max(isi)
        # if m_isi == 0:
        #     m_isi = 1
        # ax.bar(bins[:-1], height=isi / m_isi, width=0.8 * np.diff(bins)[0], color='grey')
        ax.bar(bins[:-1], height=isi, width=0.8 * np.diff(bins)[0], color='grey')
        set_axis_style(ax, xlabel=xlabel, ylabel=ylabel)

        return fig

    def plot_cluster_amplitude(self, cluster_idx, ax=None, xlabel='T in session (s)', ylabel='Amp (uV)'):

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        else:
            fig = ax.get_figure()

        spikes = filter_spikes_by_cluster_idx(self.spikes, cluster_idx)
        clusters = filter_clusters_by_cluster_idx(self.clusters, cluster_idx)
        amp_norm = self.rms_chns[clusters.channels]

        ax.scatter(spikes.times, spikes.amps * 1e6, color='grey', s=2)
        ax.set_xlim(-10, np.max(self.spikes.times))
        ax2 = ax.twinx()
        ax2.scatter(spikes.times, spikes.amps / amp_norm, color='grey', s=0)

        ax.spines['top'].set_visible(False)
        ax2.spines['top'].set_visible(False)

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax2.set_ylabel('Amp (a.u.)', fontsize=12)

        return fig

    # dl = DataLoader()
    # dl.session_init('decc8d40-cf74-4263-ae9d-a0cc68b47e86')
    # dl.get_session_details()

    # pid = list(DATA_DIR.iterdir())[0]

    # clusters = load_clusters(pid)
    # channels = load_channels(pid)
    # spikes = load_spikes(pid)
    # trials = load_trials(pid)
    # # cluster_waveforms = load_cluster_waveforms(pid)

    # cluster_idx = clusters.cluster_id[10]
    # trial_idx = len(trials) // 2

    # # fig = plot_session_raster(spikes, trials, cluster_idx, trial_idx)

    # plot_spikes_fr_vs_depth(clusters, cluster_idx)

    # plt.show()