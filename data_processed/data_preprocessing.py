import h5py
from scipy import interpolate  # for resampling
from scipy.signal import butter, lfilter  # for filtering
import os, glob
import pandas as pd
import numpy as np
from collections import OrderedDict, defaultdict

script_dir = os.path.dirname(os.path.realpath(__file__))

# from helpers import *
from utils.print_utils import *
from utils.dict_utils import *

#######################################
############ CONFIGURATION ############
#######################################

# Define where outputs will be saved.
output_dir = os.path.join(script_dir, 'data_processed')
output_filepath = os.path.join(output_dir, 'data_processed_allStreams_60hz_onlyForehand_skill_level.hdf5') # output file name
annotation_data_filePath = 'src/Annotation Data Classification.xlsx' # directory of annotation data xlsx file
# output_filepath = None

# Define column selection/aggregation helpers for reuse.
def identity(data):
    return data


def sum_columns(start: int, end: int):
    def _fn(data: np.ndarray) -> np.ndarray:
        return data[..., start:end].sum(axis=-1, keepdims=True)

    return _fn


def select_columns(start: int, end: int):
    def _fn(data: np.ndarray) -> np.ndarray:
        return data[..., start:end]

    return _fn


# Define the modalities to use. Order matches the final feature matrix channel order.
device_streams_for_features = [
    ('moticon-insole', 'left-totalForce', identity),
    ('moticon-insole', 'left-pressure', sum_columns(0, 4)),
    ('moticon-insole', 'left-pressure', sum_columns(8, 16)),
    ('moticon-insole', 'right-totalForce', identity),
    ('moticon-insole', 'right-pressure', sum_columns(0, 4)),
    ('moticon-insole', 'right-pressure', sum_columns(8, 16)),
    ('cgx-aim-leg-emg', 'emg-values', identity),
    ('pns-joint', 'Euler-angle', select_columns(0, 3)),
    ('pns-joint', 'Euler-angle', select_columns(27, 30)),
    ('pns-joint', 'Euler-angle', select_columns(49, 52)),
    ('gforce-upperarm-emg', 'emg-values', identity),
    ('gforce-lowerarm-emg', 'emg-values', identity),
    ('pns-joint', 'Euler-angle', select_columns(48, 51)),
]

# De-duplicate device/stream pairs to avoid repeated reads.
device_streams_unique = list(dict.fromkeys((device, stream) for device, stream, _ in device_streams_for_features))
device_streams_by_device = defaultdict(set)
for device, stream, _ in device_streams_for_features:
    device_streams_by_device[device].add(stream)

# Specify the input data.
# data_root_dir = os.path.join(script_dir, 'Data_Archive')
data_root_dir = 'Data_Archive/'

data_folders_bySubject = OrderedDict([
    ('Sub00', os.path.join(data_root_dir, 'Sub00')),
    ('Sub01', os.path.join(data_root_dir, 'Sub01')),
    ('Sub02', os.path.join(data_root_dir, 'Sub02')),
    ('Sub03', os.path.join(data_root_dir, 'Sub03')),
    ('Sub04', os.path.join(data_root_dir, 'Sub04')),
    ('Sub05', os.path.join(data_root_dir, 'Sub05')),
    ('Sub06', os.path.join(data_root_dir, 'Sub05')),
    ('Sub07', os.path.join(data_root_dir, 'Sub07')),
    ('Sub08', os.path.join(data_root_dir, 'Sub08')),
    ('Sub09', os.path.join(data_root_dir, 'Sub09')),
    ('Sub10', os.path.join(data_root_dir, 'Sub10')),
    ('Sub11', os.path.join(data_root_dir, 'Sub11')),
    ('Sub12', os.path.join(data_root_dir, 'Sub12')),
    ('Sub13', os.path.join(data_root_dir, 'Sub13')),
    ('Sub14', os.path.join(data_root_dir, 'Sub14')),
    ('Sub15', os.path.join(data_root_dir, 'Sub15')),
    ('Sub16', os.path.join(data_root_dir, 'Sub16')),
    ('Sub17', os.path.join(data_root_dir, 'Sub17')),
    ('Sub18', os.path.join(data_root_dir, 'Sub18')),
    ('Sub19', os.path.join(data_root_dir, 'Sub19')),
    ('Sub20', os.path.join(data_root_dir, 'Sub20')),
    ('Sub21', os.path.join(data_root_dir, 'Sub21')),
    ('Sub22', os.path.join(data_root_dir, 'Sub22')),
    ('Sub23', os.path.join(data_root_dir, 'Sub23')),
    ('Sub24', os.path.join(data_root_dir, 'Sub24')),
])

# Specify the labels to include.  These should match the labels in the HDF5 files.
baseline_label = 'None'
activities_to_classify = [  # Total Number is 3
    baseline_label,
    'Forehand Clear',
    'Backhand Driving',
]

baseline_index = activities_to_classify.index(baseline_label)
# Some older experiments may have had different labels.
#  Each entry below maps the new name to a list of possible old names.
activities_renamed = {
    'Forehand Clear': ['Forehand Clear'], # Change name to Forehand clear
    'Backhand Driving': ['Backhand Driving'],
}

# Define segmentation parameters.
resampled_Fs = 60  # define a resampling rate for all sensors to interpolate
num_segments_per_subject = 10
num_baseline_segments_per_subject = 10  # num_segments_per_subject*(max(1, len(activities_to_classify)-1))
segment_duration_s = 2.5
segment_length = int(round(resampled_Fs * segment_duration_s))
buffer_startActivity_s = 0.01
buffer_endActivity_s = 0.01

# Data subset control: True keeps only Forehand Clear, no Backhand/Baseline.
ONLY_FOREHAND = True

# Define filtering parameters.
filter_cutoff_emg_Hz = 5
filter_cutoff_emg_cognionics_Hz = 20
filter_cutoff_pressure_Hz = 5
filter_cutoff_gaze_Hz = 5

# Make the output folder if needed.
if output_dir is not None:
    os.makedirs(output_dir, exist_ok=True)
    print('\n')
    print('Saving outputs to')
    print(output_filepath)
    print('\n')

################################################
############ INTERPOLATE AND FILTER ############
################################################

# Will filter each column of the data.
def lowpass_filter(data, cutoff, Fs, order=4):
    nyq = max(0.5 * Fs, 1e-6)
    normal_cutoff = cutoff / nyq
    if normal_cutoff >= 1.0:
        normal_cutoff = 0.99
    elif normal_cutoff <= 0:
        normal_cutoff = 1e-6
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, data.T).T
    return y


def estimate_sampling_rate(time_s: np.ndarray, fallback: float = 60.0) -> float:
    if time_s.size < 2:
        return fallback
    duration = float(time_s[-1] - time_s[0])
    if duration <= 0:
        return fallback
    Fs = float((time_s.size - 1) / duration)
    if not np.isfinite(Fs) or Fs <= 0:
        return fallback
    return Fs

def convert_to_nan(arr, difff, time):
    for i in range(len(arr) - time):
        for j in range(len(arr[0])):
            diff = abs(arr[i, j] - arr[i + time, j])
            if diff > difff:
                arr[i, j] = np.nan
    return arr

# Load the original data.
data_bySubject = {}
for (subject_id, data_folder) in data_folders_bySubject.items():
    data_bySubject[subject_id] = []
    hdf_filepaths = glob.glob(os.path.join(data_folder, '**/*.hdf5'), recursive=True)
    for hdf_filepath in hdf_filepaths:
        data_bySubject[subject_id].append({})
        hdf_file = h5py.File(hdf_filepath, 'r')
        # Add the activity label information.
        have_all_streams = True
        # try:
        #     device_name = 'experiment-activities'
        #     stream_name = 'activities'
        #     data_bySubject[subject_id][-1].setdefault(device_name, {})
        #     data_bySubject[subject_id][-1][device_name].setdefault(stream_name, {})
        #     for key in ['time_s', 'data']:
        #         data_bySubject[subject_id][-1][device_name][stream_name][key] = hdf_file[device_name][stream_name][key][
        #                                                                         :]
        #     num_activity_entries = len(data_bySubject[subject_id][-1][device_name][stream_name]['time_s'])
        #     if num_activity_entries == 0:
        #         have_all_streams = False
        #     elif data_bySubject[subject_id][-1][device_name][stream_name]['time_s'][0] == 0:
        #         have_all_streams = False
        # except KeyError:
        #     have_all_streams = False
        # Load data for each of the streams that will be used as features.
        for (device_name, stream_name) in device_streams_unique:
            data_bySubject[subject_id][-1].setdefault(device_name, {})
            data_bySubject[subject_id][-1][device_name].setdefault(stream_name, {})
            for key in ['time_s', 'data']:
                try:
                    data_bySubject[subject_id][-1][device_name][stream_name][key] = hdf_file[device_name][stream_name][
                                                                                        key][:]
                except KeyError:
                    have_all_streams = False
        if not have_all_streams:
            data_bySubject[subject_id].pop()
        hdf_file.close()

# print(data_bySubject)
# Filter data.
for (subject_id, file_datas) in data_bySubject.items():
    for (data_file_index, file_data) in enumerate(file_datas):
        # Filter EMG data.
        for gforce_key in ['gforce-lowerarm-emg', 'gforce-upperarm-emg']:
            if gforce_key in file_data:
                t = file_data[gforce_key]['emg-values']['time_s']
                Fs = estimate_sampling_rate(t, fallback=resampled_Fs)
                data_stream = file_data[gforce_key]['emg-values']['data'][:, :]
                y = np.abs(data_stream)
                y = lowpass_filter(y, filter_cutoff_emg_Hz, Fs)
                # for i in range(len(data_stream[0])):
                #     plt.plot(t-t[0], data_stream[:, i], label=gforce_key+'_raw')
                #     plt.plot(t-t[0], y[:, i], label=gforce_key+'_preprocessed')
                #     plt.legend()
                #     plt.show()
                #     plt.clf()
                #     plt.plot(t[500:900] - t[0], data_stream[500:900, i], label=gforce_key + '_raw')
                #     plt.plot(t[500:900] - t[0], y[500:900, i], label=gforce_key + '_preprocessed')
                #     plt.legend()
                #
                #     plt.show()
                #     plt.clf()
                file_data[gforce_key]['emg-values']['data'] = y
        for cognionics_key in ['cgx-aim-leg-emg']:
            if cognionics_key in file_data:
                t = file_data[cognionics_key]['emg-values']['time_s']
                Fs = estimate_sampling_rate(t, fallback=resampled_Fs)
                data_stream = file_data[cognionics_key]['emg-values']['data'][:, :]
                data_stream = np.abs(data_stream)
                # Correcting the bounce value
                y = convert_to_nan(data_stream, difff=80, time=5)
                y[y > 26000] = np.nan
                # y[y < -26000] = np.nan
                # y[y < -26000] = np.nan
                df = pd.DataFrame(y)
                # print(df.isnull().sum())
                for ii in range(len(df.loc[0])):
                    df.loc[:, ii] = df.loc[:, ii].fillna(df.loc[:, ii].median())
                    # print(df.loc[:, ii].mean())
                # print(df.isnull().sum())
                y = df.to_numpy()
                y = lowpass_filter(y, filter_cutoff_emg_cognionics_Hz, Fs)
                # for i in range(len(data_stream[0])):
                #     # print('max', np.amax(data_stream[:, i]))
                #     # print('min', np.amin(data_stream[:, i]))
                #     plt.plot(t-t[0], data_stream[:, i], label=cognionics_key+'_raw_channel' + str(i+1))
                #     plt.plot(t - t[0], y[:, i], label=cognionics_key + '_preprocessed_channel'+ str(i+1))
                #     plt.legend()
                #     plt.show()
                #     plt.clf()
                #     plt.plot(t[50000:55000] - t[0], data_stream[50000:55000, i], label=cognionics_key + '_raw_channel' + str(i+1))
                #     plt.plot(t[50000:55000] - t[0], y[50000:55000, i], label=cognionics_key + '_preprocessed_channel' + str(i+1))
                #     plt.legend()
                #     plt.show()
                #     plt.clf()
                #     plt.plot(t[50000:55000] - t[0], y[50000:55000, i], label=cognionics_key + '_preprocessed_channel' + str(i+1))
                #     plt.legend()
                #     plt.show()
                #     plt.clf()
                file_data[cognionics_key]['emg-values']['data'] = y
        moticon_streams = file_data.get('moticon-insole', {})
        moticon_stream_names = device_streams_by_device.get('moticon-insole', set())
        for moticon_key in moticon_stream_names:
            if moticon_key not in moticon_streams:
                continue
            t = file_data['moticon-insole'][moticon_key]['time_s']
            Fs = estimate_sampling_rate(t, fallback=resampled_Fs)
            data_stream = file_data['moticon-insole'][moticon_key]['data']
            if data_stream.ndim == 1:
                data_stream = data_stream[:, np.newaxis]
            y = np.abs(data_stream)
            y = lowpass_filter(y, filter_cutoff_pressure_Hz, Fs)
            file_data['moticon-insole'][moticon_key]['data'] = y
        data_bySubject[subject_id][data_file_index] = file_data

# Normalize data.
print()
for (subject_id, file_datas) in data_bySubject.items():
    print('Normalizing data for subject %s' % subject_id)
    for (data_file_index, file_data) in enumerate(file_datas):
        # Normalize gForce Pro EMG data.
        for gforce_key in ['gforce-lowerarm-emg', 'gforce-upperarm-emg']:
            if gforce_key in file_data:
                data_stream = file_data[gforce_key]['emg-values']['data'][:, :]
                min_val = 0
                max_val = 300
                y = data_stream
                # Normalize them jointly.
                y = y / ((max_val - min_val) / 2)
                # Jointly shift the baseline to -1 instead of 0.
                y = y - np.amin(y) - 1
                file_data[gforce_key]['emg-values']['data'] = y
                # plt.plot(y.reshape(y.shape[0], -1))
                # plt.show()

        # Normalize Cognionics EMG data.
        for cognionics_key in ['cgx-aim-leg-emg']:
            if cognionics_key in file_data:
                data_stream = file_data[cognionics_key]['emg-values']['data'][:, :]
                y = data_stream
                y = y / ((np.amax(y) - np.amin(y)) / 2)
                # Jointly shift the baseline to -1 instead of 0.
                y = y - np.amin(y) - 1
                file_data[cognionics_key]['emg-values']['data'] = y
                # plt.plot(y.reshape(y.shape[0], -1))
                # plt.show()

        # Normalize Perception Neuron Studio joints.
        if 'pns-joint' in file_data and 'Euler-angle' in file_data['pns-joint']:
            data_stream = file_data['pns-joint']['Euler-angle']['data'][:, :]
            y = data_stream
            min_val = -180
            max_val = 180
            # Normalize all at once since using fixed bounds anyway.
            # Preserve relative bends, such as left arm being bent more than the right.
            y = y / ((max_val - min_val) / 2)
            file_data['pns-joint']['Euler-angle']['data'] = y
            # plt.plot(y.reshape(y.shape[0], -1))
            # plt.show()

        # Normalize Moticon Pressure.
        moticon_stream_names = device_streams_by_device.get('moticon-insole', set())
        for moticon_key in moticon_stream_names:
            if 'moticon-insole' not in file_data or moticon_key not in file_data['moticon-insole']:
                continue
            data_stream = file_data['moticon-insole'][moticon_key]['data']
            if data_stream.ndim == 1:
                data_stream = data_stream[:, np.newaxis]
            y = data_stream
            # Normalize them jointly.
            y = y / ((np.amax(y) - np.amin(y)) / 2)
            # Jointly shift the baseline to -1 instead of 0.
            y = y - np.amin(y) - 1
            file_data['moticon-insole'][moticon_key]['data'] = y
            # plt.plot(y.reshape(y.shape[0], -1))
            # plt.show()

        data_bySubject[subject_id][data_file_index] = file_data

labels = []
label_indices = []
feature_chunks = []
subject_ids = []
skill_levels = []
score_annot_3_hori = []
score_annot_3_ver = []
score_annot_4 = []
score_annot_5 = []

Forehand_time_list = []
Backhand_time_list = []
NoActivity_time_list = []

df = pd.read_excel(annotation_data_filePath)

# Remove rows where the specified columns contain "NoVid"
df_filtered = df[
    (df["Annotation Level 3\n(Landing Location - Horizontal)"] != "NoVid") &
    (df["Annotation Level 3\n(Landing Location - Vertical)"] != "NoVid") &
    (df["Annotation Level 4\n(Hitting Location, major voting)"] != "NoVid") &
    (df["Annotation Level 5\n(Hitting Sound, major voting)"] != "NoVid")
    ]

print(len(df_filtered))

for (subject_id, file_datas) in data_bySubject.items():
    print('Resampling data for subject %s' % subject_id)

    df_subject = df_filtered[df_filtered["Subject Number"] == subject_id]

    df_subject_forehand = df_subject[df_subject["Annotation Level 1\n(Stroke Type)"] == 'Forehand Clear']
    df_subject_backhand = df_subject[df_subject["Annotation Level 1\n(Stroke Type)"] == 'Backhand Driving']

    for (data_file_index, file_data) in enumerate(file_datas):

        file_labels = []
        file_label_indices = []
        file_subject_ids = []
        file_skill_levels = []

        file_score_annot_3_hori = []
        file_score_annot_3_ver = []
        file_score_annot_4 = []
        file_score_annot_5 = []

        Forehand_start_time_list = df_subject_forehand['Annotation Start Time'].values.tolist()
        Forehand_stop_time_list = df_subject_forehand['Annotation Stop Time'].values.tolist()
        Backhand_start_time_list = df_subject_backhand['Annotation Start Time'].values.tolist()
        Backhand_stop_time_list = df_subject_backhand['Annotation Stop Time'].values.tolist()

        Forehand_skill_level_list = df_subject_forehand['Annotation Level 2\n(Skill Level)'].values.tolist()
        Backhand_skill_level_list = df_subject_backhand['Annotation Level 2\n(Skill Level)'].values.tolist()

        Forehand_score_annot_3_hori_list = df_subject_forehand["Annotation Level 3\n(Landing Location - Horizontal)"].values.tolist()
        Backhand_score_annot_3_hori_list = df_subject_backhand["Annotation Level 3\n(Landing Location - Horizontal)"].values.tolist()
        Forehand_score_annot_3_ver_list = df_subject_forehand["Annotation Level 3\n(Landing Location - Vertical)"].values.tolist()
        Backhand_score_annot_3_ver_list = df_subject_backhand["Annotation Level 3\n(Landing Location - Vertical)"].values.tolist()

        Forehand_score_annot_4_list = df_subject_forehand["Annotation Level 4\n(Hitting Location, major voting)"].values.tolist()
        Backhand_score_annot_4_list = df_subject_backhand["Annotation Level 4\n(Hitting Location, major voting)"].values.tolist()
        Forehand_score_annot_5_list = df_subject_forehand["Annotation Level 5\n(Hitting Sound, major voting)"].values.tolist()
        Backhand_score_annot_5_list = df_subject_backhand["Annotation Level 5\n(Hitting Sound, major voting)"].values.tolist()

        if ONLY_FOREHAND:
            Backhand_start_time_list = []
            Backhand_stop_time_list = []
            Backhand_skill_level_list = []
            Backhand_score_annot_3_hori_list = []
            Backhand_score_annot_3_ver_list = []
            Backhand_score_annot_4_list = []
            Backhand_score_annot_5_list = []

        NoActivity_start_time_list = []
        NoActivity_stop_time_list = []
        NoActivity_start_time_list.extend(Forehand_stop_time_list[0:-1])
        NoActivity_start_time_list.extend(Backhand_stop_time_list[0:-1])
        NoActivity_stop_time_list.extend(Forehand_start_time_list[1:])
        NoActivity_stop_time_list.extend(Backhand_start_time_list[1:])

        if ONLY_FOREHAND:
            NoActivity_start_time_list = []
            NoActivity_stop_time_list = []

        feature_segments_per_stream = [None] * len(device_streams_for_features)
        for device_idx, (device_name, stream_name, extractor) in enumerate(device_streams_for_features):
            stream_feature_segments = []
            data = np.squeeze(np.array(file_data[device_name][stream_name]['data']))
            time_s = np.squeeze(np.array(file_data[device_name][stream_name]['time_s']))
            
            _, unique_indices = np.unique(time_s, return_index=True)
            time_s = time_s[unique_indices]
            data = data[unique_indices]

            if data.ndim == 1:
                data = data[:, np.newaxis]

            data = extractor(data)
            if data.ndim == 1:
                data = data[:, np.newaxis]

            label_indexes = [0] * len(time_s)

            # Initialize the Number of each stroke
            Num_base = 0
            Num_clear = 0
            Num_drive = 0

            # Save the Forehand Clear Data
            highNum = 0
            backNum = 0
            baseNum = 0

            for j in range(len(Forehand_start_time_list)):
                # Save the swing time of each stroke
                Forehand_time_list.append(Forehand_stop_time_list[j]-Forehand_start_time_list[j])
                # time indexing
                high_time_indexes = np.where((time_s >= Forehand_start_time_list[j]) & (time_s <= Forehand_stop_time_list[j]))

                if len(high_time_indexes[0]) > 0:
                    target_time_s_high = np.linspace(Forehand_start_time_list[j], Forehand_stop_time_list[j],
                                        num=segment_length,
                                        endpoint=True)

                    fn_interpolate = interpolate.interp1d(
                        time_s,  # x values
                        data,  # y values
                        axis=0,  # axis of the data along which to interpolate
                        kind='slinear',  # interpolation method, such as 'linear', 'zero', 'nearest', 'quadratic', 'cubic', etc.
                        fill_value='extrapolate'  # how to handle x values outside the original range
                    )

                    data_resampled = fn_interpolate(target_time_s_high)

                    stream_feature_segments.append(
                        data_resampled.tolist())
                    highNum += 1

                    if device_idx == len(device_streams_for_features) - 1:
                        file_label_indices.append(1)
                        file_labels.append('Forehand Clear')
                        file_subject_ids.append(subject_id)
                        file_skill_levels.append(Forehand_skill_level_list[j])
                        Num_clear += 1
                        file_score_annot_3_hori.append(Forehand_score_annot_3_hori_list[j])
                        file_score_annot_3_ver.append(Forehand_score_annot_3_ver_list[j])
                        file_score_annot_4.append(Forehand_score_annot_4_list[j])
                        file_score_annot_5.append(Forehand_score_annot_5_list[j])
                for m in range(len(high_time_indexes[0])):
                    label_indexes[high_time_indexes[0][m]] = 1

            if not ONLY_FOREHAND:
                for j in range(len(Backhand_start_time_list)):
                    Backhand_time_list.append(Backhand_stop_time_list[j] - Backhand_start_time_list[j])
                    back_time_indexes = np.where(
                        (time_s >= Backhand_start_time_list[j]) & (time_s <= Backhand_stop_time_list[j]))
                    if len(back_time_indexes[0]) > 0:

                        target_time_s_back = np.linspace(Backhand_start_time_list[j], Backhand_stop_time_list[j],
                                            num=segment_length,
                                            endpoint=True)

                        fn_interpolate = interpolate.interp1d(
                            time_s,  # x values
                            data,  # y values
                            axis=0,  # axis of the data along which to interpolate
                            kind='slinear',  # interpolation method, such as 'linear', 'zero', 'nearest', 'quadratic', 'cubic', etc.
                            fill_value='extrapolate'  # how to handle x values outside the original range
                        )

                        data_resampled = fn_interpolate(target_time_s_back)

                        stream_feature_segments.append(
                            data_resampled.tolist())

                        backNum += 1
                        if device_idx == len(device_streams_for_features) - 1:
                            file_label_indices.append(0)
                            file_labels.append('Backhand Driving')
                            file_subject_ids.append(subject_id)
                            file_skill_levels.append(Backhand_skill_level_list[j])
                            Num_drive += 1
                            file_score_annot_3_hori.append(Backhand_score_annot_3_hori_list[j])
                            file_score_annot_3_ver.append(Backhand_score_annot_3_ver_list[j])
                            file_score_annot_4.append(Backhand_score_annot_4_list[j])
                            file_score_annot_5.append(Backhand_score_annot_5_list[j])
                    for m in range(len(back_time_indexes[0])):
                        label_indexes[back_time_indexes[0][m]] = 0

                # Save the Baseline Data
                for j in range(len(NoActivity_start_time_list)):
                    NoActivity_time_list.append(NoActivity_stop_time_list[j] - NoActivity_start_time_list[j])
                    no_time_indexes = np.where(
                        (time_s >= NoActivity_start_time_list[j]) & (time_s <= NoActivity_stop_time_list[j]))
                    if len(no_time_indexes[0]) > 0:

                        target_time_s_no = np.linspace(NoActivity_start_time_list[j], NoActivity_stop_time_list[j],
                                            num=segment_length,
                                            endpoint=True)

                        fn_interpolate = interpolate.interp1d(
                            time_s,  # x values
                            data,  # y values
                            axis=0,  # axis of the data along which to interpolate
                            kind='slinear',  # interpolation method, such as 'linear', 'zero', 'nearest', 'quadratic', 'cubic', etc.
                            fill_value='extrapolate'  # how to handle x values outside the original range
                        )

                        data_resampled = fn_interpolate(target_time_s_no)

                        stream_feature_segments.append(
                            data_resampled.tolist())

                        baseNum += 1
                        if device_idx == len(device_streams_for_features) - 1:
                            file_label_indices.append(2)
                            file_labels.append('Baseline')
                            file_subject_ids.append(subject_id)
                            file_skill_levels.append(-1)
                            Num_base += 1
                            file_score_annot_3_hori.append(-1)
                            file_score_annot_3_ver.append(-1)
                            file_score_annot_4.append(-1)
                            file_score_annot_5.append(-1)
                    for m in range(len(no_time_indexes[0])):
                        label_indexes[no_time_indexes[0][m]] = 2

            feature_segments_per_stream[device_idx] = stream_feature_segments

            # print("highNum")
            # print(highNum)
            # print("backNum")
            # print(backNum)
            # print("baseNum")
            # print(baseNum)
            # print("totalNum")
            # print(highNum + backNum + baseNum)

        device_arrays = []
        for segments in feature_segments_per_stream:
            segments = segments or []
            device_array = np.array(segments)
            if device_array.ndim == 2:
                device_array = device_array[:, :, np.newaxis]
            device_arrays.append(device_array)

        lengths = [arr.shape[0] for arr in device_arrays]
        if not lengths or min(lengths) == 0:
            continue

        all_equal = all(length == lengths[0] for length in lengths)
        if all_equal:
            combined_array = np.concatenate(device_arrays, axis=2)
            label_slice = slice(None)
        else:
            min_len = min(lengths)
            device_arrays = [arr[:min_len] for arr in device_arrays]
            combined_array = np.concatenate(device_arrays, axis=2)
            label_slice = slice(0, min_len)

        label_arr = np.array(file_label_indices[label_slice])
        keep_mask = label_arr == 1
        if keep_mask.sum() == 0:
            continue

        combined_array = combined_array[keep_mask]

        def _extend_filtered(target_list, source_seq):
            arr = np.array(source_seq[label_slice])
            target_list.extend(arr[keep_mask].tolist())

        _extend_filtered(label_indices, file_label_indices)
        _extend_filtered(labels, file_labels)
        _extend_filtered(subject_ids, file_subject_ids)
        _extend_filtered(skill_levels, file_skill_levels)
        _extend_filtered(score_annot_3_hori, file_score_annot_3_hori)
        _extend_filtered(score_annot_3_ver, file_score_annot_3_ver)
        _extend_filtered(score_annot_4, file_score_annot_4)
        _extend_filtered(score_annot_5, file_score_annot_5)
        feature_chunks.append(combined_array)

feature_matrices = np.concatenate([arr for arr in feature_chunks], axis=0)
print('total feature shape', feature_matrices.shape)

if output_filepath is not None:
    with h5py.File(output_filepath, 'w') as hdf_file:
        metadata = OrderedDict()
        metadata['output_dir'] = output_dir
        metadata['data_root_dir'] = data_root_dir
        metadata['data_folders_bySubject'] = data_folders_bySubject
        metadata['activities_to_classify'] = activities_to_classify
        metadata['device_streams_for_features'] = device_streams_for_features
        metadata['resampled_Fs'] = resampled_Fs
        metadata['segment_length'] = segment_length
        metadata['segment_duration_s'] = segment_duration_s
        metadata['filter_cutoff_emg_Hz'] = filter_cutoff_emg_Hz
        metadata['filter_cutoff_pressure_Hz'] = filter_cutoff_pressure_Hz
        metadata['filter_cutoff_gaze_Hz'] = filter_cutoff_gaze_Hz

        metadata = convert_dict_values_to_str(metadata, preserve_nested_dicts=False)

        # Convert labels and IDs to fixed-length byte strings
        labels_arr = np.array(labels, dtype='S')
        label_indices_arr = np.array(label_indices, dtype=np.int32)
        feature_matrices_arr = np.array(feature_matrices, dtype=np.float32)
        subject_ids_arr = np.array(subject_ids, dtype='S')

        # Convert skill level to integer labels before saving
        # Source: Excel column "Annotation Level 2 (Skill Level)"
        # Forehand/Backhand use actual levels; Baseline is -1 in the list above.
        skill_map = {
            'Beginner': 0,
            'Intermediate': 1,
            'Advanced': 2,
            'Expert': 2,
        }
        skill_levels_arr = np.array(
            [skill_map.get(str(s), -1) for s in skill_levels],
            dtype=np.int32,
        )

        # Scores may contain strings and -1; cast everything to string for storage
        score_annot_3_hori_arr = np.array(
            [str(x) for x in score_annot_3_hori],
            dtype='S',
        )
        score_annot_3_ver_arr = np.array(
            [str(x) for x in score_annot_3_ver],
            dtype='S',
        )
        score_annot_4_arr = np.array(
            [str(x) for x in score_annot_4],
            dtype='S',
        )
        score_annot_5_arr = np.array(
            [str(x) for x in score_annot_5],
            dtype='S',
        )

        # Now create datasets using the well-typed arrays
        hdf_file.create_dataset('labels', data=labels_arr)
        hdf_file.create_dataset('label_indexes', data=label_indices_arr)
        hdf_file.create_dataset('feature_matrices', data=feature_matrices_arr)
        hdf_file.create_dataset('subject_ids', data=subject_ids_arr)
        hdf_file.create_dataset('skill_levels', data=skill_levels_arr)
        hdf_file.create_dataset('score_annot_3_hori', data=score_annot_3_hori_arr)
        hdf_file.create_dataset('score_annot_3_ver', data=score_annot_3_ver_arr)
        hdf_file.create_dataset('score_annot_4', data=score_annot_4_arr)
        hdf_file.create_dataset('score_annot_5', data=score_annot_5_arr)

        hdf_file.attrs.update(metadata)

        print()
        print('Saved processed data to', output_filepath)
        print()
