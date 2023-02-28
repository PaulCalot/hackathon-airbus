import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset

HOUR = 3600
DURATION_KEY = "duration_from_last_measurement"
OBSERVATION_TIME_SPAN = 48 * HOUR
KEY_IDENTIFIER = "keys"
INDEX_KEY = "Case index"
MANEUVER_TIME_KEY = "Maneuver Time (s)"
DELTA_V_KEY = "Delta V (m/s)"
SAMPLE_INFO_KEY = "task"
SK_TYPE_KEY = "Sk type"
MANEUVERS_KEY = "maneuvers"
MAXIMUM_MEASUREMENT_COUNT_BY_SAMPLE = 1000
RESIDUAL_KEY = "residuals"
RIGHT_ASCENSION_KEY = "RA"
DECLINATION_KEY = "DEC"

class ManeuverDetectionDataset(Dataset):
    """Parser for the maneuver detection dataset

        Parameters
        ----------
        dataset_path : str
            The dataset location.
        dataset_type : str, optional (default is TRAIN).
            Either "TRAIN","VALIDATION" or "TEST". Train and Validation are created from the
            dataset file according to the validation_size_ratio. Test dataset does not
            return any maneuver information in the __iter__: function.
        validation_size_ratio:float, optional (default is 0.1)
            The size of the Validation dataset will be validation_size_ratio*total_dataset_length
            The size of the Train dataset will be (1-validation_size_ratio)*total_dataset_length
        max_size:int, optional (default is None)
            If None, the whole dataset will be kept. If specified, the dataset will not exceed this size.
        imported_dataset:dict, optional (default is None)
            If None, the dataset will be imported from the dataset_path. Else, the dictionary given (with the correct
            dataset structure) will be used. It is usefull to avoid to load in RAM twice the dataset for the validation
            and the train dataset.
        filter_samples:str, optional (default is "NO")
            Either "NO","MANEUVER_ONLY","WITHOUT_MANEUVER_ONLY". It helps to filter the relevant samples (for example,
            you may want to use only the maneuvers to train the networks to estimate the date and the dv of the maneuver)
        fixed_step:bool, optional (default is False)
            If False, the features are padded with 0 to fit a fixed size of 1000 measurements by sample.
            If True, the features are not padded (to use with evenly spaced measurements dataset where the number of
            measurements is the same for every sample)
        add_time_feature:bool, optional (default is True)
            If true, the features will have the following shape (measurements_count_by_sample,3).the first two features
            are the residuals (Right Ascension and declination). The second feature is the duration from the start of the
            observation.
            If false, the features will have the following shape (measurements_count_by_sample,2)
        """
    def __init__(self, dataset_path, dataset_type="TRAIN", validation_size_ratio=0.1, max_size=None,
                 imported_dataset=None, filter_samples="NO", fixed_step=False, add_time_feature=True):
        check_arguments(dataset_type, filter_samples)
        print("\n\n**********" + dataset_type + " DATASET *********")
        print(f"Validation/Train ratio: {validation_size_ratio}")
        print(f"Samples filtered? {filter_samples}")
        print(f"Samples evenly spaced? {fixed_step}")
        self.dataset_type = dataset_type
        self.filter_samples = filter_samples
        self.add_time_feature = add_time_feature
        self.dataset_path = dataset_path
        print("path: " + self.dataset_path)
        self.dataset = import_dataset(dataset_path, imported_dataset, filter_samples)
        self.measurements_count_by_sample = MAXIMUM_MEASUREMENT_COUNT_BY_SAMPLE if not fixed_step \
            else fixed_measurement_count_by_sample(self.dataset)
        self.length = compute_length(dataset_type, validation_size_ratio, len(self.dataset[KEY_IDENTIFIER]), max_size)
        print(f"{dataset_type} Dataset loaded. Size: {self.length}")

    def __len__(self):
        return self.length

    def __getitem__(self, torch_loader_index):
        dataset_index = torch_loader_index if (self.dataset_type == "TRAIN" or self.dataset_type == "TEST") else -torch_loader_index-1
        sample_key = self.dataset[KEY_IDENTIFIER][dataset_index]
        sample = self.dataset[sample_key]
        features = parse_sample(sample, self.measurements_count_by_sample, self.add_time_feature)
        if self.dataset_type=="TEST":
            return features,0,0.,0.
        else:
            maneuver_info = get_maneuver_info(sample)
            return features, is_maneuver(sample), maneuver_info[DELTA_V_KEY], maneuver_info[MANEUVER_TIME_KEY]

class IrregularDataset(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset, self.length = self.prepare_data(dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        return x, y # returning only the sample
    
    def prepare_data(self, dataset):
        new_data = []
        for i in range(len(dataset)):
            feature, is_maneuver, dv, time  = dataset[i]
            if(not is_maneuver):
                dv = 0.
                time = 0.
            feature = torch.tensor(feature).t().float()

            # feature[2, 1:] = feature[2, :-1] - feature[2, 1:]
            # feature[2, 0] = 0.
            feature[2, :] = feature[2, :]/OBSERVATION_TIME_SPAN # 800. # normalizing time span - 800. secondes to normalize
            new_data.append((feature, (is_maneuver, dv, time/OBSERVATION_TIME_SPAN)))
        return new_data, len(new_data)
    
    def __len__(self):
        return self.length
    

class SlidingWindowDataset(Dataset):
    def __init__(self, dataset, window_size=50, drop_last=True):
        super().__init__()
        self.window_size = window_size
        self.dataset, self.window_to_sample, self.length = split_dataset_in_window(dataset, window_size=window_size, drop_last=drop_last)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        return sample[0].t(), (sample[1], sample[2]) # channel x time series length 
    
    def __len__(self):
        return self.length

def compute_length(dataset_type, validation_size_ratio, total_dataset_length, max_size):
    length = total_dataset_length if max_size is None else min(max_size, total_dataset_length)
    if dataset_type == "TRAIN":
        return round(length * (1 - validation_size_ratio))
    elif dataset_type == "VALIDATION":
        return round(length * validation_size_ratio)
    elif dataset_type == "TEST":
        return length


def load(dataset_path):
    print("loading dataset. Ready in a minute!")
    with open(dataset_path, "r") as dataset_file:
        return json.load(dataset_file)


def is_maneuver(sample):
    return 0 if OBSERVATION_TIME_SPAN < get_maneuver_info(sample)[MANEUVER_TIME_KEY] else 1


def get_maneuver_info(sample):
    return {DELTA_V_KEY: 0., MANEUVER_TIME_KEY: 2 * OBSERVATION_TIME_SPAN} if len(
        sample[SAMPLE_INFO_KEY][MANEUVERS_KEY]) == 0 else sample[SAMPLE_INFO_KEY][MANEUVERS_KEY][0]


def get_observation_time_span(file_name):
    dates_string = file_name.split(os.path.sep)[1].split(" ")[5]
    observation_start = dates_string.split("to")[0]
    observation_end = dates_string.split("to")[1]
    return observation_end, observation_start


def get_residuals(sample, key):
    return sample[RESIDUAL_KEY][key]


def get_duration(sample):
    return sample[RESIDUAL_KEY][DURATION_KEY]


def check_arguments(dataset_type, filter_samples):
    dataset_type_keys = {"TRAIN", "TEST", "VALIDATION"}
    assert dataset_type in dataset_type_keys, f"Invalid 'dataset type' argument {dataset_type}. " \
                                              f"dataset_type should be in {dataset_type_keys}"
    filter_samples_keys = {"NO", "MANEUVER_ONLY", "WITHOUT_MANEUVER_ONLY"}
    assert filter_samples in filter_samples_keys, f"Invalid 'filter sample' argument {filter_samples}. " \
                                                  f"'filter sample' should be in {filter_samples_keys}"
    if dataset_type=="TEST":
        assert  filter_samples=="NO","Impossible to filter maneuvers, since you don't know them in the TEST dataset... " \
                                     "nice try :P "

def fixed_measurement_count_by_sample(dataset):
    return len(get_residuals(dataset[dataset[KEY_IDENTIFIER][0]], RIGHT_ASCENSION_KEY))

def parse_sample(sample, measurements_count_by_sample, time_feature):
    feature_count = 3 if time_feature else 2
    features = np.zeros((measurements_count_by_sample, feature_count))
    for i, key in enumerate([RIGHT_ASCENSION_KEY, DECLINATION_KEY]):
        residuals = get_residuals(sample, key)
        features[:len(residuals), i] = residuals
    if time_feature:
        duration = get_duration(sample)
        features[:len(duration), 2] = duration
    return features

def import_dataset(dataset_path, imported_dataset, filter_samples):
    dataset = imported_dataset if imported_dataset is not None else load(dataset_path)
    if filter_samples == "NO":
        return dataset
    elif filter_samples == "MANEUVER_ONLY":
        maneuver_keys_index = [True if is_maneuver(dataset[key]) else False for key in dataset[KEY_IDENTIFIER]]
        dataset[KEY_IDENTIFIER] = list(np.array(dataset[KEY_IDENTIFIER])[maneuver_keys_index])
    elif filter_samples == "WITHOUT_MANEUVER_ONLY":
        without_maneuver_keys_index = [True if not is_maneuver(dataset[key]) else False for key in
                                       dataset[KEY_IDENTIFIER]]
        dataset[KEY_IDENTIFIER] = list(np.array(dataset[KEY_IDENTIFIER])[without_maneuver_keys_index])
    return dataset



def get_maneuver_data_point_indexes(times, time):
    left_index = 0
    right_index = len(times) - 1
    while(left_index != right_index - 1):
        m = (left_index + right_index)//2
        if(times[m] > time):
            right_index = m
        elif(times[m] < time):
            left_index = m
        else:
            return (m, m)
    return left_index, right_index

def split_dataset_in_window(dataset, window_size, drop_last=False):
    new_data = []
    window_to_sample = []
    for i, (feature, is_maneuver, dv, time) in enumerate(dataset):
        length = feature.shape[0]
        pad_size = length % window_size
        feature = torch.nn.functional.pad(torch.tensor(feature).t().float(), pad=(0, pad_size), mode='constant', value=0.).t() # padding last dimension, to the right
        # https://pytorch.org/docs/stable/generated/torch.split.html
        splits_tupl = torch.split(feature, window_size, dim=0)
        nb_splits = len(splits_tupl)
    
        if(drop_last and pad_size > 0):
            nb_splits -= 1
    
        if(is_maneuver):
            times = feature[:, 2]
            left_index, right_index = get_maneuver_data_point_indexes(times, time)
            left_index_window = left_index // window_size
            right_index_window = right_index // window_size
        
        for k in range(nb_splits):
            if(is_maneuver and (k==left_index_window or k==right_index_window)):
                c_ = 1
                dv_ = dv
            else:
                c_ = 0
                dv_ = 0.    
            new_data.append((splits_tupl[k][:, :2], c_, dv_)) # removing time - it is useless now
            window_to_sample.append(i)
    return new_data, window_to_sample, len(new_data)