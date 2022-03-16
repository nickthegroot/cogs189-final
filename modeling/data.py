import math
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from scipy.signal import resample

ONE_MS = pd.Timedelta(1, unit="ms")


class ExperimentData(Dataset):
    def __init__(self, experiment: str, spoken=None) -> None:
        self.attention, self.raw, self.stimuli = (
            self.__load_csv(f"data/{experiment}/{x}")
            for x in ("attention.csv", "raw-values.csv", "stimuli.csv")
        )

        stimuli_val = self.stimuli.value.str.split("-")
        self.stimuli["modality"] = stimuli_val.map(lambda x: x[0])
        self.stimuli["word"] = stimuli_val.map(lambda x: x[1])
        if spoken != None:
            modality = "SAY" if spoken else "THINK"
            self.stimuli = self.stimuli[self.stimuli.modality == modality]

    def __load_csv(self, path: str):
        return pd.read_csv(path, parse_dates=["timestamp"], index_col=["timestamp"])

    def __len__(self):
        return len(self.stimuli)

    def __getitem__(self, index: int):
        stimuli = self.stimuli.iloc[index]

        # filter to 500ms prior and 2s after stimuli
        onset = stimuli.name
        values = self.attention[onset - (500 * ONE_MS) : onset + (2000 * ONE_MS)].value
        
        # average polling rate ~500Hz
        # resample to exactly 500Hz
        sample_time_s = 2.5
        desired_hz = 500
        total_samples = math.floor(sample_time_s * desired_hz) 
        values = resample(values, total_samples)

        y = stimuli.word == "YES"
        return torch.Tensor(values).unsqueeze(0), torch.Tensor([y])

def load_datasets(experiment: str, spoken: bool = None, tr_percentage = .8):
    dataset = ExperimentData(experiment, spoken)
    tr_len = math.floor(len(dataset) * tr_percentage)
    test_len = len(dataset) - tr_len
    train, test = random_split(dataset, [tr_len, test_len])
    return train, test

def get_dataloaders(experiment: str, spoken: bool = None, tr_percentage = .8):
    datasets = load_datasets(experiment, spoken, tr_percentage)
    return (
        DataLoader(x, batch_size=len(x))
        for x in datasets
    )