from pathlib import Path
from mindwave import Headset
import reactivex as rx
import reactivex.operators as ops
import pandas as pd


class Recording:
    def __init__(self, headset: Headset, stimuli: rx.Observable) -> None:
        self.__paused = True

        self.headset = headset
        self.recordings = {
            'raw-values': [],
            'waves': [],
            'attention': [],
            'stimuli': [],
        }

        stimuli.pipe(ops.timestamp()).subscribe(lambda x: self.recordings['stimuli'].append(x))

        self.__record_item(self.headset.raw_value, 'raw-values')
        self.__record_item(self.headset.waves, 'waves')
        self.__record_item(self.headset.attention, 'attention')

    def start(self):
        self.__paused = False

    def pause(self):
        self.__paused = True

    def save(self, dir: Path):
        for label, values in self.recordings.items():
            file = dir.joinpath(f'{label}.csv')
            pd.DataFrame(values).to_csv(file, index=False)

    def __record_item(self, observable: rx.Observable, label: str):
        def parse_item(item):
            # Unwrap object items
            if isinstance(item.value, dict):
                item = {
                    **item.value,
                    'timestamp': item.timestamp,
                }
            return item

        observable.pipe(
            ops.skip_while(lambda _: self.__paused),
            ops.timestamp(),
        ).subscribe(
            lambda x: self.recordings[label].append(parse_item(x))
        )