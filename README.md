# COGS 189 Final Project
Final project for UCSD's Winter '22 session of **COGS 189: Brain Computer Interfaces**. The goal of this project is to research whether or not imagined/spoken speech can be accurately classified using a low-cost EEG headset (NeuroSky Mindwave Mobile).

## Authors
- Nicholas DeGroot
- James Wang

## Setup
1. Install Python 3.10.2 (optionally via ASDF: `asdf install python 3.10.2`)
2. Install main dependencies via Poetry.
```sh
poetry install
```
3. Install your optimized version of PyTorch based on your system. Instructions can be found [here](https://pytorch.org/get-started/locally/).
```sh
# Example for CUDA 11.3 systems
poetry run pip3 install torch==1.11.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```
4. Install PyTorch lightning
```sh
poetry run pip3 install pytorch-lightning
```

### Recording Data (MacOS)
1. Connect a NeuroSky Mindwave Mobile via Bluetooth
2. Determine the serial port of the headset (usually, this is under `/dev/tty.MindWaveMobile`)
3. Configure experiment parameters in `experiment/main.py`
   1. Number of trials
   2. NeuroSky headset serial port
4. Start the experiment via `poetry run experiment/main.py`

### Recording Data (Windows)
1. Connect a NeuroSky Mindwave Mobile via Bluetooth
2. Open a COM port to the headset
3. Configure experiment parameters in `experiment/main.py`
   1. Number of trials
   2. NeuroSky headset COM port
4. Start the experiment via `poetry run experiment/main.py`

### Running the Model
1. Place data inside `modeling/data/`
2. Run the model using the following command:
```
poetry run python main.py --model $MODEL_TYPE --experiment $SUBJECT_NAME --spoken=${1 if spoken, else 0}
```

## Prior Research
Inspiration for this research comes from Young-Eun Lee and Seo-Hyun Lee and their paper: _[EEG-Transformer: Self-attention from Transformer Architecture for Decoding EEG of Imagined Speech](https://arxiv.org/abs/2112.09239)_.