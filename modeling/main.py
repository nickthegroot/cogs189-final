import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from data import get_dataloaders
from trans_model import TransformerModel
from eeg_model import EEGTransformer

parser = argparse.ArgumentParser()
# Model Args
parser.add_argument('--model', type=str,
    help='Model being used')

# Experiment Args
parser.add_argument('--experiment', type=str,
    help='Participant to train on')
parser.add_argument('--spoken', type=int, default=0,
    help='Train on spoken data (default=imagined)')


args = vars(parser.parse_args())
if __name__ == '__main__':
    match (args['model']):
        case 'transformer':
            model = TransformerModel(hz=500, dim=10)
        case 'eegnet':
            model = EEGTransformer(C=1, T=1250)
        case _:
            raise NotImplementedError

    train_loader, test_loader = get_dataloaders(args['experiment'], bool(args['spoken']))

    early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=20)
    trainer = pl.Trainer(
        callbacks=[early_stop],
        log_every_n_steps=1,
    )
    trainer.fit(model, train_loader, test_loader)