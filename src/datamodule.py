import pytorch_lightning as pl


class LightningDataModule(pl.LightningDataModule):
    def __init__(self):
        super(LightningDataModule, self).__init__()