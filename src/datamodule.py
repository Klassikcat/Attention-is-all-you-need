import pytorch_lightning as pl


class LightningDataModule(pl.LightningDataModule):
    def __init__(self):
        """
        initializer of Pytorch Lightning DataModule
        """
        super(LightningDataModule, self).__init__()

    def setup(self, stage: Optional[str] = None) -> None:
        return None

    def train_dataloader(self) -> None:
        raise NotImplementedError

    def val_dataloader(self) -> None:
        raise NotImplementedError

    def test_dataloader(self) -> None:
        raise NotImplementedError