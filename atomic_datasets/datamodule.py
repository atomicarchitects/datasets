class DataModule:

    def __init__(self, dataset: Dataset, batch_size: int, splits: Dict[str, np.ndarray]):
        self.dataset = dataset
        self.batch_size = batch_size
        self.splits = splits

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=SubsetRandomSampler(self.splits["train"]),
        )