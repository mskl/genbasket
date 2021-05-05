import tensorflow
import time
import pandas as pd
import numpy as np
from typing import Optional


class StringGenerator(tensorflow.keras.utils.Sequence):
    def __init__(
        self,
        df: pd.DataFrame,
        batch_size: int,
        history_size: int,
        training: Optional[bool] = True,
    ):
        # Log the time to debug speed
        self.start_time = time.time()

        self.df = df
        self.batch_size = batch_size
        self.history_size = history_size
        self.training = training

        # Do the preprocessing of the whole dataset
        self.unique_user_count = None
        self.unique_user_ids = None

        self.user_metadata = None
        self.user_baskets = None
        self.user_baskets_count = None
        self.user_baskets_pointers = None

        self.init_dataset()

        # Generate indexes and shuffle them
        self.indexes = None
        self.on_epoch_end()

    def time_log(self, *args, **kwargs):
        """Print the log timed from the initialization."""
        print(f"{time.time() - self.start_time:0.4f}s \t|", *args, **kwargs)

    def init_dataset(self):
        """Initializes the generator. Does dataset preprocessing."""
        self.time_log("Generator initialization was started.")
        self.df = self.df.sort_values(by=["uid", "date"])

        self.time_log("Creating the user transactions.")
        usergroup = self.df.groupby("uid")
        self.unique_user_count = len(usergroup)
        self.unique_user_ids = usergroup.uid.first()

        # Create the baskets and initialize the pointers
        self.user_metadata = usergroup.date
        self.user_baskets = usergroup.itemid.apply(np.hstack)
        self.user_baskets_count = usergroup.itemid.count().values

        # Start on random position, not on the oldest vectors
        a = np.zeros(len(self.user_baskets_count))
        b = np.maximum(a, self.pointers_max(self.user_baskets_count) - 1)
        self.user_baskets_pointers = np.round(
            b * np.random.uniform(low=0, high=1, size=len(a))
        ).astype("int32")

        # FIXME: del self.df
        self.time_log("Generator initialization done.")

    def pointers_max(self, counts):
        """Maximum value the pointer can become."""
        return np.max((counts - self.history_size, np.zeros(len(counts))), axis=0)

    def pointers_increase(self, pointers, counts):
        """Keep increasing the pointers. Once they cross a line we reset them to 0."""
        increased = pointers + 1
        limit = self.pointers_max(counts)

        survive = increased < limit
        return increased * survive

    def __len__(self):
        """Number of epochs (remainder of the last one is left out to keep the sizes consistent)."""
        return int(np.floor(len(self.user_baskets) / self.batch_size))

    def __call__(self, batch_size):
        """Allows to use the size of batch when calling the training."""
        self.batch_size = batch_size
        return self

    def on_epoch_end(self):
        """Updates indexes after each epoch. Do not shuffle when not in  training mode."""
        self.indexes = np.arange(self.unique_user_count)
        if self.training:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        """Get the batch at given index. Oldest values are at the beginning, newest are at the end."""
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        batch_baskets = self.user_baskets[indexes]
        batch_metadata = self.user_metadata[indexes]

        # Counts of elements in each of the baskets in batch
        batch_counts = self.user_baskets_count[indexes]
        batch_pointers = self.user_baskets_pointers[indexes]

        target_baskets = NotImplemented
        target_metadata = NotImplemented

        # TODO: Do the subsampling of the lists here when in training
        if self.training:
            self.user_baskets_pointers[indexes] = self.pointers_increase(batch_pointers, batch_counts)

        return (
            zip(batch_baskets, batch_metadata),
            zip(target_baskets, target_metadata)
        )
