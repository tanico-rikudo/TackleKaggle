import numpy as np
import pandas as pd
class common:
    def reduce_mem_usage(df, verbose=True):
        numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
        start_mem = df.memory_usage().sum() / 1024 ** 2
        for col in df.columns:
            col_type = df[col].dtypes
            if col_type in numerics:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == "int":
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
                else:
                    if (
                        c_min > np.finfo(np.float16).min
                        and c_max < np.finfo(np.float16).max
                    ):
                        df[col] = df[col].astype(np.float16)
                    elif (
                        c_min > np.finfo(np.float32).min
                        and c_max < np.finfo(np.float32).max
                    ):
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
        end_mem = df.memory_usage().sum() / 1024 ** 2
        if verbose:
            print(
                "Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)".format(
                    end_mem, 100 * (start_mem - end_mem) / start_mem
                )
            )
        return df
    
    
    
    def encode_categorical(df, cols, fillna=False):
        for col in cols:
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(
                df[col].fillna("MISSING") if fillna else df[col]
            )
        return df
    
    def plot_cv_indices(cv, X, y, dt_col, lw=10):
        n_splits = cv.get_n_splits()
        _, ax = plt.subplots(figsize=(20, n_splits))

        # Generate the training/testing visualizations for each CV split
        for ii, (tr, tt) in enumerate(cv.split(X=X, y=y)):
            # Fill in indices with the training/test groups
            indices = np.array([np.nan] * len(X))
            indices[tt] = 1
            indices[tr] = 0

            # Visualize the results
            ax.scatter(
                X[dt_col],
                [ii + 0.5] * len(indices),
                c=indices,
                marker="_",
                lw=lw,
                cmap=plt.cm.coolwarm,
                vmin=-0.2,
                vmax=1.2,
            )

        # Formatting
        MIDDLE = 15
        LARGE = 20
        ax.set_xlabel("Datetime", fontsize=LARGE)
        ax.set_xlim([X[dt_col].min(), X[dt_col].max()])
        ax.set_ylabel("CV iteration", fontsize=LARGE)
        ax.set_yticks(np.arange(n_splits) + 0.5)
        ax.set_yticklabels(list(range(n_splits)))
        ax.invert_yaxis()
        ax.tick_params(axis="both", which="major", labelsize=MIDDLE)
        ax.set_title("{}".format(type(cv).__name__), fontsize=LARGE)
        return ax

    
    class CustomTimeSeriesSplitter:
        def __init__(self, n_splits=5, train_days=80, test_days=20, dt_col="date"):
            self.n_splits = n_splits
            self.train_days = train_days
            self.test_days = test_days
            self.dt_col = dt_col

        def split(self, X, y=None, groups=None):
            sec = (X[self.dt_col] - X[self.dt_col][0]).dt.total_seconds()
            duration = sec.max() - sec.min()

            train_sec = 3600 * 24 * self.train_days
            test_sec = 3600 * 24 * self.test_days
            total_sec = test_sec + train_sec
            step = (duration - total_sec) / (self.n_splits - 1)

            train_start = 0
            for idx in range(self.n_splits):
                train_start = idx * step
                train_end = train_start + train_sec
                test_end = train_end + test_sec

                if idx == self.n_splits - 1:
                    test_mask = sec >= train_end
                else:
                    test_mask = (sec >= train_end) & (sec < test_end)

                train_mask = (sec >= train_start) & (sec < train_end)
                test_mask = (sec >= train_end) & (sec < test_end)

                yield sec[train_mask].index.values, sec[test_mask].index.values

        def get_n_splits(self):
            return self.n_splits
