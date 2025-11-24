def handle_values(df):
    import pandas as pd
    import numpy as np

    df = df.fillna(0)
    df = df.replace([np.inf,-np.inf],0)

    return df
