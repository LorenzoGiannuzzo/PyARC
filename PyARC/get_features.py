import pandas as pd
import numpy as np

class GetFeatures:

    @staticmethod
    def spot_tou(main_df, tou_df):
        merged_df = pd.merge(main_df, tou_df[['Hour', 'ToU']], on='Hour', how='left')
        main_df['ToU'] = merged_df['ToU']

        return main_df



