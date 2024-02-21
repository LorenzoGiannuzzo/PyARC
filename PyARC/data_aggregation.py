import pandas as pd


class Aggregator:

    def expand_dataframe(merged_data):
        # Create a dataframe with hours from 0 to 23
        hours_df = pd.DataFrame({'Hour': range(24)})

        # Create a dataframe with all combinations of User, Year, Month, Day, and Hour
        expanded_data = pd.merge(merged_data.assign(key=1), hours_df.assign(key=1), on='key').drop('key', axis=1)

        # Add the "Day" column calculated based on the month
        expanded_data['Day'] = expanded_data.groupby(['User', 'Year', 'Month'])['Hour'].transform(
            lambda x: (x // 24) + 1)

        return expanded_data

    def load_profile_generator(df):
        # Iterate through unique elements in the "main ToU" column
        for main_ToU_value in df['main ToU'].unique():
            # Find the name of the column corresponding to "main ToU"
            column_name = main_ToU_value

            # Multiply the "weight" column by the value of the column corresponding to "main ToU"
            df['load'] = df['weight'] * df[column_name]

        return df

    def aggregate_load(df):
        # Aggrega la colonna "load" per ogni combinazione di "Year", "Month", "Day", e "Hour_y"
        aggregated_df = df.groupby(['Year', 'Month', 'Day', 'Hour_y'])['load'].sum().reset_index()

        # Rinomina la colonna "load" aggregata
        aggregated_df = aggregated_df.rename(columns={'load': 'Aggregate load', 'Hour_y': 'Hour'})

        return aggregated_df



