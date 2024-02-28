# Import necessary libraries
import pandas as pd
import os

class Aggregator:
    @staticmethod
    def expand_dataframe(merged_data):
        """
        Expand the dataframe to include Day, and Hour.

        Parameters:
        - merged_data (DataFrame): Input dataframe with merged data.

        Returns:
        - expanded_data (DataFrame): Expanded dataframe with added Hour and Day columns.
        """
        try:
            # Create a dataframe with hours from 0 to 23
            hours_df = pd.DataFrame({'Hour': range(24)})
            output_csv_path = os.path.join(os.path.dirname(__file__), "..", "test", "test2.csv")

            # Esporta il DataFrame in formato CSV
            merged_data.to_csv(output_csv_path,index=False)
            # Create a dataframe with all combinations of User, Year, Month, Day, and Hour
            expanded_data = pd.merge(merged_data.assign(key=1), hours_df.assign(key=1), on='key').drop('key', axis=1)

            # Add the "Day" column calculated based on the month
            expanded_data['Day'] = expanded_data.groupby(['User', 'Year', 'Month'])['Hour'].transform(
                lambda x: (x // 24) + 1)
            output_csv_path = os.path.join(os.path.dirname(__file__), "..", "test", "test1.csv")

            # Esporta il DataFrame in formato CSV
            expanded_data[['User','Month','Day','Hour','F1','Monthly_consumption']].to_csv(output_csv_path, index=False)
            return expanded_data

        except Exception as e:
            raise ValueError(f"Error in expand_dataframe: {str(e)}")

    @staticmethod
    def load_profile_generator(df):
        """
        Generate load profiles based on the multiplication of the "weight" column by "main ToU" values.

        Parameters:
        - df (DataFrame): Input dataframe.

        Returns:
        - df (DataFrame): Updated dataframe with load profiles.
        """
        try:
            # Iterate through unique elements in the "main ToU" column
            for main_ToU_value in df['main ToU'].unique():
                # Find the name of the column corresponding to "main ToU"
                column_name = main_ToU_value

                # Multiply the "weight" column by the value of the column corresponding to "main ToU"
                df['load'] = df['weight'] * df[column_name]
            print(df['load'])
            return df

        except Exception as e:
            raise ValueError(f"Error in load_profile_generator: {str(e)}")

    @staticmethod
    def aggregate_load(df):
        """
        Aggregate the "load" column for each combination of "Year", "Month", "Day", and "Hour_y".

        Parameters:
        - df (DataFrame): Input dataframe with load profiles.

        Returns:
        - aggregated_df (DataFrame): Aggregated dataframe with renamed columns.
        """
        try:
            # Aggregate the "load" column for each combination of "Year", "Month", "Day", and "Hour_y"
            aggregated_df = df.groupby(['Year', 'Month', 'Hour_y'])['load'].sum().reset_index()

            # Rename the aggregated "load" column
            aggregated_df = aggregated_df.rename(columns={'load': 'Aggregate load', 'Hour_y': 'Hour'})
            print(aggregated_df['Aggregate load'])
            return aggregated_df

        except Exception as e:
            raise ValueError(f"Error in aggregate_load: {str(e)}")
