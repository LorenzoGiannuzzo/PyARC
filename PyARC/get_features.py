import pandas as pd
import numpy as np
from itertools import permutations


class GetFeatures:

    @staticmethod
    def spot_tou(main_df, tou_df):
        # Merge main_df with tou_df on 'Hour' column
        merged_df = pd.merge(main_df, tou_df[['Hour', 'ToU']], on='Hour', how='left')
        main_df['ToU'] = merged_df['ToU']

        # Count unique elements in the "ToU" column
        tou_counts = tou_df['ToU'].value_counts().reset_index()
        tou_counts.columns = ['ToU', 'Count']

        # Sort by count in ascending order
        tou_counts = tou_counts.sort_values(by='Count')

        # Assign ranks to the "Extension" column based on the count
        tou_counts['Extension'] = range(1, len(tou_counts) + 1)

        # Add the "Extension" column to the main dataframe
        main_df = pd.merge(main_df, tou_counts[['ToU', 'Extension']], on='ToU', how='left')

        return main_df

    @staticmethod
    def get_features(df):
        # Get unique ToU values in the dataframe
        unique_tou_values = df['ToU'].unique()

        for tou_value in unique_tou_values:
            tou_sum_column_name = f'{tou_value}'
            tou_filtered_df = df[df['ToU'] == tou_value]
            tou_sum_df = tou_filtered_df.groupby(['User', 'Year', 'Month'])[['Consumption']].sum().reset_index()
            tou_sum_df.rename(columns={'Consumption': tou_sum_column_name}, inplace=True)
            df = pd.merge(df, tou_sum_df, on=['User', 'Year', 'Month'], how='left')

        # Create the "Monthly_consumption" column as the sum of the dynamically created columns
        df['Monthly_consumption'] = df[[f'{tou_value}' for tou_value in unique_tou_values]].sum(axis=1)

        # Create columns as ratios between the created columns and "Monthly_consumption"
        for tou_value in unique_tou_values:
            ratio_col_name = f'{tou_value}_Ratio_Monthly'
            df[ratio_col_name] = df[f'{tou_value}'] / df['Monthly_consumption']

        return df

    @staticmethod
    def create_permutation_ratios(df):
        unique_to_u_values = df['ToU'].unique()
        extension_values = df.groupby('ToU')['Extension'].max().reset_index()

        # Add the "Monthly_consumption" column as the sum of the columns created by X
        df['Monthly_consumption'] = df[[f'{tou_value}' for tou_value in unique_to_u_values]].sum(axis=1)

        for perm in permutations(unique_to_u_values, 2):
            numerator, denominator = perm
            ratio_col_name = f'{numerator}_{denominator}_Ratio'

            # Condition for comparing extensions
            if extension_values[extension_values['ToU'] == numerator]['Extension'].values[0] > \
                    extension_values[extension_values['ToU'] == denominator]['Extension'].values[0]:
                # Calculate the ratio between the columns created by X
                df[ratio_col_name] = df[f'{numerator}'] / df[f'{denominator}']

        return df

    @staticmethod
    def get_selected_features_and_cluster(df):
        # List of columns to select
        columns_to_select = ['Cluster', 'Monthly_consumption']

        # Add dynamically created columns
        unique_to_u_values = df['ToU'].unique()
        for tou_value in unique_to_u_values:
            columns_to_select.append(tou_value)
            columns_to_select.append(f'{tou_value}_Ratio_Monthly')

        unique_to_u_values = df['ToU'].unique()
        extension_values = df.groupby('ToU')['Extension'].max().reset_index()

        for perm in permutations(unique_to_u_values, 2):
            numerator, denominator = perm
            ratio_col_name = f'{numerator}_{denominator}_Ratio'

            # Condition for comparing extensions
            if extension_values[extension_values['ToU'] == numerator]['Extension'].values[0] > \
                    extension_values[extension_values['ToU'] == denominator]['Extension'].values[0]:
                columns_to_select.append(ratio_col_name)

        # Filter the DataFrame for selected columns
        selected_df = df[columns_to_select]

        selected_df = selected_df.drop_duplicates()
        selected_df = selected_df.replace([np.inf, -np.inf], np.nan).dropna()

        return selected_df

    @staticmethod
    def get_features2(df):
        # Select only numeric columns (excluding "User", "Year", "Month")
        numeric_columns = df.select_dtypes(include=['number']).columns.difference(["User", "Year", "Month"])

        # Add the "Monthly_consumption" column as the sum of numeric columns
        df['Monthly_consumption'] = df[numeric_columns].sum(axis=1)

        # Create all possible combinations of numeric columns, including "Monthly_consumption"
        all_columns = numeric_columns.union(['Monthly_consumption'])
        column_combinations = pd.DataFrame()

        # Calculate the ratio for each pair of columns
        for col1 in all_columns:
            for col2 in all_columns:
                if col1 != col2:
                    # Define the name of the ratio column
                    ratio_column_name = '{}_{}_Ratio'.format(col1, col2)

                    # Adjust the name if "Monthly_consumption" is involved
                    if "Monthly_consumption" in [col1, col2]:
                        ratio_column_name = '{}_Ratio_Monthly'.format(
                            col1) if col2 == "Monthly_consumption" else '{}_Ratio_Monthly'.format(col2)

                    # Calculate the ratio and add it to the new dataframe
                    column_combinations[ratio_column_name] = df[col1] / df[col2]

        # Concatenate the original dataframe with the new columns
        new_dataframe = pd.concat([df, column_combinations], axis=1)

        return new_dataframe

    @staticmethod
    def identify_main_ToU(df):
        # Select only rows with "Hour" in the range from 8 to 20
        df_interval = df[(df['Hour'] >= 10) & (df['Hour'] <= 18)]

        # Count occurrences of each "ToU" in the selected interval
        ToU_counts = df_interval['ToU'].value_counts()

        # Find the most frequent "ToU" and its count
        main_ToU = ToU_counts.idxmax()
        main_ToU_frequency = ToU_counts.max()

        # Calculate the total duration of the interval for the main "ToU"
        duration_main_ToU = df_interval[df_interval['ToU'] == main_ToU].shape[0]

        # Calculate the extension per unique cluster
        unique_clusters = df_interval['Cluster'].nunique()
        extension_per_cluster = duration_main_ToU / unique_clusters

        # Create new columns "main ToU" and "Extension" per unique cluster
        df['main ToU'] = main_ToU
        df['Extension'] = extension_per_cluster

        return df

    @staticmethod
    def calculate_sum_column(df):
        # Filter the dataframe for the most frequent "ToU"
        df_main_ToU = df[df['ToU'] == df['main ToU']]

        # Calculate the sum of elements in the "Centroid" column for each unique value in the "Cluster" column
        sum_per_cluster = df_main_ToU.groupby('Cluster')['Centroid'].sum()

        # Merge the sum for each cluster into the original dataframe
        df = pd.merge(df, sum_per_cluster.reset_index(name='sum'), on='Cluster', how='left')

        return df

    @staticmethod
    def calculate_weight_coefficient(df):
        # Calculate the "weight" column as "Centroid" / (Extension * sum)
        df['weight'] = df['Centroid'] / (df['Extension'] * df['sum'])

        return df

    @staticmethod
    def numeric_to_words(df):
        # Map numbers to words
        word_mapping = {
            1: 'One',
            2: 'Two',
            3: 'Three',
            4: 'Four',
            5: 'Five',
            6: 'Six',
            7: 'Seven',
            8: 'Eight',
            9: 'Nine'
        }

        # Apply the mapping to the specified column
        df['Cluster'] = df['Cluster'].map(word_mapping)

        return df





