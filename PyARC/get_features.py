import pandas as pd
import numpy as np
from itertools import permutations

class GetFeatures:

    @staticmethod
    def spot_tou(main_df, tou_df):
        # Merge
        merged_df = pd.merge(main_df, tou_df[['Hour', 'ToU']], on='Hour', how='left')
        main_df['ToU'] = merged_df['ToU']

        # Count degli elementi unici nella colonna "ToU"
        tou_counts = tou_df['ToU'].value_counts().reset_index()
        tou_counts.columns = ['ToU', 'Count']

        # Ordina in base al conteggio in modo crescente
        tou_counts = tou_counts.sort_values(by='Count')

        # Assegna i ranghi alla colonna "Extension" in base al conteggio
        tou_counts['Extension'] = range(1, len(tou_counts) + 1)

        # Aggiungi la colonna "Extension" al dataframe principale
        main_df = pd.merge(main_df, tou_counts[['ToU', 'Extension']], on='ToU', how='left')

        return main_df

    @staticmethod
    def get_features(df):
        unique_tou_values = df['ToU'].unique()

        for tou_value in unique_tou_values:
            tou_sum_column_name = f'{tou_value}'
            tou_filtered_df = df[df['ToU'] == tou_value]
            tou_sum_df = tou_filtered_df.groupby(['User', 'Year', 'Month'])[['Consumption']].sum().reset_index()
            tou_sum_df.rename(columns={'Consumption': tou_sum_column_name}, inplace=True)
            df = pd.merge(df, tou_sum_df, on=['User', 'Year', 'Month'], how='left')

        # Creazione della colonna "Monthly_consumption" come somma delle colonne create dalla funzione
        df['Monthly_consumption'] = df[[f'{tou_value}' for tou_value in unique_tou_values]].sum(axis=1)

        # Creazione di colonne come rapporto tra le colonne create e "Monthly_consumption"
        for tou_value in unique_tou_values:
            ratio_col_name = f'{tou_value}_Monthly_Ratio'
            df[ratio_col_name] = df[f'{tou_value}'] / df['Monthly_consumption']

        return df

    import pandas as pd
    from itertools import permutations

    @staticmethod
    def create_permutation_ratios(df):
        unique_to_u_values = df['ToU'].unique()
        extension_values = df.groupby('ToU')['Extension'].max().reset_index()

        for perm in permutations(unique_to_u_values, 2):
            numerator, denominator = perm
            ratio_col_name = f'{numerator}_{denominator}_Ratio'

            # Condizione di confronto delle estensioni
            if extension_values[extension_values['ToU'] == numerator]['Extension'].values[0] > \
                    extension_values[extension_values['ToU'] == denominator]['Extension'].values[0]:
                # Calcola il rapporto tra le colonne
                df[ratio_col_name] = df[f'{numerator}'] / df[f'{denominator}']

        return df





