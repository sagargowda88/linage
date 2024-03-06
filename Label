import pandas as pd

def label_csv(csv_file):
    # Read CSV file
    data = pd.read_csv(csv_file)

    # Add new column 'class' to dataframe
    data['class'] = ''

    # Apply heuristics and pattern matching to assign labels
    for index, row in data.iterrows():
        if 'missing' in str(row).lower():
            data.at[index, 'class'] = 'missing'
        elif 'mismatch' in str(row).lower():
            data.at[index, 'class'] = 'mismatch'
        elif 'checksum' in str(row).lower():
            data.at[index, 'class'] = 'checksum'
        else:
            data.at[index, 'class'] = 'reconciliation'

    # Save the modified dataframe back to CSV
    data.to_csv('labeled_' + csv_file, index=False)

# Example usage
label_csv('input.csv')
