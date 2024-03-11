import pandas as pd

def label_csv(csv_file):
    # Read CSV file
    data = pd.read_csv(csv_file)

    # Add new column 'class' to dataframe
    data['class'] = ''

    # Apply heuristics and pattern matching to assign labels
    for index, row in data.iterrows():
        if any(keyword.lower() in str(cell).lower() for cell in row for keyword in ['missing']):
            data.at[index, 'class'] = 'missing'
        elif any(keyword.lower() in str(cell).lower() for cell in row for keyword in ['invalid', 'like', 'duplicate']):
            data.at[index, 'class'] = 'checksum'
        elif any(keyword.lower() in str(cell).lower() for cell in row for keyword in ['mismatch', 'match']):
            data.at[index, 'class'] = 'mismatch'
        else:
            data.at[index, 'class'] = 'others'

    # Save the modified dataframe back to CSV
    data.to_csv('labeled_' + csv_file, index=False)

# Example usage
label_csv('input.csv')
