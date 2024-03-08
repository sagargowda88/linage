import re

def extract_column_names(sql_query):
    # Define the pattern to find column names
    pattern = r'SELECT\s+(.*?)\s+FROM'

    # Use regular expression to find column names
    match = re.search(pattern, sql_query, re.IGNORECASE)

    if match:
        # Extract column names and split them by commas
        column_names = match.group(1).split(',')
        # Remove leading and trailing whitespaces from column names
        column_names = [name.strip() for name in column_names]
        return column_names
    else:
        return None

# Example usage:
sql_query = "SELECT id, name, age FROM employees"
column_names = extract_column_names(sql_query)
print(column_names)
