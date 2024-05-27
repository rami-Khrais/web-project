import pandas as pd

data_path = '/web_analysis/archive/CNN_Articels_clean_2/CNN_Articels_clean.csv'
data = pd.read_csv(data_path)

def remove_null_rows(input_csv, output_csv=None):
    """
    Remove rows containing any null values from a CSV file.

    Parameters:
    input_csv (str): Path to the input CSV file.
    output_csv (str, optional): Path to save the cleaned CSV file. If None, the file is not saved.

    Returns:
    pd.DataFrame: DataFrame without rows containing null values.
    """
    # Load the CSV data into a pandas DataFrame
    df = pd.read_csv(input_csv)

    # Remove rows containing any null values
    cleaned_df = df.dropna()

    # If an output path is provided, save the cleaned DataFrame to a new CSV file
    if output_csv:
        cleaned_df.to_csv(output_csv, index=False)

    return cleaned_df

def clean_author_column(df, output_csv=None):
    """
    Split the values in the 'Author' column on commas and keep only the first segment.

    """
    # Load the CSV data into a pandas DataFrame
    
    # Split the 'Author' column on commas and keep only the first segment
    df['Author'] = df['Author'].apply(lambda x: x.split(',')[0] if pd.notnull(x) else x)
    
    # If an output path is provided, save the modified DataFrame to a new CSV file
    if output_csv:
        df.to_csv(output_csv, index=False)
    
    return df




cleaned_data = remove_null_rows(data_path)
modified_data = clean_author_column(cleaned_data)
modified_data = modified_data.drop(columns=['Index'])
modified_data.to_csv(data_path.replace('CNN_Articels_clean.csv','cleaned_data.csv'),index=False)


