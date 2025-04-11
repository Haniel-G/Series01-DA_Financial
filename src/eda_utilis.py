# Tools for Exploratory Data Analysis

# Importing libraries 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys

# Debugging.
from src.exception import CustomException
from pathlib import Path
from src.logger import *
from collections import defaultdict

# Add the src directory to the system path for module imports
sys.path.append(os.path.abspath(os.path.join('..', 'src')))

# Warnings.
from warnings import filterwarnings
filterwarnings('ignore')

# Defining a color palette for visualizations
palette=sns.color_palette(
    ['#023050', '#0080b6', '#0095c7', '#90a4ae', '#6a3d9a', '#8f4f4f', '#e31a1c',
    '#e85d10', '#ff8210', '#ff9c35']
)

# 1. Function to load a dataset from a file
def load_data(file_path: str | Path) -> dict:
    """
    Loads a dataset from .csv or .xlsx file and returns structured dictionary.
    
    Parameters:
        file_path (str or Path): Path to the file to be loaded.

    Returns:
        dict: A structured dictionary containing:
            - "data": A DataFrame (if .csv) or a dictionary of DataFrames (if .xlsx).
            - "metadata": Information about the loaded file (file name, number of rows and columns, or sheet names).

    Raises:
        FileNotFoundError: If the specified file is not found.
        ValueError: If the file format is not supported.
        RuntimeError: If an unexpected error occurs during file loading.

    Example:
        dataset = load_data("financial_data.xlsx")
        print(dataset["metadata"])  # Prints metadata information
        print(dataset["data"]["Sheet1"].head())  # Prints first rows of "Sheet1"
    """
    file_path = Path(file_path).expanduser().resolve()
    
    if not file_path.exists():
        logging.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")
    
    file_extension = file_path.suffix.lower()
    
    try:

        if file_extension == ".xlsx":

            with pd.ExcelFile(file_path, engine="openpyxl") as xls:
                sheets = {
                    sheet: pd.read_excel(xls, sheet_name=sheet) for sheet in xls.sheet_names
                }
            logging.info(f"Excel file loaded. Sheets: {list(sheets.keys())}")

            return {
                "data": sheets, "metadata": {
                    "file": str(file_path), "sheets": list(sheets.keys())
                }
            }
        
        elif file_extension == ".csv":
            df = pd.read_csv(file_path, encoding="utf-8", on_bad_lines="skip")
            logging.info(f"CSV file loaded. Shape: {df.shape}")
            return df
        
        else:
            raise ValueError("Unsupported file format. Use .xlsx or .csv")
    
    except Exception as e:
        logging.error(f"Error loading {file_path}: {e}")
        raise RuntimeError(f"Error loading file: {e}")


# 2. Function to display general information about the dataset
def display_info(df):
    """
    Displays general information and descriptive statistics about the dataset.

    Parameters:
        df (pd.DataFrame): DataFrame to be analyzed.

    Output:
        - Prints dataset structure (df.info()).
        - Prints descriptive statistics (df.describe()).

    Example:
        display_info(df)

    """

    #logging.info("Displaying dataset information.")
    print("\n🔹 Dataset Info:\n")
    print(df.info())
    print('-'*10)
    print("\n🔹 Descriptive Statistics:\n")
    print(df.describe())


# 3. Function to handle missing values with different strategies
def handle_missing_values(df, strategy=None):
    """
    Handles missing values using different strategies.

    Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        strategy (str, optional): Strategy for handling missing values.
            Options:
            - 'mean': Replace missing values with the column mean.
            - 'median': Replace missing values with the column median.
            - 'mode': Replace missing values with the most frequent value in the column.
            - 'drop': Remove rows with missing values.
            - None: Only displays the number of missing values.

    Returns:
        pd.DataFrame: The DataFrame with missing values handled.

    Example:
        df_cleaned = handle_missing_values(df, strategy="mean")

    """
    logging.info("Handling missing values.")
    print("\n◇ Missing Values Before Treatment:\n")
    print(df.isnull().sum())
    
    if strategy is None:
        return df
    
    strategies = {
        "mean": df.fillna(df.mean()),
        "median": df.fillna(df.median()),
        "mode": df.fillna(df.mode().iloc[0]),
        "drop": df.dropna()
    }
    
    if strategy in strategies:
        df = strategies[strategy]
    
    else:
        logging.error("Invalid missing value strategy.")
        raise ValueError("Invalid strategy. Choose from: 'mean', 'median', 'mode', 'drop'.")
    
    print("\n◇ Missing Values After Treatment:\n")
    print(df.isnull().sum())
    
    return df


# 4. Function to convert date columns
def convert_dates(df, columns):
    """
    Converts specified columns to datetime format.

    Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        columns (list): List of column names to be converted to datetime.

    Returns:
        pd.DataFrame: DataFrame with converted date columns.


    Notes:
        - Uses the format '%d/%m/%Y', ignoring conversion errors.
        - Invalid values are converted to NaT (Not a Time).

    Example:

        df = convert_dates(df, ["transaction_date"])

    """
    logging.info("Converting date columns.")
    df[columns] = df[columns].apply(pd.to_datetime, format='%d/%m/%Y', errors='coerce')
    return df


# 6. Function to format currency
def format_currency(dataset, tables=None, columns=None, format_str="R$ {:,.2f}"):
    """
    Formats specified numeric columns as currency in a dictionary of DataFrames or a single DataFrame.

    Parameters:
        dataset (dict or pd.DataFrame): Dictionary containing tables as DataFrames or a single DataFrame.
        tables (list, optional): List of table names to process (only if dataset is a dict).
        columns (list): List of column names to format.
        format_str (str): Format string for currency (default: "R$ {:,.2f}").

    Returns:
        dict or pd.DataFrame: The dataset with formatted currency values.
    """

    if columns is None:
        raise ValueError("You must specify at least one column to format.")

    formatted_data = {}

    if isinstance(dataset, dict) and 'data' in dataset:  # Check if 'data' key exists
        if tables is None:
            raise ValueError("You must specify 'tables' when working with multiple tables.")

        for table in tables:
            if table in dataset['data']:  # Access the correct dictionary level
                df = dataset['data'][table].copy()

                for column in columns:
                    if column in df.columns:
                        df[column] = df[column].apply(
                            lambda x: format_str.format(x).replace(',', 'X').replace('.', ',').replace('X', '.')
                            if isinstance(x, (int, float)) else x
                        )

                formatted_data[table] = df
            else:
                raise ValueError(
                    f"Table '{table}' not found in dataset['data']." 
                    f"Available tables: {list(dataset['data'].keys())}"
                )

        return formatted_data

    elif isinstance(dataset, pd.DataFrame):  
        if tables is not None:
            raise ValueError(
                "The 'tables' parameter should not be used when the dataset is a single DataFrame."
            )

        df = dataset.copy()
        for column in columns:
            if column in df.columns:
                df[column] = df[column].apply(
                    lambda x: format_str.format(x).replace(',', 'X').replace('.', ',').replace('X', '.')
                    if isinstance(x, (int, float)) else x
                )

        return df

    else:
        raise TypeError(
            "The dataset must be a dictionary with a 'data' key or a single DataFrame."
        )


# 7. Function to visualize the distribution of numeric columns
def visualize_plots(dataset, tables=None, columns=None, plot_types=None, 
                    color='#023047', bins=30, kde=False, hue=None, 
                    figsize=(18, 8), outliers=False, mean=None, text_y=1):
    """
    Generates multiple plots for selected columns, supporting both multiple tables (dict of DataFrames)
    and single DataFrame structures.

    Parameters:
        dataset (dict or pd.DataFrame): Either a dictionary containing tables as DataFrames or a single DataFrame.
        tables (list, optional): List of table names to search for columns (only if dataset is a dict).
        columns (list): List of numeric column names to visualize.
        plot_types (list, optional): Type of plot for each column ('histogram', 'boxplot', 'barplot').
        color (str, optional): Color of the plots (default: '#023047').
        bins (int, optional): Number of bins for histograms (default: 30).
        kde (bool, optional): Whether to include KDE in histograms (default: False).
        hue (str, optional): Column for hue differentiation (default: None).
        figsize (tuple, optional): Figure size (default: (18, 8)).
        outliers (bool, optional): Whether to show boxplots for outliers.
        barplot (bool, optional): Whether to generate barplots.
        mean (str, optional): Column name for calculating the mean in barplots.
        text_y (int, optional): Adjusts text position in barplots.
    """
    
    selected_data = {}

    if isinstance(dataset, dict) and 'data' in dataset:  

        if tables is None:
            raise ValueError(
                "You must specify 'tables' when working with multiple tables."
            )

        for table in tables:
            if table in dataset['data']:  
                df = dataset['data'][table]
                for column in columns:
                    if column in df.columns:
                        selected_data[f"{table} - {column}"] = df[column]

        if not selected_data:
            raise ValueError("None of the specified columns were found.")

    elif isinstance(dataset, pd.DataFrame):  

        if tables is not None:
            raise ValueError(
                "The 'tables' parameter should not be used when the dataset is a single DataFrame."
            )

        for column in columns:
            if column in dataset.columns:
                selected_data[column] = dataset[column]

    else:
        raise TypeError(
            "The dataset must be a dictionary of tables or a single DataFrame."
        )

    if not selected_data:
        raise ValueError("None of the specified columns were found.")

    num_features = len(selected_data)
    num_cols = min(3, num_features)
    num_rows = -(-num_features // num_cols)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = np.array(axes).reshape(num_rows, num_cols)

    for i, (column_name, column_data) in enumerate(selected_data.items()):

        row, col = divmod(i, num_cols)
        ax = axes[row, col]
        plot_type = 'boxplot' if outliers else (plot_types[i] if plot_types and i < len(plot_types) else 'histogram')

        if plot_type == 'histogram':
            sns.histplot(column_data, kde=kde, bins=bins, color=color, ax=ax, stat='density')
            x_max = np.percentile(column_data, 99.9)
            ax.set_xlim(column_data.min(), x_max)
            ax.set_title(f'Distribution of {column_name}', fontsize=12)
            ax.set_xlabel(column_name, fontsize=10)
            ax.set_ylabel('Density', fontsize=10)

        elif plot_type == 'boxplot' or outliers:
            if isinstance(column_data, pd.Series) and not column_data.isnull().all():
                sns.boxplot(x=column_data, color=color, ax=ax)
                ax.set_title(f'Boxplot of {column_name}', fontsize=12)
                ax.set_xlabel(column_name, fontsize=10)
            
            else:
                print(
                    f"Warning: Column '{column_name}' is empty or contains only NaN values."
                )

        elif plot_type == 'barplot':

            if column_data.dtype == 'object' or column_data.nunique() < 15:  
                value_counts = column_data.value_counts(normalize=True) * 100  
                sns.barplot(
                    x=value_counts.values, y=value_counts.index, ax=ax, color=color
                )
                ax.set_title(f'Distribution of {column_name}', fontsize=12)
                ax.set_xlabel('Percentage (%)', fontsize=10)
            
            else:

                if mean:
                    data_grouped = dataset.groupby([column_name])[[mean]].mean().reset_index()
                    data_grouped[mean] = round(data_grouped[mean], 2)
                    sns.barplot(
                        x=data_grouped[mean], y=data_grouped[column_name], ax=ax, 
                        color=color
                    )
                    
                    for index, value in enumerate(data_grouped[mean]):
                        ax.text(
                            value + text_y, index, f'{value:.1f}', va='center', 
                            fontsize=10
                        )

    for j in range(i + 1, num_rows * num_cols):  
        fig.delaxes(axes.flatten()[j])

    plt.tight_layout()
    plt.show()


# 8. Function to identify outliers using IQR
def detect_outliers(dataset, tables=None, columns=None, verbose=False, 
                    strategy=None, plot=False):
    """
    Identifies outliers in specified numeric columns using the IQR method,
    supports multiple tables, and optionally generates boxplots for visualization.

    Parameters:
        dataset (dict or pd.DataFrame): Either a dictionary containing tables as DataFrames or a single DataFrame.
        tables (list, optional): List of table names to search for columns (only if dataset is a dict).
        columns (list): List of numeric column names to check for outliers.
        verbose (bool, optional): If True, prints a summary of detected outliers.
        strategy (str, optional): Determines the action with outliers:
            - 'remove': Removes outliers from the dataset.
            - 'análise separada': Creates statistics separately for all data, without outliers, and only outliers.
        plot (bool, optional): If True, generates boxplots of the detected outliers.

    Returns:
        - If strategy='remove': Cleaned DataFrame without outliers.
        - If strategy='análise separada': DataFrame with statistical summary.
        - If verbose=True: Prints outlier summary.
        - Otherwise: None.
    """

    selected_data = {}
    
    # Check if the dataset is a dictionary of tables or a single DataFrame
    if isinstance(dataset, dict) and 'data' in dataset:  
        
        if tables is None:
            raise ValueError(
                "You must specify 'tables' when working with multiple tables."
            )

        for table in tables:

            if table in dataset['data']:  
                df = dataset['data'][table]

                for column in columns:
                    if column in df.columns:
                        selected_data.setdefault(table, {})[column] = df[column]

    elif isinstance(dataset, pd.DataFrame):  

        if tables is not None:
            raise ValueError(
                "The 'tables' parameter should not be used when the dataset is a single DataFrame."
            )

        for column in columns:
            if column in dataset.columns:
                selected_data.setdefault("Dataset", {})[column] = dataset[column]
    
    else:
        raise TypeError(
            "The dataset must be a dictionary of tables or a single DataFrame."
        )

    if not selected_data:
        raise ValueError("None of the specified columns were found.")
    
    summary = defaultdict(list)
    cleaned_data = dataset.copy() if strategy == 'remove' else None
    stats_list = []

    # Iterate through the selected data to identify outliers
    for table, columns_data in selected_data.items():

        for column, column_data in columns_data.items():
            q1, q3 = column_data.quantile([0.25, 0.75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = column_data[
                (column_data < lower_bound) | (column_data > upper_bound)
            ]

            non_outliers = column_data[
                (column_data >= lower_bound) & (column_data <= upper_bound)
            ]

            percentage = round((len(outliers) / len(column_data)) * 100, 2)
            
            summary[table].append(
                f"    - {column}: {len(outliers)} outliers ({percentage}%)"
            )

            if strategy == 'remove':
                cleaned_data['data'][table] = dataset['data'][table][
                    (dataset['data'][table][column] >= lower_bound) & 
                    (dataset['data'][table][column] <= upper_bound)
                ]
                
            
            elif strategy == 'separate analysis':
                stats_df = pd.DataFrame({
                    'Métrica': [
                        'Mean', 'Median', 'Standard Deviation', 'Minimum', 'Maximum', 'Count'
                    ],
                    'All Data': [
                        column_data.mean(), column_data.median(), column_data.std(),
                        column_data.min(), column_data.max(), len(column_data)
                    ],
                    'Sem Outliers': [
                        non_outliers.mean(), non_outliers.median(), non_outliers.std(),
                        non_outliers.min(), non_outliers.max(), len(non_outliers)
                    ],
                    'Apenas Outliers': [
                        outliers.mean(), outliers.median(), outliers.std(),
                        outliers.min(), outliers.max(), len(outliers)
                    ]
                })
                # stats_df.insert(0, 'Tabela', table)
                # stats_df.insert(1, 'Coluna', column)
                stats_list.append(stats_df)

            if plot:
                plt.figure(figsize=(8, 5))
                sns.boxplot(x=column_data)
                plt.title(f'Boxplot - {table} - {column}')
                plt.show()

    if verbose and summary:

        for table, details in summary.items():
            print(f"◇ {table}")
            print("\n".join(details))

    if strategy == 'remove':
        return cleaned_data
    
    if strategy == 'separate analysis' and stats_list:
        return pd.concat(stats_list, ignore_index=True)

    return None  # Default case if no specific strategy is selected


# 9. Function to plot 
def plot_categorical_distribution(df_dict, columns, cols=3, figsize=(24, 12), 
                                  show_xlabel=False, show_ylabel=False):
    """
    Generates bar plots for multiple categorical variables from multiple DataFrames.

    Parameters:
    - df_dict: Dictionary where keys are table names and values are DataFrames.
    - columns: List of categorical column names to plot.
    - cols: Number of columns in the subplot grid.
    - figsize: Tuple specifying the figure size.
    - show_xlabel: Boolean, whether to show x-axis labels.
    - show_ylabel: Boolean, whether to show y-axis labels.
    """
    plot_data = []

    # Collect valid columns from each table
    for table_name, df in df_dict.items():

        for col in columns:

            if col in df.columns:
                plot_data.append((df[col], f"{col} ({table_name})"))

    # Calculate grid size
    num_plots = len(plot_data)
    rows = -(-num_plots // cols)  # Ceiling division to get the number of rows

    # Create subplots
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()  # Flatten axes for easier iteration

    # Generate bar plots
    for i, (column_data, title) in enumerate(plot_data):

        ax = axes[i]
        sns.countplot(y=column_data, order=column_data.value_counts().index, palette=palette, ax=ax)
        ax.set_title(title, fontsize=14)

        # Conditionally set axis labels
        if not show_xlabel:
            ax.set_xlabel("")
        if not show_ylabel:
            ax.set_ylabel("")

        # Add value labels
        for p in ax.patches:
            ax.annotate(f"{p.get_width():.0f}", (p.get_width(), p.get_y() + p.get_height()/2),
                        ha='left', va='center', fontsize=12, color='black')

    # Remove unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


# 10. Function to calculate the coefficient of variation
def coefficient_of_variation(df, columns):
    """
    Calculates the coefficient of variation for specified columns in a DataFrame.

    The coefficient of variation is a standardized measure of dispersion, calculated as the ratio of the standard deviation to the mean. It is useful for comparing the variability of datasets with different units or scales.

    Parameters:
    - df: DataFrame containing the data.
    - columns: List of column names for which the coefficient of variation will be calculated.

    Returns:
    - A dictionary where keys are column names and values are the corresponding coefficients of variation. If the mean of a column is zero, the coefficient is set to NaN to avoid division by zero.

    Logs:
    - Logs the coefficient of variation for each column.
    - Logs a warning if a specified column is not found in the DataFrame.
    """
    
    coef_vars = {}
    
    for column in columns:

        if column in df.columns:
            mean = df[column].mean()
            std_dev = df[column].std()
            
            if mean != 0:
                coef_var = std_dev / mean
            else:
                coef_var = float('nan')  # Set to NaN to avoid division by zero
            
            coef_vars[column] = coef_var
            logging.info(f"Coefficient of Variation for '{column}': {coef_var:.2f}")
        
        else:
            logging.warning(f"Column '{column}' not found in DataFrame.")
    
    return coef_vars


# 11. Function to save the dataset
def save_data(data, output_path, explicit_path=False):
    """
    Saves the dataset as an Excel (.xlsx) or CSV (.csv) file without modifying the original dataset.

    This function takes a dictionary containing multiple sheets (DataFrames) and saves them into a new file. 
    If the dataset contains multiple sheets, it will be saved as an Excel file with separate sheets. 
    If a CSV format is specified, only the first sheet will be saved.

    Parameters:
        data (dict): Dictionary containing the dataset.
            - Must include a "data" key where sheet names are keys and DataFrames are values.
        output_path (str): File path where the dataset will be saved (.xlsx or .csv).

    Returns:
        None: Writes the output file to the specified path.

    Raises:
        ValueError: If the dataset structure is incorrect or contains no valid sheets.
        RuntimeError: If an error occurs while saving the file.

    Logs:
        - Logs a success message when the file is saved.
        - Logs a warning if a sheet is not a DataFrame and is ignored.
        - Logs an error if saving fails.

    Example:
        save_data(dataset, "processed_data.xlsx")

    Raises:
    - ValueError: If the dataset is not structured correctly or contains no valid sheets.
    - RuntimeError: If an error occurs while saving the file.
    """
    
    # Check if the dataset contains multiple sheets
    if not isinstance(data, dict) or 'data' not in data:
        raise ValueError(
            "The dataset must be a dictionary containing 'data' with the sheets."
        )

    sheets = data['data']  # Extracting only the sheets

    if not sheets:
        raise ValueError("The dataset does not contain any sheets to save.")

    file_extension = os.path.splitext(output_path)[1].lower()

    try:

        if file_extension == ".xlsx":

            with pd.ExcelWriter(output_path) as writer:

                for sheet_name, df in sheets.items():

                    if isinstance(df, pd.DataFrame):
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
                        
                    else:
                        logging.warning(
                            f"Warning: '{sheet_name}' is not a DataFrame and was ignored."
                        )
            
            if explicit_path:
                logging.info(f"Excel file successfully saved at: {output_path}")

            else: 
                logging.info(f"Excel file successfully saved.")

        elif file_extension == ".csv":
            first_sheet_name = list(sheets.keys())[0]  # Taking the first sheet
            sheets[first_sheet_name].to_csv(output_path, index=False)
            logging.info(f"CSV file successfully saved at: {output_path}")

        else:
            raise ValueError("Unsupported file format. Please provide a .xlsx or .csv file.")

    except Exception as e:
        logging.error(f"Error saving file: {e}")
        raise RuntimeError(f"Error saving file: {e}")

    
