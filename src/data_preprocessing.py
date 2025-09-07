"""
Data preprocessing module for surgery duration prediction.
Handles data loading, cleaning, and feature engineering.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import RAW_DATA_FILE, PROCESSED_DATA_DIR, RANDOM_SEED

def load_excel_data(file_path=None):
    """
    Load Excel data and analyze its structure.
    
    Args:
        file_path (str): Path to Excel file. If None, uses config default.
        
    Returns:
        dict: Dictionary containing dataframes for each sheet and metadata
    """
    if file_path is None:
        file_path = RAW_DATA_FILE
    
    print(f"Loading data from: {file_path}")
    
    # Read Excel file
    excel_file = pd.ExcelFile(file_path)
    sheet_names = excel_file.sheet_names
    
    print(f"Found {len(sheet_names)} sheet(s): {sheet_names}")
    
    # Load all sheets
    dataframes = {}
    for sheet_name in sheet_names:
        print(f"Loading sheet: {sheet_name}")
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        dataframes[sheet_name] = df
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Sample data:")
        print(df.head(3))
        print("-" * 50)
    
    return {
        'sheets': dataframes,
        'sheet_names': sheet_names,
        'file_path': file_path
    }

def identify_target_column(df, possible_names=None):
    """
    Identify the target column (surgery duration).
    
    Args:
        df (pd.DataFrame): Input dataframe
        possible_names (list): List of possible target column names
        
    Returns:
        str: Name of the target column
    """
    if possible_names is None:
        possible_names = [
            'duration', 'duration_min', 'surgery_duration', 'surgery_time',
            'operation_time', 'procedure_time', 'time_minutes', 'minutes',
            'surgery_length', 'operation_duration'
        ]
    
    # Check for exact matches
    for name in possible_names:
        if name in df.columns:
            return name
    
    # Check for partial matches
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in ['duration', 'time', 'length']):
            return col
    
    # If no match found, return None
    return None

def clean_and_standardize_data(df, target_col=None):
    """
    Clean and standardize the data.
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_col (str): Name of the target column
        
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    print("Starting data cleaning and standardization...")
    
    # Create a copy to avoid modifying original
    df_clean = df.copy()
    
    # 1. Standardize column names
    df_clean.columns = [col.strip().lower().replace(' ', '_') for col in df_clean.columns]
    
    # 2. Identify and rename target column
    if target_col is None:
        target_col = identify_target_column(df_clean)
    
    if target_col:
        if target_col != 'duration_min':
            df_clean = df_clean.rename(columns={target_col: 'duration_min'})
            print(f"Renamed target column '{target_col}' to 'duration_min'")
    else:
        print("Warning: No target column identified!")
    
    # 3. Handle missing values
    print("Handling missing values...")
    missing_info = df_clean.isnull().sum()
    print("Missing values per column:")
    print(missing_info[missing_info > 0])
    
    # 4. Basic data type conversion
    print("Converting data types...")
    
    # Convert numeric columns
    numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if col != 'duration_min':  # Don't convert target yet
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Convert date/time columns
    date_columns = []
    priority_date_columns = ['op_startdttm_fix', 'op_enddttm_fix', 'room_enter_dttm_fix', 'room_exit_dttm_fix']
    
    # First, try to convert priority date columns
    for col in priority_date_columns:
        if col in df_clean.columns:
            try:
                df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
                date_columns.append(col)
                print(f"Converted priority date column: {col}")
            except:
                print(f"Failed to convert priority date column: {col}")
    
    # Then convert other date-like columns
    for col in df_clean.columns:
        if col not in date_columns and any(keyword in col.lower() for keyword in ['date', 'time', 'start', 'end']):
            try:
                df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
                date_columns.append(col)
                print(f"Converted {col} to datetime")
            except:
                pass
    
    # 5. Create time-based features
    if date_columns:
        print("Creating time-based features...")
        # Use the first priority date column if available, otherwise use the first available
        primary_date_col = date_columns[0] if date_columns else None
        if primary_date_col:
            df_clean = create_time_features(df_clean, primary_date_col)
            print(f"Created time features based on: {primary_date_col}")
    
    # 6. Handle categorical variables
    print("Processing categorical variables...")
    categorical_columns = df_clean.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        if col != 'duration_min':
            # Remove leading/trailing whitespace
            df_clean[col] = df_clean[col].astype(str).str.strip()
            # Replace empty strings with NaN
            df_clean[col] = df_clean[col].replace('', np.nan)
    
    print("Data cleaning completed!")
    return df_clean

def create_time_features(df, date_column):
    """
    Create time-based features from date column.
    
    Args:
        df (pd.DataFrame): Input dataframe
        date_column (str): Name of the date column
        
    Returns:
        pd.DataFrame: Dataframe with additional time features
    """
    df_copy = df.copy()
    
    # Extract time components
    df_copy[f'{date_column}_hour'] = df_copy[date_column].dt.hour
    df_copy[f'{date_column}_day_of_week'] = df_copy[date_column].dt.dayofweek
    df_copy[f'{date_column}_is_weekend'] = df_copy[date_column].dt.dayofweek.isin([5, 6])
    df_copy[f'{date_column}_month'] = df_copy[date_column].dt.month
    df_copy[f'{date_column}_quarter'] = df_copy[date_column].dt.quarter
    
    # Create time of day categories
    def categorize_hour(hour):
        if 6 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 18:
            return 'afternoon'
        elif 18 <= hour < 24:
            return 'evening'
        else:
            return 'night'
    
    df_copy[f'{date_column}_time_of_day'] = df_copy[f'{date_column}_hour'].apply(categorize_hour)
    
    return df_copy

def analyze_data_quality(df):
    """
    Analyze data quality and generate summary statistics.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        dict: Dictionary containing data quality metrics
    """
    print("Analyzing data quality...")
    
    quality_metrics = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
        'unique_values': {col: df[col].nunique() for col in df.columns},
        'numeric_summary': {},
        'categorical_summary': {}
    }
    
    # Numeric columns summary
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        quality_metrics['numeric_summary'][col] = {
            'mean': df[col].mean(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max(),
            'median': df[col].median()
        }
    
    # Categorical columns summary
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        value_counts = df[col].value_counts()
        quality_metrics['categorical_summary'][col] = {
            'unique_count': len(value_counts),
            'top_values': value_counts.head(5).to_dict(),
            'is_high_cardinality': len(value_counts) > 50
        }
    
    return quality_metrics

def save_processed_data(df, filename='hernia_clean.csv'):
    """
    Save processed data to CSV.
    
    Args:
        df (pd.DataFrame): Dataframe to save
        filename (str): Output filename
        
    Returns:
        str: Path to saved file
    """
    output_path = PROCESSED_DATA_DIR / filename
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to: {output_path}")
    return output_path

def main():
    """
    Main function to run the complete data preprocessing pipeline.
    """
    print("=" * 60)
    print("SURGERY DURATION PREDICTION - DATA PREPROCESSING")
    print("=" * 60)
    
    # 1. Load data
    data_info = load_excel_data()
    
    # 2. Process main sheet (assume first sheet is main data)
    main_sheet_name = data_info['sheet_names'][0]
    main_df = data_info['sheets'][main_sheet_name]
    
    print(f"\nProcessing main sheet: {main_sheet_name}")
    
    # 3. Clean and standardize data
    cleaned_df = clean_and_standardize_data(main_df)
    
    # 4. Analyze data quality
    quality_metrics = analyze_data_quality(cleaned_df)
    
    # 5. Save processed data
    output_path = save_processed_data(cleaned_df)
    
    # 6. Print summary
    print("\n" + "=" * 60)
    print("DATA PREPROCESSING SUMMARY")
    print("=" * 60)
    print(f"Original shape: {data_info['sheets'][main_sheet_name].shape}")
    print(f"Cleaned shape: {cleaned_df.shape}")
    print(f"Output file: {output_path}")
    
    # Print key findings
    if 'duration_min' in cleaned_df.columns:
        duration_stats = quality_metrics['numeric_summary'].get('duration_min', {})
        if duration_stats:
            print(f"\nTarget variable (duration_min) statistics:")
            print(f"  Mean: {duration_stats.get('mean', 'N/A'):.2f} minutes")
            print(f"  Std: {duration_stats.get('std', 'N/A'):.2f} minutes")
            print(f"  Range: {duration_stats.get('min', 'N/A')} - {duration_stats.get('max', 'N/A')} minutes")
    
    return cleaned_df, quality_metrics, output_path

if __name__ == "__main__":
    main()
