import pandas as pd
import os

# 1. Load the file
file_path = './data/phishing_data.csv'
print(f"Reading {file_path}...")

try:
    # Try reading with different encodings (emails often have weird characters)
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding='latin-1')

    print(f"Original Columns: {list(df.columns)}")

    # 2. Rename Columns automatically
    # We look for common names and rename them to 'text' and 'label'
    df.columns = [c.lower() for c in df.columns] # Make all lowercase
    
    # Map common text column names
    if 'text' not in df.columns:
        if 'sms' in df.columns: df = df.rename(columns={'sms': 'text'})
        elif 'content' in df.columns: df = df.rename(columns={'content': 'text'})
        elif 'body' in df.columns: df = df.rename(columns={'body': 'text'})
        elif 'message' in df.columns: df = df.rename(columns={'message': 'text'})
    
    # Map common label column names
    if 'label' not in df.columns:
        if 'label_num' in df.columns: df = df.rename(columns={'label_num': 'label'})
        elif 'class' in df.columns: df = df.rename(columns={'class': 'label'})
        elif 'type' in df.columns: df = df.rename(columns={'type': 'label'})

    # 3. Fix Label Values (Must be 0 or 1)
    # If labels are 'ham'/'spam' or 'safe'/'phishing', convert them.
    if df['label'].dtype == 'object':
        print("Detected text labels. Converting to numbers...")
        # Dictionary of common terms
        mapping = {
            'ham': 0, 'safe': 0, 'legit': 0, 'innocent': 0,
            'spam': 1, 'phishing': 1, 'malicious': 1, 'bad': 1
        }
        df['label'] = df['label'].map(mapping)

    # 4. Final Clean
    df = df[['text', 'label']] # Keep only what we need
    df = df.dropna() # Remove empty rows
    
    # 5. Save Over the old file
    df.to_csv(file_path, index=False)
    
    print("------------------------------------------------")
    print(f"SUCCESS! Data cleaned.")
    print(f"Total Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"First row example: {df.iloc[0].to_dict()}")
    print("You are ready to run train.py now.")

except Exception as e:
    print(f"ERROR: {e}")
    print("Please check your CSV file manually.")