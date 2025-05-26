import pandas as pd
import numpy as np
import hashlib
import spacy
import re 

# Load scispaCy model 
# pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_md-0.5.4.tar.gz
SCISPACY_MODEL = 'en_core_sci_md' 
nlp_scispaCy = spacy.load(SCISPACY_MODEL)
print(f"Successfully loaded scispaCy model for NER: {SCISPACY_MODEL}")

# --- Function to hash PII columns ---
def hash_pii_columns(df, columns_to_hash):
    """Hashes specified columns in a DataFrame using SHA-256."""
    print(f"\nHashing PII columns: {columns_to_hash}...")
    df_anonymized = df.copy()
    for column in columns_to_hash:
        if column in df_anonymized.columns:
            # Ensure data is string and encode to bytes before hashing
            df_anonymized[column] = df_anonymized[column].astype(str).apply(
                lambda x: hashlib.sha256(x.encode()).hexdigest()
            )
            print(f"  Column '{column}' hashed.")
        else:
            print(f"  Warning: Column '{column}' not found for hashing.")
    return df_anonymized

# --- Function to redact PII in text using spaCy NER ---
def redact_pii_in_text(text, nlp_pipeline):
    """
    Redacts PII (names, dates, locations) in a text string.
    Returns the redacted text.
    """
    if nlp_pipeline is None or not text or pd.isna(text):
        return text

    doc = nlp_pipeline(str(text))
    redacted_text = str(text) # Start with the original text

    # Iterate over entities in reverse to avoid index shifting issues during replacement
    for ent in reversed(doc.ents):
        # redact Names (PERSON), Dates (DATE), Locations (LOC, GPE)
        if ent.label_ in ['PERSON', 'DATE', 'LOC', 'GPE', 'FAC', 'NORP', 'ORG']: # Added ORG, FAC, NORP as often PII-related
            # Replace entity text with [REDACTED]
            redacted_text = redacted_text[:ent.start_char] + '[REDACTED]' + redacted_text[ent.end_char:]
            
    # Add regex for emails, phone numbers if not caught by NER
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    phone_pattern = r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
    redacted_text = re.sub(email_pattern, '[REDACTED_EMAIL]', redacted_text)
    redacted_text = re.sub(phone_pattern, '[REDACTED_PHONE]', redacted_text)
            
    return redacted_text

# --- Main Anonymization Function ---
def anonymize_patient_data(df_input, pii_columns_to_hash, text_column_to_redact, nlp_pipeline_for_redaction):
    """
    Applies PII hashing and text redaction to the DataFrame.
    """
    print("Starting data anonymization process...")
    df_processed = df_input.copy()

    # Mask PII: hash specified columns [cite: 11]
    df_processed = hash_pii_columns(df_processed, pii_columns_to_hash)

    # Redact PII in doctor_notes using spaCy NER [cite: 12]
    if text_column_to_redact in df_processed.columns and nlp_pipeline_for_redaction is not None:
        print(f"\nRedacting PII in '{text_column_to_redact}' column...")
        df_processed[text_column_to_redact + '_redacted'] = df_processed[text_column_to_redact].apply(
            lambda x: redact_pii_in_text(x, nlp_pipeline_for_redaction)
        )
        print(f"  PII redaction in '{text_column_to_redact}' complete. New column: '{text_column_to_redact}_redacted'.")
        
        df_processed[text_column_to_redact] = df_processed[text_column_to_redact + '_redacted']
        df_processed.drop(columns=[text_column_to_redact + '_redacted'], inplace=True)

    # elif nlp_pipeline_for_redaction is None:
    #     print(f"  Skipping PII redaction in '{text_column_to_redact}' as scispaCy model was not loaded.")
    else:
        print(f"  Warning: Text column '{text_column_to_redact}' not found for PII redaction.")
        
    return df_processed

# --- Main Execution ---
if __name__ == '__main__':
    # Load the data
    df = pd.read_csv('data/patient_behavior_data.csv', delimiter=';')
    print("\nOriginal DataFrame (first 3 rows):")
    print(df.head(3))

    # Define PII columns to hash and text column for redaction
    pii_to_hash = ['patient_id', 'name', 'surname']
    notes_column = 'doctor_notes'

    # Perform anonymization
    anonymized_df = anonymize_patient_data(df, 
                                           pii_columns_to_hash=pii_to_hash, 
                                           text_column_to_redact=notes_column,
                                           nlp_pipeline_for_redaction=nlp_scispaCy)

    print("\nAnonymized DataFrame (first 3 rows with redacted notes):")
    print(anonymized_df[['patient_id', 'name', 'surname', notes_column + '_redacted' if notes_column + '_redacted' in anonymized_df.columns else notes_column]].head(3))

    # Export anonymized_patients.csv
    output_csv_path = 'data/anonymized_patients.csv'
    try:
        # Select columns for export - exclude original notes if redacted version exists and is preferred
        cols_to_export = anonymized_df.columns.tolist()
        if notes_column + '_redacted' in anonymized_df.columns and notes_column in cols_to_export:
            anonymized_df_export = anonymized_df.drop(columns=[notes_column])
            anonymized_df_export.rename(columns={notes_column + '_redacted': notes_column}, inplace=True)

        anonymized_df.to_csv(output_csv_path, index=False, sep=';')
        print(f"\nSuccessfully exported anonymized data to: {output_csv_path}")
    except Exception as e:
        print(f"\nError exporting anonymized data: {e}")