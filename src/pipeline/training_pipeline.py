import pandas as pd
import numpy as np
import pickle

# Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, mean_absolute_error, classification_report, mean_squared_error, accuracy_score 

# Deep Learning imports (TensorFlow/Keras)
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, concatenate, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

# Constants (paths, model parameters)
PROCESSED_DATA_PATH = "data/processed/processed_patient_behavior_data.csv" 
# Assuming EDA notebook saves its output here and Dockerfile copies `data/`
# If using raw data and script does all preprocessing, change to "data/raw/patient_behavior_data.csv"

# For text processing
MAX_WORDS = 10000 # Max number of words in tokenizer
MAX_SEQ_LENGTH = 200 # Max length of sequences for padding
EMBEDDING_DIM = 100

# For model comparison output
COMPARISON_RESULTS_FILE = "model_comparison_results.txt"

# --- 1. Load Data ---
def load_data(csv_path):
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print("Data loaded successfully.")
    df.info()
    return df

# --- 2. Feature Preparation ---
def prepare_tabular_features(df_input, target_column):
    print(f"\nPreparing tabular features for target: '{target_column}'...")
    df = df_input.copy()

    y = df[target_column]
    # Shift labels if they are in range like -2 to 2 for classification
    if y.min() < 0 and (target_column == 'concentration' or target_column == 'impulsivity' or target_column == 'mood'): # Add other behavioral targets if they are used as classification target
        print(f"Shifting target '{target_column}' by adding 2 to make labels non-negative for classification.")
        y = y + 2
    
    behavioral_targets = ['concentration', 'impulsivity', 'mood', 'sleep', 'appetite', 'distractibility', 'hyperactivity']
    # potential_other_targets = [col for col in behavioral_targets if col != target_column] # This logic might be too aggressive if some behavioral are features
   
    # Adjust based on actual features used in notebook's X_tab_rf
    numerical_features = ['bmi', 'weight', 'height', 'systolic', 'diastolic', 'is_medicated', 'dose_mg'] 
    categorical_features = ['gender', 'medication', 'bmi_category', 'bp_category']
    
    # Ensure features exist in df
    numerical_features = [col for col in numerical_features if col in df.columns]
    categorical_features = [col for col in categorical_features if col in df.columns]

    X_tabular_selected = df[numerical_features + categorical_features]
    
    print(f"  Selected numerical features: {numerical_features}")
    print(f"  Selected categorical features: {categorical_features}")

    transformers_list = []
    if numerical_features:
        numerical_transformer = StandardScaler()
        transformers_list.append(('num', numerical_transformer, numerical_features))
    if categorical_features:
        categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=True) # sparse=True generally ok for RF
        transformers_list.append(('cat', categorical_transformer, categorical_features))

    preprocessor = ColumnTransformer(
        transformers=transformers_list,
        remainder='passthrough' 
    )
    print("Tabular preprocessor created.")
    return X_tabular_selected, y, preprocessor

def prepare_text_features(df_input, text_column_name, target_column_name=None):
    print(f"\nPreparing text features from '{text_column_name}'...")
    df = df_input.copy()
    texts = df[text_column_name].astype(str).values # Ensure string type
    
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<unk>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    X_text = pad_sequences(sequences, maxlen=MAX_SEQ_LENGTH, padding='post', truncating='post')
    
    print(f"  Vocabulary size: {len(tokenizer.word_index)}")
    print(f"  Shape of padded sequences: {X_text.shape}")

    y_text = None
    if target_column_name and target_column_name in df.columns:
        y_text = df[target_column_name].values
        # Shift labels if they are in range like -2 to 2 for classification
        if y_text.min() < 0 and (target_column_name == 'concentration' or target_column_name == 'impulsivity' or target_column_name == 'mood'): # Add other behavioral targets
            print(f"Shifting text target '{target_column_name}' by adding 2 to make labels non-negative for classification.")
            y_text = y_text + 2
    
    # Save tokenizer for potential use in CLSTM if inputs need separate tokenization
    # with open('tokenizer.pkl', 'wb') as handle:
    #     pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # print("Tokenizer saved to tokenizer.pkl")

    return X_text, y_text, tokenizer # Return tokenizer if needed later

# --- 3. Random Forest Model ---
def train_random_forest(X_tabular, y_tabular, preprocessor):
    print("\n--- Training Random Forest for 'concentration' ---")
    X_train, X_test, y_train, y_test = train_test_split(X_tabular, y_tabular, test_size=0.2, random_state=42)

    # Assuming 'concentration' is categorical (0, 1, 2 etc.)
    # If it's regression, use RandomForestRegressor and different metrics
    model_rf = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))]) # Added class_weight
    
    print("Training RF model...")
    model_rf.fit(X_train, y_train)
    
    print("Evaluating RF model...")
    y_pred = model_rf.predict(X_test)
    
    # For classification:
    f1 = f1_score(y_test, y_pred, average='weighted') # Use 'weighted' or 'macro' for multi-class
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print(f"Random Forest - F1 Score (weighted): {f1:.4f}")
    print(f"Random Forest - Accuracy: {acc:.4f}")
    print("Classification Report:\n", report)
    
    # If 'concentration' were regression target:
    # mae = mean_absolute_error(y_test, y_pred)
    # mse = mean_squared_error(y_test, y_pred)
    # r2 = r2_score(y_test, y_pred)
    # print(f"Random Forest - MAE: {mae:.4f}, MSE: {mse:.4f}, R2: {r2:.4f}")
    
    # For the comparison table, we need specific metrics. Assuming classification for 'concentration'.
    return {"model": "Random Forest (concentration)", "F1_weighted": f1, "Accuracy": acc}


# --- 4. LSTM Model ---
def train_lstm(X_text_lstm, y_lstm, tokenizer):
    print("\n--- Training LSTM for 'impulsivity' from doctor_notes ---")
    X_train, X_test, y_train, y_test = train_test_split(X_text_lstm, y_lstm, test_size=0.2, random_state=42)

    # Assuming 'impulsivity' is also categorical (e.g., 0, 1, 2 ratings)
    # If regression, change loss, activation, and metrics
    num_classes_impulsivity = len(np.unique(y_lstm))

    input_layer = Input(shape=(MAX_SEQ_LENGTH,))
    embedding_layer = Embedding(input_dim=min(MAX_WORDS, len(tokenizer.word_index) + 1), 
                                output_dim=EMBEDDING_DIM, 
                                input_length=MAX_SEQ_LENGTH)(input_layer)
    lstm_layer = LSTM(64, dropout=0.2, recurrent_dropout=0.2)(embedding_layer)
    dense_layer = Dense(32, activation='relu')(lstm_layer)
    output_layer = Dense(num_classes_impulsivity, activation='softmax')(dense_layer) # Softmax for multi-class

    model_lstm = Model(inputs=input_layer, outputs=output_layer)
    model_lstm.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) # sparse_categorical for integer targets
    
    model_lstm.summary()
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    print("Training LSTM model...")
    model_lstm.fit(X_train, y_train, 
                   epochs=10, # Keep epochs low for quick Docker runs, or use early stopping effectively
                   batch_size=32, 
                   validation_data=(X_test, y_test),
                   callbacks=[early_stopping],
                   verbose=1) # Set to 1 or 2 for logs in Docker

    print("Evaluating LSTM model...")
    loss, acc = model_lstm.evaluate(X_test, y_test, verbose=0)
    
    # For classification 'impulsivity':
    y_pred_probs = model_lstm.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"LSTM - Loss: {loss:.4f}, Accuracy: {acc:.4f}, F1 Score (weighted): {f1:.4f}")
    print("Classification Report (Impulsivity):\n", classification_report(y_test, y_pred, zero_division=0))
    
    return {"model": "LSTM (impulsivity from notes)", "F1_weighted": f1, "Accuracy": acc, "Loss": loss}

# --- 5. CLSTM Model ---
def train_clstm(X_tabular_clstm, X_text_clstm, y_clstm, preprocessor_clstm, tokenizer_clstm, target_name="concentration"):
    print(f"\n--- Training CLSTM for '{target_name}' from combined features ---")
    
    # Split data first (tabular and text must be aligned)
    X_tab_train, X_tab_test, X_text_train, X_text_test, y_train, y_test = train_test_split(
        X_tabular_clstm, X_text_clstm, y_clstm, test_size=0.2, random_state=42
    )

    # Preprocess tabular data *after* split to prevent data leakage from test to train through scaler/encoder
    print("Preprocessing CLSTM tabular data (fitting on train split)...")
    X_tab_train_processed = preprocessor_clstm.fit_transform(X_tab_train)
    X_tab_test_processed = preprocessor_clstm.transform(X_tab_test)
    
    # If preprocessor outputs sparse matrix and dense is needed by Keras:
    if hasattr(X_tab_train_processed, "toarray"):
        X_tab_train_processed = X_tab_train_processed.toarray()
        X_tab_test_processed = X_tab_test_processed.toarray()

    num_classes_clstm = len(np.unique(y_clstm)) # Assuming categorical target

    # Tabular input branch
    input_tabular = Input(shape=(X_tab_train_processed.shape[1],), name='tabular_input')
    dense_tabular = Dense(32, activation='relu')(input_tabular)
    # dropout_tabular = Dropout(0.3)(dense_tabular) # Optional

    # Text input branch (CNN + LSTM)
    input_text = Input(shape=(MAX_SEQ_LENGTH,), name='text_input')
    embedding_text = Embedding(input_dim=min(MAX_WORDS, len(tokenizer_clstm.word_index) + 1), 
                               output_dim=EMBEDDING_DIM, 
                               input_length=MAX_SEQ_LENGTH)(input_text)
    conv_text = Conv1D(filters=64, kernel_size=5, activation='relu')(embedding_text)
    pool_text = GlobalMaxPooling1D()(conv_text) # Or MaxPooling1D
    # lstm_text = LSTM(64)(embedding_text) # Simpler alternative to CNN+Pool if preferred
    # For CLSTM, usually CNN captures local features, then LSTM on top, or just a strong text encoder
    # For now, let's use the CNN -> Pool -> (optional LSTM on CNN output) -> Dense
    # Or simply CNN -> Pool as a feature extractor
    dense_text_features = Dense(32, activation='relu')(pool_text) 


    # Concatenate branches
    concatenated = concatenate([dense_tabular, dense_text_features]) # or [dense_tabular, lstm_text]
    
    dropout_all = Dropout(0.5)(concatenated)
    output_combined = Dense(num_classes_clstm, activation='softmax', name='output')(dropout_all) # Softmax for multi-class

    model_clstm = Model(inputs=[input_tabular, input_text], outputs=output_combined)
    model_clstm.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    model_clstm.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True) # Increased patience
    
    print("Training CLSTM model...")
    model_clstm.fit([X_tab_train_processed, X_text_train], y_train,
                    epochs=15, # Keep epochs reasonable for Docker run
                    batch_size=32,
                    validation_data=([X_tab_test_processed, X_text_test], y_test),
                    callbacks=[early_stopping],
                    verbose=1)

    print("Evaluating CLSTM model...")
    loss, acc = model_clstm.evaluate([X_tab_test_processed, X_text_test], y_test, verbose=0)
    
    y_pred_probs = model_clstm.predict([X_tab_test_processed, X_text_test])
    y_pred = np.argmax(y_pred_probs, axis=1)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"CLSTM ({target_name}) - Loss: {loss:.4f}, Accuracy: {acc:.4f}, F1 Score (weighted): {f1:.4f}")
    print(f"Classification Report (CLSTM - {target_name}):\n", classification_report(y_test, y_pred, zero_division=0))

    return {"model": f"CLSTM ({target_name} from combined)", "F1_weighted": f1, "Accuracy": acc, "Loss": loss}


# --- 6. Main Orchestration & Model Comparison ---
def main():
    print("=== Starting Model Training Pipeline ===")
    df_full = load_data(PROCESSED_DATA_PATH)

    all_metrics = []

    # --- Random Forest for 'concentration' ---
    # Ensure 'concentration' is in the df and is suitable for classification (e.g. int type)
    if 'concentration' in df_full.columns:
        X_tab_rf, y_rf, preprocessor_rf = prepare_tabular_features(df_full, target_column='concentration')
        if X_tab_rf is not None and y_rf is not None and preprocessor_rf is not None:
            # y_rf is already shifted by prepare_tabular_features
            rf_metrics = train_random_forest(X_tab_rf, y_rf, preprocessor_rf)
            all_metrics.append(rf_metrics)
    else:
        print("Skipping Random Forest: 'concentration' column not found.")

    # --- LSTM for 'impulsivity' using 'processed_notes' (or 'doctor_notes') ---
    # Using 'processed_notes' if available from EDA, otherwise 'doctor_notes'
    text_col_for_lstm = 'processed_notes' if 'processed_notes' in df_full.columns else 'doctor_notes'
    if 'impulsivity' in df_full.columns and text_col_for_lstm in df_full.columns:
        print(f"Attempting to prepare LSTM features with text_col: {text_col_for_lstm} and target: impulsivity")
        X_text_lstm, y_lstm, tokenizer_lstm = prepare_text_features(df_full, text_column_name=text_col_for_lstm, target_column_name='impulsivity')
        print(f"LSTM features prepared. X_text_lstm is None: {X_text_lstm is None}. y_lstm is None: {y_lstm is None}")
        if y_lstm is not None:
            print(f"y_lstm head after preparation (first 5): {y_lstm[:5]}")

        if X_text_lstm is not None and y_lstm is not None:
             # Ensure y_lstm is integer type for sparse_categorical_crossentropy
             # y_lstm is already shifted by prepare_text_features
            y_lstm = y_lstm.astype(int)
            print(f"y_lstm head after astype(int) (first 5): {y_lstm[:5]}")
            lstm_metrics = train_lstm(X_text_lstm, y_lstm, tokenizer_lstm)
            all_metrics.append(lstm_metrics)
    else:
        print(f"Skipping LSTM: 'impulsivity' or '{text_col_for_lstm}' column not found.")

    # --- CLSTM for 'concentration'  using combined features ---
    clstm_target = 'concentration' 
    text_col_for_clstm = 'processed_notes' if 'processed_notes' in df_full.columns else 'doctor_notes'

    if clstm_target in df_full.columns and text_col_for_clstm in df_full.columns:
        # Prepare tabular features for CLSTM target
        X_tab_clstm, y_clstm_tabular_part, preprocessor_clstm = prepare_tabular_features(df_full, target_column=clstm_target)
        
        # Prepare text features for CLSTM (y is not strictly needed here if y_clstm_tabular_part is the sole target)
        X_text_clstm, _, tokenizer_clstm = prepare_text_features(df_full, text_column_name=text_col_for_clstm) # Use same tokenizer or retrain? For consistency, using same for now.

        if X_tab_clstm is not None and y_clstm_tabular_part is not None and preprocessor_clstm is not None and X_text_clstm is not None:
            # Align X_tab_clstm and X_text_clstm by index if they were derived from differently processed DFs (should be okay if both from df_full)
            # Ensure target y_clstm_tabular_part is integer for sparse_categorical_crossentropy
            # y_clstm_tabular_part is already shifted by prepare_tabular_features
            y_clstm_tabular_part = y_clstm_tabular_part.astype(int)
            clstm_metrics = train_clstm(X_tab_clstm, X_text_clstm, y_clstm_tabular_part, preprocessor_clstm, tokenizer_clstm, target_name=clstm_target)
            all_metrics.append(clstm_metrics)
    else:
        print(f"Skipping CLSTM: '{clstm_target}' or '{text_col_for_clstm}' column not found.")

    # --- Print Model Comparison Table ---
    print("\n\n=== Model Comparison Summary ===")
    if all_metrics:
        comparison_df = pd.DataFrame(all_metrics)
        print(comparison_df.to_string()) # Print full DF to console
        
        # Save to a text file for the Docker logs deliverable
        with open(COMPARISON_RESULTS_FILE, "w") as f:
            f.write("Model Comparison Summary:\n")
            f.write(comparison_df.to_string())
        print(f"\nComparison results also saved to {COMPARISON_RESULTS_FILE}")
    else:
        print("No models were trained, so no comparison available.")
        
    print("\n=== Model Training Pipeline Finished ===")

if __name__ == '__main__':
    main() 