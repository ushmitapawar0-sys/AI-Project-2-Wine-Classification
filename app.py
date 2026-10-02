import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib
import os

# Setup Streamlit page configuration
st.set_page_config(page_title="Data Classification Using AI", page_icon="🤖", layout="wide")

# Apply custom styling (clean, professional, beginner friendly)
st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    h1 {color: #2c3e50;}
    h2, h3 {color: #34495e;}
    .stButton>button {background-color: #3498db; color: white; border-radius: 5px; border: none;}
    </style>
""", unsafe_allow_html=True)

st.title("🤖 Data Classification Using AI")
st.markdown("**A complete beginner-friendly but professional ML internship project.**")

# Sidebar for navigation
st.sidebar.title("Navigation")
options = ["1. Dataset Upload & Overview", 
           "2. Data Cleaning & EDA", 
           "3. Model Training & Evaluation", 
           "4. Prediction System"]
choice = st.sidebar.radio("Go to step:", options)

# Helper function to cache data
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

if choice == "1. Dataset Upload & Overview":
    st.header("📂 1. Dataset Upload & Overview")
    
    uploaded_file = st.file_uploader("Upload your CSV dataset", type=['csv'])
    
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        st.session_state['raw_data'] = df
        
        st.subheader("Dataset Overview")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Number of rows:** {df.shape[0]}")
            st.write(f"**Number of columns:** {df.shape[1]}")
        with col2:
            st.write("**Missing Values:**", df.isnull().sum().sum())
            st.write("**Duplicate Rows:**", df.duplicated().sum())
        
        st.subheader("First 5 Rows")
        st.dataframe(df.head())
        
        st.subheader("Column Names & Data Types")
        col_types = pd.DataFrame(df.dtypes, columns=['Data Type']).reset_index().rename(columns={'index': 'Column Name'})
        st.dataframe(col_types, use_container_width=True)
        
        st.subheader("Statistical Summary")
        st.dataframe(df.describe())

elif choice == "2. Data Cleaning & EDA":
    st.header("🧹 2. Data Cleaning & Exploratory Data Analysis (EDA)")
    
    if 'raw_data' not in st.session_state:
        st.warning("Please upload a dataset in Step 1 first.")
    else:
        df = st.session_state['raw_data'].copy()
        
        st.subheader("Data Cleaning")
        clean_df = df.copy()
        
        # Handling Missing Values & Duplicates gracefully
        missing_count = clean_df.isnull().sum().sum()
        dup_count = clean_df.duplicated().sum()
        
        if missing_count > 0:
            st.write(f"Found **{missing_count}** missing values. Imputing numerical columns with median and categorical with mode.")
            for col in clean_df.columns:
                if clean_df[col].isnull().sum() > 0:
                    if clean_df[col].dtype == 'object':
                        clean_df[col].fillna(clean_df[col].mode()[0], inplace=True)
                    else:
                        clean_df[col].fillna(clean_df[col].median(), inplace=True)
        else:
            st.success("No missing values found.")
            
        if dup_count > 0:
            st.write(f"Found **{dup_count}** duplicate rows. Dropping duplicates.")
            clean_df.drop_duplicates(inplace=True)
        else:
            st.success("No duplicate rows found.")
        
        # Encoding categorical variables
        st.write("**Encoding Categorical Columns:** Automatically encoding textual data to numeric using Label Encoding so algorithms can process it.")
        label_encoders = {}
        for col in clean_df.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            clean_df[col] = le.fit_transform(clean_df[col].astype(str))
            label_encoders[col] = le
        st.session_state['label_encoders'] = label_encoders
        
        st.session_state['clean_data'] = clean_df
        st.write("Data cleaning complete. Here is the cleaned data preview:")
        st.dataframe(clean_df.head())
        
        # Target Selection
        st.subheader("Feature & Target Selection")
        target_col = st.selectbox("Select the Target Column for Classification:", clean_df.columns)
        st.session_state['target_col'] = target_col
        
        # Feature detection
        features = [col for col in clean_df.columns if col != target_col]
        st.write(f"**Selected Target:** `{target_col}`")
        st.write(f"**Identified Features ({len(features)}):** {', '.join(features)}")
        
        st.subheader("Exploratory Data Analysis (EDA)")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("### Target Class Distribution")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.countplot(x=target_col, data=clean_df, ax=ax, palette='viridis')
            plt.xticks(rotation=45)
            st.pyplot(fig)
            
        with col2:
            st.write("### Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(6, 5))
            corr = clean_df.corr()
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax, annot_kws={"size": 7})
            st.pyplot(fig)
            
        st.write("### Numerical Feature Distributions (Boxplots)")
        num_cols = clean_df.select_dtypes(include=np.number).columns
        num_cols = [c for c in num_cols if c != target_col]
        
        if len(num_cols) > 0:
            cols = st.columns(3)
            for i, col in enumerate(num_cols[:9]): # Limit to 9 for clean display
                with cols[i % 3]:
                    fig, ax = plt.subplots(figsize=(4, 3))
                    sns.boxplot(y=clean_df[col], ax=ax, color='skyblue')
                    ax.set_title(col)
                    st.pyplot(fig)

elif choice == "3. Model Training & Evaluation":
    st.header("🧠 3. Model Training & Evaluation")
    
    if 'clean_data' not in st.session_state or 'target_col' not in st.session_state:
        st.warning("Please complete Step 2 (Data Cleaning) first.")
    else:
        df = st.session_state['clean_data']
        target_col = st.session_state['target_col']
        
        # Train-test split
        st.subheader("Train-Test Split")
        st.write("Splitting data into 80% for training and 20% for testing using a fixed random state for reproducible results.")
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if len(y.unique()) > 1 else None)
        st.write(f"- Training data shape: **{X_train.shape}**")
        st.write(f"- Testing data shape: **{X_test.shape}**")
        
        # Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        st.session_state['scaler'] = scaler
        
        # Determine if binary or multiclass
        is_multiclass = len(np.unique(y)) > 2
        avg_method = 'weighted' if is_multiclass else 'binary'
        st.info(f"Detected **{'Multiclass' if is_multiclass else 'Binary'}** Classification.")
        
        # Model Selection
        st.subheader("Train Classification Models")
        models_to_train = st.multiselect("Select Models to Train:", 
                                         ["Logistic Regression", "Decision Tree", "K-Nearest Neighbors", "Random Forest", "Support Vector Machine"],
                                         default=["Logistic Regression", "Decision Tree", "K-Nearest Neighbors", "Random Forest", "Support Vector Machine"])
        
        if st.button("Train Selected Models"):
            model_dict = {
                "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "K-Nearest Neighbors": KNeighborsClassifier(),
                "Random Forest": RandomForestClassifier(random_state=42),
                "Support Vector Machine": SVC(probability=True, random_state=42)
            }
            
            trained_models = {}
            results = []
            
            with st.spinner("Training models..."):
                for model_name in models_to_train:
                    model = model_dict[model_name]
                    
                    # Apply feature scaling for models that are distance/gradient based
                    train_X = X_train_scaled if model_name in ["Logistic Regression", "K-Nearest Neighbors", "Support Vector Machine"] else X_train
                    test_X = X_test_scaled if model_name in ["Logistic Regression", "K-Nearest Neighbors", "Support Vector Machine"] else X_test
                    
                    model.fit(train_X, y_train)
                    y_pred = model.predict(test_X)
                    
                    trained_models[model_name] = model
                    
                    acc = accuracy_score(y_test, y_pred)
                    prec = precision_score(y_test, y_pred, average=avg_method, zero_division=0)
                    rec = recall_score(y_test, y_pred, average=avg_method, zero_division=0)
                    f1 = f1_score(y_test, y_pred, average=avg_method, zero_division=0)
                    
                    results.append({
                        "Model": model_name,
                        "Accuracy": acc,
                        "Precision": prec,
                        "Recall": rec,
                        "F1 Score": f1
                    })
                    
                    st.session_state[f'cm_{model_name}'] = confusion_matrix(y_test, y_pred)
                    st.session_state[f'cr_{model_name}'] = classification_report(y_test, y_pred)
            
            st.session_state['trained_models'] = trained_models
            st.session_state['results_df'] = pd.DataFrame(results)
            st.session_state['feature_names'] = X.columns.tolist()
            st.success("All selected models trained and evaluated successfully!")
        
        # Display Results
        if 'results_df' in st.session_state:
            st.subheader("Model Comparison")
            res_df = st.session_state['results_df']
            
            # Format dataframe for display
            styled_df = res_df.style.highlight_max(subset=['Accuracy', 'Precision', 'Recall', 'F1 Score'], color='#a8e6cf', axis=0)
            st.dataframe(styled_df, use_container_width=True)
            
            # Find best model
            best_model_name = res_df.loc[res_df['Accuracy'].idxmax()]['Model']
            st.write(f"🏆 **Recommended Model based on Accuracy:** `{best_model_name}`")
            st.session_state['best_model'] = best_model_name
            
            # Confusion Matrix Viewer
            st.subheader("Model Evaluation & Confusion Matrix")
            st.write("""
            **Understanding the Confusion Matrix:**
            - **True Positive (TP):** The model correctly predicted the positive class.
            - **True Negative (TN):** The model correctly predicted the negative class.
            - **False Positive (FP):** The model incorrectly predicted the positive class (Type I Error).
            - **False Negative (FN):** The model incorrectly predicted the negative class (Type II Error).
            
            *(Note: For multiclass classification, the matrix displays the actual vs. predicted counts across all different classes.)*
            """)
            
            selected_cm_model = st.selectbox("Select model to view Confusion Matrix & Report:", res_df['Model'].tolist())
            cm = st.session_state[f'cm_{selected_cm_model}']
            cr = st.session_state[f'cr_{selected_cm_model}']
            
            col1, col2 = st.columns([1, 1])
            with col1:
                fig, ax = plt.subplots(figsize=(6, 5))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel('Predicted Label')
                ax.set_ylabel('True Label')
                ax.set_title(f'Confusion Matrix: {selected_cm_model}')
                st.pyplot(fig)
            with col2:
                st.write("**Classification Report:**")
                st.text(cr)

            # Option to save model using joblib
            st.subheader("💾 Export Model")
            model_to_save = st.session_state['trained_models'][best_model_name]
            joblib.dump(model_to_save, "best_model.pkl")
            joblib.dump(st.session_state['scaler'], "scaler.pkl")
            st.success(f"Best model '{best_model_name}' and scaler have been saved locally using **Joblib**.")

elif choice == "4. Prediction System":
    st.header("🔮 4. Predict on New Data")
    
    if 'trained_models' not in st.session_state:
        st.warning("Please train models in Step 3 first.")
    else:
        st.write("Use a trained model to make predictions on new or unseen data.")
        
        best_model_name = st.session_state.get('best_model', list(st.session_state['trained_models'].keys())[0])
        selected_model_name = st.selectbox("Select Model for Prediction:", list(st.session_state['trained_models'].keys()), index=list(st.session_state['trained_models'].keys()).index(best_model_name))
        
        model = st.session_state['trained_models'][selected_model_name]
        scaler = st.session_state['scaler']
        features = st.session_state['feature_names']
        target_col = st.session_state['target_col']
        clean_data = st.session_state['clean_data']
        
        st.subheader("Enter Feature Values:")
        
        input_data = {}
        cols = st.columns(3)
        
        for i, feature in enumerate(features):
            with cols[i % 3]:
                # If feature was categorically encoded, let user pick from original strings
                if feature in st.session_state.get('label_encoders', {}):
                    le = st.session_state['label_encoders'][feature]
                    original_classes = le.classes_
                    val = st.selectbox(f"{feature}", original_classes)
                    input_data[feature] = le.transform([val])[0]
                else:
                    # Numerical input
                    min_val = float(clean_data[feature].min())
                    max_val = float(clean_data[feature].max())
                    mean_val = float(clean_data[feature].mean())
                    val = st.number_input(f"{feature}", min_value=min_val, max_value=max_val, value=mean_val)
                    input_data[feature] = val
                    
        if st.button("Predict Result", use_container_width=True):
            input_df = pd.DataFrame([input_data])
            
            # Apply scaling for distance/gradient based models
            if selected_model_name in ["Logistic Regression", "K-Nearest Neighbors", "Support Vector Machine"]:
                input_processed = scaler.transform(input_df)
            else:
                input_processed = input_df
                
            pred = model.predict(input_processed)
            
            # Revert prediction to original label if target was encoded
            final_pred = pred[0]
            if target_col in st.session_state.get('label_encoders', {}):
                le_target = st.session_state['label_encoders'][target_col]
                final_pred = le_target.inverse_transform([pred[0]])[0]
            
            st.markdown(f"### 🎯 Predicted Result: **{final_pred}**")
            
            # Display Prediction Probability if supported
            if hasattr(model, "predict_proba"):
                try:
                    prob = model.predict_proba(input_processed)
                    st.write("**Prediction Probabilities:**")
                    
                    classes = model.classes_
                    if target_col in st.session_state.get('label_encoders', {}):
                        classes = st.session_state['label_encoders'][target_col].inverse_transform(classes)
                        
                    prob_df = pd.DataFrame(prob, columns=classes)
                    st.dataframe(prob_df.style.format("{:.2%}"))
                except Exception as e:
                    pass
