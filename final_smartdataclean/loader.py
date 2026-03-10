import pandas as pd

def load_file(uploaded_file):
    """
    Robust file loader for CSV / Excel
    Handles empty files safely
    """

    try:
        # Detect file type
        filename = uploaded_file.name.lower()

        if filename.endswith(".csv"):
            df = pd.read_csv(uploaded_file)

        elif filename.endswith(".xlsx") or filename.endswith(".xls"):
            df = pd.read_excel(uploaded_file)

        else:
            raise ValueError("Unsupported file format")

        # Safety check
        if df.empty or df.shape[1] == 0:
            raise ValueError("Uploaded file has no usable columns")

        return df

    except pd.errors.EmptyDataError:
        raise ValueError("The uploaded file is empty. Please upload a valid dataset.")

    except Exception as e:
        raise ValueError(f"Failed to load file: {str(e)}")



'''import pandas as pd

def load_file(uploaded_file):
    if uploaded_file.name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".xlsx"):
        return pd.read_excel(uploaded_file)
    else:
        raise ValueError("Unsupported file format")'''
