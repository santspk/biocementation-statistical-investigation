# ---------------------------------------------
# Split dataset into Training (80%) and Testing (20%)
# Randomly selected and saved separately
# ---------------------------------------------
import pandas as pd
from sklearn.model_selection import train_test_split

# ========= USER INPUTS =========
excel_path = "Biocementation_Data_Analysis_Cleaned.xlsx"  # input dataset
train_save_path = "Biocementation_Data_Train.xlsx"
test_save_path = "Biocementation_Data_Test.xlsx"
test_size = 0.20   # 20% for testing
random_seed = 42   # for reproducibility
# ===============================

# Load dataset
df = pd.read_excel(excel_path)

# Split into train and test
train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_seed, shuffle=True)

# Save to Excel
train_df.to_excel(train_save_path, index=False)
test_df.to_excel(test_save_path, index=False)

# Summary output
print("==== Dataset Split Summary ====")
print(f"Total samples     : {len(df)}")
print(f"Training samples  : {len(train_df)} ({(1 - test_size) * 100:.0f}%)")
print(f"Testing samples   : {len(test_df)} ({test_size * 100:.0f}%)")
print(f"Random seed used  : {random_seed}")
print(f"Saved training set: {train_save_path}")
print(f"Saved testing set : {test_save_path}")
