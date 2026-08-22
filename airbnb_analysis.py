# ======================================================================================================================
# 1. IMPORT LIBRARIES
# ======================================================================================================================

import pandas as pd

# ======================================================================================================================
# 2. LOAD RAW DATA
# ======================================================================================================================

# Load the original Airbnb dataset into a Pandas DataFrame
df = pd.read_csv(
    r"C:\Users\prajw\OneDrive\Documents\Airbnb_project\airbnb_raw_data.csv"
)

# ======================================================================================================================
# 3. DATA UNDERSTANDING
# ======================================================================================================================

# -------------------- Basic Dataset Information --------------------

# Check the number of rows and columns
print("Shape:", df.shape)

# Display all column names
print("\nColumns:")
print(df.columns.tolist())


# -------------------- Data Types & Missing Values --------------------

# Check the data type of every column
print("\nData Types:")
print(df.dtypes)

# Check the number of missing values in each column
print("\nMissing Values:")
print(df.isnull().sum())

# Check whether the dataset contains duplicate rows
print("\nDuplicate Rows:", df.duplicated().sum())


# -------------------- Missing Value Investigation --------------------

# Display listings where price is missing
print("\nRows with missing price:")
print(df[df["price"].isna()][
    ["id", "name", "neighbourhood", "room_type", "price"]
])

# Display the listing where minimum_nights is missing
print("\nRows with missing minimum_nights:")
print(df[df["minimum_nights"].isna()][
    ["id", "name", "room_type", "price", "minimum_nights"]
])

# Check listings where last_review is missing
print("\nRows with missing last_review:")
print(df[df["last_review"].isna()][
    ["id", "number_of_reviews", "last_review", "reviews_per_month"]
])


# -------------------- Sample Data --------------------

# Display the first 5 rows to understand the structure of the data
print("\nFirst 5 rows:")
print(df.head())


# -------------------- Numerical Statistics --------------------

# Generate descriptive statistics for numerical columns
# Includes count, mean, standard deviation, minimum,
# quartiles (25%, 50%, 75%), and maximum
print("\nBasic Statistics:")
print(df.describe())


# -------------------- Categorical Data Analysis --------------------

# Count the number of listings for each room type
print("\nRoom Types:")
print(df["room_type"].value_counts())

# Display the 10 neighbourhoods with the highest number of listings
print("\nTop 10 Neighbourhoods:")
print(df["neighbourhood"].value_counts().head(10))

# Count the number of unique hosts in the dataset
print("\nUnique Hosts:")
print(df["host_name"].nunique())


# ============================================================================================================================
# 4. DATA CLEANING
# =============================================================================================================================

# Cleaning steps will be added here after completing
# the initial data understanding and quality checks.
df = df.drop(columns=["neighbourhood_group", "license"])

# Check the shape after removing the empty columns
print("\nShape after removing empty columns:", df.shape)

# -------------------- Investigate Missing Price --------------------

# Count the number of listings with missing prices
missing_price = df["price"].isna().sum()
print("\nMissing price values:", missing_price)

# Display the listing that has both missing price and minimum nights
print("\nListing with missing price and minimum nights:")
print(df[
    df["price"].isna() & df["minimum_nights"].isna()
][[
    "id",
    "name",
    "room_type",
    "price",
    "minimum_nights"
]]
)

# -------------------- Handle Missing Minimum Nights --------------------

# Remove the single listing where both price and minimum_nights are missing
# These are important fields for our Airbnb analysis, so we cannot reliably
# use this listing for the main analysis
df = df.dropna(subset=["minimum_nights"])
print("\nShape after removing missing minimum_nights:", df.shape)

# -------------------- Check ID Columns --------------------

# Check whether ID columns contain any decimal values
# IDs should normally be whole numbers
print("\nID columns with decimal values:")
print(
    df[df["id"] % 1 != 0]["id"].head()
)
print(
    df[df["host_id"] % 1 != 0]["host_id"].head()
)
print(
    df[df["host_profile_id"] % 1 != 0]["host_profile_id"].head()
)

#-------------------- Convert ID Columns --------------------

#Convert ID columns from float to integer
#IDs are identifiers, so whole-number format is more appropriate
df["id"] = df["id"].astype("int64")
df["host_id"] = df["host_id"].astype("int64")
df["host_profile_id"] = df["host_profile_id"].astype("int64")

# Check the updated data types
print("\nUpdated ID data types:")
print(df[["id", "host_id", "host_profile_id"]].dtypes)

# -------------------- Check Numerical Values --------------------

# Check the minimum and maximum values of important Airbnb fields
# This helps identify impossible or suspicious values before cleaning

print("\nPrice range:")
print("Minimum:", df["price"].min())
print("Maximum:", df["price"].max())

print("\nMinimum nights range:")
print("Minimum:", df["minimum_nights"].min())
print("Maximum:", df["minimum_nights"].max())

print("\nAvailability range:")
print("Minimum:", df["availability_365"].min())
print("Maximum:", df["availability_365"].max())

print("\nNumber of reviews range:")
print("Minimum:", df["number_of_reviews"].min())
print("Maximum:", df["number_of_reviews"].max())

# -------------------- Check Text Consistency --------------------

# Check for leading or trailing spaces in important text columns
# Extra spaces can create duplicate categories during analysis

text_columns = [
    "name",
    "host_name",
    "neighbourhood",
    "room_type"
]
for column in text_columns:
    spaces = df[column].astype(str).str.strip().ne(df[column].astype(str))
    print(f"\n{column} - values with extra spaces:", spaces.sum())

# -------------------- Convert Review Date --------------------

# Convert last_review from text to datetime format
# The dataset uses day-month-year format
# Missing or invalid dates are converted to NaT
df["last_review"] = pd.to_datetime(
    df["last_review"],
    dayfirst=True,
    errors="coerce"
)
# Check the updated data type
print("\nUpdated last_review data type:")
print(df["last_review"].dtype)

# -------------------- Final Missing Value Check --------------------

# Count remaining missing values in every column
# This helps us confirm what is still missing after cleaning
print("\nRemaining Missing Values:")
print(df.isnull().sum())


# -------------------- Final Data Type Corrections --------------------

# Convert minimum_nights to integer because it represents a whole number
df["minimum_nights"] = df["minimum_nights"].astype("int64")

# -------------------- Convert Review Date --------------------

# Convert last_review from text to datetime format
# The dataset uses day-month-year format
# Missing or invalid dates are converted to NaT
df["last_review"] = pd.to_datetime(
    df["last_review"],
    dayfirst=True,
    errors="coerce"
)


# -------------------- Final Data Type Corrections --------------------

# Convert minimum_nights to integer because it represents whole nights
df["minimum_nights"] = df["minimum_nights"].astype("int64")


# -------------------- Final Missing Value Check --------------------

# Count remaining missing values after cleaning
print("\nRemaining Missing Values:")
print(df.isnull().sum())


# =====================================================================================================================
# 5. VALIDATION
# =====================================================================================================================

# -------------------- Dataset Shape --------------------

# Confirm the final number of rows and columns
print("\nFinal Shape:")
print(df.shape)

# -------------------- Duplicate Check --------------------

# Confirm that no duplicate rows remain
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# -------------------- Data Type Check --------------------

# Confirm the final data types
print("\nFinal Data Types:")
print(df.dtypes)

# -------------------- Range Validation --------------------

# Confirm important numerical columns contain valid ranges
print("\nAvailability Range:")
print(df["availability_365"].min(), "to", df["availability_365"].max())

print("\nMinimum Nights Range:")
print(df["minimum_nights"].min(), "to", df["minimum_nights"].max())

print("\nPrice Range:")
print(df["price"].min(), "to", df["price"].max())


# ============================================================
# 6. SAVE CLEANED DATA
# =====================================================================================================================

# Save the cleaned dataset as a new CSV file
# The original raw dataset remains unchanged
output_path = (
    r"C:\\Users\\prajw\\OneDrive\\Documents\\Airbnb_project\\airbnb_cleaned_data.csv"
)

df.to_csv(output_path, index=False)

print("\nCleaned dataset saved successfully.")
print("File:", output_path)


# =====================================================================================================================
# 7. FINAL FILE CHECK
# =====================================================================================================================

# Load the saved cleaned CSV again to verify the exported file
cleaned_check = pd.read_csv(r"C:\\Users\\prajw\\OneDrive\\Documents\\Airbnb_project\\airbnb_cleaned_data.csv")

# Confirm the saved file has the expected number of rows and columns
print("\nSaved File Shape:")
print(cleaned_check.shape)

# Confirm that the saved file can be read successfully
print("\nSaved File Loaded Successfully:", not cleaned_check.empty)


print(cleaned_check["id"].head(20))
print(cleaned_check["id"].dtype)

print(cleaned_check["id"].min())
print(cleaned_check["id"].max())


