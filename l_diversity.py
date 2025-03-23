import pandas as pd
import numpy as np
import anjana.anonymity as anonymity

# Example utility function to generate age intervals.
def generate_intervals(values, min_val, max_val, interval):
    intervals = []
    for v in values:
        lower_bound = (v // interval) * interval
        upper_bound = lower_bound + interval - 1
        intervals.append(f"{lower_bound}-{upper_bound}")
    return np.array(intervals)

# Sample DataFrame with an identifier column, quasi-identifiers, and sensitive attributes.
df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
    'ZipCode': ['12345', '12346', '12347', '12348', '12349'],
    'Age': [23, 34, 45, 23, 34],
    'Gender': ['F', 'M', 'M', 'F', 'M'],
    'Occupation': ['Engineer', 'Doctor', 'Lawyer', 'Artist', 'Teacher']
})

# Define identifiers, quasi-identifiers, and sensitive attribute.
# Here, 'Name' is an identifier, 'ZipCode' and 'Age' are quasi-identifiers,
# and 'Occupation' is chosen as the sensitive attribute (to which l-diversity will be applied).
ident = ['Name']
quasi_ident = ['ZipCode', 'Age']
sens_att = 'Occupation'

# Build hierarchies dictionary.
hierarchies = {}

# --- Hierarchy for ZipCode (or zipcode) ---
zipcode_field = None
if 'ZipCode' in quasi_ident and 'ZipCode' in df.columns:
    zipcode_field = 'ZipCode'
elif 'zipcode' in quasi_ident and 'zipcode' in df.columns:
    zipcode_field = 'zipcode'
if zipcode_field:
    hierarchies[zipcode_field] = {
        0: df[zipcode_field].values,                                        # Original values
        1: df[zipcode_field].astype(str).str[:-1] + "*",                     # Remove last digit
        2: df[zipcode_field].astype(str).str[:-2] + "**",                    # Remove last two digits
        3: df[zipcode_field].astype(str).str[:-3] + "***",                   # Remove last three digits
        4: np.array(["*"] * len(df[zipcode_field].values))                   # Full suppression as last resort
    }

# --- Hierarchy for Age (or age) ---
age_field = None
if 'Age' in quasi_ident and 'Age' in df.columns:
    age_field = 'Age'
elif 'age' in quasi_ident and 'age' in df.columns:
    age_field = 'age'
if age_field:
    hierarchies[age_field] = {
        0: df[age_field].values,                                            # Original ages
        1: generate_intervals(df[age_field].values, 0, 100, 5),              # 5-year intervals
        2: generate_intervals(df[age_field].values, 0, 100, 10),             # 10-year intervals
    }

# --- Hierarchies for sensitive attributes ---
# For Gender:
gender_field = None
if 'Gender' in df.columns and ('Gender' in sens_att or 'gender' in sens_att):
    gender_field = 'Gender'
if gender_field:
    hierarchies[gender_field] = {
        0: df[gender_field].values,                                         # Original gender values
        1: np.array(["*"] * len(df[gender_field].values))                   # Full suppression
    }

# For Occupation:
occupation_field = None
if 'Occupation' in df.columns and ('Occupation' in sens_att or 'occupation' in sens_att):
    occupation_field = 'Occupation'
if occupation_field:
    hierarchies[occupation_field] = {
        0: df[occupation_field].values,                                     # Original occupation values
        1: df[occupation_field].astype(str).str[0] + "*",                     # Partial generalization: first letter + "*"
        2: np.array(["*"] * len(df[occupation_field].values))                 # Full suppression
    }

# --- Call the l-diversity anonymization function ---
# Specify k-anonymity and l-diversity levels, and maximum suppression level (as a percentage).
k = 2         # Each quasi-identifier combination must appear at least 2 times.
l_div = 2     # Each group must have at least 2 distinct sensitive values.
supp_level = 20  # Up to 20% of records can be suppressed if necessary.

anonymized_df = anonymity.l_diversity(
    data=df,
    ident=ident,
    quasi_ident=quasi_ident,
    sens_att=sens_att,
    k=k,
    l_div=l_div,
    supp_level=supp_level,
    hierarchies=hierarchies
)

print("Anonymized DataFrame:")
print(anonymized_df)
