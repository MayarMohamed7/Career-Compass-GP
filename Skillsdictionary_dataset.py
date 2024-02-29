import pandas as pd
import html

df = pd.read_csv('skill2vec_10K.csv')

melted_df = pd.melt(df, value_vars=df.columns)

# The 'value' column now contains all the skills. We can rename it to 'Skills'
# and remove any NaN values or duplicates if necessary.
skills_df = melted_df[['value']].rename(columns={'value': 'Skills'}).dropna().drop_duplicates()
skills_df.to_csv('skills_dataset_single_column.csv', index=False)
print(skills_df.head())
df = pd.read_csv('skills_dataset_single_column.csv')

df_cleaned = df.dropna().drop_duplicates()

def unescape_html(s):
    return html.unescape(s)

# Apply the unescape function to the 'Skills' column
df_cleaned['Skills'] = df_cleaned['Skills'].apply(unescape_html)
df_cleaned.to_csv('first_trans_try.csv', index=False)
print(df_cleaned.head())