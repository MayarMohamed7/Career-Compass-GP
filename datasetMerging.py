import pandas as pd

# Load datasets
jobs = pd.read_csv("/content/drive/MyDrive/dataset_grad/jobs.csv")
dataScrappingjob_data = pd.read_csv("/content/drive/MyDrive/dataset_grad/dataScrappingjob_data.csv", encoding='ISO-8859-1')

# Select relevant columns
jobs = jobs[["Job Title", "Skills"]]
dataScrappingjob_data = dataScrappingjob_data[["Job Title", "Skills"]]

# Merge datasets based on "Job Title" using outer join
merged_data = pd.merge(jobs, dataScrappingjob_data, on="Job Title", how="inner")

# Group by "Job Title" and concatenate skills
merged_data["Skills_y"] = merged_data["Skills_y"].fillna('')
merged_data = merged_data.groupby("Job Title")["Skills_y"].apply(lambda x: ' '.join(str(skill) for skill in x)).reset_index()

# Rename the "Skills_y" column to "Skills"
merged_data.rename(columns={"Skills_y": "Skills"}, inplace=True)

# Save merged dataset to a new file with only "Skills" and "Job Title" columns
merged_data.to_csv("mergeddata_OuterJoin.csv", index=False)
