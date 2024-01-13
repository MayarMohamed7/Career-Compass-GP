import pandas as pd

# Load your skills dataset
# Replace 'your_dataset.csv' with the actual filename or path
df = pd.read_csv('your_dataset.csv')

# Assuming your skills are in a column named 'skills'
skills_column = 'skills'

# Create a skills dictionary
skills_dict = {}

# Iterate through each row in the dataset
for index, row in df.iterrows():
    # Split skills if they are comma-separated or use another delimiter
    skills_list = row[skills_column].split(',')

    # Add each skill to the dictionary
    for skill in skills_list:
        # Remove leading and trailing whitespaces
        skill = skill.strip()
        
        # Check if the skill is already in the dictionary
        if skill not in skills_dict:
            # If not, add it with an empty list as the value
            skills_dict[skill] = []
        
        # Append any additional information you have about the skill
        # For example, you might have a column 'category' in your dataset
        # skills_dict[skill].append(row['category'])

# Now, skills_dict contains a dictionary where keys are skills, and values are lists of additional information about each skill.

# Example usage:
# Print information about a specific skill
print("Information about 'Python':", skills_dict.get('Python', 'Skill not found'))

# Print all skills and their information
for skill, info in skills_dict.items():
    print(f"Skill: {skill}, Information: {info}")
