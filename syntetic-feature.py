import numpy as np
import pandas as pd

# Number of users
num_users = 58228

# Generate features
np.random.seed(42)  # For reproducibility
ages = np.random.randint(10, 65, size=num_users)  # Ages between 18 and 65
genders = np.random.choice([0, 1], size=num_users).astype(int)  # Binary encoding: 0 for male, 1 for female
genders = np.eye(2)[genders].astype(int)  # Convert to one-hot encoding
occupations = np.eye(4)[np.random.choice(4, size=num_users)]  # 4 categories, one-hot encoded
education_levels = np.eye(3)[np.random.choice(3, size=num_users)].astype(int)  # 3 education levels, one-hot encoded
home_locations = np.eye(4)[np.random.choice(4, size=num_users)].astype(int)  # 4 regions, one-hot encoded
frequent_places = np.random.randint(0, 2, size=(num_users, 3)).astype(int)  # Binary for 3 places
visit_counts = np.random.randint(0, 50, size=(num_users, 3)) / 50  # Normalize counts
distances = np.random.uniform(0, 100, size=num_users) / 100  # Normalize distance
regions = np.eye(3)[np.random.choice(3, size=num_users)].astype(int)  # Region affiliation, one-hot encoded
mutual_friends = np.random.randint(0, 50, size=num_users) / 50  # Normalize mutual friends
interests = np.random.randint(0, 2, size=(num_users, 3)).astype(int)  # Binary vector for interests
hobbies = np.random.randint(0, 2, size=(num_users, 3)).astype(int)  # Binary vector for hobbies
activity_engagement = np.random.uniform(0, 1, size=num_users).astype(int)  # Engagement score
recency_activity = np.random.randint(0, 30, size=num_users).astype(int)  # Days since last activity
recency_activity = (30 - recency_activity) / 30  # Normalize to [0, 1]

# Combine features into a DataFrame for better readability
data = np.hstack([
    ages.reshape(-1, 1), genders, occupations, 
    education_levels, home_locations, 
    frequent_places, visit_counts, distances.reshape(-1, 1), regions, 
    mutual_friends.reshape(-1, 1), interests, hobbies, 
    activity_engagement.reshape(-1, 1), recency_activity.reshape(-1, 1)
])

columns = [
    "Age", 
    "Male", "Female", "Occ_1", "Occ_2", "Occ_3", "Occ_4", 
     "Edu_1", "Edu_2", "Edu_3", "Loc_1", "Loc_2", "Loc_3", "Loc_4", 
    "Place_1", "Place_2", "Place_3", "Visit_1", "Visit_2", "Visit_3", 
    "Distance", "Reg_1", "Reg_2", "Reg_3", "Mutual_Friends", 
    "Interest_1", "Interest_2", "Interest_3", "Hobby_1", "Hobby_2", "Hobby_3", 
    "Engagement", "Recency"
]

df = pd.DataFrame(data, columns=columns)
#Age normalization
min_age, max_age = df['Age'].min(), df['Age'].max()
df['Age'] = (df['Age'] - min_age) / (max_age - min_age)

df.to_csv("synthetic_social_network_features.csv", index=False)
print("File saved as 'synthetic_social_network_features.csv'")