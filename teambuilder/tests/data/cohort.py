import pandas as pd
import phantombunch as pb

n = 250  # number of students in the cohort

# Create cohort.
while not pb.valid_cohort(cohort := pb.cohort(n)):
    cohort = pb.cohort(250)

# Create marks.
marks = pb.assignment(
    usernames=cohort.username, mean=65, sd=6, fail_probability=0.02, feedback=False
)

# Add marks to cohort.
cohort = pd.merge(cohort, marks, on="username")

# Merge first and last names.
cohort["name"] = cohort.first_name + " " + cohort.last_name

# Keep only relevant columns.
cohort = cohort[
    [
        "username",
        "name",
        "course",
        "gender",
        "mark",
    ]
]

cohort.to_csv("cohort.csv", index=False)
