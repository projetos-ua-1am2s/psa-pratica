from person_tracker import PersonTracker

# 1. Initialize your tracker
tracker = PersonTracker()

# 2. Run the validation method you built!
# Make sure "data.yaml" matches the name of your dataset config file
tracker.validate(data_config="Human_Dataset_v2/data.yaml")