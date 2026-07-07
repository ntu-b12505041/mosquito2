# Shared settings for local CPU deep-learning runs.
#
# Change this placeholder to your local PTB-XL folder.
# Accepted:
#   1. PTB-XL root folder containing ptbxl_database.csv, scp_statements.csv, records100/
#   2. The records100 folder itself
$PTBXL_DATA_DIR = "<<<PTBXL_DATA_DIR>>>"

# Full PTB-XL deep run settings.
# 0 means full dataset. Use a smaller number like 1000 if you only want a quick sanity run.
$DEEP_MAX_RECORDS = 0

# Formal experiment setting used in the project.
$DEEP_EPOCHS = 20

# CPU-friendly batch size. Increase only if your RAM is comfortable.
$DEEP_BATCH_SIZE = 32
