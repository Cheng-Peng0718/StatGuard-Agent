import os

# Original read-only data path
ORIGINAL_DATA_PATH = "data/test_data_2.csv"

# Sandbox workspace directory
WORKSPACE_DIR = "workspace"

# Writable working copy for the agent
WORKING_DATA_PATH = os.path.join(WORKSPACE_DIR, "working_data.csv")