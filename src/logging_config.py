import logging
import os
from datetime import datetime

# Step 1: create logs folder
LOG_DIR = "/tmp/logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Step 2: create log file name
log_file = f"log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

# Step 3: FULL correct path (only once!)
LOG_FILE_PATH = os.path.join(LOG_DIR, log_file)

# Step 4: configure logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.info("Logging is working")