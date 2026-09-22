"""
Fraud Detection Project: Shared Configuration
Loads database credentials and API keys from environment variables (via a
local .env file) instead of hardcoding them in scripts. This file itself
contains NO secrets and is safe to commit to GitHub.

Requires: pip install python-dotenv
"""

import os
from dotenv import load_dotenv

load_dotenv()  # reads variables from a .env file in the same folder, if present

DB_CONFIG = {
    "user": os.environ.get("DB_USER", "postgres"),
    "password": os.environ.get("DB_PASSWORD"),
    "host": os.environ.get("DB_HOST", "localhost"),
    "port": os.environ.get("DB_PORT", "5432"),
    "database": os.environ.get("DB_NAME", "fraud_detection"),
}

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if DB_CONFIG["password"] is None:
    raise ValueError(
        "DB_PASSWORD not set. Create a .env file (see .env.example) with your "
        "PostgreSQL password before running this script."
    )