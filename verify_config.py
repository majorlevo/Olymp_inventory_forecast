import os
from dotenv import load_dotenv

# Setup: Create a temporary .env file
test_env_file = ".env.test_verification"
with open(test_env_file, "w") as f:
    f.write("TEST_VERIFICATION_VAR=from_file\n")

print("--- Starting Verification ---")

# Test 1: Verify .env loading (Simulate no system var)
if "TEST_VERIFICATION_VAR" in os.environ:
    del os.environ["TEST_VERIFICATION_VAR"]

load_dotenv(dotenv_path=test_env_file)
val = os.getenv("TEST_VERIFICATION_VAR")
if val == "from_file":
    print("✅ PASS: Loaded from .env file when strict system var is missing.")
else:
    print(f"❌ FAIL: Expected 'from_file', got '{val}'")

# Test 2: Verify System Variable Precedence
# reset
if "TEST_VERIFICATION_VAR" in os.environ:
    del os.environ["TEST_VERIFICATION_VAR"]

# Set "System" variable
os.environ["TEST_VERIFICATION_VAR"] = "from_system"

# Load again (should NOT override)
load_dotenv(dotenv_path=test_env_file)
val = os.getenv("TEST_VERIFICATION_VAR")
if val == "from_system":
    print("✅ PASS: System variable took precedence over .env file.")
else:
    print(f"❌ FAIL: Expected 'from_system', got '{val}' (Did .env override it?)")

# Cleanup
if os.path.exists(test_env_file):
    os.remove(test_env_file)
print("--- Verification Complete ---")
