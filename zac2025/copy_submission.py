import json

# Read source file
with open('d:/ZALO-AI-2025/submission.json', 'r') as f:
    data = json.load(f)

# Write to destination
with open('d:/ZALO-AI-2025/zac2025/jupyter_submission.json', 'w') as f:
    json.dump(data, f, indent=2)

print("✓ Copied submission.json to jupyter_submission.json")
