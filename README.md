# UPI Fraud Detection Environment

## Overview
Real-world environment for AI agents to learn UPI payment fraud detection.

## State Space (6 features)
- `amount`: Transaction amount (₹100-100k)
- `time`: Hour of day (0-23)  
- `location_risk`: 0-1 (merchant location risk)
- `device_risk`: 0-1 (device fingerprint risk)
- `user_trust`: 0-1 (user reputation score)
- `failed_attempts`: Recent failed payments (0-10)

## Action Space
- `0`: Approve transaction
- `1`: Reject transaction  
- `2`: Flag for manual review

## Tasks (3 levels)
1. **Easy**: Safe transactions (10 steps)
2. **Medium**: Mixed patterns (20 steps)
3. **Hard**: Complex fraud cases (30 steps)

## Setup & Run
```bash
pip install -r requirements.txt
python inference.py
```

## Expected Output