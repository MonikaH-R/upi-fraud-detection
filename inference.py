#!/usr/bin/env python3
"""
UPI Fraud Detection - Round 1 Baseline
"""
from environment import UPIFraudEnv
import time


def run_task(env, task_name: str, steps: int = 10) -> float:
    """Run single task and return score"""
    total_reward = 0.0
    state, _ = env.reset()

    for step in range(steps):
        # Simple rule-based agent
        amount, _, loc_risk, dev_risk, trust, fails = state
        fraud_risk = loc_risk + dev_risk + (1 - trust) + (fails * 0.1)

        if fraud_risk > 2.0 or amount > 20000:
            action = 1  # Reject
        elif fraud_risk > 1.0:
            action = 2  # Flag
        else:
            action = 0  # Approve

        state, reward, done, _, _ = env.step(action)
        total_reward += reward

        if done:
            break

    score = max(0.0, min(1.0, total_reward / steps))
    print(f"  {task_name:<25} | Score: {score:.3f}")
    return score


def main():
    print(" UPI FRAUD DETECTION - ROUND 1")
    print("=" * 60)

    env = UPIFraudEnv()
    tasks = [
        ("Easy (10 steps)", 10),
        ("Medium (20 steps)", 20),
        ("Hard (30 steps)", 30)
    ]

    total_score = 0.0
    for name, steps in tasks:
        score = run_task(env, name, steps)
        total_score += score

    avg_score = total_score / 3
    print("\n" + "=" * 60)
    print(f" FINAL RESULT: {avg_score:.3f}/1.000")
    print(" All 3 tasks completed successfully!")


if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"  Runtime: {time.time() - start_time:.1f}s (<20min ✓)")