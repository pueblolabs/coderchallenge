import json
import pandas as pd
import matplotlib.pyplot as plt

# IMPORTANT: We need the actual calculation function to test it.
# This assumes 'calculate_reimbursement.py' exists and has the function.
from calculate_reimbursement import calculate_reimbursement

def analyze():
    """
    Loads public cases, runs the current model, calculates errors, and generates plots.
    """
    print("--- Running Error Analysis ---")

    # 1. Load the data
    with open("public_cases.json") as f:
        data = json.load(f)
    
    # Flatten the JSON data into a pandas DataFrame
    df = pd.json_normalize(data)
    df.rename(columns={
        'input.trip_duration_days': 'days',
        'input.miles_traveled': 'miles',
        'input.total_receipts_amount': 'receipts',
        'expected_output': 'expected'
    }, inplace=True)

    print(f"Loaded {len(df)} public cases.")

    # 2. Compute predictions and errors
    # We pass a placeholder for case_index since our current model doesn't use it yet.
    df['predicted'] = df.apply(lambda row: calculate_reimbursement(
        row['days'], row['miles'], row['receipts'], case_index=0
    ), axis=1)
    
    df['error'] = df['predicted'] - df['expected']
    
    # 3. Compute helper features for plotting
    df['receipts_per_day'] = df['receipts'] / df['days']
    df['miles_per_day'] = df['miles'] / df['days']

    avg_error = df['error'].abs().mean()
    print(f"Calculated Average Error on public set: ${avg_error:.2f}")

    # 4. Generate plots
    print("Generating residual plots... (Close the plot window to continue)")

    # Plot 1: Error vs. Trip Duration
    plt.figure(figsize=(10, 6))
    plt.scatter(df['days'], df['error'], alpha=0.5)
    plt.axhline(0, color='red', linestyle='--')
    plt.title('Error vs. Trip Duration (days)')
    plt.xlabel('Trip Duration (days)')
    plt.ylabel('Error (Predicted - Expected)')
    plt.grid(True)
    plt.show()

    # Plot 2: Error vs. Receipts per Day
    plt.figure(figsize=(10, 6))
    plt.scatter(df['receipts_per_day'], df['error'], alpha=0.5)
    plt.axhline(0, color='red', linestyle='--')
    plt.title('Error vs. Receipts per Day ($)')
    plt.xlabel('Receipts per Day ($)')
    plt.ylabel('Error (Predicted - Expected)')
    plt.grid(True)
    plt.show()

    print("--- Analysis Complete ---")
    print("Review the plots to identify patterns in the errors.")

if __name__ == "__main__":
    analyze()