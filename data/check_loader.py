from loader import load_history_sales

try:
    print("Attempting to load data...")
    df = load_history_sales()
    
    print("\n✅ SUCCESS: Data loaded.")
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    
    # Critical Check: Does 'single_tickets' exist and is it a number?
    if 'single_tickets' in df.columns:
        print(f"Sample ticket counts: {df['single_tickets'].head().tolist()}")
        print("Type check passed.")
    else:
        print("❌ FAILURE: 'single_tickets' column is missing!")

except Exception as e:
    print(f"\n❌ CRASHED: {e}")
