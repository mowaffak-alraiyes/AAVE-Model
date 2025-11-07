"""
Fix incorrect position entry price.
If the entry price is clearly wrong (too low/high), correct it to current market price.
"""

import json
from data_fetcher import AAVEDataFetcher
from datetime import datetime

def fix_position():
    """Fix position with incorrect entry price."""
    
    # Get current AAVE price
    fetcher = AAVEDataFetcher(source='coingecko')
    df = fetcher.fetch_data(days=1)
    if df is None or len(df) == 0:
        print("❌ Could not fetch current price")
        return
    
    current_price = df['close'].iloc[-1]
    print(f"Current AAVE Price: ${current_price:.2f}")
    
    # Load position
    try:
        with open('paper_trading_positions.json', 'r') as f:
            positions = json.load(f)
    except:
        print("❌ No positions file found")
        return
    
    if 'AAVE' not in positions:
        print("❌ No AAVE position found")
        return
    
    position = positions['AAVE']
    old_price = position['entry_price']
    quantity = position['quantity']
    
    print(f"\nCurrent Position:")
    print(f"  Quantity: {quantity:.4f} AAVE")
    print(f"  Entry Price: ${old_price:.2f}")
    print(f"  Position Value: ${quantity * old_price:.2f}")
    
    # Check if price is clearly wrong (more than 50% off)
    if abs(old_price - current_price) / current_price > 0.5:
        print(f"\n⚠️  Entry price seems incorrect!")
        print(f"  Current market: ${current_price:.2f}")
        print(f"  Stored entry: ${old_price:.2f}")
        print(f"  Difference: {abs(old_price - current_price) / current_price * 100:.1f}%")
        
        response = input(f"\nFix entry price to ${current_price:.2f}? (y/n): ")
        if response.lower() == 'y':
            # Update position
            positions['AAVE']['entry_price'] = current_price
            positions['AAVE']['entry_time'] = datetime.now().isoformat()
            
            with open('paper_trading_positions.json', 'w') as f:
                json.dump(positions, f, indent=2)
            
            print(f"\n✅ Position updated!")
            print(f"  New Entry Price: ${current_price:.2f}")
            print(f"  New Position Value: ${quantity * current_price:.2f}")
        else:
            print("❌ Cancelled")
    else:
        print(f"\n✅ Entry price looks reasonable (within 50% of current price)")

if __name__ == "__main__":
    fix_position()


