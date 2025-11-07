"""
Paper Trading Integration for AAVE Model
Automatically executes trades based on model signals on paper trading platforms.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import time
import json
import os
import subprocess
from best_model_alert_monitor import BestModelAlertMonitor


def send_paper_trading_notification(title, message):
    """Send macOS notification for paper trading events."""
    try:
        message_clean = message.replace('\n', ' | ')
        script = f'''
        display notification "{message_clean}" with title "{title}" sound name "Glass"
        '''
        subprocess.run(['osascript', '-e', script], check=True, capture_output=True)
        return True
    except:
        return False


class PaperTradingPlatform:
    """Base class for paper trading platforms."""
    
    def __init__(self, initial_balance=10000):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions = {}  # {symbol: {'quantity': float, 'entry_price': float, 'entry_time': datetime}}
        self.trade_history = []
        self.commission = 0.0085  # 0.85% Robinhood commission
        
    def place_order(self, symbol, side, quantity, price, stop_loss=None, take_profit=None):
        """Place an order (buy or sell)."""
        raise NotImplementedError
    
    def get_balance(self):
        """Get current account balance."""
        return self.balance
    
    def get_positions(self):
        """Get current open positions."""
        return self.positions
    
    def get_trade_history(self):
        """Get trade history."""
        return self.trade_history


class LocalPaperTrader(PaperTradingPlatform):
    """
    Local paper trading simulator.
    No external API needed - runs entirely locally.
    """
    
    def __init__(self, initial_balance=10000, commission=0.0085):
        super().__init__(initial_balance)
        self.commission = commission
        self.orders_file = 'paper_trading_orders.json'
        self.positions_file = 'paper_trading_positions.json'
        self.history_file = 'paper_trading_history.json'
        self._load_state()
    
    def _load_state(self):
        """Load saved state from files."""
        try:
            if os.path.exists(self.positions_file):
                with open(self.positions_file, 'r') as f:
                    data = json.load(f)
                    self.positions = {k: {
                        'quantity': v['quantity'],
                        'entry_price': v['entry_price'],
                        'entry_time': datetime.fromisoformat(v['entry_time'])
                    } for k, v in data.items()}
            
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r') as f:
                    self.trade_history = json.load(f)
        except:
            pass
    
    def _save_state(self):
        """Save current state to files."""
        try:
            # Save positions
            positions_data = {k: {
                'quantity': v['quantity'],
                'entry_price': v['entry_price'],
                'entry_time': v['entry_time'].isoformat()
            } for k, v in self.positions.items()}
            
            with open(self.positions_file, 'w') as f:
                json.dump(positions_data, f, indent=2)
            
            # Save history
            with open(self.history_file, 'w') as f:
                json.dump(self.trade_history, f, indent=2)
        except Exception as e:
            print(f"⚠️  Error saving state: {e}")
    
    def place_order(self, symbol, side, quantity, price, stop_loss=None, take_profit=None):
        """
        Place a paper trade order.
        
        Args:
            symbol: Trading symbol (e.g., 'AAVE')
            side: 'buy' or 'sell'
            quantity: Amount to trade
            price: Execution price
            stop_loss: Stop-loss price (optional)
            take_profit: Take-profit price (optional)
            
        Returns:
            dict with order details
        """
        if side.lower() == 'buy':
            # Calculate cost including commission
            cost = quantity * price * (1 + self.commission)
            
            if cost > self.balance:
                return {
                    'success': False,
                    'error': f'Insufficient balance. Need ${cost:.2f}, have ${self.balance:.2f}'
                }
            
            # Execute buy
            self.balance -= cost
            
            if symbol in self.positions:
                # Add to existing position (average price)
                old_qty = self.positions[symbol]['quantity']
                old_price = self.positions[symbol]['entry_price']
                new_qty = old_qty + quantity
                avg_price = ((old_qty * old_price) + (quantity * price)) / new_qty
                
                self.positions[symbol] = {
                    'quantity': new_qty,
                    'entry_price': avg_price,
                    'entry_time': self.positions[symbol]['entry_time'],
                    'stop_loss': stop_loss,
                    'take_profit': take_profit
                }
            else:
                # New position
                self.positions[symbol] = {
                    'quantity': quantity,
                    'entry_price': price,
                    'entry_time': datetime.now(),
                    'stop_loss': stop_loss,
                    'take_profit': take_profit
                }
            
            order = {
                'success': True,
                'order_id': f"BUY_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                'symbol': symbol,
                'side': 'buy',
                'quantity': quantity,
                'price': price,
                'cost': cost,
                'commission': cost - (quantity * price),
                'timestamp': datetime.now().isoformat(),
                'stop_loss': stop_loss,
                'take_profit': take_profit
            }
            
            self.trade_history.append(order)
            self._save_state()
            
            return order
        
        elif side.lower() == 'sell':
            if symbol not in self.positions:
                return {
                    'success': False,
                    'error': f'No position in {symbol}'
                }
            
            position = self.positions[symbol]
            available_qty = position['quantity']
            
            if quantity > available_qty:
                quantity = available_qty  # Sell all
            
            # Calculate proceeds (minus commission)
            proceeds = quantity * price * (1 - self.commission)
            self.balance += proceeds
            
            # Calculate P&L
            pnl = (price - position['entry_price']) * quantity
            pnl_pct = ((price / position['entry_price']) - 1) * 100
            
            # Update position
            if quantity >= available_qty:
                # Close entire position
                del self.positions[symbol]
            else:
                # Partial close
                self.positions[symbol]['quantity'] -= quantity
            
            order = {
                'success': True,
                'order_id': f"SELL_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                'symbol': symbol,
                'side': 'sell',
                'quantity': quantity,
                'price': price,
                'proceeds': proceeds,
                'commission': (quantity * price) - proceeds,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'timestamp': datetime.now().isoformat()
            }
            
            self.trade_history.append(order)
            self._save_state()
            
            return order
        
        return {'success': False, 'error': 'Invalid side'}
    
    def check_stop_loss_take_profit(self, symbol, current_price):
        """Check if stop-loss or take-profit should be triggered."""
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        stop_loss = position.get('stop_loss')
        take_profit = position.get('take_profit')
        
        # Check stop-loss
        if stop_loss and current_price <= stop_loss:
            return self.place_order(symbol, 'sell', position['quantity'], stop_loss)
        
        # Check take-profit
        if take_profit and current_price >= take_profit:
            return self.place_order(symbol, 'sell', position['quantity'], take_profit)
        
        return None
    
    def get_portfolio_value(self, current_price):
        """Get total portfolio value."""
        positions_value = sum(
            pos['quantity'] * current_price 
            for pos in self.positions.values()
        )
        return self.balance + positions_value
    
    def get_performance(self, current_price):
        """Get performance metrics."""
        portfolio_value = self.get_portfolio_value(current_price)
        total_return = ((portfolio_value - self.initial_balance) / self.initial_balance) * 100
        
        closed_trades = [t for t in self.trade_history if t.get('pnl') is not None]
        winning_trades = [t for t in closed_trades if t['pnl'] > 0]
        losing_trades = [t for t in closed_trades if t['pnl'] < 0]
        
        win_rate = (len(winning_trades) / len(closed_trades) * 100) if closed_trades else 0
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        
        return {
            'initial_balance': self.initial_balance,
            'current_balance': self.balance,
            'portfolio_value': portfolio_value,
            'total_return': total_return,
            'total_trades': len(closed_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'reward_risk_ratio': abs(avg_win / avg_loss) if avg_loss != 0 else 0
        }


class AutoPaperTrader:
    """
    Automatic paper trading system that executes trades based on model signals.
    """
    
    def __init__(self, initial_balance=10000, platform='local'):
        """
        Initialize auto paper trader.
        
        Args:
            initial_balance: Starting balance
            platform: 'local' (default) or 'alpaca' (requires API keys)
        """
        self.model_monitor = BestModelAlertMonitor()
        
        if platform == 'local':
            self.platform = LocalPaperTrader(initial_balance=initial_balance)
        elif platform == 'alpaca':
            # TODO: Implement Alpaca integration
            raise NotImplementedError("Alpaca integration coming soon")
        else:
            raise ValueError(f"Unknown platform: {platform}")
        
        self.symbol = 'AAVE'
        self.last_signal_time = None
        self.active_trades = {}  # Track active trades with stop-loss/take-profit
    
    def execute_signal(self, signal_data):
        """
        Execute a trading signal.
        
        Args:
            signal_data: Signal data from model (dict with entry_price, stop_loss, etc.)
        """
        if signal_data['signal'] != 'BUY':
            return None
        
        # Place buy order
        order = self.platform.place_order(
            symbol=self.symbol,
            side='buy',
            quantity=signal_data['quantity'],
            price=signal_data['entry_price'],
            stop_loss=signal_data['stop_loss'],
            take_profit=signal_data['take_profit']
        )
        
        if order['success']:
            print(f"\n✅ ORDER EXECUTED:")
            print(f"   Symbol: {self.symbol}")
            print(f"   Side: BUY")
            print(f"   Quantity: {order['quantity']:.6f}")
            print(f"   Price: ${order['price']:.2f}")
            print(f"   Cost: ${order['cost']:.2f}")
            print(f"   Stop-Loss: ${signal_data['stop_loss']:.2f}")
            print(f"   Take-Profit: ${signal_data['take_profit']:.2f}")
            print(f"   Balance: ${self.platform.balance:.2f}")
            
            # Send notification
            message = (
                f"BUY {order['quantity']:.4f} @ ${order['price']:.2f} | "
                f"Cost: ${order['cost']:.2f} | "
                f"Stop: ${signal_data['stop_loss']:.2f} | "
                f"Target: ${signal_data['take_profit']:.2f} | "
                f"Balance: ${self.platform.balance:.2f}"
            )
            send_paper_trading_notification("📈 Paper Trade Executed", message)
            
            # Track active trade
            self.active_trades[order['order_id']] = {
                'symbol': self.symbol,
                'entry_price': signal_data['entry_price'],
                'stop_loss': signal_data['stop_loss'],
                'take_profit': signal_data['take_profit'],
                'quantity': signal_data['quantity']
            }
        
        return order
    
    def check_active_trades(self, current_price):
        """Check and execute stop-loss/take-profit for active trades."""
        if self.symbol in self.platform.positions:
            result = self.platform.check_stop_loss_take_profit(self.symbol, current_price)
            if result and result['success']:
                print(f"\n🔄 TRADE CLOSED:")
                print(f"   Symbol: {self.symbol}")
                print(f"   Side: {result['side'].upper()}")
                print(f"   Quantity: {result['quantity']:.6f}")
                print(f"   Price: ${result['price']:.2f}")
                if 'pnl' in result:
                    print(f"   P&L: ${result['pnl']:.2f} ({result['pnl_pct']:.2f}%)")
                print(f"   Balance: ${self.platform.balance:.2f}")
                
                # Send notification
                if 'pnl' in result:
                    pnl_emoji = "✅" if result['pnl'] > 0 else "❌"
                    message = (
                        f"{pnl_emoji} {result['side'].upper()} {result['quantity']:.4f} @ ${result['price']:.2f} | "
                        f"P&L: ${result['pnl']:.2f} ({result['pnl_pct']:.2f}%) | "
                        f"Balance: ${self.platform.balance:.2f}"
                    )
                    send_paper_trading_notification("🔄 Paper Trade Closed", message)
                
                return result
        return None
    
    def run_cycle(self):
        """Run one cycle: check for signals and execute trades."""
        # Check for new signals (disable notifications in model monitor)
        # Temporarily disable notifications by patching the send_alert method
        original_send_alert = self.model_monitor.send_alert
        self.model_monitor.send_alert = lambda *args, **kwargs: None  # Disable notifications
        signal = self.model_monitor.check_for_signals()
        self.model_monitor.send_alert = original_send_alert  # Restore
        
        if signal:
            # Check if we already have a position
            if self.symbol not in self.platform.positions:
                # Execute new trade
                self.execute_signal(signal)
                self.last_signal_time = datetime.now()
            else:
                print(f"⏸️  Already have position in {self.symbol}, skipping new signal")
        
        # Check active trades for stop-loss/take-profit
        if self.symbol in self.platform.positions:
            # Get current price
            try:
                df = self.model_monitor.fetcher.fetch_data(days=1, timeframe='5m')
                if df is not None and len(df) > 0:
                    current_price = df['close'].iloc[-1]
                    self.check_active_trades(current_price)
            except:
                pass
    
    def run_continuous(self, interval_minutes=5):
        """Run continuously, checking for signals every N minutes."""
        print("="*70)
        print("AUTO PAPER TRADING SYSTEM")
        print("="*70)
        print(f"Platform: Local Paper Trader")
        print(f"Initial Balance: ${self.platform.initial_balance:,.2f}")
        print(f"Symbol: {self.symbol}")
        print(f"Check Interval: {interval_minutes} minutes")
        print("="*70)
        print("\n🔄 Starting automatic trading...")
        print("Press Ctrl+C to stop\n")
        
        try:
            while True:
                self.run_cycle()
                
                # Show performance
                try:
                    df = self.model_monitor.fetcher.fetch_data(days=1, timeframe='5m')
                    if df is not None and len(df) > 0:
                        current_price = df['close'].iloc[-1]
                        perf = self.platform.get_performance(current_price)
                        
                        print(f"\n📊 PERFORMANCE ({datetime.now().strftime('%H:%M:%S')}):")
                        print(f"   Portfolio Value: ${perf['portfolio_value']:.2f}")
                        print(f"   Total Return: {perf['total_return']:.2f}%")
                        print(f"   Win Rate: {perf['win_rate']:.1f}%")
                        print(f"   Trades: {perf['total_trades']}")
                        if perf['total_trades'] > 0:
                            print(f"   Avg Win: ${perf['avg_win']:.2f}")
                            print(f"   Avg Loss: ${perf['avg_loss']:.2f}")
                            print(f"   R:R Ratio: {perf['reward_risk_ratio']:.2f}:1")
                        
                        # Send notification for significant updates (every 10th cycle or on trades)
                        if not hasattr(self, '_last_notification_cycle'):
                            self._last_notification_cycle = 0
                            self._last_trade_count = 0
                        
                        # Notify on new trades or every 10 cycles (50 minutes)
                        if (perf['total_trades'] > self._last_trade_count) or \
                           ((self._last_notification_cycle % 10 == 0) and perf['total_trades'] > 0):
                            message = (
                                f"Return: {perf['total_return']:.2f}% | "
                                f"Win Rate: {perf['win_rate']:.1f}% | "
                                f"Trades: {perf['total_trades']} | "
                                f"Value: ${perf['portfolio_value']:.2f}"
                            )
                            send_paper_trading_notification("📊 Paper Trading Update", message)
                            self._last_trade_count = perf['total_trades']
                        
                        self._last_notification_cycle += 1
                except:
                    pass
                
                time.sleep(interval_minutes * 60)
        
        except KeyboardInterrupt:
            print("\n\n🛑 Stopping auto trader...")
            self.show_final_performance()
    
    def show_final_performance(self):
        """Show final performance summary."""
        try:
            df = self.model_monitor.fetcher.fetch_data(days=1, timeframe='5m')
            current_price = df['close'].iloc[-1] if df is not None and len(df) > 0 else 0
            perf = self.platform.get_performance(current_price)
            
            print("\n" + "="*70)
            print("FINAL PERFORMANCE")
            print("="*70)
            print(f"Initial Balance: ${perf['initial_balance']:,.2f}")
            print(f"Final Portfolio Value: ${perf['portfolio_value']:,.2f}")
            print(f"Total Return: {perf['total_return']:.2f}%")
            print(f"\nTrade Statistics:")
            print(f"  Total Trades: {perf['total_trades']}")
            print(f"  Winning Trades: {perf['winning_trades']}")
            print(f"  Losing Trades: {perf['losing_trades']}")
            print(f"  Win Rate: {perf['win_rate']:.1f}%")
            if perf['total_trades'] > 0:
                print(f"  Avg Win: ${perf['avg_win']:.2f}")
                print(f"  Avg Loss: ${perf['avg_loss']:.2f}")
                print(f"  Reward:Risk: {perf['reward_risk_ratio']:.2f}:1")
            print("="*70)
        except Exception as e:
            print(f"Error showing performance: {e}")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Auto Paper Trading System')
    parser.add_argument('--balance', type=float, default=10000, help='Initial balance')
    parser.add_argument('--interval', type=int, default=5, help='Check interval in minutes')
    parser.add_argument('--once', action='store_true', help='Run once and exit')
    
    args = parser.parse_args()
    
    trader = AutoPaperTrader(initial_balance=args.balance)
    
    if args.once:
        trader.run_cycle()
        trader.show_final_performance()
    else:
        trader.run_continuous(interval_minutes=args.interval)


if __name__ == "__main__":
    main()

