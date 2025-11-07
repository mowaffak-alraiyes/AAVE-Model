"""
Risk Management System for AAVE Trading
Implements stop-loss, take-profit, and 3:1 reward:risk ratio
"""

import pandas as pd
import numpy as np


class RiskManager:
    """Manages risk, position sizing, stop-loss, and take-profit."""
    
    def __init__(self, 
                 reward_risk_ratio=3.0,
                 target_gain_pct=0.04,  # 4% target (3-5% range)
                 max_loss_pct=0.0133,  # 1.33% max loss (for 3:1 ratio: 4%/3)
                 position_size=3500,  # Average investment 3-4k
                 atr_multiplier=2.0):
        """
        Initialize risk manager.
        
        Args:
            reward_risk_ratio: Target reward:risk ratio (default 3:1)
            target_gain_pct: Target gain percentage (default 4%)
            max_loss_pct: Maximum loss percentage (auto-calculated from ratio)
            position_size: Average position size in USD
            atr_multiplier: ATR multiplier for stop-loss calculation
        """
        self.reward_risk_ratio = reward_risk_ratio
        self.target_gain_pct = target_gain_pct
        self.max_loss_pct = max_loss_pct if max_loss_pct else target_gain_pct / reward_risk_ratio
        self.position_size = position_size
        self.atr_multiplier = atr_multiplier
    
    def calculate_stop_loss(self, entry_price, atr=None, use_atr=True):
        """
        Calculate stop-loss price.
        
        Args:
            entry_price: Entry price
            atr: Average True Range value
            use_atr: Use ATR-based stop or percentage-based
            
        Returns:
            Stop-loss price
        """
        # Always enforce max loss percentage as hard limit
        max_stop = entry_price * (1 - self.max_loss_pct)
        
        if use_atr and atr is not None and atr > 0:
            # Use ATR-based stop (more dynamic), but cap at max loss
            atr_stop = entry_price - (atr * self.atr_multiplier)
            # Use the tighter (higher) stop-loss
            return max(atr_stop, max_stop)
        else:
            # Use percentage-based stop
            return max_stop
    
    def calculate_take_profit(self, entry_price, stop_loss):
        """
        Calculate take-profit price based on reward:risk ratio.
        
        Args:
            entry_price: Entry price
            stop_loss: Stop-loss price
            
        Returns:
            Take-profit price
        """
        risk = entry_price - stop_loss
        reward = risk * self.reward_risk_ratio
        take_profit = entry_price + reward
        
        # Also check if it meets target gain
        target_price = entry_price * (1 + self.target_gain_pct)
        
        # Use the higher of the two (ensures both 3:1 ratio and target gain)
        return max(take_profit, target_price)
    
    def calculate_position_size(self, entry_price, stop_loss, max_risk_amount=None, commission=0.0085):
        """
        Calculate position size based on risk, accounting for commission.
        
        Args:
            entry_price: Entry price
            stop_loss: Stop-loss price
            max_risk_amount: Maximum amount to risk (default: position_size * max_loss_pct)
            commission: Trading commission (default 0.85% for Robinhood)
            
        Returns:
            Position size in USD and quantity
        """
        if max_risk_amount is None:
            max_risk_amount = self.position_size * self.max_loss_pct
        
        # Account for commission on entry
        # Effective entry price = entry_price * (1 + commission)
        effective_entry = entry_price * (1 + commission)
        risk_per_share = effective_entry - stop_loss
        
        if risk_per_share <= 0:
            return 0, 0
        
        # Calculate quantity based on max risk
        quantity = max_risk_amount / risk_per_share
        
        # Calculate position size (including commission)
        position_value = quantity * effective_entry
        
        # Cap at average position size
        if position_value > self.position_size * 1.5:  # Allow 50% over for good setups
            position_value = self.position_size * 1.5
            quantity = position_value / effective_entry
        
        return position_value, quantity
    
    def validate_trade(self, entry_price, stop_loss, take_profit, atr=None):
        """
        Validate if trade meets risk criteria.
        
        Args:
            entry_price: Entry price
            stop_loss: Stop-loss price
            take_profit: Take-profit price
            atr: Average True Range
            
        Returns:
            Tuple of (is_valid, reason)
        """
        # Calculate actual risk and reward
        risk = entry_price - stop_loss
        reward = take_profit - entry_price
        
        if risk <= 0:
            return False, "Invalid stop-loss (above entry price)"
        
        # Check reward:risk ratio
        actual_ratio = reward / risk
        if actual_ratio < self.reward_risk_ratio * 0.9:  # Allow 10% tolerance
            return False, f"Reward:risk ratio too low ({actual_ratio:.2f}, need {self.reward_risk_ratio})"
        
        # Check target gain
        gain_pct = (take_profit / entry_price - 1) * 100
        if gain_pct < 3.0:
            return False, f"Target gain too low ({gain_pct:.2f}%, need 3-5%)"
        if gain_pct > 6.0:
            return False, f"Target gain too high ({gain_pct:.2f}%, need 3-5%)"
        
        # Check max loss
        loss_pct = (1 - stop_loss / entry_price) * 100
        if loss_pct > self.max_loss_pct * 100 * 1.1:  # Allow 10% tolerance
            return False, f"Stop-loss too wide ({loss_pct:.2f}%, max {self.max_loss_pct*100:.2f}%)"
        
        return True, f"Valid trade: {actual_ratio:.2f}:1 ratio, {gain_pct:.2f}% target gain"
    
    def get_trade_plan(self, entry_price, atr=None, use_atr=True):
        """
        Get complete trade plan with all risk parameters.
        
        Args:
            entry_price: Entry price
            atr: Average True Range
            use_atr: Use ATR for stop-loss
            
        Returns:
            Dictionary with trade plan
        """
        stop_loss = self.calculate_stop_loss(entry_price, atr, use_atr)
        take_profit = self.calculate_take_profit(entry_price, stop_loss)
        position_value, quantity = self.calculate_position_size(entry_price, stop_loss)
        
        is_valid, reason = self.validate_trade(entry_price, stop_loss, take_profit, atr)
        
        risk = entry_price - stop_loss
        reward = take_profit - entry_price
        reward_risk_ratio = reward / risk if risk > 0 else 0
        
        return {
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'position_value': position_value,
            'quantity': quantity,
            'risk_amount': risk * quantity,
            'reward_amount': reward * quantity,
            'reward_risk_ratio': reward_risk_ratio,
            'target_gain_pct': (take_profit / entry_price - 1) * 100,
            'max_loss_pct': (1 - stop_loss / entry_price) * 100,
            'is_valid': is_valid,
            'validation_reason': reason
        }


if __name__ == "__main__":
    # Test risk manager
    rm = RiskManager(reward_risk_ratio=3.0, target_gain_pct=0.04, position_size=3500)
    
    entry = 200.0
    atr = 5.0
    
    plan = rm.get_trade_plan(entry, atr)
    print("Trade Plan:")
    for key, value in plan.items():
        print(f"  {key}: {value}")

