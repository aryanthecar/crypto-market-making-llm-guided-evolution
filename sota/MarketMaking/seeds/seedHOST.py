import numpy as np
from numba import njit
from hftbacktest import BUY, SELL, GTX, LIMIT

# Define the data type for recording output stats from the model.
out_dtype = np.dtype([
    ('inventory_shift', 'f8'),
    ('mid_price_tick', 'f8'),
])

@njit
def ho_stoll_market_maker(hbt, recorder):
    """
    This function contains the core logic for the Ho and Stoll market maker.
    It's decorated with @njit for performance.
    """
    tick_size = hbt.depth(0).tick_size
    
    # Pre-allocate a numpy array to record our output statistics.
    out = np.zeros(10_000_000, out_dtype)
    
    # Time step counter
    t = 0

    # --- Strategy Parameters ---
    # The base bid-ask spread the dealer wants to earn, in ticks.
    desired_spread_tick = 2.5  # e.g., 2.5 ticks on each side of the mid-price
    
    # The risk parameter (lambda) that determines how aggressively to shift quotes.
    # A higher value means a larger price shift for the same inventory.
    risk_parameter = 0.1
    
    # Order quantity and maximum position limits.
    order_qty = 1
    max_position = 50

    # Main loop: checks every 100 milliseconds.
    while hbt.elapse(100_000_000) == 0:
        # Clear any trades that have occurred since the last check.
        hbt.clear_last_trades(0)
        # Clear any inactive (filled or canceled) orders from our order management.
        hbt.clear_inactive_orders(0)

        # Get the current state of the market and our agent.
        depth = hbt.depth(0)
        position = hbt.position(0)
        orders = hbt.orders(0)

        best_bid_tick = depth.best_bid_tick
        best_ask_tick = depth.best_ask_tick

        # If the book is empty, we cannot determine a price, so we wait.
        if not (np.isfinite(best_bid_tick) and np.isfinite(best_ask_tick)):
            continue

        # Approximate the fundamental value with the current mid-price.
        mid_price_tick = (best_bid_tick + best_ask_tick) / 2.0

        #--------------------------------------------------------
        # Computes bid price and ask price using Ho and Stoll logic.

        half_spread_tick = desired_spread_tick / 2.0
        
        # This is the core of the model: calculate the price shift based on inventory.
        inventory_shift = risk_parameter * position
        
        # Calculate the base quotes around the mid-price.
        base_bid_tick = mid_price_tick - half_spread_tick
        base_ask_tick = mid_price_tick + half_spread_tick
        
        # Apply the inventory shift to both bid and ask.
        bid_price_tick = np.round(base_bid_tick - inventory_shift)
        ask_price_tick = np.round(base_ask_tick - inventory_shift)
        
        # Convert ticks back to a price value.
        bid_price = bid_price_tick * tick_size
        ask_price = ask_price_tick * tick_size

        #--------------------------------------------------------
        # Updates quotes in the market.

        # Cancel existing orders if they don't match our newly calculated prices.
        order_values = orders.values()
        while order_values.has_next():
            order = order_values.get()
            if order.cancellable:
                if (order.side == BUY and order.price != bid_price) or \
                   (order.side == SELL and order.price != ask_price):
                    hbt.cancel(0, order.order_id, False)

        # Submit new orders if our position is within limits and no order
        # already exists at the target price.
        if position < max_position and np.isfinite(bid_price):
            # Using price as the order ID is a common technique for simple market makers
            # to prevent placing duplicate orders at the same price level.
            bid_price_as_order_id = round(bid_price / tick_size)
            if bid_price_as_order_id not in orders:
                hbt.submit_buy_order(0, bid_price_as_order_id, bid_price, order_qty, GTX, LIMIT, False)
        
        if position > -max_position and np.isfinite(ask_price):
            ask_price_as_order_id = round(ask_price / tick_size)
            if ask_price_as_order_id not in orders:
                hbt.submit_sell_order(0, ask_price_as_order_id, ask_price, order_qty, GTX, LIMIT, False)

        #--------------------------------------------------------
        # Records variables and stats for analysis.

        out[t].inventory_shift = inventory_shift
        out[t].mid_price_tick = mid_price_tick
        
        t += 1
        
        # Ensure we don't exceed the allocated array size.
        if t >= len(out):
            raise Exception("Output array size exceeded.")

        # Records the current state for overall performance calculation.
        recorder.record(hbt)
        
    return out[:t]

class Model:
    """
    A wrapper class for the Ho and Stoll market making strategy
    to make it compatible with the backtesting framework.
    """
    def __init__(self):
        """
        Initializes the Model. Parameters are hardcoded inside the
        Numba-jitted function for performance, following the template's pattern.
        """
        pass
    
    def run(self, hbt, recorder):
        """
        Executes the Ho and Stoll market making strategy.
        
        Args:
            hbt: The HftBacktest instance.
            recorder: The Recorder instance for tracking performance.
        """
        return ho_stoll_market_maker(hbt, recorder)