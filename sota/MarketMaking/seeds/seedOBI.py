import numpy as np
from numba import njit, uint64
from numba.typed import Dict
from hftbacktest import BUY, SELL, GTX, LIMIT
from hftbacktest import BUY_EVENT
import logging

out_dtype = np.dtype([
    ('half_spread_tick', 'f8'),
    ('skew', 'f8'),
    ('imbalance', 'f8'),
    ('mid_price', 'f8'),
    ('position', 'f8')
])

# --OPTION--
@njit
def obi_mm(hbt, stat, half_spread, skew, c1, looking_depth, interval, window, order_qty_dollar, max_position_dollar, grid_num, grid_interval, roi_lb, roi_ub):
    """
    Order Book Imbalance Market Making Strategy
    """
    asset_no = 0
    imbalance_timeseries = np.full(30_000_000, np.nan, np.float64)

    tick_size = hbt.depth(0).tick_size
    lot_size = hbt.depth(0).lot_size

    t = 0
    roi_lb_tick = int(round(roi_lb / tick_size))
    roi_ub_tick = int(round(roi_ub / tick_size))

    while hbt.elapse(interval) == 0:
        hbt.clear_inactive_orders(asset_no)

        depth = hbt.depth(asset_no)
        position = hbt.position(asset_no)
        orders = hbt.orders(asset_no)

        best_bid = depth.best_bid
        best_ask = depth.best_ask

        mid_price = (best_bid + best_ask) / 2.0

        sum_ask_qty = 0.0
        from_tick = max(depth.best_ask_tick, roi_lb_tick)
        upto_tick = min(int(np.floor(mid_price * (1 + looking_depth) / tick_size)), roi_ub_tick)
        for price_tick in range(from_tick, upto_tick):
            sum_ask_qty += depth.ask_depth[price_tick - roi_lb_tick]

        sum_bid_qty = 0.0
        from_tick = min(depth.best_bid_tick, roi_ub_tick)
        upto_tick = max(int(np.ceil(mid_price * (1 - looking_depth) / tick_size)), roi_lb_tick)
        for price_tick in range(from_tick, upto_tick, -1):
            sum_bid_qty += depth.bid_depth[price_tick - roi_lb_tick]

        imbalance_timeseries[t] = sum_bid_qty - sum_ask_qty

        # Standardizes the order book imbalance timeseries for a given window
        m = np.nanmean(imbalance_timeseries[max(0, t + 1 - window):t + 1])
        s = np.nanstd(imbalance_timeseries[max(0, t + 1 - window):t + 1])
        alpha = np.divide(imbalance_timeseries[t] - m, s)

        #--------------------------------------------------------
        # Computes bid price and ask price.

        order_qty = max(round((order_qty_dollar / mid_price) / lot_size) * lot_size, lot_size)
        fair_price = mid_price + c1 * alpha

        normalized_position = position / order_qty

        reservation_price = fair_price - skew * normalized_position

        bid_price = min(np.round(reservation_price - half_spread), best_bid)
        ask_price = max(np.round(reservation_price + half_spread), best_ask)

        bid_price = np.floor(bid_price / tick_size) * tick_size
        ask_price = np.ceil(ask_price / tick_size) * tick_size

        #--------------------------------------------------------
        # Updates quotes.

        # Creates a new grid for buy orders.
        new_bid_orders = Dict.empty(np.uint64, np.float64)
        if position * mid_price < max_position_dollar and np.isfinite(bid_price):
            for i in range(grid_num):
                bid_price_tick = round(bid_price / tick_size)

                # order price in tick is used as order id.
                new_bid_orders[uint64(bid_price_tick)] = bid_price

                bid_price -= grid_interval

        # Creates a new grid for sell orders.
        new_ask_orders = Dict.empty(np.uint64, np.float64)
        if position * mid_price > -max_position_dollar and np.isfinite(ask_price):
            for i in range(grid_num):
                ask_price_tick = round(ask_price / tick_size)

                # order price in tick is used as order id.
                new_ask_orders[uint64(ask_price_tick)] = ask_price

                ask_price += grid_interval

        order_values = orders.values()
        while order_values.has_next():
            order = order_values.get()
            # Cancels if a working order is not in the new grid.
            if order.cancellable:
                if (
                    (order.side == BUY and order.order_id not in new_bid_orders)
                    or (order.side == SELL and order.order_id not in new_ask_orders)
                ):
                    hbt.cancel(asset_no, order.order_id, False)

        for order_id, order_price in new_bid_orders.items():
            # Posts a new buy order if there is no working order at the price on the new grid.
            if order_id not in orders:
                hbt.submit_buy_order(asset_no, order_id, order_price, order_qty, GTX, LIMIT, False)

        for order_id, order_price in new_ask_orders.items():
            # Posts a new sell order if there is no working order at the price on the new grid.
            if order_id not in orders:
                hbt.submit_sell_order(asset_no, order_id, order_price, order_qty, GTX, LIMIT, False)

        t += 1

        if t >= len(imbalance_timeseries):
            raise Exception

        # Records the current state for stat calculation.
        stat.record(hbt)

# --OPTION--
# Order Book Imbalance (OBI) Market Making Strategy
class Model:
    """
    Order Book Imbalance Market Making Strategy
    
    This model uses order book imbalance as an alpha signal to dynamically
    adjust bid and ask prices. The strategy:
    
    1. Calculates order book imbalance by comparing bid vs ask quantities
    2. Standardizes the imbalance using a rolling window
    3. Uses the standardized imbalance to adjust spreads and skew
    4. Posts orders with dynamic pricing based on market microstructure
    
    Key features:
    - Dynamic spread adjustment based on order flow imbalance
    - Inventory management through position-based skew
    - Standardized imbalance calculation for robust signals
    - Configurable depth analysis and update intervals
    """
    
    def __init__(self):
        """
        Initialize the OBI model with default parameters.
        
        The model uses order book imbalance as the primary alpha signal
        and adjusts pricing dynamically based on market microstructure.
        """
        pass
    
    def run(self, hbt, recorder):
        """
        Execute the Order Book Imbalance market making strategy.
        
        This method implements the core OBI strategy that:
        - Calculates order book imbalance in real-time
        - Standardizes the imbalance signal
        - Adjusts bid/ask spreads based on imbalance
        - Manages inventory through dynamic skew
        
        Args:
            hbt: HftBacktest instance
            recorder: Recorder instance for tracking performance
        """
        # Strategy parameters from the documentation - this is what LLMGE will tweak
        half_spread = 10
        skew = 2
        c1 = 20
        looking_depth = 0.001  # 0.1% from the mid price
        interval = 500_000_000  # 500ms
        window = 600_000_000_000 // interval  # 10min
        order_qty_dollar = 25_000
        max_position_dollar = order_qty_dollar * 20
        grid_num = 1
        grid_interval = hbt.depth(0).tick_size
        roi_lb = 0.0
        roi_ub = 100000.0
        
        return obi_mm(
            hbt,
            recorder,
            half_spread,
            skew,
            c1,
            looking_depth,
            interval,
            window,
            order_qty_dollar,
            max_position_dollar,
            grid_num,
            grid_interval,
            roi_lb,
            roi_ub
        )