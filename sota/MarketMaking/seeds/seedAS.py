import numpy as np
from numba import njit
from hftbacktest import BUY, SELL, GTX, LIMIT
from hftbacktest import BUY_EVENT

out_dtype = np.dtype([
    ('half_spread_tick', 'f8'),
    ('skew', 'f8'),
    ('volatility', 'f8'),
    ('A', 'f8'),
    ('k', 'f8')
])

@njit
def measure_trading_intensity(order_arrival_depth, out):
    max_tick = 0
    for depth in order_arrival_depth:
        if not np.isfinite(depth):
            continue
        tick = round(depth / 0.5) - 1
        if tick < 0 or tick >= len(out):
            continue
        out[:tick] += 1
        max_tick = max(max_tick, tick)
    return out[:max_tick]

@njit
def linear_regression(x, y):
    sx = np.sum(x); sy = np.sum(y)
    sx2 = np.sum(x ** 2); sxy = np.sum(x * y)
    w = len(x)
    slope = (w * sxy - sx * sy) / (w * sx2 - sx**2)
    intercept = (sy - slope * sx) / w
    return slope, intercept

# --OPTION--
@njit
def as_market_maker(hbt, recorder):
    """
    Avellaneda–Stoikov market maker wired like the GLFT seed, but tuned for higher ROI:
      - narrower spreads (smaller gamma, tau)
      - peg quotes to top-of-book (best bid/ask) to maximize maker fills
      - larger order size with tighter inventory cap
    """
    tick_size = hbt.depth(0).tick_size

    arrival_depth = np.full(10_000_000, np.nan, np.float64)
    mid_price_chg = np.full(10_000_000, np.nan, np.float64)
    out = np.zeros(10_000_000, out_dtype)

    t = 0
    prev_mid_price_tick = np.nan
    mid_price_tick = np.nan

    tmp = np.zeros(500, np.float64)
    ticks = np.arange(len(tmp)) + 0.5

    A = np.nan
    k = np.nan
    volatility = np.nan

    gamma = 0.02
    tau   = 0.2   

    order_qty = 0.01   
    max_position = 2     

    peg_to_top = True

    while hbt.elapse(100_000_000) == 0:
        depth = hbt.depth(0)

        if not np.isnan(mid_price_tick):
            max_depth = -np.inf
            for last_trade in hbt.last_trades(0):
                trade_price_tick = last_trade.px / tick_size
                if last_trade.ev & BUY_EVENT == BUY_EVENT:
                    max_depth = np.nanmax([trade_price_tick - mid_price_tick, max_depth])
                else:
                    max_depth = np.nanmax([mid_price_tick - trade_price_tick, max_depth])
            arrival_depth[t] = max_depth

        hbt.clear_last_trades(0)
        hbt.clear_inactive_orders(0)

        depth = hbt.depth(0)
        position = hbt.position(0)
        orders = hbt.orders(0)

        best_bid_tick = depth.best_bid_tick
        best_ask_tick = depth.best_ask_tick
        prev_mid_price_tick = mid_price_tick
        mid_price_tick = (best_bid_tick + best_ask_tick) / 2.0

        mid_price_chg[t] = mid_price_tick - prev_mid_price_tick

        if t % 50 == 0:
            if t >= 6000 - 1:
                tmp[:] = 0.0
                lam = measure_trading_intensity(arrival_depth[t + 1 - 6000:t + 1], tmp)
                if len(lam) > 2:
                    lam = lam[:70] / 600.0
                    valid = lam > 0
                    if np.any(valid) and np.sum(valid) > 2:
                        x = ticks[:len(lam)][valid]
                        y = np.log(lam[valid])
                        k_, logA = linear_regression(x, y)
                        A = np.exp(logA)
                        k = -k_ if k_ < 0 else np.abs(k_)

                volatility = np.nanstd(mid_price_chg[t + 1 - 6000:t + 1]) * np.sqrt(10.0)

        if not np.isfinite(volatility):
            out[t].half_spread_tick = np.nan
            out[t].skew = np.nan
            out[t].volatility = volatility
            out[t].A = A
            out[t].k = k
            t += 1
            if t >= len(arrival_depth) or t >= len(mid_price_chg) or t >= len(out):
                raise Exception
            recorder.record(hbt)
            continue

        k_eff = k if (np.isfinite(k) and k > 1e-12) else 1e-6

        sig2_tau = (volatility * volatility) * tau
        half_spread_tick = (1.0 / gamma) * np.log(1.0 + (gamma / k_eff)) + 0.5 * gamma * sig2_tau

        skew_coeff = gamma * sig2_tau
        reservation_price_tick = mid_price_tick - skew_coeff * position

        raw_bid_tick = np.round(reservation_price_tick - half_spread_tick)
        raw_ask_tick = np.round(reservation_price_tick + half_spread_tick)

        if peg_to_top:
            bid_price_tick = best_bid_tick 
            ask_price_tick = best_ask_tick
        else:
            bid_price_tick = np.minimum(raw_bid_tick, best_bid_tick)
            ask_price_tick = np.maximum(raw_ask_tick, best_ask_tick)

        bid_price = bid_price_tick * tick_size
        ask_price = ask_price_tick * tick_size

        order_values = orders.values()
        while order_values.has_next():
            order = order_values.get()
            if order.cancellable:
                if ((order.side == BUY and order.price != bid_price) or
                    (order.side == SELL and order.price != ask_price)):
                    hbt.cancel(0, order.order_id, False)

        if position < max_position and np.isfinite(bid_price):
            bid_id = round(bid_price / tick_size)
            if bid_id not in orders:
                hbt.submit_buy_order(0, bid_id, bid_price, order_qty, GTX, LIMIT, False)

        if position > -max_position and np.isfinite(ask_price):
            ask_id = round(ask_price / tick_size)
            if ask_id not in orders:
                hbt.submit_sell_order(0, ask_id, ask_price, order_qty, GTX, LIMIT, False)

        out[t].half_spread_tick = half_spread_tick
        out[t].skew = skew_coeff
        out[t].volatility = volatility
        out[t].A = A
        out[t].k = k

        t += 1
        if t >= len(arrival_depth) or t >= len(mid_price_chg) or t >= len(out):
            raise Exception

        recorder.record(hbt)

    return out[:t]

class Model:
    """
    Avellaneda–Stoikov Market Making Strategy
    """

    def __init__(self):
        pass

    def run(self, hbt, recorder):
        return as_market_maker(hbt, recorder)
