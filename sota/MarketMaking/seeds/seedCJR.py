import numpy as np
from numba import njit
from hftbacktest import BUY, SELL, GTX, LIMIT


@njit
def h_exp(delta, kappa):
    return np.exp(-kappa * max(0.0, delta))

@njit
def h_exp_prime(delta, kappa):
    delta = max(0.0, delta)
    return -kappa * np.exp(-kappa * delta)

@njit
def h_exp_second(delta, kappa):
    delta = max(0.0, delta)
    return (kappa**2) * np.exp(-kappa * delta)

@njit
def delta0_exponential(kappa):
    return 1.0 / max(1e-8, kappa)

@njit
def B_coeff(delta0, kappa, lam):
    h0 = h_exp(delta0, kappa)
    hp = h_exp_prime(delta0, kappa)
    hpp = h_exp_second(delta0, kappa)
    denom = 2.0 * lam * h0 + delta0 * hpp
    return hp / max(1e-12, denom)

@njit
def E_int_alpha(alpha_now, lam_buy, lam_sell, tau, zeta, rho, eps_buy_mean, eps_sell_mean):
    if tau <= 0:
        return 0.0
    term_cur = alpha_now * (1.0 - np.exp(-zeta * tau)) / zeta
    int_ba = (tau / zeta) - (1.0 - np.exp(-zeta * tau)) / (zeta**2)
    term_jumps = rho * (eps_buy_mean * lam_buy - eps_sell_mean * lam_sell) * int_ba
    return term_cur + term_jumps

@njit
def optimal_deltas(t, S, q, X, lam_buy, lam_sell, k_buy, k_sell, alpha, 
                  beta, theta, eta, nu, rho, xi, theta_k, eta_k, nu_k, 
                  zeta, sigma_a, eps_buy_mean, eps_sell_mean, sigma_s, phi, T):
    tau = max(0.0, T - t)
    
    d0_sell = delta0_exponential(k_sell)
    d0_buy = delta0_exponential(k_buy)
    B_sell = B_coeff(d0_sell, k_sell, lam_sell)
    B_buy = B_coeff(d0_buy, k_buy, lam_buy)
    E_alpha = E_int_alpha(alpha, lam_buy, lam_sell, tau, zeta, rho, eps_buy_mean, eps_sell_mean)
    b_alpha0 = (1.0 - np.exp(-zeta * tau)) / zeta
    inv_term_sell = (1.0 - 2.0 * q) * tau
    inv_term_buy = (1.0 + 2.0 * q) * tau
    dir_sell = (+eps_sell_mean) * b_alpha0
    dir_buy = (-eps_buy_mean) * b_alpha0
    delta_sell = d0_sell + B_sell * (E_alpha + phi * (dir_sell + inv_term_sell))
    delta_buy = d0_buy + B_buy * (E_alpha + phi * (dir_buy + inv_term_buy))
    return max(0.0, delta_sell), max(0.0, delta_buy)

# --OPTION--
@njit
def cjr_market_maker(hbt, recorder):
    tick_size = hbt.depth(0).tick_size
    
    # CJR parameters
    beta = 5.0
    theta = 5.0
    eta = 2.0
    nu = 1.0
    rho = 0.5
    xi = 2.0
    theta_k = 2.0
    eta_k = 0.6
    nu_k = 0.4
    zeta = 5.0
    sigma_a = 0.0
    eps_buy_mean = 0.02
    eps_sell_mean = 0.02
    sigma_s = 0.1
    phi = 0.002
    T = 60.0
    
    # Initialize CJR state
    t = 0.0
    S = 100.0
    q = 0.0
    X = 0.0
    lam_buy = 5.0
    lam_sell = 5.0
    k_buy = 2.0
    k_sell = 2.0
    alpha = 0.0
    
    order_qty = 1
    max_position = 50
    
    # Checks every 100 milliseconds like GLTP
    while hbt.elapse(100_000_000) == 0:
        hbt.clear_last_trades(0)
        hbt.clear_inactive_orders(0)
        
        depth = hbt.depth(0)
        position = hbt.position(0)
        orders = hbt.orders(0)
        
        best_bid_tick = depth.best_bid_tick
        best_ask_tick = depth.best_ask_tick
        mid_price_tick = (best_bid_tick + best_ask_tick) / 2.0
        
        if mid_price_tick > 0:
            S = mid_price_tick * tick_size
        
        # Update CJR state
        dt = 0.1  # 100ms
        t += dt
        q = position
        
        # Simple state evolution (without stochastic terms for deterministic behavior)
        # In practice, you would use proper random number generation
        
        # Compute optimal deltas
        delta_sell, delta_buy = optimal_deltas(
            t, S, q, X, lam_buy, lam_sell, k_buy, k_sell, alpha,
            beta, theta, eta, nu, rho, xi, theta_k, eta_k, nu_k,
            zeta, sigma_a, eps_buy_mean, eps_sell_mean, sigma_s, phi, T
        )
        
        # Calculate bid and ask prices
        reservation_price = S
        bid_price_tick = np.minimum(np.round((reservation_price - delta_buy) / tick_size), best_bid_tick)
        ask_price_tick = np.maximum(np.round((reservation_price + delta_sell) / tick_size), best_ask_tick)
        
        bid_price = bid_price_tick * tick_size
        ask_price = ask_price_tick * tick_size
        
        # Cancel orders if they differ from the updated bid and ask prices
        order_values = orders.values()
        while order_values.has_next():
            order = order_values.get()
            if order.cancellable:
                if (
                    (order.side == BUY and order.price != bid_price)
                    or (order.side == SELL and order.price != ask_price)
                ):
                    hbt.cancel(0, order.order_id, False)
        
        # Submit new orders if within position limits
        if position < max_position and np.isfinite(bid_price):
            bid_price_as_order_id = round(bid_price / tick_size)
            if bid_price_as_order_id not in orders:
                hbt.submit_buy_order(0, bid_price_as_order_id, bid_price, order_qty, GTX, LIMIT, False)
        
        if position > -max_position and np.isfinite(ask_price):
            ask_price_as_order_id = round(ask_price / tick_size)
            if ask_price_as_order_id not in orders:
                hbt.submit_sell_order(0, ask_price_as_order_id, ask_price, order_qty, GTX, LIMIT, False)
        
        # Records the current state for stat calculation
        recorder.record(hbt)
    
    return None


class Model:
    """
    CJR Market Making Strategy
    
    This model implements the Cartea-Jaimungal-Ricci optimal market making strategy
    with stochastic volatility and order flow.
    """
    
    def __init__(self):
        """
        Initialize the CJR model with trading parameters.
        """
        pass
    
    def run(self, hbt, recorder):
        """
        Execute the CJR market making strategy.
        
        This method should be called with hftbacktest's backtester and recorder.
        It's decorated with @njit when called from the evaluation script.
        
        Args:
            hbt: HftBacktest instance
            recorder: Recorder instance for tracking performance
        """
        return cjr_market_maker(hbt, recorder)
