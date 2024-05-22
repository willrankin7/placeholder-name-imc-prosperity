from datamodel import OrderDepth, UserId, TradingState, Order, Symbol, Trade, Listing, Observation, ProsperityEncoder
from typing import List, Any
import string
import jsonpickle
import json
import math
import numpy as np

class trader_data_python_object:
    def __init__(self):
        self.starfruit_prices = []
        self.past_spreads = []
        self.bs_past_spreads = []
        self.bot_positions_roses = {}
        self.last_trades = {}

# Constant Variables
POSITION_LIMITS = {"AMETHYSTS": 20, "STARFRUIT": 20, "ORCHIDS": 100, "CHOCOLATE": 250, "STRAWBERRIES": 350, "ROSES": 60, "GIFT_BASKET": 60, "COCONUT": 300, "COCONUT_COUPON": 600}
PRODUCTS = ["AMETHYSTS", "STARFRUIT", "ORCHIDS", "CHOCOLATE", "STRAWBERRIES", "ROSES", "GIFT_BASKET", "COCONUT", "COCONUT_COUPON"]

# STARFRUIT CONSTANTS
STARFRUIT_COEFFICIENTS = [0.143715, 0.135610, 0.173796, 0.232805, 0.313696]
STARFRUIT_INTERCEPT =  1.908945
STARFRUIT_PRICE_DEPTH = 5
STARFRUIT_SPREAD = [1,1]

# AMETHYSTS CONSTANTS
AMETHYSTS_STABLE_PRICE = 10000
AMETHYSTS_SPREAD = [1,1]

# ORCHID CONSTANTS
ORCHID_ARB_MARGIN = 0.9

# ROUND 3 CONSTANTS
BASKET_COMPONENTS = {"CHOCOLATE": 4, "STRAWBERRIES": 6, "ROSES": 1}
BASKET_MEAN_PREMIUM = 379.49
BASKET_PREMIUM_STD = 76.42
BASKET_STDEV_MULTIPLIER = 0.5

# ROUND 4 CONSTANTS
COUPON_STRIKE_PRICE = 10000
CURRENT_DAY = 4
VOLATILITY_COCONUT = 0.16
COCONUT_SPREAD = [-5,5]
COCONUT_SPREAD_DEPTH = 12
"""
BT Values:
    - 5 seems to work best for spread
    - 12 seems to work best for depth (487k)
"""
class Trader:
    def follow_rhianna(self, order_book, rhianna_buy, rhianna_sell, current_positions, rhianna_position):
        orders = []
        if rhianna_position == 0:
            best_ask, _ = list(order_book["ROSES"].sell_orders.items())[0]
            max_buy = POSITION_LIMITS["ROSES"] - current_positions["ROSES"]
            orders.append(Order("ROSES", best_ask + 2, max_buy))
        if rhianna_position < 0:
            best_bid, _ = list(order_book["ROSES"].buy_orders.items())[0]
            max_sell = -POSITION_LIMITS["ROSES"] - current_positions["ROSES"]
            orders.append(Order("ROSES", best_bid - 2, max_sell))
        return orders

    def norm_cdf(self, x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    def black_scholes_call_price(self, S, K, r, T, sigma):
        # S: spot price
        # K: strike price
        # r: risk-free rate
        # T: time to maturity (in years)
        # sigma: volatility of underlying asset
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)

        return S * self.norm_cdf(d1) - K * math.exp(-r * T) * self.norm_cdf(d2)

    def compute_coconut_price(self, order_book, day, timestamp):
        # Calculate fair price for option
        # Spot price of coconut
        spot_price = (list(order_book["COCONUT"].buy_orders.items())[0][0] + list(order_book["COCONUT"].sell_orders.items())[0][0]) / 2
        # Strike price of option
        strike_price = COUPON_STRIKE_PRICE
        # Risk-free rate
        rf_rate = 0
        # Time to maturity
        tt_maturity = (250 - (day - 1 + (timestamp/1000000)))/252
        # Volatility
        sigma = VOLATILITY_COCONUT
        # Calculate fair price based on BS model
        fair_price = self.black_scholes_call_price(spot_price, strike_price, rf_rate, tt_maturity, sigma)
 
        return fair_price
    
    def compute_coconut_spread(self, order_book, fair_price):
        try:
            bid_coupon, _ = list(order_book["COCONUT_COUPON"].buy_orders.items())[0]
            ask_coupon, _ = list(order_book["COCONUT_COUPON"].sell_orders.items())[0]
            mid_price_coupon = (bid_coupon + ask_coupon) / 2
        except:
            mid_price_coupon = 0

        logger.print(f"Fair price: {fair_price}, Market price: {mid_price_coupon}")

        price_spread = fair_price - mid_price_coupon
        
        return price_spread

    def compute_coconut_orders(self, order_book, spread, current_positions):
        orders = []
        cpos = current_positions["COCONUT_COUPON"]
        if spread < COCONUT_SPREAD[0]:
            # Compare to the best bid and ask prices
            best_bid, _ = list(order_book["COCONUT_COUPON"].buy_orders.items())[0]
            max_sell = -POSITION_LIMITS["COCONUT_COUPON"] - cpos
            orders.append(Order("COCONUT_COUPON", best_bid, max_sell))

        if spread > COCONUT_SPREAD[1]:
            best_ask, _ = list(order_book["COCONUT_COUPON"].sell_orders.items())[0]
            max_buy = POSITION_LIMITS["COCONUT_COUPON"] - cpos
            orders.append(Order("COCONUT_COUPON", best_ask, max_buy))     
        
        logger.print(f"COCONUT_COUPON Orders: {orders}, cpos: {cpos}")
        
        return orders

    def compute_basket_spread(self, order_book):
        mid_prices = {"CHOCOLATE": 0, "STRAWBERRIES": 0, "ROSES": 0, "GIFT_BASKET": 0}
        best_bids = {"CHOCOLATE": 0, "STRAWBERRIES": 0, "ROSES": 0, "GIFT_BASKET": 0}
        best_asks = {"CHOCOLATE": 0, "STRAWBERRIES": 0, "ROSES": 0, "GIFT_BASKET": 0}

        for product in mid_prices:
            best_bids[product], _ = list(order_book[product].buy_orders.items())[0]
            best_asks[product], _ = list(order_book[product].sell_orders.items())[0]
            mid_prices[product] = (best_bids[product] + best_asks[product]) / 2

        # Calculate the residual of the basket
        spread = mid_prices["GIFT_BASKET"] - mid_prices["CHOCOLATE"]*4 - mid_prices["STRAWBERRIES"]*6 - mid_prices["ROSES"] - BASKET_MEAN_PREMIUM
        return spread
   
    def compute_basket_orders(self, order_book, current_positions):
        orders = {"GIFT_BASKET": [], "CHOCOLATE": [], "STRAWBERRIES": [], "ROSES": []}
        best_bids = {"CHOCOLATE": 0, "STRAWBERRIES": 0, "ROSES": 0, "GIFT_BASKET": 0}
        best_asks = {"CHOCOLATE": 0, "STRAWBERRIES": 0, "ROSES": 0, "GIFT_BASKET": 0}

        for product in best_bids:
            best_bids[product], _ = list(order_book[product].buy_orders.items())[0]
            best_asks[product], _ = list(order_book[product].sell_orders.items())[0]

        trade_at = BASKET_PREMIUM_STD * BASKET_STDEV_MULTIPLIER

        average_spread = self.compute_basket_spread(order_book)

        # If residual is positive, sell baskets
        if average_spread > trade_at:
            order_vol_baskets = -POSITION_LIMITS["GIFT_BASKET"] - current_positions["GIFT_BASKET"]
            
            if order_vol_baskets != 0:
                orders["GIFT_BASKET"].append(Order("GIFT_BASKET", best_bids["GIFT_BASKET"] - 2, order_vol_baskets))

        elif average_spread < -trade_at:
            order_vol_baskets = POSITION_LIMITS["GIFT_BASKET"] - current_positions["GIFT_BASKET"]
            
            if order_vol_baskets != 0:
                orders["GIFT_BASKET"].append(Order("GIFT_BASKET", best_asks["GIFT_BASKET"] + 2, order_vol_baskets))
        return orders
 
    def compute_orchids_orders(self, product, order_book, observation):
        orders = []
        os_ask_price = observation.askPrice
        transport = observation.transportFees
        import_tariff = observation.importTariff

        best_bid_local, _ = list(order_book[product].buy_orders.items())[0]
        net_os_ask_price = os_ask_price + transport + import_tariff
                
        orders.append(Order(product, round(net_os_ask_price + 1), -100))
        return orders

    def compute_orders(self, product, fair_price, order_book, cpos, spread):
        orders = []
        ipos = cpos
        sorders = order_book[product].sell_orders.items()
        borders = order_book[product].buy_orders.items()
        best_buy, vol = list(order_book[product].buy_orders.items())[0]
        best_sell, vol = list(order_book[product].sell_orders.items())[0]

        undercut_buy = best_buy + 1
        undercut_sell = best_sell - 1

        optimal_buy = fair_price - spread[0]
        optimal_sell = fair_price + spread[1]

        cumulative_buy = 0
        cumulative_sell = 0

        bid_price = min(undercut_buy, optimal_buy)
        ask_price = max(undercut_sell, optimal_sell)

        if len(sorders) != 0:
            for price, volume in sorders:
                if (price < fair_price or price == fair_price and cpos < 0) and cpos < POSITION_LIMITS[product]:
                    current_buy_position = ipos + cumulative_buy
                    order_qty = min(-volume, POSITION_LIMITS[product] - cpos, POSITION_LIMITS[product] - current_buy_position)
                    orders.append(Order(product, price, order_qty))
                    cpos += order_qty
                    cumulative_buy += order_qty

        if len(borders) != 0:
            for price, volume in borders:
                if (price > fair_price or price == fair_price and cpos > 0) and cpos > -POSITION_LIMITS[product] and ipos > -POSITION_LIMITS[product]:
                    current_sell_position = ipos + cumulative_sell
                    order_qty = max(-volume, -POSITION_LIMITS[product] - cpos, -POSITION_LIMITS[product] - current_sell_position)
                    orders.append(Order(product, price, order_qty))
                    cpos += order_qty
                    cumulative_sell += order_qty
        border, sorder = None, None
        border, sorder = self.market_make(product, fair_price, bid_price, ask_price, cumulative_buy, cumulative_sell, cpos, ipos)
        if border:
            orders.append(border)
        if sorder:
            orders.append(sorder)

        #print(f"PRODUCT: {product}, Orders: {orders}, ipos: {ipos}")
        return orders
    
    def market_make(self, product, mid_price, bid_price, ask_price, cumulative_buy, cumulative_sell, cpos, ipos):
        border_price = None
        orders = []
        current_buy_position = ipos + cumulative_buy
        if cpos < POSITION_LIMITS[product]:
            border_qty = min(2*POSITION_LIMITS[product], POSITION_LIMITS[product] - cpos, POSITION_LIMITS[product] - current_buy_position)
            border_price = bid_price

        sorder_price = None
        current_sell_position = ipos + cumulative_sell
        if cpos > -POSITION_LIMITS[product]:
            sorder_qty = max(-2*POSITION_LIMITS[product], -POSITION_LIMITS[product] - cpos, -POSITION_LIMITS[product] - current_sell_position)
            sorder_price = ask_price
        
        if border_price:
            border = (Order(product, border_price, border_qty))
        else:
            border = None
        if sorder_price:
            sorder = (Order(product, sorder_price, sorder_qty))
        else:
            sorder = None
        return border, sorder

    def calculate_next_price_starfruit(self, last_prices, coefficients, intercept, depth):
        # Calculate the next price of starfruit based on the last 4 prices
        next_price = intercept
        for i in range(depth):
            next_price += last_prices[i] * coefficients[i]
        return int(round(next_price))

    def run(self, state: TradingState):
        # Initialial conditions and variables.
        try:
            td = jsonpickle.decode(state.traderData)
        except:
            td = trader_data_python_object()
        current_positions = {}
        for product in PRODUCTS:
            current_positions[product] = state.position.get(product, 0)
        result={}   
        
        # Amethysts
        result["AMETHYSTS"] = self.compute_orders("AMETHYSTS", AMETHYSTS_STABLE_PRICE, state.order_depths, current_positions["AMETHYSTS"], AMETHYSTS_SPREAD)
        
        # Starfruit
        # Update the last DEPTH prices of starfruit
        best_ask_starfruit, vol = list(state.order_depths["STARFRUIT"].sell_orders.items())[0]
        best_bid_starfruit, vol = list(state.order_depths["STARFRUIT"].buy_orders.items())[0]
        mid_price_starfruit = (best_ask_starfruit + best_bid_starfruit) / 2

        if len(td.starfruit_prices) == STARFRUIT_PRICE_DEPTH:
            td.starfruit_prices.pop(0)
    
        td.starfruit_prices.append((mid_price_starfruit))
        last_prices = td.starfruit_prices

        # If we have DEPTH price points, calculate the next price and place orders
        if len(last_prices) == STARFRUIT_PRICE_DEPTH:
            starfruit_next_price = self.calculate_next_price_starfruit(last_prices, STARFRUIT_COEFFICIENTS, STARFRUIT_INTERCEPT, STARFRUIT_PRICE_DEPTH)
            result["STARFRUIT"] = self.compute_orders("STARFRUIT", starfruit_next_price, state.order_depths, current_positions["STARFRUIT"], STARFRUIT_SPREAD)

        # ORCHIDS
        result["ORCHIDS"] = self.compute_orchids_orders("ORCHIDS", state.order_depths, state.observations.conversionObservations["ORCHIDS"])

        conversions = -current_positions["ORCHIDS"]
        
        # BASKETS
        basket_spread = self.compute_basket_spread(state.order_depths)

        basket_orders = self.compute_basket_orders(state.order_depths, current_positions)
        result["GIFT_BASKET"] = basket_orders["GIFT_BASKET"]

        # ROSES
        rhianna_buy = False
        rhianna_sell = False
        state.market_trades["ROSES"] = state.market_trades.get("ROSES", [])
        td.last_trades["ROSES"] = td.last_trades.get("ROSES", [])
        if str(td.last_trades["ROSES"]) != str(state.market_trades["ROSES"]):
            logger.print("UPDATE SIGNAL")
            for trade in state.market_trades["ROSES"]:
                td.bot_positions_roses[trade.seller] = td.bot_positions_roses.get(trade.seller, 0) - trade.quantity
                td.bot_positions_roses[trade.buyer] = td.bot_positions_roses.get(trade.buyer, 0) + trade.quantity
        try:
            rhianna_position = td.bot_positions_roses["Rhianna"]
            result["ROSES"] = self.follow_rhianna(state.order_depths, rhianna_buy, rhianna_sell, current_positions, rhianna_position)
        except:
            pass



        # Coconuts
        
        day = 5 # Change this in future!
        coconut_fair_price = self.compute_coconut_price(state.order_depths, day, state.timestamp)
        coconut_spread = self.compute_coconut_spread(state.order_depths, coconut_fair_price)

        if len(td.bs_past_spreads) == COCONUT_SPREAD_DEPTH:
            td.bs_past_spreads.pop(0)
        td.bs_past_spreads.append(coconut_spread)

        avg_coconut_spread = np.mean(td.bs_past_spreads) + 0.92

        result["COCONUT_COUPON"] = self.compute_coconut_orders(state.order_depths, avg_coconut_spread, current_positions)
        
        td.last_trades = state.market_trades

        # Encode traderData to be passed to the next round
        traderData = jsonpickle.encode(td)
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData
    

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(self.to_json([
            self.compress_state(state, ""),
            self.compress_orders(orders),
            conversions,
            "",
            "",
        ]))

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ]))

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing["symbol"], listing["product"], listing["denomination"]])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append([
                    trade.symbol,
                    trade.price,
                    trade.quantity,
                    trade.buyer,
                    trade.seller,
                    trade.timestamp,
                ])

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sunlight,
                observation.humidity,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[:max_length - 3] + "..."

logger = Logger()