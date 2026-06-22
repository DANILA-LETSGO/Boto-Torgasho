//+------------------------------------------------------------------+
//|                                          Antigravity_Scalper.mq4 |
//|                             Copyright 2026, Antigravity AI Team. |
//|                                             https://antigravity  |
//+------------------------------------------------------------------+
#property copyright   "Copyright 2026, Antigravity AI Team."
#property link        "https://antigravity"
#property version     "1.00"
#property description "Bollinger Bands & RSI Scalper with ATR Risk Management"
#property strict

//--- Input Parameters - Strategy
input string      Section_Strategy  = "=== STRATEGY SETTINGS ==="; // ----------------------------
input int         BBPeriod          = 20;         // Bollinger Bands Period
input double      BBDeviation       = 2.0;        // Bollinger Bands Deviation
input int         RSIPeriod         = 14;         // RSI Period
input double      RSIOverbought     = 70.0;       // RSI Overbought Level
input double      RSIOversold       = 30.0;       // RSI Oversold Level
input bool        UseRSIFilter      = true;       // Use RSI filter for entries
input bool        UseTrendFilter    = false;      // Filter trades by long-term trend (200 EMA)
input int         TrendEMAPeriod    = 200;        // Trend Filter EMA Period (if UseTrendFilter=true)

//--- Input Parameters - Risk Management
input string      Section_Risk      = "=== RISK & MONEY MANAGEMENT ==="; // ----------------------------
input double      RiskPercent       = 1.5;        // Risk Percent per Trade (0 for Fixed Lots)
input double      FixedLotSize      = 0.1;        // Fixed Lot Size (if RiskPercent = 0)
input int         ATRPeriod         = 14;         // ATR Period for SL/TP
input double      ATR_SL_Multiplier = 2.0;        // ATR Multiplier for Stop Loss
input double      ATR_TP_Multiplier = 2.0;        // ATR Multiplier for Take Profit
input double      MinStopLossPoints = 100.0;      // Minimum Stop Loss (in Points)
input double      MaxStopLossPoints = 800.0;      // Maximum Stop Loss (in Points)
input double      MaxSpreadPoints   = 30.0;       // Maximum allowed Spread (in Points)

//--- Input Parameters - Trade Management (BE / Trailing)
input string      Section_Trailing  = "=== TRADE MANAGEMENT ==="; // ----------------------------
input bool        UseBreakEven      = true;       // Enable Break-Even
input double      BreakEvenStartATR = 1.0;        // Break-Even Start (ATR Multiplier)
input double      BreakEvenBuffer   = 10.0;       // Break-Even Profit Buffer (in Points)
input bool        UseTrailingStop   = true;       // Enable Trailing Stop
input double      TrailingStartATR  = 1.5;        // Trailing Start (ATR Multiplier)
input double      TrailingStepATR   = 0.5;        // Trailing Step (ATR Multiplier)

//--- Input Parameters - Session Filter
input string      Section_Session   = "=== SESSION FILTER ==="; // ----------------------------
input bool        UseSessionFilter  = true;       // Enable Session Filter
input int         StartHour         = 9;          // Start Hour (Broker Time)
input int         StartMinute       = 0;          // Start Minute
input int         EndHour           = 21;         // End Hour (Broker Time)
input int         EndMinute         = 0;          // End Minute

//--- Input Parameters - System Settings
input string      Section_System    = "=== SYSTEM SETTINGS ==="; // ----------------------------
input int         MagicNumber       = 888123;     // Magic Number
input int         Slippage          = 3;          // Slippage (in Points)
input bool        TradeOnNewBarOnly = true;       // Execute trades only at bar open

//--- Global Variables
datetime lastBarTime = 0;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   // Check settings
   if(BBPeriod <= 0 || BBDeviation <= 0)
   {
      Alert("Error: BBPeriod and BBDeviation must be greater than 0!");
      return(INIT_PARAMETERS_INCORRECT);
   }
   
   if(UseRSIFilter && (RSIOverbought <= 50.0 || RSIOversold >= 50.0))
   {
      Alert("Error: RSI Overbought should be > 50 and RSI Oversold should be < 50!");
      return(INIT_PARAMETERS_INCORRECT);
   }

   // Initialize lastBarTime to the current bar time
   lastBarTime = iTime(Symbol(), Period(), 0);
   
   Print("Antigravity_Scalper initialized successfully on ", Symbol(), ", Period: ", Period());
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   Print("Antigravity_Scalper deinitialized. Reason code: ", reason);
}

//+------------------------------------------------------------------+
//| Calculate open positions for current symbol and magic number      |
//+------------------------------------------------------------------+
int GetOpenPositionsCount()
{
   int count = 0;
   for(int i = 0; i < OrdersTotal(); i++)
   {
      if(OrderSelect(i, SELECT_BY_POS, MODE_TRADES))
      {
         if(OrderSymbol() == Symbol() && OrderMagicNumber() == MagicNumber)
         {
            count++;
         }
      }
   }
   return count;
}

//+------------------------------------------------------------------+
//| Calculate optimal lot size based on account risk and Stop Loss   |
//+------------------------------------------------------------------+
double CalculateLotSize(double sl_points)
{
   if(RiskPercent <= 0) 
      return NormalizeDouble(FixedLotSize, 2);
      
   double balance = AccountBalance();
   double riskAmount = balance * (RiskPercent / 100.0);
   
   double tickValue = MarketInfo(Symbol(), MODE_TICKVALUE);
   double tickSize = MarketInfo(Symbol(), MODE_TICKSIZE);
   double lotStep = MarketInfo(Symbol(), MODE_LOTSTEP);
   double minLot = MarketInfo(Symbol(), MODE_MINLOT);
   double maxLot = MarketInfo(Symbol(), MODE_MAXLOT);
   
   if(tickValue <= 0 || tickSize <= 0 || sl_points <= 0)
      return NormalizeDouble(FixedLotSize, 2);
      
   // In MQL4, tickValue is the profit in account currency for 1 lot when the price moves by 1 tickSize.
   // Point value per lot = tickValue * (Point / tickSize)
   double point = Point;
   double pointValue = tickValue * (point / tickSize);
   
   double calculatedLots = riskAmount / (sl_points * pointValue);
   
   // Align to lot step
   calculatedLots = MathFloor(calculatedLots / lotStep) * lotStep;
   
   if(calculatedLots < minLot) calculatedLots = minLot;
   if(calculatedLots > maxLot) calculatedLots = maxLot;
   
   return NormalizeDouble(calculatedLots, 2);
}

//+------------------------------------------------------------------+
//| Check if a new bar has opened                                    |
//+------------------------------------------------------------------+
bool IsNewBar()
{
   datetime currentBarTime = iTime(Symbol(), Period(), 0);
   if(currentBarTime != lastBarTime)
   {
      lastBarTime = currentBarTime;
      return true;
   }
   return false;
}

//+------------------------------------------------------------------+
//| Move Stop Loss to Break-Even when target is reached              |
//+------------------------------------------------------------------+
void ApplyBreakEven()
{
   if(!UseBreakEven) return;
   
   double atr = iATR(Symbol(), Period(), ATRPeriod, 1);
   if(atr <= 0) return;
   
   double breakEvenStartPoints = BreakEvenStartATR * atr / Point;
   double stopLevel = MarketInfo(Symbol(), MODE_STOPLEVEL);
   
   for(int i = OrdersTotal() - 1; i >= 0; i--)
   {
      if(OrderSelect(i, SELECT_BY_POS, MODE_TRADES))
      {
         if(OrderSymbol() == Symbol() && OrderMagicNumber() == MagicNumber)
         {
            if(OrderType() == OP_BUY)
            {
               double currentProfitPoints = (Bid - OrderOpenPrice()) / Point;
               if(currentProfitPoints >= breakEvenStartPoints)
               {
                  double newStopLoss = OrderOpenPrice() + (BreakEvenBuffer * Point);
                  newStopLoss = NormalizeDouble(newStopLoss, Digits);
                  
                  // Check if current stop loss is less than newStopLoss and price is far enough from new SL
                  if((OrderStopLoss() < newStopLoss) && (Bid - newStopLoss) >= (stopLevel * Point))
                  {
                     if(!OrderModify(OrderTicket(), OrderOpenPrice(), newStopLoss, OrderTakeProfit(), 0, Blue))
                     {
                        Print("Error modifying breakeven for BUY order #", OrderTicket(), ": ", GetLastError());
                     }
                     else
                     {
                        Print("Moved BUY order #", OrderTicket(), " to breakeven at ", newStopLoss);
                     }
                  }
               }
            }
            else if(OrderType() == OP_SELL)
            {
               double currentProfitPoints = (OrderOpenPrice() - Ask) / Point;
               if(currentProfitPoints >= breakEvenStartPoints)
               {
                  double newStopLoss = OrderOpenPrice() - (BreakEvenBuffer * Point);
                  newStopLoss = NormalizeDouble(newStopLoss, Digits);
                  
                  // Check if current stop loss is greater than newStopLoss (or 0) and price is far enough from new SL
                  if((OrderStopLoss() > newStopLoss || OrderStopLoss() == 0) && (newStopLoss - Ask) >= (stopLevel * Point))
                  {
                     if(!OrderModify(OrderTicket(), OrderOpenPrice(), newStopLoss, OrderTakeProfit(), 0, Red))
                     {
                        Print("Error modifying breakeven for SELL order #", OrderTicket(), ": ", GetLastError());
                     }
                     else
                     {
                        Print("Moved SELL order #", OrderTicket(), " to breakeven at ", newStopLoss);
                     }
                  }
               }
            }
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Apply trailing stop loss                                         |
//+------------------------------------------------------------------+
void ApplyTrailingStop()
{
   if(!UseTrailingStop) return;
   
   double atr = iATR(Symbol(), Period(), ATRPeriod, 1);
   if(atr <= 0) return;
   
   double trailingStartPoints = TrailingStartATR * atr / Point;
   double trailingStepPoints = TrailingStepATR * atr / Point;
   double stopLevel = MarketInfo(Symbol(), MODE_STOPLEVEL);
   
   if(trailingStartPoints < stopLevel) trailingStartPoints = stopLevel;
   if(trailingStepPoints < 1) trailingStepPoints = 1;
   
   for(int i = OrdersTotal() - 1; i >= 0; i--)
   {
      if(OrderSelect(i, SELECT_BY_POS, MODE_TRADES))
      {
         if(OrderSymbol() == Symbol() && OrderMagicNumber() == MagicNumber)
         {
            if(OrderType() == OP_BUY)
            {
               double currentProfitPoints = (Bid - OrderOpenPrice()) / Point;
               if(currentProfitPoints >= trailingStartPoints)
               {
                  double newStopLoss = Bid - (TrailingStartATR * atr);
                  newStopLoss = NormalizeDouble(newStopLoss, Digits);
                  
                  // Check if the new stop loss is higher than current stop loss
                  if((OrderStopLoss() < newStopLoss || OrderStopLoss() == 0) && (Bid - newStopLoss) >= (stopLevel * Point))
                  {
                     // Only modify if step threshold is met to prevent excessive order modifications
                     if(OrderStopLoss() == 0 || (newStopLoss - OrderStopLoss()) >= (trailingStepPoints * Point))
                     {
                        if(!OrderModify(OrderTicket(), OrderOpenPrice(), newStopLoss, OrderTakeProfit(), 0, Blue))
                        {
                           Print("Error modifying trailing stop for BUY order #", OrderTicket(), ": ", GetLastError());
                        }
                     }
                  }
               }
            }
            else if(OrderType() == OP_SELL)
            {
               double currentProfitPoints = (OrderOpenPrice() - Ask) / Point;
               if(currentProfitPoints >= trailingStartPoints)
               {
                  double newStopLoss = Ask + (TrailingStartATR * atr);
                  newStopLoss = NormalizeDouble(newStopLoss, Digits);
                  
                  // Check if the new stop loss is lower than current stop loss
                  if((OrderStopLoss() > newStopLoss || OrderStopLoss() == 0) && (newStopLoss - Ask) >= (stopLevel * Point))
                  {
                     // Only modify if step threshold is met
                     if(OrderStopLoss() == 0 || (OrderStopLoss() - newStopLoss) >= (trailingStepPoints * Point))
                     {
                        if(!OrderModify(OrderTicket(), OrderOpenPrice(), newStopLoss, OrderTakeProfit(), 0, Red))
                        {
                           Print("Error modifying trailing stop for SELL order #", OrderTicket(), ": ", GetLastError());
                        }
                     }
                  }
               }
            }
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Check if current broker time is within allowed trading hours     |
//+------------------------------------------------------------------+
bool IsWithinTradingHours()
{
   if(!UseSessionFilter) return true;
   
   datetime now = TimeCurrent();
   int currentHour = TimeHour(now);
   int currentMinute = TimeMinute(now);
   
   int nowMinutes = currentHour * 60 + currentMinute;
   int startMinutes = StartHour * 60 + StartMinute;
   int endMinutes = EndHour * 60 + EndMinute;
   
   if(startMinutes < endMinutes)
   {
      return (nowMinutes >= startMinutes && nowMinutes < endMinutes);
   }
   else // Overnight sessions (e.g., from 22:00 to 06:00)
   {
      return (nowMinutes >= startMinutes || nowMinutes < endMinutes);
   }
}

//+------------------------------------------------------------------+
//| OnTick function                                                  |
//+------------------------------------------------------------------+
void OnTick()
{
   // Basic health checks
   if(Bars < 100 || !IsTradeAllowed())
      return;
      
   // 1. Process active trade management (Breakeven & Trailing Stop) on every tick
   ApplyBreakEven();
   ApplyTrailingStop();
   
   // 2. Check if a new bar has opened (if configured to only trade at bar start)
   if(TradeOnNewBarOnly && !IsNewBar())
      return;
      
   // 3. Check if we already have an open position
   if(GetOpenPositionsCount() > 0)
      return;
      
   // 3b. Check session trading hours
   if(!IsWithinTradingHours())
      return;
      
   // 4. Spread filter
   double spread = Ask - Bid;
   double spreadPoints = spread / Point;
   if(spreadPoints > MaxSpreadPoints)
   {
      return; // Skip trading if spread is too high
   }
   
   // 5. Calculate indicators (evaluated on closed bars to prevent repaint issues)
   double bbLower1 = iBands(Symbol(), Period(), BBPeriod, BBDeviation, 0, PRICE_CLOSE, MODE_LOWER, 1);
   double bbUpper1 = iBands(Symbol(), Period(), BBPeriod, BBDeviation, 0, PRICE_CLOSE, MODE_UPPER, 1);
   
   double rsi1 = iRSI(Symbol(), Period(), RSIPeriod, PRICE_CLOSE, 1);
   double atr1 = iATR(Symbol(), Period(), ATRPeriod, 1);
   
   // Trend Filter
   bool isTrendUp = true;
   bool isTrendDown = true;
   
   if(UseTrendFilter && TrendEMAPeriod > 0)
   {
      double trendEma1 = iMA(Symbol(), Period(), TrendEMAPeriod, 0, MODE_EMA, PRICE_CLOSE, 1);
      isTrendUp = (Close[1] > trendEma1);
      isTrendDown = (Close[1] < trendEma1);
   }
   
   // RSI Filter
   bool isRsiBuy = true;
   bool isRsiSell = true;
   
   if(UseRSIFilter)
   {
      isRsiBuy = (rsi1 <= RSIOversold);
      isRsiSell = (rsi1 >= RSIOverbought);
   }
   
   // 6. Signal evaluation
   bool signalBuy  = (Close[1] < bbLower1) && isRsiBuy && isTrendUp;
   bool signalSell = (Close[1] > bbUpper1) && isRsiSell && isTrendDown;
   
   double stopLevel = MarketInfo(Symbol(), MODE_STOPLEVEL) * Point;
   
   // 7. Order Execution
   if(signalBuy)
   {
      // Calculate Stop Loss
      double sl_dist = atr1 * ATR_SL_Multiplier;
      if(sl_dist < MinStopLossPoints * Point) sl_dist = MinStopLossPoints * Point;
      if(sl_dist > MaxStopLossPoints * Point) sl_dist = MaxStopLossPoints * Point;
      if(sl_dist < stopLevel) sl_dist = stopLevel;
      
      double sl_price = Ask - sl_dist;
      sl_price = NormalizeDouble(sl_price, Digits);
      
      // Calculate Take Profit
      double tp_dist = atr1 * ATR_TP_Multiplier;
      if(tp_dist < stopLevel) tp_dist = stopLevel;
      
      double tp_price = Ask + tp_dist;
      tp_price = NormalizeDouble(tp_price, Digits);
      
      // Calculate Lot Size based on Risk
      double sl_points = sl_dist / Point;
      double lots = CalculateLotSize(sl_points);
      
      int ticket = OrderSend(Symbol(), OP_BUY, lots, Ask, Slippage, sl_price, tp_price, "Antigravity Buy", MagicNumber, 0, Blue);
      if(ticket < 0)
      {
         Print("Failed to open BUY order. Error: ", GetLastError());
      }
      else
      {
         Print("BUY Order #", ticket, " opened. Lots: ", lots, ", Entry: ", Ask, ", SL: ", sl_price, ", TP: ", tp_price);
      }
   }
   else if(signalSell)
   {
      // Calculate Stop Loss
      double sl_dist = atr1 * ATR_SL_Multiplier;
      if(sl_dist < MinStopLossPoints * Point) sl_dist = MinStopLossPoints * Point;
      if(sl_dist > MaxStopLossPoints * Point) sl_dist = MaxStopLossPoints * Point;
      if(sl_dist < stopLevel) sl_dist = stopLevel;
      
      double sl_price = Bid + sl_dist;
      sl_price = NormalizeDouble(sl_price, Digits);
      
      // Calculate Take Profit
      double tp_dist = atr1 * ATR_TP_Multiplier;
      if(tp_dist < stopLevel) tp_dist = stopLevel;
      
      double tp_price = Bid - tp_dist;
      tp_price = NormalizeDouble(tp_price, Digits);
      
      // Calculate Lot Size based on Risk
      double sl_points = sl_dist / Point;
      double lots = CalculateLotSize(sl_points);
      
      int ticket = OrderSend(Symbol(), OP_SELL, lots, Bid, Slippage, sl_price, tp_price, "Antigravity Sell", MagicNumber, 0, Red);
      if(ticket < 0)
      {
         Print("Failed to open SELL order. Error: ", GetLastError());
      }
      else
      {
         Print("SELL Order #", ticket, " opened. Lots: ", lots, ", Entry: ", Bid, ", SL: ", sl_price, ", TP: ", tp_price);
      }
   }
}
//+------------------------------------------------------------------+
