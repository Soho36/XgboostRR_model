//+------------------------------------------------------------------+
//| Red-Green Breakout EA: dynamic exit + time filters              |
//+------------------------------------------------------------------+
#property strict

input double Lots           = 1.0;
input double RiskReward     = 1.0;
input int    Slippage       = 5;
input double StopLossOffsetPoints = 0.0;  // SL offset (neg=lower, pos=higher)
input double EntryLimitOffsetTicks = 1.0; // Buy Stop Limit price below the stop trigger, in ticks
input bool   UseLimitExitAfterRR = false; // Place SellLimit after candle-close RR is reached
input double ProfitTargetOffsetTicks = 0.0; // SellLimit offset from trigger candle close, in ticks

// ======== CANDLE RANGE FILTER ========
input bool   UseCandleRangeFilter = false;  // ENABLE/DISABLE CANDLE RANGE FILTER
input double MaxCandleRange       = 50.0;   // Maximum allowed candle range in points
input double MinCandleRange       = 5.0;    // Minimum allowed candle range in points

// flattening end session
input bool   UseFlattenEnd     = true;	// USE FLATTENING END SESSION
input int    FlattenHourEnd    = 23;
input int    FlattenMinuteEnd  = 30;

// ======== TIME WINDOW FILTERING ========
// no-trading window (block new trades between these times) - Mixed intervals with session borders
input bool   UseTradeWindow   = true;	// USE TIME TRADE WINDOW

// ========== SESSION 1: MARKET CLOSED (00:00-01:00) ==========
input bool W0000W0100 = false;  // 00:00–01:00 (Market Closed)

// ========== SESSION 2: MORNING SESSION (01:00-10:00) ==========
// 01:00-02:00 split into 30-min intervals
input bool W0100W0130 = false;  // 01:00–01:30 (Morning Session)
input bool W0130W0200 = false;  // 01:30–02:00 (Morning Session)

// 1-hour intervals for 02:00-10:00
input bool W0200W0300 = false;  // 02:00–03:00 (Morning Session)
input bool W0300W0400 = false;  // 03:00–04:00 (Morning Session)
input bool W0400W0500 = false;  // 04:00–05:00 (Morning Session)
input bool W0500W0600 = false;  // 05:00–06:00 (Morning Session)
input bool W0600W0700 = false;  // 06:00–07:00 (Morning Session)
input bool W0700W0800 = false;  // 07:00–08:00 (Morning Session)
input bool W0800W0900 = false;  // 08:00–09:00 (Morning Session)
input bool W0900W1000 = false;  // 09:00–10:00 (Morning Session)

// ========== SESSION 3: MAIN SESSION (10:00-23:00) ==========
input bool W1000W1100 = false;  // 10:00–11:00 (Main Session)
input bool W1100W1200 = false;  // 11:00–12:00 (Main Session)
input bool W1200W1300 = false;  // 12:00–13:00 (Main Session)
input bool W1300W1400 = false;  // 13:00–14:00 (Main Session)
input bool W1400W1500 = false;  // 14:00–15:00 (Main Session)
input bool W1500W1600 = false;  // 15:00–16:00 (Main Session)
input bool W1600W1700 = false;  // 16:00–17:00 (Main Session)
input bool W1700W1800 = false;  // 17:00–18:00 (Main Session)
input bool W1800W1900 = false;  // 18:00–19:00 (Main Session)
input bool W1900W2000 = false;  // 19:00–20:00 (Main Session)
input bool W2000W2100 = false;  // 20:00–21:00 (Main Session)
input bool W2100W2200 = false;  // 21:00–22:00 (Main Session)
input bool W2200W2300 = false;  // 22:00–23:00 (Main Session)

// ========== SESSION 4: EVENING SESSION (23:00-00:00) ==========
input bool W2300W2330 = false;  // 23:00–23:30 (Evening Session)
input bool W2330W0000 = false;  // 23:30–00:00 (Evening Session)

bool windows[26] =  // Total slots: 1 + 2 + 8 + 13 + 2 = 26
{
   // ===== SESSION 1: MARKET CLOSED (00:00-01:00) =====
   W0000W0100,
   
   // ===== SESSION 2: MORNING SESSION (01:00-10:00) =====
   W0100W0130, W0130W0200,
   W0200W0300, W0300W0400, W0400W0500, W0500W0600,
   W0600W0700, W0700W0800, W0800W0900, W0900W1000,
   
   // ===== SESSION 3: MAIN SESSION (10:00-23:00) =====
   W1000W1100, W1100W1200, W1200W1300, W1300W1400,
   W1400W1500, W1500W1600, W1600W1700, W1700W1800,
   W1800W1900, W1900W2000, W2000W2100, W2100W2200,
   W2200W2300,
   
   // ===== SESSION 4: EVENING SESSION (23:00-00:00) =====
   W2300W2330, W2330W0000
};

// Session names for display purposes
string GetSessionName(int slot)
{
   if(slot == 0) return "MARKET CLOSED";
   if(slot >= 1 && slot <= 10) return "MORNING";
   if(slot >= 11 && slot <= 23) return "MAIN";
   if(slot >= 24 && slot <= 25) return "EVENING";
   return "UNKNOWN";
}

// Global variables
bool   g_exitLimitPlaced = false;
double g_signalLow  = 0.0;
bool g_wasInPosition = false;
double g_initialEntry = 0.0;
double g_initialRisk  = 0.0;
bool   g_initialSet   = false;

// ======== MAE / MFE (FLOATING PNL BASED) ========
double g_maeMoney   = 0.0;   // most negative floating PnL
double g_mfeMoney   = 0.0;   // most positive floating PnL
bool   g_tracking   = false;
ulong  g_ticket     = 0;
double g_candleRange = 0.0;

datetime g_entryTime = 0;
string   g_csvName   = "trade_stats.csv";

// ======== HELPER FUNCTIONS ========

bool IsFlattenTimeEnd(datetime barOpen)
{
   MqlDateTime dt; TimeToStruct(barOpen, dt);
   return (dt.hour == FlattenHourEnd && dt.min == FlattenMinuteEnd);
}

bool IsTradeWindow(datetime barOpen)
{
   if(!UseTradeWindow)
      return true;

   MqlDateTime dt;
   TimeToStruct(barOpen, dt);
   
   int totalMinutes = dt.hour * 60 + dt.min;
   int slot;
   
   // Session 1: 00:00-01:00 - 1-hour interval
   if(totalMinutes < 60)
   {
      slot = 0;
   }
   // Session 2: 01:00-10:00 - mixed intervals
   else if(totalMinutes >= 60 && totalMinutes < 600) // 01:00 to 10:00
   {
      if(totalMinutes < 120) // 01:00-02:00 (30-min slots)
      {
         slot = 1 + ((totalMinutes - 60) / 30);
      }
      else // 02:00-10:00 (hourly slots)
      {
         slot = 3 + (dt.hour - 2);  // slots 3-10
      }
   }
   // Session 3: 10:00-23:00 - hourly intervals
   else if(totalMinutes >= 600 && totalMinutes < 1380) // 10:00 to 23:00
   {
      slot = 11 + (dt.hour - 10);  // slots 11-23
   }
   // Session 4: 23:00-00:00 - 30-min intervals
   else // 23:00-00:00
   {
      slot = 24 + ((totalMinutes - 1380) / 30);  // slots 24-25
   }
   
   return windows[slot];
}

void DisplayTradeWindowStatus(datetime barOpen)
{
   if(!UseTradeWindow) return;
   
   MqlDateTime dt;
   TimeToStruct(barOpen, dt);
   
   int totalMinutes = dt.hour * 60 + dt.min;
   string sessionName = "";
   string borderLine = "";
   
   if(totalMinutes < 60)
   {
      sessionName = "MARKET CLOSED";
      borderLine = "═══════════════════════════════════════════════";
   }
   else if(totalMinutes >= 60 && totalMinutes < 600)
   {
      sessionName = "MORNING SESSION";
      if(totalMinutes == 60) // Start of morning session
         borderLine = "═══════════════ MORNING SESSION START ═══════════════";
   }
   else if(totalMinutes >= 600 && totalMinutes < 1380)
   {
      sessionName = "MAIN SESSION";
      if(totalMinutes == 600) // Start of main session
         borderLine = "════════════════ MAIN SESSION START ════════════════";
   }
   else
   {
      sessionName = "EVENING SESSION";
      if(totalMinutes == 1380) // Start of evening session
         borderLine = "═══════════════ EVENING SESSION START ═══════════════";
   }
   
   if(borderLine != "")
      Print(borderLine);
}

bool IsCandleInRange(double high, double low)
{
   if(!UseCandleRangeFilter)
      return true;
   
   // Calculate candle range in points
   double rangePoints = (high - low) / _Point;
   
   // Check if range is within allowed limits
   if(rangePoints > MaxCandleRange)
   {
      Print("⚠️ Candle range filter: Range ", DoubleToString(rangePoints, 2), " points > Max ", DoubleToString(MaxCandleRange, 2), " points - Skipping");
      return false;
   }
   
   if(rangePoints < MinCandleRange)
   {
      Print("⚠️ Candle range filter: Range ", DoubleToString(rangePoints, 2), " points < Min ", DoubleToString(MinCandleRange, 2), " points - Skipping");
      return false;
   }
   
   return true;
}

// ======== CSV FUNCTIONS ========
void SaveTradeStats(double realized, datetime entryTime, datetime exitTime, double candleRange)
{
   int f = FileOpen(g_csvName, FILE_READ|FILE_WRITE|FILE_CSV|FILE_SHARE_WRITE);
   if(f == INVALID_HANDLE)
   {
      Print("File open failed ", GetLastError());
      return;
   }

   if(FileSize(f) == 0)
      FileWrite(f, "ticket", "entry_time", "exit_time", "mae_money", "mfe_money", "trade_profit", "candle_range");

   FileSeek(f, 0, SEEK_END);
   FileWrite(
      f,
      (long)g_ticket,
      TimeToString(entryTime, TIME_DATE|TIME_SECONDS),
      TimeToString(exitTime, TIME_DATE|TIME_SECONDS),
      g_maeMoney,
      g_mfeMoney,
      realized,
      candleRange
   );

   FileClose(f);
}

// ======== TRADE MANAGEMENT FUNCTIONS ========
void CancelAllOrders()
{
   for(int i = OrdersTotal() - 1; i >= 0; --i)
   {
      ulong ticket = OrderGetTicket(i);
      if(ticket == 0) continue;
      if(!OrderSelect(ticket)) continue;

      MqlTradeRequest req = {};
      MqlTradeResult  res = {};
      req.action = TRADE_ACTION_REMOVE;
      req.order  = ticket;

      if(!OrderSend(req, res))
         Print("❌ CancelAllOrders failed ticket=", ticket, " err=", GetLastError());
      else
         Print("✅ Cancelled order ticket=", ticket);
   }
}

void CloseAllPositions()
{
   for(int i = PositionsTotal() - 1; i >= 0; --i)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(!PositionSelectByTicket(ticket)) continue;

      string sym = PositionGetString(POSITION_SYMBOL);
      double vol = PositionGetDouble(POSITION_VOLUME);
      long   typ = PositionGetInteger(POSITION_TYPE);

      MqlTradeRequest req = {};
      MqlTradeResult  res = {};
      req.action    = TRADE_ACTION_DEAL;
      req.symbol    = sym;
      req.volume    = vol;
      req.deviation = Slippage;

      if(typ == POSITION_TYPE_BUY)
      {
         req.type  = ORDER_TYPE_SELL;
         req.price = SymbolInfoDouble(sym, SYMBOL_BID);
      }
      else
      {
         req.type  = ORDER_TYPE_BUY;
         req.price = SymbolInfoDouble(sym, SYMBOL_ASK);
      }

      if(!OrderSend(req, res))
         Print("❌ CloseAllPositions failed pos#", ticket, " err=", GetLastError());
      else
         Print("✅ Closed position #", ticket);
   }
}

void CancelOldBuyStops()
{
   for(int i = OrdersTotal() - 1; i >= 0; --i)
   {
      ulong ticket = OrderGetTicket(i);
      if(ticket == 0) continue;
      if(!OrderSelect(ticket)) continue;

      int type = (int)OrderGetInteger(ORDER_TYPE);
      if(type != ORDER_TYPE_BUY_STOP && type != ORDER_TYPE_BUY_STOP_LIMIT) continue;

      MqlTradeRequest req = {};
      MqlTradeResult  res = {};
      req.action = TRADE_ACTION_REMOVE;
      req.order  = ticket;

      if(!OrderSend(req, res))
         Print("❌ Failed to cancel pending entry ticket=", ticket, " err=", GetLastError());
      else
         Print("✅ Cancelled pending entry ticket=", ticket);
   }
}

bool HasSellLimitForSymbol()
{
   for(int i = OrdersTotal() - 1; i >= 0; --i)
   {
      ulong ticket = OrderGetTicket(i);
      if(ticket == 0) continue;
      if(!OrderSelect(ticket)) continue;

      if(OrderGetString(ORDER_SYMBOL) != _Symbol) continue;

      int type = (int)OrderGetInteger(ORDER_TYPE);
      if(type == ORDER_TYPE_SELL_LIMIT)
         return true;
   }

   return false;
}

void CancelAllSellLimits()
{
   for(int i = OrdersTotal() - 1; i >= 0; --i)
   {
      ulong ticket = OrderGetTicket(i);
      if(ticket == 0) continue;
      if(!OrderSelect(ticket)) continue;

      if(OrderGetString(ORDER_SYMBOL) != _Symbol) continue;

      int type = (int)OrderGetInteger(ORDER_TYPE);
      if(type != ORDER_TYPE_SELL_LIMIT) continue;

      MqlTradeRequest req = {};
      MqlTradeResult  res = {};
      req.action = TRADE_ACTION_REMOVE;
      req.order  = ticket;

      if(!OrderSend(req, res))
         Print("❌ Failed to cancel SellLimit ticket=", ticket, " err=", GetLastError());
      else
         Print("🧹 Cancelled SellLimit ticket=", ticket);
   }
}

void ManageOpenPosition()
{
   if(!PositionSelect(_Symbol)) return;

   double vol = PositionGetDouble(POSITION_VOLUME);
   long   typ = PositionGetInteger(POSITION_TYPE);

   if(typ != POSITION_TYPE_BUY) return;

   if(!g_initialSet || g_initialRisk <= 0.0)
   {
      Print("⚠️ Initial reference not set → skipping TP logic");
      return;
   }

   double barClose = iClose(_Symbol, _Period, 1);

   double target = g_initialEntry + g_initialRisk * RiskReward;

   if(barClose >= target)
   {
      if(UseLimitExitAfterRR)
      {
         if(g_exitLimitPlaced || HasSellLimitForSymbol())
         {
            g_exitLimitPlaced = true;
            Print("⏳ Fixed R:R reached → SellLimit exit already placed");
            return;
         }

         double tickSize = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
         if(tickSize <= 0.0)
            tickSize = _Point;

         double limitPrice = barClose + ProfitTargetOffsetTicks * tickSize;
         limitPrice = MathCeil(limitPrice / tickSize) * tickSize;
         limitPrice = NormalizeDouble(limitPrice, _Digits);

         double ask     = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
         double minDist = SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL) * _Point;

         if(limitPrice <= ask + minDist)
         {
            Print("⚠️ SellLimit exit price invalid/too close: barClose=", barClose,
                  " ProfitTargetOffsetTicks=", ProfitTargetOffsetTicks,
                  " limitPrice=", limitPrice,
                  " ask=", ask,
                  " minDist=", minDist);
            return;
         }


         Print("✅ Fixed R:R reached → placing SellLimit exit @", limitPrice);

         MqlTradeRequest req = {};
         MqlTradeResult  res = {};
         req.action       = TRADE_ACTION_PENDING;
         req.symbol       = _Symbol;
         req.volume       = vol;
         req.type         = ORDER_TYPE_SELL_LIMIT;
         req.price        = limitPrice;
         req.deviation    = Slippage;
         req.type_filling = ORDER_FILLING_RETURN;

         if(!OrderSend(req, res))
            Print("❌ SellLimit exit failed err=", GetLastError());
         else
         {
            g_exitLimitPlaced = true;
            Print("✅ SellLimit exit placed @", limitPrice, " lots=", vol);
         }

         return;
      }

      Print("✅ Fixed R:R reached → closing ALL positions");

      MqlTradeRequest req = {};
      MqlTradeResult  res = {};
      req.action    = TRADE_ACTION_DEAL;
      req.symbol    = _Symbol;
      req.volume    = vol; // closes full net position
      req.type      = ORDER_TYPE_SELL;
      req.price     = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      req.deviation = Slippage;

      if(!OrderSend(req, res))
         Print("❌ Close fail err=", GetLastError());
      else
         Print("✅ Position closed at fixed R:R");
   }
   else
   {
      Print("⏳ Waiting fixed R:R → target=", target);
   }
}

// ======== DISPLAY CURRENT SETTINGS ========
void DisplaySettings()
{
   Print("╔════════════════════════════════════════════════════════════╗");
   Print("║                    EA SETTINGS                             ║");
   Print("╚════════════════════════════════════════════════════════════╝");
   Print("Lots: ", Lots, ", RiskReward: ", RiskReward);
   Print("StopLossOffsetPoints: ", StopLossOffsetPoints, " points (negative=below red candle low, positive=above)");
   Print("EntryLimitOffsetTicks: ", EntryLimitOffsetTicks, " ticks below the Buy Stop trigger");
   Print("UseLimitExitAfterRR: ", UseLimitExitAfterRR ? "true" : "false",
         ", ProfitTargetOffsetTicks: ", ProfitTargetOffsetTicks);

   // Display candle range filter settings
   Print("┌────────────────────────────────────────────────────────────┐");
   Print("│ CANDLE RANGE FILTER                                        │");
   Print("├────────────────────────────────────────────────────────────┤");
   Print("│ Status: ", UseCandleRangeFilter ? "ENABLED" : "DISABLED");
   if(UseCandleRangeFilter)
   {
      Print("│ Max Range: ", MaxCandleRange, " points");
      Print("│ Min Range: ", MinCandleRange, " points");
   }
   Print("└────────────────────────────────────────────────────────────┘");

   // Display time window settings
   Print("┌────────────────────────────────────────────────────────────┐");
   Print("│ TIME WINDOW FILTERING - SESSIONS                           │");
   Print("├────────────────────────────────────────────────────────────┤");
   Print("│ Status: ", UseTradeWindow ? "ENABLED" : "DISABLED");
   if(UseTradeWindow)
   {
      Print("├────────────────────────────────────────────────────────────┤");
      Print("│ SESSION 1: MARKET CLOSED (00:00-01:00) - 1 hour           │");
      Print("├────────────────────────────────────────────────────────────┤");
      Print("│ SESSION 2: MORNING (01:00-10:00)                          │");
      Print("│   - 01:00-02:00: 30-min intervals                         │");
      Print("│   - 02:00-10:00: 1-hour intervals                         │");
      Print("├────────────────────────────────────────────────────────────┤");
      Print("│ SESSION 3: MAIN (10:00-23:00) - 1-hour intervals          │");
      Print("├────────────────────────────────────────────────────────────┤");
      Print("│ SESSION 4: EVENING (23:00-00:00) - 30-min intervals       │");
   }
   Print("└────────────────────────────────────────────────────────────┘");

   // Display flattening times
   Print("┌────────────────────────────────────────────────────────────┐");
   Print("│ FLATTEN TIMES                                              │");
   Print("├────────────────────────────────────────────────────────────┤");
   Print("│ End of Session: ", UseFlattenEnd ? "Yes (" + (string)FlattenHourEnd + ":" + (string)FlattenMinuteEnd + ")" : "No");
   Print("└────────────────────────────────────────────────────────────┘");
}

// ======== EA CORE ========
int OnInit()
{
   // Auto-name the stats file per RiskReward so multiple RR backtests
   // don't overwrite each other (e.g. trade_stats_rr_1.0.csv, _rr_2.5.csv ...)
   g_csvName = "trade_stats_rr_" + DoubleToString(RiskReward, 1) + ".csv";
   Print("Trade stats will be written to: ", g_csvName);

   // Delete previous stats file if exists
   if(FileIsExist(g_csvName))
   {
      if(FileDelete(g_csvName))
         Print("Old trade_stats.csv deleted");
      else
         Print("Failed to delete old CSV. Error=", GetLastError());
   }

   DisplaySettings();
   return(INIT_SUCCEEDED);
}


void OnTick()

{
   // ---- TRACK FLOATING MAE / MFE (tick-based, broker-safe) ----
   if(PositionSelect(_Symbol))
   {
      double floating = PositionGetDouble(POSITION_PROFIT);

      if(!g_tracking)
      {
         g_tracking  = true;
         g_ticket    = PositionGetInteger(POSITION_TICKET);
         g_entryTime = (datetime)PositionGetInteger(POSITION_TIME);
         g_maeMoney  = floating;
         g_mfeMoney  = floating;

         Print("Tracking started ticket=", g_ticket);
      }

      // 🔹 update excursions
      g_maeMoney = MathMin(g_maeMoney, floating);
      g_mfeMoney = MathMax(g_mfeMoney, floating);
   }
   else if(g_tracking)
   {
      // --- include final realized PnL into MAE/MFE ---
      double realized = 0.0;
      
      // --- include datetime ---
      datetime exitTime = 0;

      if(HistorySelect(g_entryTime - 86400, TimeCurrent()))
      {
         for(int i = HistoryDealsTotal() - 1; i >= 0; i--)
         {
            ulong deal = HistoryDealGetTicket(i);

            if(HistoryDealGetInteger(deal, DEAL_POSITION_ID) != g_ticket)
               continue;

            double profit = HistoryDealGetDouble(deal, DEAL_PROFIT);
            realized += profit;

            if(HistoryDealGetInteger(deal, DEAL_ENTRY) == DEAL_ENTRY_OUT)
            {
               exitTime = (datetime)HistoryDealGetInteger(deal, DEAL_TIME);
               break; // ← exit deal found, stop
            }
         }
      }

      // realized PnL IS the last excursion
      g_maeMoney = MathMin(g_maeMoney, realized);
      g_mfeMoney = MathMax(g_mfeMoney, realized);

      SaveTradeStats(realized, g_entryTime, exitTime, g_candleRange);

      g_tracking = false;
   }

   static datetime lastBar = 0;
   datetime barOpen = iTime(_Symbol, _Period, 0);
	
	bool isInPosition = PositionSelect(_Symbol);
	
	// 🔹 NEW POSITION DETECTED → initialize R:R reference
	if(isInPosition && !g_wasInPosition)
	{
	   g_exitLimitPlaced = false;

	   double entry = PositionGetDouble(POSITION_PRICE_OPEN);
	   double sl    = PositionGetDouble(POSITION_SL);

	   if(g_signalLow > 0.0 && entry > g_signalLow)
	   {
		  g_initialEntry = entry;
		  g_initialRisk  = entry - g_signalLow;
		  g_initialSet   = true;

		  Print("📌 Initial trade locked (universal): Entry=", g_initialEntry, " Risk=", g_initialRisk);
	   }
	   else if(sl > 0.0 && entry > sl)
	   {
		  g_initialEntry = entry;
		  g_initialRisk  = entry - sl;
		  g_initialSet   = true;

		  Print("📌 Initial trade locked (fallback from actual SL): Entry=", g_initialEntry, " Risk=", g_initialRisk);
	   }
	   else
	   {
		  Print("⚠️ Invalid SL or entry → cannot compute risk");
	   }
	}

	// Position CLOSED -> remove pending exits and reset trade state
	if(!isInPosition && g_wasInPosition)
	{
	   Print("📤 Position closed → cleanup");

	   CancelAllSellLimits();

	   g_signalLow  = 0.0;
	   g_exitLimitPlaced = false;

	   // 🔹 reset initial reference
	   g_initialEntry = 0.0;
	   g_initialRisk  = 0.0;
	   g_initialSet   = false;
	}

	// update state
	g_wasInPosition = isInPosition;
	
   if(barOpen == lastBar) return;
   lastBar = barOpen;

   // Display session borders
   DisplayTradeWindowStatus(barOpen);

   // 🔹 flatten end of session
   if(UseFlattenEnd && IsFlattenTimeEnd(barOpen))
   {
      Print("🌙 Flatten cutoff reached → closing everything");
      CloseAllPositions();
      CancelAllOrders();
      return;
   }

   // 🔹 manage existing position
   if(PositionsTotal() > 0)
   {
      ManageOpenPosition();
      return;
   }

   // 🔹 time window check (ENTRY ONLY)
   if(!IsTradeWindow(barOpen))
   {
      Print("⏱ Outside trading window → no new entries");
      CancelOldBuyStops();
      return;
   }

   // Red candle setup (only if all filters pass)
   double o1 = iOpen(_Symbol, _Period, 1);
   double h1 = iHigh(_Symbol, _Period, 1);
   double l1 = iLow(_Symbol, _Period, 1);
   double c1 = iClose(_Symbol, _Period, 1);

   // 🔹 Candle range filter check
   if(!IsCandleInRange(h1, l1))
   {
      Print("⚠️ Candle range outside allowed limits - Skipping signal");
      CancelOldBuyStops();
      return;
   }
   
   
   // BUY STOP LIMIT: trigger at previous red candle high, limit below it by the configured offset
    if(c1 < o1)
	{
	   Print("🔴 Red candle → place Buy Stop Limit");

	   CancelOldBuyStops();

	   double entry = h1;
	   double stop  = NormalizeDouble(l1 + StopLossOffsetPoints, _Digits);
	   double risk  = entry - l1;
	   g_candleRange = h1 - l1;

	   double tickSize = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
	   if(tickSize <= 0.0)
	      tickSize = _Point;

	   double entryLimitOffset = MathMax(0.0, EntryLimitOffsetTicks);
	   double limitPrice = entry - entryLimitOffset * tickSize;
	   limitPrice = MathFloor(limitPrice / tickSize) * tickSize;
	   limitPrice = NormalizeDouble(limitPrice, _Digits);

	   if(entry <= stop)
	   {
	      Print("⚠️ Invalid risk after SL offset: entry=", entry,
	            " redLow=", l1,
	            " StopLossOffsetPoints=", StopLossOffsetPoints,
	            " adjustedSL=", stop);
	      return;
	   }

	   if(risk <= 0.0)
	   {
	      Print("⚠️ Invalid original candle risk: entry=", entry,
	            " redLow=", l1);
	      return;
	   }

	   if(limitPrice <= stop)
	   {
	      Print("⚠️ Buy Stop Limit price must remain above SL: trigger=", entry,
	            " limit=", limitPrice,
	            " SL=", stop,
	            " EntryLimitOffsetTicks=", EntryLimitOffsetTicks);
	      return;
	   }

	   // Store the original signal low for the fixed R:R calculation
	   g_signalLow  = l1;

	   MqlTradeRequest req = {};
	   MqlTradeResult  res = {};
	   req.action       = TRADE_ACTION_PENDING;
	   req.symbol       = _Symbol;
	   req.volume       = Lots;
	   req.type         = ORDER_TYPE_BUY_STOP_LIMIT;
	   req.price        = entry;
	   req.stoplimit    = limitPrice;
	   req.sl           = stop;
	   req.deviation    = Slippage;
	   req.type_filling = ORDER_FILLING_RETURN;

	   if(!OrderSend(req, res))
		  Print("❌ Place Buy Stop Limit failed: trigger=", entry,
		        " limit=", limitPrice,
		        " err=", GetLastError(),
		        " retcode=", res.retcode);
	   else
		  Print("🚀 Buy Stop Limit placed: trigger=", entry, " limit=", limitPrice);
	}
}
