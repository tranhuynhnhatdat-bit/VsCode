//+------------------------------------------------------------------+
//| Composable_Strategy_5.mq5                                        |
//|                                                                  |
//| Faithful MQL5 port of the Python backtesting engine for          |
//| results/strategy_5 (ComposableStrategy).                         |
//|                                                                  |
//| Written in the StrategyQuant X idiom (see Strategy_QuantX_       |
//| reference) but using ONLY built-in MQL5 indicators (iATR,        |
//| iBands).                                                         |
//|                                                                  |
//| Strategy (from results/strategy_5/strategy.md — the source of    |
//| truth):                                                          |
//|   Symbol : XAUUSD                                                |
//|   Signal : H1 01:00 bar -> BUY on first M1 tick at/after 02:00   |
//|            (ASK + spread) when ANY of:                           |
//|              Close < BB_Lower(19, 1.0)                           |
//|              Close < Open                                        |
//|   Exit   : first M1 tick at/after 23:00, same calendar day the   |
//|            position was opened (CLOSE at BID)                    |
//|   SL     : 3.0 x ATR(37) H1, SL = ask - distance (fixed)         |
//|   Sizing : lots = risk / (sl_distance * tick_value / tick_size)  |
//|            floored to volume_step, REJECTED outside [min,max]    |
//|   Single position; Wed & Fri ONLY (MQL5: Wed=3, Fri=5)           |
//|                                                                  |
//| IMPORTANT: MQL5 weekday numbers differ from Python: Wed=3,Fri=5. |
//|            Default InpSessionDays = "3,5".  Do NOT revert to     |
//|            "2,4".                                                |
//|                                                                  |
//| BB parity: engine uses population std (ddof=0) on Close, bands   |
//| = SMA +/- stddev*std. Matches iBands(symbol, H1, 19, 1.0, 0,     |
//| PRICE_CLOSE). Lower band = MODE_LOWER buffer (2).                |
//+------------------------------------------------------------------+
#property copyright "Composable Strategy 5"
#property version   "1.00"
#property strict

#include <Trade/Trade.mqh>

//+------------------------------------------------------------------+
// -- Inputs (defaults = strategy_5 params)
//+------------------------------------------------------------------+
input double InpRiskMoney    = 100.0;   // Risk money per trade (USD)
input double InpSlAtr        = 3.0;     // SL = InpSlAtr * ATR(InpAtrPeriod)
input int    InpAtrPeriod    = 37;      // ATR period (H1)
input int    InpBbPeriod     = 19;      // Bollinger period (H1, PRICE_CLOSE)
input double InpBbStdDev     = 1.0;     // Bollinger stddev deviation
input int    InpEntryHour    = 1;       // H1 signal bar hour (close known at +1h)
input int    InpExitHour     = 22;      // H1 exit bar hour (close known at +1h)
input string InpSessionDays  = "3,5";   // Session days (MQL5: 0=Sun..6=Sat) -> Wed=3, Fri=5
input int    InpMagicNumber  = 5;       // Magic number

//+------------------------------------------------------------------+
// -- Constants
//+------------------------------------------------------------------+
#define ATR_1 0      // iATR(_Symbol, PERIOD_H1, InpAtrPeriod)
#define BANDS_1 1    // iBands(_Symbol, PERIOD_H1, InpBbPeriod, InpBbStdDev, 0, PRICE_CLOSE)
#define BANDS_LOWER 2 // lower band buffer (MODE_LOWER = 2)

//+------------------------------------------------------------------+
// -- Variables
//+------------------------------------------------------------------+
int indicatorHandles[];
int magicNumber;

CTrade g_trade;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   magicNumber = InpMagicNumber;

   //--- Trade settings: buy at ASK, sell at BID, magic filtering
   g_trade.SetExpertMagicNumber(magicNumber);
   g_trade.SetDeviationInPoints(50);

   //--- Create the built-in ATR + Bollinger handles on H1
   if(!initIndicators())
      return(INIT_FAILED);

   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   for(int i = 0; i < ArraySize(indicatorHandles); i++)
     {
      if(indicatorHandles[i] != INVALID_HANDLE)
         IndicatorRelease(indicatorHandles[i]);
     }
  }

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
   //--- Do we have enough history on H1?
   if(Bars(_Symbol, PERIOD_H1) < InpAtrPeriod + 10)
      return;
   if(Bars(_Symbol, PERIOD_M1) < 5)
      return;

   //--- Current forming M1 bar (timeframe-agnostic). Entry/exit fire on
   //--- the FIRST tick whose M1 bar time is at/after the target hour.
   datetime curTime = iTime(_Symbol, PERIOD_M1, 0);
   if(curTime <= 0)
      return;

   MqlDateTime dt;
   TimeToStruct(curTime, dt);
   int dayOfWeek = dt.day_of_week;   // 0=Sun..6=Sat
   int hour      = dt.hour;

   //--- Is today a session day?
   if(!IsSessionDay(dayOfWeek))
      return;

   //--- ENTRY: at/after 02:00 the 01:00 H1 bar has just closed.
   //--- Condition: (Close < BB_Lower) OR (Close < Open) -> BUY.
   if(hour == InpEntryHour + 1)
     {
      //--- Only if no position is open (single position rule, magic-filtered)
      if(HasOpenPosition())
         return;

      //--- The H1 signal bar opened today at InpEntryHour:00.
      MqlDateTime dtSignal = dt;
      dtSignal.hour = InpEntryHour;
      dtSignal.min  = 0;
      dtSignal.sec  = 0;
      datetime signalOpenTime = StructToTime(dtSignal);

      int h1Shift = iBarShift(_Symbol, PERIOD_H1, signalOpenTime, true);
      if(h1Shift < 0)
         return;

      double h1Open  = iOpen(_Symbol, PERIOD_H1, h1Shift);
      double h1Close = iClose(_Symbol, PERIOD_H1, h1Shift);
      if(h1Open <= 0.0 || h1Close <= 0.0)
         return;

      //--- Condition A (or): Close < BB_Lower(19, 1.0).
      bool condBelowBands = false;
      double bbLower = sqGetIndicatorBufferValue(BANDS_1, BANDS_LOWER, h1Shift);
      if(bbLower > 0.0)
        {
         if(h1Close < bbLower)
            condBelowBands = true;
        }

      //--- Condition B (or): Close < Open (bearish H1 bar)
      bool condCloseBelowOpen = (h1Close < h1Open);

      //--- OR combination; if neither holds, no entry.
      if(!condBelowBands && !condCloseBelowOpen)
         return;

      //--- SL distance = InpSlAtr * ATR(37) on that H1 bar
      double atrVal = sqGetIndicatorValue(ATR_1, h1Shift);
      if(atrVal <= 0.0)
         return;
      double slDistance = InpSlAtr * atrVal;

      //--- Entry price = ASK (open + spread), matching the engine
      double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);

      //--- SL absolute price = ask - distance (long)
      double slPrice = ask - slDistance;

      //--- Lot size from fixed risk money (matches engine _compute_sizes)
      double size = sqMMFixedAmount(ask, slDistance);
      if(size <= 0.0)
         return;   // rejected (outside volume min/max)

      //--- Send BUY
      if(!g_trade.Buy(size, _Symbol, ask, slPrice, 0.0, "Composable_5"))
         Print("Buy failed, error ", GetLastError());
      return;
     }

    //--- EXIT: at/after 23:00 the 22:00 H1 bar has just closed.
    if(hour == InpExitHour + 1)
      {
       if(!HasOpenPosition())
          return;

       //--- Close at market (BID), matching the engine's exit fill
       if(!g_trade.PositionClose(_Symbol, 0))
          Print("Close failed, error ", GetLastError());
       return;
      }
  }

//+------------------------------------------------------------------+
//| Create indicator handles (built-in indicators only)              |
//+------------------------------------------------------------------+
bool initIndicators()
  {
   ArrayResize(indicatorHandles, 2);
   indicatorHandles[ATR_1]   = iATR(_Symbol, PERIOD_H1, InpAtrPeriod);
   indicatorHandles[BANDS_1] = iBands(_Symbol, PERIOD_H1, InpBbPeriod,
                                      InpBbStdDev, 0, PRICE_CLOSE);

   for(int a = 0; a < ArraySize(indicatorHandles); a++)
     {
      if(indicatorHandles[a] == INVALID_HANDLE)
        {
         Print("Failed to create handle of the indicator, error code ", GetLastError());
         return(false);
        }
     }
   return(true);
  }

//+------------------------------------------------------------------+
//| Read a built-in indicator value at a given shift (buffer 0)      |
//+------------------------------------------------------------------+
double sqGetIndicatorValue(int indyIndex, int shift)
  {
   if(indyIndex < 0 || indyIndex >= ArraySize(indicatorHandles))
      return(0.0);
   if(indicatorHandles[indyIndex] == INVALID_HANDLE)
      return(0.0);

   double buf[];
   ArraySetAsSeries(buf, true);
   if(CopyBuffer(indicatorHandles[indyIndex], 0, shift, 1, buf) < 1)
      return(0.0);
   return(buf[0]);
  }

//+------------------------------------------------------------------+
//| Read a built-in indicator value at a given shift AND a given     |
//| buffer (for multi-line indicators)                               |
//+------------------------------------------------------------------+
double sqGetIndicatorBufferValue(int indyIndex, int bufferIndex, int shift)
  {
   if(indyIndex < 0 || indyIndex >= ArraySize(indicatorHandles))
      return(0.0);
   if(indicatorHandles[indyIndex] == INVALID_HANDLE)
      return(0.0);

   double buf[];
   ArraySetAsSeries(buf, true);
   if(CopyBuffer(indicatorHandles[indyIndex], bufferIndex, shift, 1, buf) < 1)
      return(0.0);
   return(buf[0]);
  }

//+------------------------------------------------------------------+
//| Check whether today is a session day                             |
//| MQL5 weekday numbering: 0=Sun,1=Mon,2=Tue,3=Wed,4=Thu,5=Fri,6=Sat |
//| (differs from Python: Wed=2,Fri=4 -> MQL5 Wed=3,Fri=5)           |
//+------------------------------------------------------------------+
bool IsSessionDay(int dayOfWeek)
  {
   string parts[];
   int n = StringSplit(InpSessionDays, ',', parts);
   for(int i = 0; i < n; i++)
     {
      int d = (int)StringToInteger(parts[i]);
      if(d == dayOfWeek)
         return(true);
     }
   return(false);
  }

//+------------------------------------------------------------------+
//| Is there an open position for this symbol + magic?               |
//+------------------------------------------------------------------+
bool HasOpenPosition()
  {
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0)
         continue;
      if(PositionSelectByTicket(ticket))
        {
         if(PositionGetString(POSITION_SYMBOL) == _Symbol
            && PositionGetInteger(POSITION_MAGIC) == magicNumber)
            return(true);
        }
     }
   return(false);
  }

//+------------------------------------------------------------------+
//| Lot size from fixed risk money (matches engine _compute_sizes)   |
//| lots = risk_money / (sl_distance * tick_value / tick_size)       |
//| floor to volume_step; rejected (0) if outside [min,max]          |
//+------------------------------------------------------------------+
double sqMMFixedAmount(double openPrice, double slDistance)
  {
   if(slDistance <= 0.0)
      return(0.0);

   double tickValue  = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double tickSize   = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   double volumeStep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   double volumeMin  = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double volumeMax  = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);

   if(tickValue <= 0.0 || tickSize <= 0.0 || volumeStep <= 0.0)
      return(0.0);

   //--- lots = risk_money / (sl_distance * tick_value / tick_size)
   double lots = InpRiskMoney / (slDistance * tickValue / tickSize);

   //--- floor to volume_step (epsilon avoids float floor artifacts)
   lots = MathFloor(lots / volumeStep + 1e-9) * volumeStep;

   //--- reject if outside [min, max] (never clamp)
   if(lots < volumeMin || lots > volumeMax)
      return(0.0);

   return(lots);
  }
//+------------------------------------------------------------------+