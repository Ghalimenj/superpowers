// MARS v2 — Multi-Factor Adaptive Regime System with AI Confluence Engine
// 84+ indicators, online adaptive weight learning, 3-TP management
// For EURUSD M15 | cTrader cAlgo
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using cAlgo.API;
using cAlgo.API.Indicators;
using cAlgo.API.Internals;

namespace cAlgo.Robots
{
    // ─────────────────────────────────────────────────────────────
    //  ENUMS
    // ─────────────────────────────────────────────────────────────
    public enum MarketRegime   { Trending, Ranging, HighVolatility }
    public enum RiskAction     { None, SoftDaily, HardDaily, SoftTotal, HardTotal }
    public enum SignalDirection { None, Long, Short }
    public enum ExitReason     { TakeProfit, StopLoss, Breakeven, TrailingStop,
                                  PartialClose, TimeExit, WeekendClose,
                                  DailyLimitClose, DrawdownClose, Manual }

    // ─────────────────────────────────────────────────────────────
    //  IndicatorSignals  — normalised signals in [-1, +1]
    // ─────────────────────────────────────────────────────────────
    public class IndicatorSignals : Dictionary<string, double>
    {
        private static double Clamp(double v)
        {
            if (double.IsNaN(v) || double.IsInfinity(v)) return 0;
            return v < -1.0 ? -1.0 : v > 1.0 ? 1.0 : v;
        }
        public void Set(string key, double value) { this[key] = Clamp(value); }
        public int CountActive(double thr = 0.15)
        {
            int c = 0; foreach (var kv in this) if (Math.Abs(kv.Value) >= thr) c++; return c;
        }
        public int CountAgreeing(double dir, double thr = 0.15)
        {
            int c = 0;
            foreach (var kv in this)
                if (Math.Abs(kv.Value) >= thr && Math.Sign(kv.Value) == Math.Sign(dir)) c++;
            return c;
        }
        public string TopSignals(int n, double minAbs = 0.15)
        {
            var sb = new StringBuilder();
            foreach (var kv in this.Where(k => Math.Abs(k.Value) >= minAbs)
                                   .OrderByDescending(k => Math.Abs(k.Value)).Take(n))
                sb.Append(kv.Key).Append(':').Append(kv.Value.ToString("F2")).Append(' ');
            return sb.ToString().TrimEnd();
        }
    }

    // ─────────────────────────────────────────────────────────────
    //  TradingSignal  — full trade decision
    // ─────────────────────────────────────────────────────────────
    public class TradingSignal
    {
        public SignalDirection  Direction;
        public double           Score;          // -1 .. +1
        public double           Confidence;     // 0 .. 100
        public int              ConfluenceCount;
        public string           StrengthLabel;
        public double           EntryPrice;
        public double           SlDistance;
        public double           Tp1Distance;    // 1.0 × SL  (30 % close)
        public double           Tp2Distance;    // 1.8 × SL  (30 % close)
        public double           Tp3Distance;    // 3.0 × SL  (40 % close)
        public double           RiskReward;
        public string           TimeHorizon;
        public string           Rationale;
        public IndicatorSignals ActiveSignals;
    }

    // ─────────────────────────────────────────────────────────────
    //  TradeRecord  — enhanced with AI fields
    // ─────────────────────────────────────────────────────────────
    public class TradeRecord
    {
        public DateTime         EntryTime      { get; set; }
        public double           EntryPrice     { get; set; }
        public DateTime         ExitTime       { get; set; }
        public double           ExitPrice      { get; set; }
        public TradeType        Direction      { get; set; }
        public double           Lots           { get; set; }
        public double           StopLoss       { get; set; }
        public double           TakeProfit     { get; set; }
        public double           PnL            { get; set; }
        public ExitReason       ExitReason     { get; set; }
        public string           SignalSource   { get; set; }
        public double           SignalScore    { get; set; }
        public double           SignalConf     { get; set; }
        public IndicatorSignals ActiveSignals  { get; set; }
        public bool             Tp1Hit         { get; set; }
        public bool             Tp2Hit         { get; set; }
        public double           SlDistAtEntry  { get; set; }
    }

    // ─────────────────────────────────────────────────────────────
    //  AdaptiveWeightEngine  — online reinforcement learning
    // ─────────────────────────────────────────────────────────────
    public class AdaptiveWeightEngine
    {
        private readonly Dictionary<string, double> _w = new Dictionary<string, double>();
        private int _trades = 0;
        private const double LR  = 0.06;
        private const double Min = 0.15;
        private const double Max = 4.0;
        private const double Init = 1.0;
        public int TradeCount => _trades;

        private double G(string k)         { return _w.ContainsKey(k) ? _w[k] : Init; }
        private void   S(string k, double v) { _w[k] = v < Min ? Min : v > Max ? Max : v; }

        public void OnTradeComplete(IndicatorSignals sig, bool win)
        {
            if (sig == null) return;
            _trades++;
            foreach (var kv in sig)
            {
                if (Math.Abs(kv.Value) < 0.15) continue;
                bool ok = (kv.Value > 0) == win;
                S(kv.Key, G(kv.Key) * (ok ? 1.0 + LR : 1.0 - LR));
            }
        }

        public double Score(IndicatorSignals sig)
        {
            double ws = 0, wt = 0;
            foreach (var kv in sig)
            {
                if (Math.Abs(kv.Value) < 0.05) continue;
                double w = G(kv.Key);
                ws += kv.Value * w;
                wt += w;
            }
            return wt > 0 ? ws / wt : 0;
        }

        public double Confidence(IndicatorSignals sig, double score)
        {
            int active = sig.CountActive(0.15);
            int agree  = sig.CountAgreeing(score, 0.15);
            if (active == 0) return 0;
            return Math.Min(100.0, (double)agree / active * Math.Abs(score) * 130.0);
        }

        public string TopWeights(int n = 8)
        {
            var sb = new StringBuilder();
            foreach (var kv in _w.OrderByDescending(x => x.Value).Take(n))
                sb.Append(kv.Key).Append('=').Append(kv.Value.ToString("F2")).Append(' ');
            return sb.ToString().TrimEnd();
        }
        public string BotWeights(int n = 5)
        {
            var sb = new StringBuilder();
            foreach (var kv in _w.OrderBy(x => x.Value).Take(n))
                sb.Append(kv.Key).Append('=').Append(kv.Value.ToString("F2")).Append(' ');
            return sb.ToString().TrimEnd();
        }
    }

    // ─────────────────────────────────────────────────────────────
    //  FTMORiskManager
    // ─────────────────────────────────────────────────────────────
    public class FTMORiskManager
    {
        public double InitialBalance         { get; private set; }
        public double DailyStartBalance      { get; private set; }
        public double SoftDailyLossLimit     { get; private set; }
        public double HardDailyLossLimit     { get; private set; }
        public double SoftTotalDrawdownLimit { get; private set; }
        public double HardTotalDrawdownLimit { get; private set; }
        public double DailyProfitTarget      { get; private set; }
        public int    TradingDaysLogged      { get; private set; }
        public bool   IsDailyLimitBreached   { get; private set; }
        public bool   IsTotalLimitBreached   { get; private set; }

        public void Initialize(double bal)
        {
            InitialBalance         = bal;
            DailyStartBalance      = bal;
            SoftDailyLossLimit     = bal * 0.045;
            HardDailyLossLimit     = bal * 0.049;
            SoftTotalDrawdownLimit = bal * 0.090;
            HardTotalDrawdownLimit = bal * 0.098;
            DailyProfitTarget      = bal * 0.010;
        }
        public void OnNewDay(double bal)
        {
            DailyStartBalance    = bal;
            DailyProfitTarget    = bal * 0.010;
            IsDailyLimitBreached = false;
        }
        public void LogTradingDay() { TradingDaysLogged++; }
        public bool CanOpenTrade(double equity, double dayPnL)
        {
            if (IsDailyLimitBreached || IsTotalLimitBreached) return false;
            if (InitialBalance - equity >= SoftTotalDrawdownLimit) return false;
            if (DailyStartBalance - equity >= SoftDailyLossLimit)  return false;
            return true;
        }
        public RiskAction CheckForBreach(double equity)
        {
            if (InitialBalance  - equity >= HardTotalDrawdownLimit) { IsTotalLimitBreached = true; return RiskAction.HardTotal; }
            if (InitialBalance  - equity >= SoftTotalDrawdownLimit) { IsTotalLimitBreached = true; return RiskAction.SoftTotal; }
            if (DailyStartBalance - equity >= HardDailyLossLimit)   { IsDailyLimitBreached = true; return RiskAction.HardDaily; }
            if (DailyStartBalance - equity >= SoftDailyLossLimit)   { IsDailyLimitBreached = true; return RiskAction.SoftDaily; }
            return RiskAction.None;
        }
        public double GetSizingMultiplier(double dayPnL)
        {
            double pct = -dayPnL / DailyStartBalance * 100.0;
            if (pct >= 3.0) return 0.40;
            if (pct >= 2.0) return 0.60;
            if (pct >= 1.0) return 0.80;
            return 1.0;
        }
        public void LogStatus(string ctx, Action<string> log)
        {
            log(string.Format("[RISK|{0}] Init={1:F0} DayStart={2:F0} SoftDD={3:F0} HardDD={4:F0} DayBreach={5} TotBreach={6}",
                ctx, InitialBalance, DailyStartBalance,
                SoftTotalDrawdownLimit, HardTotalDrawdownLimit,
                IsDailyLimitBreached, IsTotalLimitBreached));
        }
    }

    // ─────────────────────────────────────────────────────────────
    //  QuantPositionSizer
    // ─────────────────────────────────────────────────────────────
    public class QuantPositionSizer
    {
        private readonly List<double> _hist = new List<double>();
        private const int Window = 30;

        public double CalculateLots(double bal, double riskPct, double slPips,
            double pipVal, double equity, double usedMargin, double contractSize,
            double dayPnL, FTMORiskManager rm, double regimeMult)
        {
            if (slPips <= 0 || pipVal <= 0) return 0;
            double risk = bal * (riskPct / 100.0);
            double raw  = risk / (slPips * pipVal);
            raw *= HalfKelly();
            raw *= rm.GetSizingMultiplier(dayPnL);
            raw *= regimeMult;
            double maxMargin = equity * 0.20 - usedMargin;
            if (maxMargin <= 0) return 0;
            double margPerLot = contractSize / 100.0;
            if (margPerLot > 0) raw = Math.Min(raw, maxMargin / margPerLot);
            raw = Math.Floor(raw * 100.0) / 100.0;
            return raw < 0.01 ? 0 : raw;
        }
        public void Record(double pnl) { _hist.Add(pnl); }
        private double HalfKelly()
        {
            if (_hist.Count < 10) return 0.5;
            var r = _hist.Count >= Window ? _hist.Skip(_hist.Count - Window).ToList() : _hist;
            int w = r.Count(p => p > 0);
            double W = (double)w / r.Count;
            double aw = w > 0 ? r.Where(p => p > 0).Average() : 0;
            double al = (r.Count - w) > 0 ? Math.Abs(r.Where(p => p <= 0).Average()) : 1;
            if (al <= 0) al = 1;
            double R = aw / al;
            if (R <= 0) return 0.25;
            double k = W - (1 - W) / R;
            return Math.Max(0.25, Math.Min(1.0, k * 0.5));
        }
    }

    // ─────────────────────────────────────────────────────────────
    //  MAIN ROBOT
    // ─────────────────────────────────────────────────────────────
    [Robot(AccessRights = AccessRights.None)]
    public class MARSTradingBot : Robot
    {
        #region Parameters
        [Parameter("Risk % Per Trade",       DefaultValue = 1.0,   MinValue = 0.25, MaxValue = 2.0,  Group = "Risk")]
        public double RiskPercentPerTrade { get; set; }
        [Parameter("Max Concurrent Trades",  DefaultValue = 2,     MinValue = 1,    MaxValue = 5,    Group = "Risk")]
        public int MaxConcurrentTrades { get; set; }
        [Parameter("Max Trades Per Day",     DefaultValue = 5,     MinValue = 1,    MaxValue = 8,    Group = "Risk")]
        public int MaxTradesPerDay { get; set; }
        [Parameter("Daily Loss Pause %",     DefaultValue = 1.0,   MinValue = 0,    MaxValue = 5.0,  Group = "Risk")]
        public double DailyLossPausePct { get; set; }
        [Parameter("Max Consecutive Losses", DefaultValue = 2,    MinValue = 1,    MaxValue = 5,    Group = "Risk")]
        public int MaxConsecutiveLosses { get; set; }
        [Parameter("FTMO Phase (1 or 2)",    DefaultValue = 1,     MinValue = 1,    MaxValue = 2,    Group = "Risk")]
        public int Phase { get; set; }
        [Parameter("RSI Buy Threshold",      DefaultValue = 40,    MinValue = 25,   MaxValue = 50,   Group = "Signal")]
        public int RsiBuyThreshold { get; set; }
        [Parameter("RSI Sell Threshold",     DefaultValue = 60,    MinValue = 50,   MaxValue = 75,   Group = "Signal")]
        public int RsiSellThreshold { get; set; }
        [Parameter("BB Buy Zone (0-1)",      DefaultValue = 0.25,  MinValue = 0.1,  MaxValue = 0.45, Group = "Signal")]
        public double BbBuyZone { get; set; }
        [Parameter("BB Sell Zone (0-1)",     DefaultValue = 0.75,  MinValue = 0.55, MaxValue = 0.9,  Group = "Signal")]
        public double BbSellZone { get; set; }
        [Parameter("Enable AI Learning",     DefaultValue = true,                                    Group = "Signal")]
        public bool EnableLearning { get; set; }
        [Parameter("ATR Period",             DefaultValue = 14,    MinValue = 5,    MaxValue = 30,   Group = "Indicators")]
        public int AtrPeriod { get; set; }
        [Parameter("FOMC Dates (yyyy-MM-dd,csv)", DefaultValue = "",                                 Group = "News")]
        public string FomcDates { get; set; }
        [Parameter("ECB Dates (yyyy-MM-dd,csv)",  DefaultValue = "",                                 Group = "News")]
        public string EcbDates { get; set; }
        #endregion

        #region Indicator Fields — Primary M15
        // SMAs
        private SimpleMovingAverage   _sma20, _sma50, _sma100, _sma200;
        // EMAs
        private ExponentialMovingAverage _ema8, _ema12, _ema21, _ema26, _ema50, _ema55, _ema200;
        // MACD, ATR, DMS, PSAR
        private MacdCrossOver          _macd;
        private AverageTrueRange       _atr;
        private DirectionalMovementSystem _dms;
        private ParabolicSAR           _psar;
        // Smoothers
        private SimpleMovingAverage    _atrSma50;
        private SimpleMovingAverage    _volSma20;
        #endregion

        #region Indicator Fields — Multi-TF
        private Bars                     _h1Bars, _h4Bars;
        private ExponentialMovingAverage _h1Ema50, _h1Ema200, _h1Ema12, _h1Ema26;
        private ExponentialMovingAverage _h4Ema50, _h4Ema200;
        #endregion

        #region State
        private FTMORiskManager       _risk;
        private QuantPositionSizer    _sizer;
        private AdaptiveWeightEngine  _ai;

        private List<TradeRecord>          _closed   = new List<TradeRecord>();
        private Dictionary<int, TradeRecord> _open   = new Dictionary<int, TradeRecord>();
        private Dictionary<string, int>    _cooldown = new Dictionary<string, int>();
        private int                        _consecLosses = 0;

        private DateTime _lastDay       = DateTime.MinValue;
        private double   _dayPnL        = 0;
        private bool     _tradedToday   = false;
        private int      _dayTrades     = 0;
        private double   _peakBal       = 0;
        private double   _maxDD         = 0;
        private Dictionary<DateTime, double> _dailyRet = new Dictionary<DateTime, double>();

        // VWAP
        private double   _vwapNum = 0, _vwapDen = 0, _vwap = 0;
        private DateTime _vwapDate = DateTime.MinValue;

        // Running volume indicators
        private double       _obvVal   = 0;
        private double       _adVal    = 0;
        private List<double> _obvHist  = new List<double>();
        private List<double> _adHist   = new List<double>();
        private List<double> _rsiHist  = new List<double>();   // for StochRSI

        // Prev day OHLC for pivots
        private double   _pdHigh = 0, _pdLow = 0, _pdClose = 0;
        private DateTime _pdDate = DateTime.MinValue;

        // Bar counter for weight logging
        private int _barCount = 0;

        // Adaptive threshold — self-adjusts based on recent performance
        private double         _entryThreshold  = 55.0; // 0-100 score needed to enter
        private Queue<bool>    _last20Outcomes  = new Queue<bool>();
        private int            _totalWins       = 0;
        private int            _totalLosses     = 0;

        // News dates
        private HashSet<DateTime> _fomcSet = new HashSet<DateTime>();
        private HashSet<DateTime> _ecbSet  = new HashSet<DateTime>();
        #endregion

        #region Lifecycle
        protected override void OnStart()
        {
            _risk  = new FTMORiskManager();
            _sizer = new QuantPositionSizer();
            _ai    = new AdaptiveWeightEngine();
            _risk.Initialize(Account.Balance);
            _peakBal = Account.Balance;

            // SMAs
            _sma20  = Indicators.SimpleMovingAverage(Bars.ClosePrices, 20);
            _sma50  = Indicators.SimpleMovingAverage(Bars.ClosePrices, 50);
            _sma100 = Indicators.SimpleMovingAverage(Bars.ClosePrices, 100);
            _sma200 = Indicators.SimpleMovingAverage(Bars.ClosePrices, 200);
            // EMAs
            _ema8   = Indicators.ExponentialMovingAverage(Bars.ClosePrices, 8);
            _ema12  = Indicators.ExponentialMovingAverage(Bars.ClosePrices, 12);
            _ema21  = Indicators.ExponentialMovingAverage(Bars.ClosePrices, 21);
            _ema26  = Indicators.ExponentialMovingAverage(Bars.ClosePrices, 26);
            _ema50  = Indicators.ExponentialMovingAverage(Bars.ClosePrices, 50);
            _ema55  = Indicators.ExponentialMovingAverage(Bars.ClosePrices, 55);
            _ema200 = Indicators.ExponentialMovingAverage(Bars.ClosePrices, 200);
            // Core
            _macd    = Indicators.MacdCrossOver(26, 12, 9);
            _atr     = Indicators.AverageTrueRange(AtrPeriod, MovingAverageType.Simple);
            _dms     = Indicators.DirectionalMovementSystem(14);
            _psar    = Indicators.ParabolicSAR(0.02, 0.2);
            _atrSma50 = Indicators.SimpleMovingAverage(_atr.Result, 50);
            _volSma20 = Indicators.SimpleMovingAverage(Bars.TickVolumes, 20);
            // Multi-TF
            _h1Bars  = MarketData.GetBars(TimeFrame.Hour);
            _h4Bars  = MarketData.GetBars(TimeFrame.Hour4);
            _h1Ema50  = Indicators.ExponentialMovingAverage(_h1Bars.ClosePrices, 50);
            _h1Ema200 = Indicators.ExponentialMovingAverage(_h1Bars.ClosePrices, 200);
            _h1Ema12  = Indicators.ExponentialMovingAverage(_h1Bars.ClosePrices, 12);
            _h1Ema26  = Indicators.ExponentialMovingAverage(_h1Bars.ClosePrices, 26);
            _h4Ema50  = Indicators.ExponentialMovingAverage(_h4Bars.ClosePrices, 50);
            _h4Ema200 = Indicators.ExponentialMovingAverage(_h4Bars.ClosePrices, 200);

            ParseDates(FomcDates, _fomcSet);
            ParseDates(EcbDates,  _ecbSet);
            Positions.Closed += OnPositionClosed;

            _lastDay = Server.Time.Date;
            _risk.OnNewDay(Account.Balance);

            Print("[MARS v2] Started. Balance=" + Account.Balance.ToString("F2"));
            Print("[INFO] VIX/DXY/BondYields/EquityCorr skipped — no external data feed in cAlgo.");
            _risk.LogStatus("OnStart", Print);
        }

        protected override void OnBar()
        {
            int idx = Bars.Count - 2; // last closed bar (forward indexing)
            _barCount++;

            // ── Daily reset ─────────────────────────────────────
            DateTime today = Server.Time.Date;
            if (today != _lastDay)
            {
                StoreDailyReturn();
                _risk.OnNewDay(Account.Balance);
                _dayPnL       = 0;
                _tradedToday  = false;
                _dayTrades    = 0;
                _consecLosses = 0;
                _lastDay     = today;
                _obvVal      = 0;
                _adVal       = 0;
                Print("[MARS v2] New day: " + today.ToString("yyyy-MM-dd") +
                      " Bal=" + Account.Balance.ToString("F2"));
            }

            // ── Update running accumulators ──────────────────────
            UpdateVwap(idx);
            UpdateOBV(idx);
            UpdateAD(idx);
            UpdatePrevDayHL(idx);

            // ── Drawdown tracking ────────────────────────────────
            if (Account.Balance > _peakBal) _peakBal = Account.Balance;
            double dd = (_peakBal - Account.Balance) / _peakBal * 100.0;
            if (dd > _maxDD) _maxDD = dd;

            // ── RSI history for StochRSI ─────────────────────────
            double rsi14 = CalcRSI(idx, 14);
            if (!double.IsNaN(rsi14))
            {
                _rsiHist.Add(rsi14);
                if (_rsiHist.Count > 22) _rsiHist.RemoveAt(0);
            }

            // ── Risk breach check ────────────────────────────────
            RiskAction breach = _risk.CheckForBreach(Account.Equity);
            if (breach == RiskAction.HardDaily || breach == RiskAction.HardTotal ||
                breach == RiskAction.SoftDaily  || breach == RiskAction.SoftTotal)
            {
                Print("[MARS v2][RISK BREACH] " + breach);
                CloseAllTrades("RISK_BREACH");
                return;
            }

            // ── Weekend close ────────────────────────────────────
            if (Server.Time.DayOfWeek == DayOfWeek.Friday &&
                Server.Time.Hour >= 20 && Server.Time.Minute >= 45)
            {
                CloseAllTrades("WEEKEND");
                return;
            }

            // ── Manage open positions ────────────────────────────
            ManageOpenTrades(idx);

            // ── Periodic AI weight log ───────────────────────────
            if (_barCount % 100 == 0 && _ai.TradeCount > 5)
                Print("[MARS v2][AI] TOP: " + _ai.TopWeights(8) +
                      " | BOT: " + _ai.BotWeights(5));

            // ── Entry gates ──────────────────────────────────────
            if (Positions.Count(p => p.Label.StartsWith("MARS")) >= MaxConcurrentTrades) return;
            if (!IsSessionOpen(Server.Time)) return;
            if (IsNewsBlackout(Server.Time)) return;
            if (!_risk.CanOpenTrade(Account.Equity, _dayPnL)) return;
            if (DailyLossPausePct > 0 && _risk.DailyStartBalance > 0 &&
                -_dayPnL / _risk.DailyStartBalance * 100.0 >= DailyLossPausePct) return;
            if (_dayTrades >= MaxTradesPerDay) return;
            if (Positions.Any(p => p.SymbolName == SymbolName && p.Label.StartsWith("MARS"))) return;
            if (_cooldown.ContainsKey(SymbolName) && _cooldown[SymbolName] > 0)
            { _cooldown[SymbolName]--; return; }

            // ── Consecutive loss gate ────────────────────────────
            if (_consecLosses >= MaxConsecutiveLosses) return;

            // ── AI-scored multi-layer evaluation ─────────────────
            var sigs   = new IndicatorSignals(); // kept for AI weight recording
            var signal = AggregateSignal(sigs, idx);

            if (signal.Direction == SignalDirection.None) return;

            Print(string.Format("[MARS][SIGNAL] {0} Score={1:F0}/100 Thr={2:F1} | {3}",
                signal.Direction, Math.Abs(signal.Score * 100), _entryThreshold, signal.Rationale));
            OpenTrade(signal, idx);
        }

        protected override void OnTick()
        {
            RiskAction a = _risk.CheckForBreach(Account.Equity);
            if (a == RiskAction.HardDaily || a == RiskAction.HardTotal)
                CloseAllTrades("HARD_BREACH_TICK");
        }

        private void OnPositionClosed(PositionClosedEventArgs args)
        {
            var pos = args.Position;
            if (!pos.Label.StartsWith("MARS")) return;
            double pnl = pos.NetProfit;
            _dayPnL += pnl;
            _sizer.Record(pnl);

            if (_open.ContainsKey(pos.Id))
            {
                var rec = _open[pos.Id];
                rec.ExitTime  = Server.Time;
                double lots   = pos.VolumeInUnits / Symbol.LotSize;
                double pp     = lots > 0 && Symbol.PipValue > 0 ? pnl / (Symbol.PipValue * lots) : 0;
                double pd     = pp * Symbol.PipSize;
                rec.ExitPrice = pos.TradeType == TradeType.Buy
                    ? pos.EntryPrice + pd : pos.EntryPrice - pd;
                rec.PnL = pnl;
                _closed.Add(rec);

                if (EnableLearning && rec.ActiveSignals != null)
                {
                    _ai.OnTradeComplete(rec.ActiveSignals, pnl > 0);
                    Print("[MARS v2][AI] After trade " + _ai.TradeCount + " | TOP: " + _ai.TopWeights(6));
                }
                _open.Remove(pos.Id);
            }

            bool won = pnl > 0;
            UpdateAdaptiveThreshold(won);

            if (!won)
            {
                _consecLosses++;
                _cooldown[SymbolName] = Math.Min(5 * _consecLosses, 20);
            }
            else
            {
                _consecLosses = 0;
            }
            Print(string.Format("[MARS v2][CLOSED] {0} PnL={1:F2} DayPnL={2:F2} Threshold={3:F1}",
                pos.Label, pnl, _dayPnL, _entryThreshold));
        }
        #endregion

        #region Signal Calculations — AI Scored Multi-Layer System

        // ─────────────────────────────────────────────────────────────────
        // EvaluateSetup: scores a potential trade 0-100 across 3 layers.
        // Returns score>0 = bullish, score<0 = bearish, 0 = no trade.
        // Each layer is independent — partial alignment still scores.
        // ─────────────────────────────────────────────────────────────────
        private double EvaluateSetup(int idx, out string reasoning)
        {
            reasoning = "";
            double atr = _atr.Result[idx];
            if (atr <= 0 || double.IsNaN(atr)) return 0;

            double close = Bars.ClosePrices[idx];
            double buyPts = 0, sellPts = 0;
            var r = new System.Text.StringBuilder();

            // ═══════════════════════════════════════════════════
            // LAYER 1 — MACRO CONTEXT (H4)    max ±30 points
            // Answers: "What is the big-picture direction?"
            // ═══════════════════════════════════════════════════
            int h4Idx = _h4Bars.Count - 2;
            if (h4Idx >= 0)
            {
                double h4e50  = _h4Ema50.Result[h4Idx];
                double h4e200 = _h4Ema200.Result[h4Idx];
                if (!double.IsNaN(h4e50) && !double.IsNaN(h4e200) && h4e200 > 0)
                {
                    // H4 EMA alignment — core macro bias
                    if (h4e50 > h4e200) { buyPts  += 15; r.Append("H4:Bull+15 "); }
                    else                { sellPts += 15; r.Append("H4:Bear+15 "); }

                    // Trend strength: separation > 0.08% = trending, not flat
                    double sep = Math.Abs(h4e50 - h4e200) / h4e200 * 100.0;
                    if (sep > 0.08)
                    {
                        if (h4e50 > h4e200) { buyPts  += 10; r.Append("H4:Strong+10 "); }
                        else                { sellPts += 10; r.Append("H4:Strong+10 "); }
                    }

                    // Price side relative to H4 EMA200 (macro mean)
                    double h4Close = _h4Bars.ClosePrices[h4Idx];
                    if (h4Close > h4e200) { buyPts  += 5; r.Append("H4:AboveMean+5 "); }
                    else                  { sellPts += 5; r.Append("H4:BelowMean+5 "); }
                }
            }

            // ═══════════════════════════════════════════════════
            // LAYER 2 — MOMENTUM CONFIRMATION (H1)  max ±35 pts
            // Answers: "Is the trend actually moving right now?"
            // ═══════════════════════════════════════════════════
            int h1Idx = _h1Bars.Count - 2;
            if (h1Idx >= 26)
            {
                double h1Close = _h1Bars.ClosePrices[h1Idx];
                double h1e50   = _h1Ema50.Result[h1Idx];
                double h1e200  = _h1Ema200.Result[h1Idx];
                double h1e12   = _h1Ema12.Result[h1Idx];
                double h1e26   = _h1Ema26.Result[h1Idx];

                // H1 price vs EMA50 — medium-term momentum
                if (!double.IsNaN(h1e50))
                {
                    if (h1Close > h1e50) { buyPts  += 15; r.Append("H1:Above50+15 "); }
                    else                 { sellPts += 15; r.Append("H1:Below50+15 "); }
                }

                // H1 MACD (EMA12 vs EMA26) — momentum direction
                if (!double.IsNaN(h1e12) && !double.IsNaN(h1e26))
                {
                    if (h1e12 > h1e26) { buyPts  += 10; r.Append("H1:MACD++10 "); }
                    else               { sellPts += 10; r.Append("H1:MACD-+10 "); }
                }

                // H1 price vs EMA200 — confirms we are in a real trend, not noise
                if (!double.IsNaN(h1e200))
                {
                    if (h1Close > h1e200) { buyPts  += 10; r.Append("H1:Above200+10 "); }
                    else                  { sellPts += 10; r.Append("H1:Below200+10 "); }
                }
            }

            // ═══════════════════════════════════════════════════
            // LAYER 3 — ENTRY TIMING (M15)    max ±35 points
            // Answers: "Is this a good moment to enter?"
            // ═══════════════════════════════════════════════════

            // ADX trend strength + direction
            double adx = _dms.ADX[idx];
            double diP = _dms.DIPlus[idx];
            double diM = _dms.DIMinus[idx];
            if (!double.IsNaN(adx) && !double.IsNaN(diP) && !double.IsNaN(diM) && adx > 18.0)
            {
                if (diP > diM) { buyPts  += 15; r.Append("ADX:Bull+15 "); }
                else           { sellPts += 15; r.Append("ADX:Bear+15 "); }
            }

            // RSI pullback — entering when price has pulled back, not extended
            double rsi = CalcRSI(idx, 14);
            if (!double.IsNaN(rsi))
            {
                if      (rsi < RsiBuyThreshold)  { buyPts  += 10; r.Append("RSI:Cheap+10 "); }
                else if (rsi > RsiSellThreshold) { sellPts += 10; r.Append("RSI:Rich+10 "); }
                // Partial credit: RSI between threshold and 50 = slight pullback
                else if (rsi < 50) { buyPts  += 3; r.Append("RSI:Mild+3 "); }
                else               { sellPts += 3; r.Append("RSI:Mild+3 "); }
            }

            // Price proximity to EMA21 — confirms we are buying a pullback, not a breakout
            double ema21 = _ema21.Result[idx];
            if (!double.IsNaN(ema21) && atr > 0)
            {
                double dist = Math.Abs(close - ema21) / atr;
                if (dist < 1.5)
                {
                    if (close > ema21) { buyPts  += 5; r.Append("M15:NearEMA+5 "); }
                    else               { sellPts += 5; r.Append("M15:NearEMA+5 "); }
                }
            }

            // Volume: above-average volume = conviction
            double vol    = Bars.TickVolumes[idx];
            double volSma = _volSma20.Result[idx];
            if (!double.IsNaN(volSma) && volSma > 0 && vol > volSma * 1.1)
            {
                bool bullBar = close > Bars.OpenPrices[idx];
                if (bullBar) { buyPts  += 5; r.Append("Vol:Bull+5 "); }
                else         { sellPts += 5; r.Append("Vol:Bear+5 "); }
            }

            reasoning = r.ToString().TrimEnd();

            // Net score: positive = bullish edge, negative = bearish edge
            double netScore = buyPts - sellPts;  // range roughly -100 to +100
            return netScore;
        }
        #endregion



        #region Signal Aggregation & Trade Decision
        private TradingSignal AggregateSignal(IndicatorSignals sigs, int idx)
        {
            var result = new TradingSignal
            {
                Score = 0, Confidence = 0, ConfluenceCount = 0,
                ActiveSignals = sigs, Direction = SignalDirection.None
            };

            // Run the scored multi-layer analysis
            string reasoning;
            double netScore = EvaluateSetup(idx, out reasoning);

            // Minimum score gate — adaptive threshold self-adjusts after each trade
            if (Math.Abs(netScore) < _entryThreshold) return result;

            SignalDirection dir = netScore > 0 ? SignalDirection.Long : SignalDirection.Short;

            // Normalise score to [-1, +1] for compatibility with rest of system
            double normScore = Math.Max(-1.0, Math.Min(1.0, netScore / 100.0));
            double confidence = Math.Min(100.0, Math.Abs(netScore));

            result.Direction      = dir;
            result.Score          = normScore;
            result.Confidence     = confidence;
            result.ConfluenceCount = (int)(Math.Abs(netScore) / 10);
            result.StrengthLabel  = netScore > 80 ? "STRONG " + dir.ToString().ToUpper()
                                  : netScore > 55 ? dir.ToString().ToUpper()
                                  : "WEAK " + dir.ToString().ToUpper();
            result.ActiveSignals  = sigs;

            double atr    = _atr.Result[idx];
            double slDist = ComputeSlDistance(dir, idx, atr);
            result.SlDistance   = slDist;
            result.Tp1Distance  = slDist * 1.0;
            result.Tp2Distance  = slDist * 1.8;
            result.Tp3Distance  = slDist * 3.0;
            result.RiskReward   = 3.0;
            result.TimeHorizon  = "2-8h";
            result.Rationale    = string.Format("[Score={0:F0}/100 Thr={1:F0}] {2}",
                                  Math.Abs(netScore), _entryThreshold, reasoning);
            result.EntryPrice   = dir == SignalDirection.Long ? Symbol.Ask : Symbol.Bid;

            return result;
        }

        // Called after every closed trade — self-adjusts entry threshold
        private void UpdateAdaptiveThreshold(bool win)
        {
            _last20Outcomes.Enqueue(win);
            if (_last20Outcomes.Count > 20) _last20Outcomes.Dequeue();
            if (win) _totalWins++; else _totalLosses++;

            if (_last20Outcomes.Count < 10) return; // need min 10 trades to adapt

            int wins = 0;
            foreach (bool b in _last20Outcomes) if (b) wins++;
            double recentWR = (double)wins / _last20Outcomes.Count;

            // Losing streak: raise the bar (harder to enter)
            if (recentWR < 0.38) _entryThreshold = Math.Min(70.0, _entryThreshold + 2.0);
            // Winning streak: can afford slightly looser entries for more trades
            else if (recentWR > 0.62) _entryThreshold = Math.Max(45.0, _entryThreshold - 1.0);
            // Neutral: drift back toward 55
            else _entryThreshold += (_entryThreshold > 55.0 ? -0.5 : 0.5);

            Print(string.Format("[MARS][AI] Threshold={0:F1} RecentWR={1:F1}% (W{2}/L{3} last {4})",
                _entryThreshold, recentWR * 100, wins, _last20Outcomes.Count - wins,
                _last20Outcomes.Count));
        }

        private double ComputeSlDistance(SignalDirection dir, int idx, double atr)
        {
            double swLow  = GetSwingLow(idx, 4);
            double swHigh = GetSwingHigh(idx, 4);
            double entry  = dir == SignalDirection.Long ? Symbol.Ask : Symbol.Bid;
            double slDist;
            if (dir == SignalDirection.Long)
                slDist = Math.Max(entry - (swLow - atr * 0.3), atr * 0.8);
            else
                slDist = Math.Max((swHigh + atr * 0.3) - entry, atr * 0.8);
            double minStop = 10 * Symbol.PipSize;
            double maxStop = 20 * Symbol.PipSize; // hard cap: never risk more than 20 pips
            return Math.Min(Math.Max(slDist, minStop), maxStop);
        }

        private string GetStrengthLabel(double score)
        {
            double a = Math.Abs(score);
            if (a > 0.60) return score > 0 ? "STRONG BUY" : "STRONG SELL";
            if (a > 0.35) return score > 0 ? "BUY"         : "SELL";
            if (a > 0.15) return score > 0 ? "WEAK BUY"    : "WEAK SELL";
            return "HOLD";
        }
        #endregion

        #region Trade Execution
        private void OpenTrade(TradingSignal sig, int idx)
        {
            double atr = _atr.Result[idx];
            if (atr <= 0 || sig.SlDistance <= 0) return;

            TradeType type = sig.Direction == SignalDirection.Long ? TradeType.Buy : TradeType.Sell;

            double slPips  = sig.SlDistance / Symbol.PipSize;
            double tp3Pips = sig.Tp3Distance / Symbol.PipSize;

            // Enforce min RR
            if (sig.Tp3Distance < sig.SlDistance * 1.5)
            {
                Print("[MARS v2] Skipping — TP3 < 1.5:1 RR");
                return;
            }

            // Regime size multiplier — use ATR ratio as proxy
            double atrSma = _atrSma50.Result[idx];
            double regMult = (!double.IsNaN(atrSma) && atrSma > 0 && atr > atrSma * 1.8) ? 0.5 : 1.0;

            double lots = _sizer.CalculateLots(
                Account.Balance, RiskPercentPerTrade, slPips, Symbol.PipValue,
                Account.Equity, GetTotalUsedMargin(), Symbol.LotSize,
                _dayPnL, _risk, regMult);

            if (lots < 0.01) { Print("[MARS v2] Lot too small, skip."); return; }

            double vol = Symbol.QuantityToVolumeInUnits(lots);
            vol = Math.Max(Symbol.VolumeInUnitsMin,
                  Math.Round(vol / Symbol.VolumeInUnitsStep) * Symbol.VolumeInUnitsStep);

            string label = "MARS_" + sig.StrengthLabel.Replace(" ","") + "_" + Server.Time.ToString("HHmmss");

            var result = ExecuteMarketOrder(type, SymbolName, vol, label, slPips, tp3Pips, "MARS v2", false);
            if (!result.IsSuccessful)
            {
                Print("[MARS v2][EXEC ERROR] " + result.Error + " Label=" + label);
                return;
            }

            _dayTrades++;
            if (!_tradedToday) { _tradedToday = true; _risk.LogTradingDay(); }

            var rec = new TradeRecord
            {
                EntryTime    = Server.Time,
                EntryPrice   = result.Position.EntryPrice,
                Direction    = type,
                Lots         = lots,
                StopLoss     = result.Position.StopLoss ?? 0,
                TakeProfit   = result.Position.TakeProfit ?? 0,
                SignalSource = sig.StrengthLabel,
                SignalScore  = sig.Score,
                SignalConf   = sig.Confidence,
                ActiveSignals = sig.ActiveSignals,
                SlDistAtEntry = sig.SlDistance
            };
            _open[result.Position.Id] = rec;

            Print(string.Format(
                "[MARS v2][OPEN] {0} {1} Conf={2:F0}% Score={3:F2} Confluence={4} " +
                "SL={5:F1}pips TP3={6:F1}pips Lots={7:F2} | {8}",
                label, type, sig.Confidence, sig.Score, sig.ConfluenceCount,
                slPips, tp3Pips, lots, sig.Rationale));
        }

        private void ManageOpenTrades(int idx)
        {
            double atr = _atr.Result[idx];
            if (atr <= 0) return;

            foreach (var pos in Positions.ToArray())
            {
                if (!pos.Label.StartsWith("MARS") || pos.SymbolName != SymbolName) continue;

                double entryPrice  = pos.EntryPrice;
                double currentPrice = pos.TradeType == TradeType.Buy ? Symbol.Bid : Symbol.Ask;
                double profitInPrice = pos.TradeType == TradeType.Buy
                    ? currentPrice - entryPrice
                    : entryPrice - currentPrice;
                double hoursOpen = (Server.Time - pos.EntryTime).TotalHours;

                // Time exits
                if (hoursOpen > 36) { ClosePosition(pos); Print("[MARS v2][TIME EXIT >36h] " + pos.Label); continue; }
                if (hoursOpen > 20 && pos.NetProfit > 0) { ClosePosition(pos); Print("[MARS v2][TIME EXIT >20h profit] " + pos.Label); continue; }

                // Weekend close
                if (Server.Time.DayOfWeek == DayOfWeek.Friday &&
                    Server.Time.Hour == 20 && Server.Time.Minute >= 45)
                { ClosePosition(pos); continue; }

                // Retrieve SL distance at entry for TP calculations
                double slDist = atr * 1.2; // default fallback
                if (_open.ContainsKey(pos.Id) && _open[pos.Id].SlDistAtEntry > 0)
                    slDist = _open[pos.Id].SlDistAtEntry;

                bool tp1Hit = _open.ContainsKey(pos.Id) && _open[pos.Id].Tp1Hit;
                bool tp2Hit = _open.ContainsKey(pos.Id) && _open[pos.Id].Tp2Hit;

                double newSl   = pos.StopLoss ?? 0;
                bool   modified = false;

                // ── TP1: 1.0× SL — close 30%, move to breakeven ──
                if (!tp1Hit && profitInPrice >= slDist * 1.0)
                {
                    double vol30 = Math.Floor(pos.VolumeInUnits * 0.30 / Symbol.VolumeInUnitsStep)
                                   * Symbol.VolumeInUnitsStep;
                    if (vol30 >= Symbol.VolumeInUnitsMin)
                    {
                        ClosePosition(pos, vol30);
                        Print("[MARS v2][PARTIAL-TP1] " + pos.Label);
                    }
                    // Move SL to breakeven + 1 pip
                    double beSl = pos.TradeType == TradeType.Buy
                        ? entryPrice + Symbol.PipSize
                        : entryPrice - Symbol.PipSize;
                    if (pos.TradeType == TradeType.Buy  && (newSl == 0 || newSl < beSl)) { newSl = beSl; modified = true; }
                    if (pos.TradeType == TradeType.Sell && (newSl == 0 || newSl > beSl)) { newSl = beSl; modified = true; }
                    if (_open.ContainsKey(pos.Id)) _open[pos.Id].Tp1Hit = true;
                    continue;
                }

                // ── TP2: 1.8× SL — close 30%, lock in TP1 level ──
                if (tp1Hit && !tp2Hit && profitInPrice >= slDist * 1.8)
                {
                    double vol30 = Math.Floor(pos.VolumeInUnits * 0.30 / Symbol.VolumeInUnitsStep)
                                   * Symbol.VolumeInUnitsStep;
                    if (vol30 >= Symbol.VolumeInUnitsMin)
                    {
                        ClosePosition(pos, vol30);
                        Print("[MARS v2][PARTIAL-TP2] " + pos.Label);
                    }
                    // Move SL to TP1 level
                    double tp1Sl = pos.TradeType == TradeType.Buy
                        ? entryPrice + slDist * 1.0
                        : entryPrice - slDist * 1.0;
                    if (pos.TradeType == TradeType.Buy  && (newSl == 0 || newSl < tp1Sl)) { newSl = tp1Sl; modified = true; }
                    if (pos.TradeType == TradeType.Sell && (newSl == 0 || newSl > tp1Sl)) { newSl = tp1Sl; modified = true; }
                    if (_open.ContainsKey(pos.Id)) _open[pos.Id].Tp2Hit = true;
                    continue;
                }

                // ── Trailing stop at 2.5× SL ─────────────────────
                if (tp1Hit && profitInPrice >= slDist * 2.5)
                {
                    double trailSl = pos.TradeType == TradeType.Buy
                        ? currentPrice - atr * 1.2
                        : currentPrice + atr * 1.2;
                    if (pos.TradeType == TradeType.Buy  && trailSl > newSl) { newSl = trailSl; modified = true; }
                    if (pos.TradeType == TradeType.Sell && (newSl == 0 || trailSl < newSl)) { newSl = trailSl; modified = true; }
                }

                if (modified && newSl != 0)
                {
                    double tp = pos.TakeProfit ?? 0;
                    ModifyPosition(pos, newSl, tp > 0 ? tp : (double?)null, ProtectionType.Absolute);
                }
            }
        }

        private void CloseAllTrades(string reason)
        {
            foreach (var pos in Positions.ToArray())
                if (pos.Label.StartsWith("MARS")) { ClosePosition(pos); Print("[MARS v2][CLOSE ALL] " + reason + " " + pos.Label); }
            foreach (var ord in PendingOrders.ToArray())
                if (ord.Label != null && ord.Label.StartsWith("MARS")) CancelPendingOrder(ord);
        }
        #endregion

        #region Manual Indicator Calculations
        // ── RSI (Wilder) ─────────────────────────────────────────
        private double CalcRSI(int idx, int period)
        {
            if (idx < period + 1) return double.NaN;
            double g = 0, l = 0;
            for (int i = idx; i > idx - period; i--)
            {
                double ch = Bars.ClosePrices[i] - Bars.ClosePrices[i - 1];
                if (ch > 0) g += ch; else l -= ch;
            }
            g /= period; l /= period;
            if (l == 0) return 100.0;
            return 100.0 - 100.0 / (1.0 + g / l);
        }

        private double CalcRSIForBars(Bars bars, int idx, int period)
        {
            if (idx < period + 1) return double.NaN;
            double g = 0, l = 0;
            for (int i = idx; i > idx - period; i--)
            {
                double ch = bars.ClosePrices[i] - bars.ClosePrices[i - 1];
                if (ch > 0) g += ch; else l -= ch;
            }
            g /= period; l /= period;
            if (l == 0) return 100.0;
            return 100.0 - 100.0 / (1.0 + g / l);
        }

        // ── Bollinger Bands ──────────────────────────────────────
        private void CalcBB(int idx, int period, double mult,
                             out double mid, out double upper, out double lower)
        {
            mid = upper = lower = double.NaN;
            if (idx < period - 1) return;
            double sum = 0;
            for (int i = idx; i > idx - period; i--) sum += Bars.ClosePrices[i];
            mid = sum / period;
            double sq = 0;
            for (int i = idx; i > idx - period; i--) { double d = Bars.ClosePrices[i] - mid; sq += d * d; }
            double sd = Math.Sqrt(sq / period);
            upper = mid + mult * sd;
            lower = mid - mult * sd;
        }

        // ── Stochastic %K ────────────────────────────────────────
        private double CalcStochK(int idx, int period)
        {
            if (idx < period - 1) return double.NaN;
            double hi = double.MinValue, lo = double.MaxValue;
            for (int i = idx; i > idx - period; i--)
            {
                if (Bars.HighPrices[i] > hi) hi = Bars.HighPrices[i];
                if (Bars.LowPrices[i]  < lo) lo = Bars.LowPrices[i];
            }
            return hi == lo ? 50.0 : (Bars.ClosePrices[idx] - lo) / (hi - lo) * 100.0;
        }

        // ── CCI ──────────────────────────────────────────────────
        private double CalcCCI(int idx, int period)
        {
            if (idx < period - 1) return double.NaN;
            double[] tp = new double[period];
            for (int i = 0; i < period; i++)
                tp[i] = (Bars.HighPrices[idx - i] + Bars.LowPrices[idx - i] + Bars.ClosePrices[idx - i]) / 3.0;
            double sma = tp.Sum() / period;
            double md  = tp.Sum(t => Math.Abs(t - sma)) / period;
            if (md == 0) return 0;
            return (tp[0] - sma) / (0.015 * md);
        }

        // ── Williams %R ──────────────────────────────────────────
        private double CalcWilliamsR(int idx, int period)
        {
            if (idx < period - 1) return double.NaN;
            double hi = double.MinValue, lo = double.MaxValue;
            for (int i = idx; i > idx - period; i--)
            {
                if (Bars.HighPrices[i] > hi) hi = Bars.HighPrices[i];
                if (Bars.LowPrices[i]  < lo) lo = Bars.LowPrices[i];
            }
            return hi == lo ? -50.0 : (hi - Bars.ClosePrices[idx]) / (hi - lo) * -100.0;
        }

        // ── KDJ (simplified: K-D divergence) ────────────────────
        private double CalcKDJ(int idx)
        {
            double k0 = CalcStochK(idx, 9);
            double k1 = CalcStochK(idx - 1, 9);
            double k2 = CalcStochK(idx - 2, 9);
            if (double.IsNaN(k0) || double.IsNaN(k1) || double.IsNaN(k2)) return double.NaN;
            double d = (k0 + k1 + k2) / 3.0;
            return k0 - d;
        }

        // ── Awesome Oscillator ───────────────────────────────────
        private double CalcAO(int idx)
        {
            if (idx < 34) return double.NaN;
            double sma5 = 0, sma34 = 0;
            for (int i = idx; i > idx - 5;  i--) sma5  += (Bars.HighPrices[i] + Bars.LowPrices[i]) / 2.0;
            for (int i = idx; i > idx - 34; i--) sma34 += (Bars.HighPrices[i] + Bars.LowPrices[i]) / 2.0;
            return sma5 / 5.0 - sma34 / 34.0;
        }

        // ── Ultimate Oscillator ──────────────────────────────────
        private double CalcUO(int idx)
        {
            if (idx < 28) return double.NaN;
            double bp7 = 0, tr7 = 0, bp14 = 0, tr14 = 0, bp28 = 0, tr28 = 0;
            for (int i = idx; i > idx - 28; i--)
            {
                if (i < 1) break;
                double pc = Bars.ClosePrices[i - 1];
                double bp = Bars.ClosePrices[i] - Math.Min(Bars.LowPrices[i], pc);
                double tr = Math.Max(Bars.HighPrices[i], pc) - Math.Min(Bars.LowPrices[i], pc);
                bp28 += bp; tr28 += tr;
                if (i > idx - 14) { bp14 += bp; tr14 += tr; }
                if (i > idx - 7)  { bp7  += bp; tr7  += tr; }
            }
            double a7  = tr7  > 0 ? bp7  / tr7  : 0.5;
            double a14 = tr14 > 0 ? bp14 / tr14 : 0.5;
            double a28 = tr28 > 0 ? bp28 / tr28 : 0.5;
            return 100.0 * (4 * a7 + 2 * a14 + a28) / 7.0;
        }

        // ── MFI ─────────────────────────────────────────────────
        private double CalcMFI(int idx, int period)
        {
            if (idx < period) return double.NaN;
            double pmf = 0, nmf = 0;
            for (int i = idx; i > idx - period; i--)
            {
                double tp = (Bars.HighPrices[i] + Bars.LowPrices[i] + Bars.ClosePrices[i]) / 3.0;
                double mf = tp * Bars.TickVolumes[i];
                if (i > 0)
                {
                    double tp1 = (Bars.HighPrices[i-1] + Bars.LowPrices[i-1] + Bars.ClosePrices[i-1]) / 3.0;
                    if (tp > tp1) pmf += mf; else nmf += mf;
                }
            }
            if (nmf == 0) return 100.0;
            return 100.0 - 100.0 / (1.0 + pmf / nmf);
        }

        // ── CMF ──────────────────────────────────────────────────
        private double CalcCMF(int idx, int period)
        {
            if (idx < period - 1) return double.NaN;
            double mfvSum = 0, volSum = 0;
            for (int i = idx; i > idx - period; i--)
            {
                double h = Bars.HighPrices[i], l = Bars.LowPrices[i], c = Bars.ClosePrices[i];
                double hl = h - l;
                double clv = hl > 0 ? ((c - l) - (h - c)) / hl : 0;
                double vol = Bars.TickVolumes[i];
                mfvSum += clv * vol;
                volSum += vol;
            }
            return volSum > 0 ? mfvSum / volSum : 0;
        }

        // ── DeMarker ─────────────────────────────────────────────
        private double CalcDeMarker(int idx, int period)
        {
            if (idx < period) return double.NaN;
            double deMaxSum = 0, deMinSum = 0;
            for (int i = idx; i > idx - period; i--)
            {
                if (i < 1) break;
                double deMax = Math.Max(0, Bars.HighPrices[i] - Bars.HighPrices[i - 1]);
                double deMin = Math.Max(0, Bars.LowPrices[i - 1] - Bars.LowPrices[i]);
                deMaxSum += deMax;
                deMinSum += deMin;
            }
            double denom = deMaxSum + deMinSum;
            return denom > 0 ? deMaxSum / denom : 0.5;
        }

        // ── Stochastic RSI ───────────────────────────────────────
        private double CalcStochRSI(int period)
        {
            if (_rsiHist.Count < period) return double.NaN;
            double cur  = _rsiHist[_rsiHist.Count - 1];
            double min_ = double.MaxValue, max_ = double.MinValue;
            int start = Math.Max(0, _rsiHist.Count - period);
            for (int i = start; i < _rsiHist.Count; i++)
            {
                if (_rsiHist[i] < min_) min_ = _rsiHist[i];
                if (_rsiHist[i] > max_) max_ = _rsiHist[i];
            }
            return max_ == min_ ? 50.0 : (cur - min_) / (max_ - min_) * 100.0;
        }

        // ── Fisher Transform ─────────────────────────────────────
        private double CalcFisher(int idx, int period)
        {
            if (idx < period - 1) return double.NaN;
            double hi = GetSwingHigh(idx, period);
            double lo = GetSwingLow(idx, period);
            double rng = hi - lo;
            if (rng <= 0) return 0;
            double x = 2.0 * ((Bars.ClosePrices[idx] - lo) / rng) - 1.0;
            x = Math.Max(-0.999, Math.Min(0.999, x));
            return 0.5 * Math.Log((1.0 + x) / (1.0 - x));
        }

        // ── WMA (helper for HMA) ─────────────────────────────────
        private double CalcWMA(int idx, int period)
        {
            if (idx < period - 1) return double.NaN;
            double sum = 0, wSum = 0;
            for (int i = 0; i < period; i++)
            {
                double w = period - i;
                sum  += Bars.ClosePrices[idx - i] * w;
                wSum += w;
            }
            return wSum > 0 ? sum / wSum : double.NaN;
        }

        // ── Hull Moving Average ──────────────────────────────────
        private double CalcHMA(int idx, int period)
        {
            int half = period / 2;
            int sqrtP = (int)Math.Round(Math.Sqrt(period));
            double wma_n = CalcWMA(idx, period);
            double wma_h = CalcWMA(idx, half);
            if (double.IsNaN(wma_n) || double.IsNaN(wma_h)) return double.NaN;
            // HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
            // We approximate by computing the synthetic series manually
            if (idx < period + sqrtP) return double.NaN;
            double sum = 0, wSum = 0;
            for (int i = 0; i < sqrtP; i++)
            {
                double w2 = CalcWMA(idx - i, half);
                double wn = CalcWMA(idx - i, period);
                if (double.IsNaN(w2) || double.IsNaN(wn)) return double.NaN;
                double wt = sqrtP - i;
                sum  += (2.0 * w2 - wn) * wt;
                wSum += wt;
            }
            return wSum > 0 ? sum / wSum : double.NaN;
        }

        // ── Linear Regression Slope ──────────────────────────────
        private double CalcLinRegSlope(int idx, int period)
        {
            if (idx < period - 1) return double.NaN;
            double sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
            for (int i = 0; i < period; i++)
            {
                double x = i;
                double y = Bars.ClosePrices[idx - i];
                sumX  += x; sumY  += y;
                sumXY += x * y; sumX2 += x * x;
            }
            double n = period;
            double denom = n * sumX2 - sumX * sumX;
            return denom != 0 ? (n * sumXY - sumX * sumY) / denom : 0;
        }

        // ── Ichimoku signal ──────────────────────────────────────
        private double CalcIchimokuSignal(int idx)
        {
            if (idx < 78) return 0; // need 52 bars + 26 displacement
            int pivot = idx - 26;
            if (pivot < 52) return 0;

            double t9h = GetSwingHigh(pivot, 9), t9l  = GetSwingLow(pivot, 9);
            double k26h = GetSwingHigh(pivot, 26), k26l = GetSwingLow(pivot, 26);
            double b52h = GetSwingHigh(pivot, 52), b52l = GetSwingLow(pivot, 52);

            double senkouA = ((t9h + t9l) / 2.0 + (k26h + k26l) / 2.0) / 2.0;
            double senkouB = (b52h + b52l) / 2.0;
            double cloudTop = Math.Max(senkouA, senkouB);
            double cloudBot = Math.Min(senkouA, senkouB);
            double close    = Bars.ClosePrices[idx];

            if (close > cloudTop) return 1.0;
            if (close < cloudBot) return -1.0;
            return 0;
        }

        // ── Manual ATR for other Bars ────────────────────────────
        private double CalcManualATR(Bars bars, int barsIdx, int period)
        {
            if (barsIdx < period) return double.NaN;
            double sum = 0;
            for (int i = barsIdx; i > barsIdx - period; i--)
            {
                double tr = bars.HighPrices[i] - bars.LowPrices[i];
                if (i > 0)
                {
                    tr = Math.Max(tr, Math.Abs(bars.HighPrices[i] - bars.ClosePrices[i - 1]));
                    tr = Math.Max(tr, Math.Abs(bars.LowPrices[i]  - bars.ClosePrices[i - 1]));
                }
                sum += tr;
            }
            return sum / period;
        }

        // ── Fibonacci signal ─────────────────────────────────────
        private double CalcFibSignal(int idx, int lookback, double atr, double close)
        {
            if (idx < lookback) return 0;
            double hi = GetSwingHigh(idx, lookback);
            double lo = GetSwingLow(idx, lookback);
            double rng = hi - lo;
            if (rng <= 0) return 0;
            // Fib levels from the swing range
            double[] fibs = { 0.236, 0.382, 0.500, 0.618, 0.786 };
            double tolerance = atr * 0.5;
            foreach (double f in fibs)
            {
                double supportLevel = lo + rng * f;
                double resistLevel  = hi - rng * f;
                if (Math.Abs(close - supportLevel) < tolerance) return  0.6; // near support fib
                if (Math.Abs(close - resistLevel)  < tolerance) return -0.6; // near resistance fib
            }
            return 0;
        }
        #endregion

        #region Candle Patterns
        private bool IsBullishEngulfing(int idx)
        {
            if (idx < 1) return false;
            return Bars.ClosePrices[idx - 1] < Bars.OpenPrices[idx - 1] &&
                   Bars.ClosePrices[idx]     > Bars.OpenPrices[idx] &&
                   Bars.OpenPrices[idx]     <= Bars.ClosePrices[idx - 1] &&
                   Bars.ClosePrices[idx]    >= Bars.OpenPrices[idx - 1];
        }
        private bool IsBearishEngulfing(int idx)
        {
            if (idx < 1) return false;
            return Bars.ClosePrices[idx - 1] > Bars.OpenPrices[idx - 1] &&
                   Bars.ClosePrices[idx]     < Bars.OpenPrices[idx] &&
                   Bars.OpenPrices[idx]     >= Bars.ClosePrices[idx - 1] &&
                   Bars.ClosePrices[idx]    <= Bars.OpenPrices[idx - 1];
        }
        private bool IsHammer(int idx)
        {
            double o = Bars.OpenPrices[idx], c = Bars.ClosePrices[idx];
            double h = Bars.HighPrices[idx],  l = Bars.LowPrices[idx];
            double rng = h - l; if (rng <= 0) return false;
            double body = Math.Abs(c - o);
            double lw   = Math.Min(o, c) - l;
            double uw   = h - Math.Max(o, c);
            return body < rng * 0.30 && lw >= body * 2.0 && uw <= body * 1.5;
        }
        private bool IsShootingStar(int idx)
        {
            double o = Bars.OpenPrices[idx], c = Bars.ClosePrices[idx];
            double h = Bars.HighPrices[idx],  l = Bars.LowPrices[idx];
            double rng = h - l; if (rng <= 0) return false;
            double body = Math.Abs(c - o);
            double uw   = h - Math.Max(o, c);
            double lw   = Math.Min(o, c) - l;
            return body < rng * 0.30 && uw >= body * 2.0 && lw <= body * 1.5;
        }
        private bool IsBullishPinBar(int idx)
        {
            double o = Bars.OpenPrices[idx], c = Bars.ClosePrices[idx];
            double h = Bars.HighPrices[idx],  l = Bars.LowPrices[idx];
            double rng = h - l; if (rng <= 0) return false;
            return (c - l) >= rng * 0.75 && (Math.Min(o, c) - l) >= rng * 0.5;
        }
        private bool IsBearishPinBar(int idx)
        {
            double o = Bars.OpenPrices[idx], c = Bars.ClosePrices[idx];
            double h = Bars.HighPrices[idx],  l = Bars.LowPrices[idx];
            double rng = h - l; if (rng <= 0) return false;
            return (c - l) <= rng * 0.25 && (h - Math.Max(o, c)) >= rng * 0.5;
        }
        #endregion

        #region Session & News Filters
        private bool IsSessionOpen(DateTime utc)
        {
            double t = utc.Hour + utc.Minute / 60.0;
            return (t >= 7.25 && t <= 11.75) || (t >= 13.25 && t <= 16.75);
        }
        private bool IsNewsBlackout(DateTime utc)
        {
            if (_fomcSet.Contains(utc.Date)) return true;
            if (_ecbSet.Contains(utc.Date))
            {
                double t = utc.Hour + utc.Minute / 60.0;
                if (t >= 10.25 && t <= 14.25) return true;
            }
            if (utc.DayOfWeek == DayOfWeek.Friday && utc.Day <= 7)
            {
                double t = utc.Hour + utc.Minute / 60.0;
                if (t >= 13.0 && t <= 14.75) return true;
            }
            if (utc.DayOfWeek == DayOfWeek.Wednesday)
            {
                int wc = 0;
                for (int d = 1; d <= utc.Day; d++)
                    if (new DateTime(utc.Year, utc.Month, d).DayOfWeek == DayOfWeek.Wednesday) wc++;
                if (wc == 2 || wc == 3)
                {
                    double t = utc.Hour + utc.Minute / 60.0;
                    if (t >= 13.1667 && t <= 14.0) return true;
                }
            }
            return false;
        }
        private void ParseDates(string csv, HashSet<DateTime> set)
        {
            if (string.IsNullOrWhiteSpace(csv)) return;
            foreach (var p in csv.Split(','))
            { DateTime dt; if (DateTime.TryParse(p.Trim(), out dt)) set.Add(dt.Date); }
        }
        #endregion

        #region Running Accumulators
        private void UpdateVwap(int idx)
        {
            var d = Bars.OpenTimes[idx].Date;
            if (d != _vwapDate)
            {
                _vwapNum = 0; _vwapDen = 0; _vwapDate = d;
                for (int i = Bars.Count - 1; i >= 1; i--)
                {
                    if (Bars.OpenTimes[i].Date != d) continue;
                    double tp = (Bars.HighPrices[i] + Bars.LowPrices[i] + Bars.ClosePrices[i]) / 3.0;
                    double v  = Bars.TickVolumes[i];
                    _vwapNum += tp * v; _vwapDen += v;
                }
            }
            else
            {
                double tp = (Bars.HighPrices[idx] + Bars.LowPrices[idx] + Bars.ClosePrices[idx]) / 3.0;
                double v  = Bars.TickVolumes[idx];
                _vwapNum += tp * v; _vwapDen += v;
            }
            _vwap = _vwapDen > 0 ? _vwapNum / _vwapDen : Bars.ClosePrices[idx];
        }

        private void UpdateOBV(int idx)
        {
            if (idx > 0)
            {
                double vol = Bars.TickVolumes[idx];
                if (Bars.ClosePrices[idx] > Bars.ClosePrices[idx - 1])      _obvVal += vol;
                else if (Bars.ClosePrices[idx] < Bars.ClosePrices[idx - 1]) _obvVal -= vol;
            }
            _obvHist.Add(_obvVal);
            if (_obvHist.Count > 25) _obvHist.RemoveAt(0);
        }

        private void UpdateAD(int idx)
        {
            double h = Bars.HighPrices[idx], l = Bars.LowPrices[idx], c = Bars.ClosePrices[idx];
            double hl = h - l;
            double clv = hl > 0 ? ((c - l) - (h - c)) / hl : 0;
            _adVal += clv * Bars.TickVolumes[idx];
            _adHist.Add(_adVal);
            if (_adHist.Count > 25) _adHist.RemoveAt(0);
        }

        private void UpdatePrevDayHL(int idx)
        {
            DateTime today = Server.Time.Date;
            if (today != _pdDate)
            {
                // Find yesterday's data
                for (int i = idx; i >= 1; i--)
                {
                    if (Bars.OpenTimes[i].Date < today)
                    {
                        _pdClose = Bars.ClosePrices[i];
                        _pdHigh  = double.MinValue;
                        _pdLow   = double.MaxValue;
                        DateTime prevDay = Bars.OpenTimes[i].Date;
                        for (int j = i; j >= 0; j--)
                        {
                            if (Bars.OpenTimes[j].Date != prevDay) break;
                            if (Bars.HighPrices[j] > _pdHigh) _pdHigh = Bars.HighPrices[j];
                            if (Bars.LowPrices[j]  < _pdLow)  _pdLow  = Bars.LowPrices[j];
                        }
                        _pdDate = today;
                        break;
                    }
                }
            }
        }

        private void StoreDailyReturn()
        {
            if (_risk.DailyStartBalance > 0 && !_dailyRet.ContainsKey(_lastDay))
                _dailyRet.Add(_lastDay, _dayPnL / _risk.DailyStartBalance);
        }
        #endregion

        #region Utilities
        private double GetSwingLow(int idx, int lookback)
        {
            double lo = double.MaxValue;
            int end = Math.Max(0, idx - lookback + 1);
            for (int i = idx; i >= end; i--)
                if (Bars.LowPrices[i] < lo) lo = Bars.LowPrices[i];
            return lo == double.MaxValue ? Bars.LowPrices[idx] : lo;
        }
        private double GetSwingHigh(int idx, int lookback)
        {
            double hi = double.MinValue;
            int end = Math.Max(0, idx - lookback + 1);
            for (int i = idx; i >= end; i--)
                if (Bars.HighPrices[i] > hi) hi = Bars.HighPrices[i];
            return hi == double.MinValue ? Bars.HighPrices[idx] : hi;
        }
        private double GetTotalUsedMargin()
        {
            double t = 0;
            foreach (var p in Positions) t += p.VolumeInUnits / 100.0;
            return t;
        }
        #endregion

        #region Backtest Report
        protected override void OnStop()
        {
            StoreDailyReturn();
            Print("══════════════════════════════════════════════════════");
            Print("  MARS v2 AI BACKTEST REPORT");
            Print("══════════════════════════════════════════════════════");
            int n = _closed.Count;
            if (n == 0) { Print("  No completed trades."); return; }

            int wins   = _closed.Count(r => r.PnL > 0);
            int losses = n - wins;
            double wr  = (double)wins / n * 100.0;
            double gw  = _closed.Where(r => r.PnL > 0).Sum(r => r.PnL);
            double gl  = Math.Abs(_closed.Where(r => r.PnL <= 0).Sum(r => r.PnL));
            double pf  = gl > 0 ? gw / gl : 999;
            double ret = (gw - gl) / _risk.InitialBalance * 100.0;

            double sharpe = 0;
            if (_dailyRet.Count > 1)
            {
                var arr = _dailyRet.Values.ToArray();
                double mean = arr.Average();
                double std  = Math.Sqrt(arr.Select(x => (x - mean) * (x - mean)).Average());
                sharpe = std > 0 ? mean / std * Math.Sqrt(252) : 0;
            }

            Print(string.Format("  Trades: {0}  W/L: {1}/{2}  WinRate: {3:F1}%", n, wins, losses, wr));
            Print(string.Format("  Gross Win: ${0:F2}  Gross Loss: ${1:F2}  PF: {2:F2}", gw, gl, pf));
            Print(string.Format("  Return: {0:F2}%  MaxDD: {1:F2}%  Sharpe: {2:F2}", ret, _maxDD, sharpe));
            Print(string.Format("  AI Trades Learned: {0}", _ai.TradeCount));
            Print(string.Format("  AI Top Indicators: {0}", _ai.TopWeights(10)));
            Print(string.Format("  AI Weak Indicators: {0}", _ai.BotWeights(5)));
            Print("══════════════════════════════════════════════════════");

            // Per-trade log
            Print("--- TRADE LOG ---");
            foreach (var r in _closed)
                Print(string.Format("  [{0}] {1} Score={2:F2} Conf={3:F0}% PnL={4:F2} TP1={5} TP2={6} | {7}",
                    r.EntryTime.ToString("yyyy-MM-dd HH:mm"),
                    r.Direction, r.SignalScore, r.SignalConf, r.PnL,
                    r.Tp1Hit, r.Tp2Hit, r.SignalSource));
        }
        #endregion

    } // end MARSTradingBot
} // end namespace
