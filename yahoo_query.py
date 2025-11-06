import pandas as pd
import yfinance as yf

TICK = "✅"
CROSS = "❌"

# ==========================================================
#                    DATA FETCHING
# ==========================================================

def get_data(symbol="CL=F", period="3mo", offset="16h30min", tz="America/New_York"):
    """Download OHLC data, align to proper daily session cutoff, and clean Yahoo quirks."""
    data = yf.download(symbol, period=period, interval="1h",
                       auto_adjust=True, progress=False).dropna()

    # --- Fix column naming quirks ---
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[1] for col in data.columns]
    if len(data.columns) >= 5 and len(set(data.columns)) == 1:
        data.columns = ["Open", "High", "Low", "Close", "Volume"][:len(data.columns)]
    elif all("_" in str(c) for c in data.columns):
        data.columns = [str(c).split("_")[-1].title() for c in data.columns]
    elif not {"Open", "High", "Low", "Close"}.issubset(set(data.columns)):
        cols = ["Open", "High", "Low", "Close"][:len(data.columns)]
        data.columns = cols + list(data.columns[len(cols):])
    data = data.rename(columns={"Adj Close": "Close", "AdjClose": "Close"})

    # --- Normalize timezone safely ---
    if data.index.tz is None:
        data.index = data.index.tz_localize("UTC")
    else:
        data.index = data.index.tz_convert("UTC")

    # --- FTSE special handling (closes 9 PM NY / 02:00 UTC) ---
    if symbol == "^FTSE":
        shifted = data.copy()
        shifted.index = shifted.index - pd.Timedelta(hours=2)
        daily = shifted.resample("1D").agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last"
        }).dropna()
        daily.index = daily.index + pd.Timedelta(hours=2)
        data = daily
    else:
        # Other assets → normal local-time resample
        data = data.tz_convert(tz)
        data = data.resample("1D", offset=offset).agg({
            "Open": "first", "High": "max", "Low": "min", "Close": "last"
        }).dropna()

    required = {"Open", "High", "Low", "Close"}
    if not required.issubset(data.columns):
        raise KeyError(f"Missing expected columns: {required - set(data.columns)}")

    return data


# ==========================================================
#                    HELPER FUNCTIONS
# ==========================================================

def get_weekly_data(data):
    return data.resample("W").agg({
        "Open": "first", "High": "max", "Low": "min", "Close": "last"
    }).dropna()

def is_inside_week(data):
    weekly = get_weekly_data(data)
    if len(weekly) < 2:
        return False
    return (weekly["High"].iloc[-1] <= weekly["High"].iloc[-2]) and \
           (weekly["Low"].iloc[-1] >= weekly["Low"].iloc[-2])

def in_previous_week_range(data):
    weekly = get_weekly_data(data)
    if len(weekly) < 2:
        return False

    # Use the *previous* completed week (not current one)
    prev_week_high = weekly["High"].iloc[-2]
    prev_week_low  = weekly["Low"].iloc[-2]

    # Get data for current (still forming) week
    current_week_number = data.index[-1].isocalendar().week
    this_week = data.loc[data.index.isocalendar().week == current_week_number]
    if this_week.empty:
        return False

    curr_high = this_week["High"].max()
    curr_low  = this_week["Low"].min()

    return (curr_high <= prev_week_high) and (curr_low >= prev_week_low)


# --- Daily logic based on last fully closed session ---
def is_high_of_month(data):  return data["High"].iloc[-2] == data["High"].tail(22).max()
def is_high_of_week(data):   return data["High"].iloc[-2] == data["High"].tail(5).max()
def is_low_of_month(data):   return data["Low"].iloc[-2]  == data["Low"].tail(22).min()
def is_low_of_week(data):    return data["Low"].iloc[-2]  == data["Low"].tail(5).min()
def is_red_day(data):        return data["Close"].iloc[-2] < data["Open"].iloc[-2] if len(data) > 1 else False
def is_green_day(data):      return data["Close"].iloc[-2] > data["Open"].iloc[-2] if len(data) > 1 else False
def is_inside_day(data):     return (data["High"].iloc[-2] <= data["High"].iloc[-3]) and \
                                    (data["Low"].iloc[-2]  >= data["Low"].iloc[-3])


# ==========================================================
#                    ANALYSIS WRAPPER
# ==========================================================

def analyze_symbol(symbol="CL=F", offset="16h30min", tz="America/New_York"):
    """Run all checks for a single symbol."""
    data = get_data(symbol, offset=offset, tz=tz)
    hom = is_high_of_month(data)
    lom = is_low_of_month(data)
    frd = is_red_day(data)
    fgd = is_green_day(data)

    # --- Compute signal ---
    if hom and frd:
        signal = "<span style='color:red;font-weight:bold;'>SHORT</span>"
    elif lom and fgd:
        signal = "<span style='color:green;font-weight:bold;'>LONG</span>"
    else:
        signal = "–"

    return {
        "Symbol": symbol,
        "Signal": signal,
        "Inside Week":  TICK if is_inside_week(data) else CROSS,
        "Inside PW Range": TICK if in_previous_week_range(data) else CROSS,
        "HOM": TICK if hom else CROSS,
        "HOW": TICK if is_high_of_week(data) else CROSS,
        "FRD": TICK if frd else CROSS,
        "LOM": TICK if lom else CROSS,
        "LOW": TICK if is_low_of_week(data) else CROSS,
        "FGD": TICK if fgd else CROSS,
        "Inside Day": TICK if is_inside_day(data) else CROSS,
    }


# ==========================================================
#                    MAIN LOGIC
# ==========================================================

def main():
    groups = {
        "Commodities": {
            "CL=F": "WTI Crude Oil",
            "GC=F": "Gold",
            "SI=F": "Silver",
            "HG=F": "Copper",
            "PL=F": "Platinum",
            "NG=F": "Natural Gas",
        },
        "Indices": {
            "^GSPC": "S&P 500",
            "^NDX": "Nasdaq 100",
            "^DJI": "Dow Jones",
            "^FTSE": "FTSE 100",
            "^N225": "Nikkei 225",
        },
        "FX": {
            "EURUSD=X": "EUR/USD",
            "GBPUSD=X": "GBP/USD",
            "USDJPY=X": "USD/JPY",
            "AUDUSD=X": "AUD/USD",
            "USDCAD=X": "USD/CAD",
        },
        "Crypto": {
            "BTC-USD": "Bitcoin",
            "ETH-USD": "Ethereum",
        }
    }

    tz_offsets = {
        "Commodities": ("America/New_York", "16h30min"),
        "FX": ("America/New_York", "17h"),
        "Crypto": ("UTC", "00h"),
        "^GSPC": ("America/New_York", "16h"),
        "^NDX": ("America/New_York", "16h"),
        "^DJI": ("America/New_York", "16h"),
        "^FTSE": ("UTC", "0h"),  # handled specially
        "^N225": ("UTC", "0h"),
    }

    html_tables = []
    no_data_template = {
        "Signal": "–",
        "Inside Week": "No data", "Inside PW Range": "No data",
        "HOM": "No data", "HOW": "No data", "FRD": "No data",
        "LOM": "No data", "LOW": "No data", "FGD": "No data",
        "Inside Day": "No data"
    }

    for group_name, symbols in groups.items():
        print(f"\n=== {group_name.upper()} ===")
        group_results = []

        for sym, name in symbols.items():
            print(f"[+] Analyzing {sym} ({name})...")
            tz, offset = tz_offsets.get(sym, tz_offsets.get(group_name, ("America/New_York", "16h30min")))

            try:
                test = yf.download(sym, period="5d", interval="1d",
                                   auto_adjust=True, progress=False)
                if test is None or test.empty:
                    raise ValueError("empty")
            except Exception:
                print(f"⚠️  {sym} returned no recent data — skipping.")
                group_results.append({"Asset": name, "Symbol": sym, **no_data_template})
                continue

            try:
                res = analyze_symbol(sym, offset=offset, tz=tz)
                res["Asset"] = name
                group_results.append(res)
            except Exception as e:
                print(f"⚠️  {sym} failed: {e}")
                row = {"Asset": name, "Symbol": sym}
                row.update({k: "Error" for k in no_data_template})
                group_results.append(row)

        df = pd.DataFrame(group_results)
        if not df.empty:
            df = df[["Asset", "Symbol", "Signal", "Inside Week", "Inside PW Range",
                     "HOM", "HOW", "FRD", "LOM", "LOW", "FGD", "Inside Day"]]
            html_tables.append(f"<h2>{group_name}</h2>{df.to_html(index=False, escape=False)}")

    # ---------------- HTML OUTPUT ---------------- #
    final_html = f"""
    <html>
    <head>
        <meta charset='utf-8'>
        <title>Market Signal Dashboard</title>
        <style>
            body {{ font-family: Arial, sans-serif; padding: 40px; background: #fafafa; color: #222; }}
            table {{ border-collapse: collapse; margin-top: 20px; width: 90%; }}
            th, td {{ border: 1px solid #ccc; padding: 8px 12px; text-align: center; }}
            th {{ background-color: #f2f2f2; }}
            td {{ font-size: 1.1em; }}
            h2 {{ margin-top: 40px; color: #333; }}
        </style>
    </head>
    <body>
        <h1>Market Signal Dashboard</h1>
        {"".join(html_tables)}
        <p>Last updated: {pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M')} UTC</p>
    </body>
    </html>
    """

    with open("index.html", "w", encoding="utf-8") as f:
        f.write(final_html)

    print("\n✅ Grouped dashboard generated: index.html")


# ==========================================================
#                    ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()
