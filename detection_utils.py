import numpy as np # for nan
from numpy.random import default_rng
from numba import njit # Just-in-time compilation (performance boost)
from scipy.interpolate import interp1d # Univariate interpolation

import holidays
import pandas as pd
import datetime as dt
import statsmodels.api as sm
import patsy

from IPython.display import display
from tqdm.notebook import trange, tqdm # For progress bars

# Configure plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from brokenaxes import brokenaxes

plt.rcParams['figure.figsize'] = (8.0, 3.0)

def load_call_data():

    halfhours = pd.read_csv("Appels entrants PFT par 0.5h  2017 - 2018.csv",
                            names=["date", "time", "numcalls"])

    # Merge date & time into a datetime object and set it as
    # the index for the dataframe.
    halfhours["date_time"] = pd.to_datetime(halfhours["date"] + ' ' + halfhours["time"], dayfirst=True)
    halfhours = halfhours.set_index("date_time")

    # Make the time column a dt.time object
    halfhours["date"] = pd.to_datetime(halfhours.date, dayfirst=True)
    halfhours["time"] = pd.to_datetime(halfhours.time)

    # Remove saturday afternoons
    isASaturday = halfhours.index.day_name() == "Saturday"
    after1pm = halfhours.index.hour >= 13
    halfhours[isASaturday & after1pm] = np.NaN

    # Remove bizarre Sunday which is thrown in
    halfhours["2017-04-02"] = np.NaN

    # Remove holidays and pseudo-holidays
    pseudoHolidays = ["2015-12-26", "2016-01-02", "2016-12-24",
                  "2016-12-31", "2017-07-15", "2017-12-23",
                 "2017-12-30"]

    dateAHoliday = np.array([d in holidays.France() or str(d) in pseudoHolidays for d in halfhours.index.date])
    halfhours["holiday"] = False
    halfhours.loc[dateAHoliday, "holiday"] = True
    halfhours["holiday"] = halfhours["holiday"].astype(np.bool)

    # Handle other early-closing/late-opening quirks
    halfhours.loc["2017-09-22 11:30":"2017-09-22","holiday"] = True
    halfhours.loc["2018-10-12 12:00":"2018-10-12","holiday"] = True
    halfhours.loc["2018-12-24 16:00":"2018-12-24","holiday"] = True
    halfhours.loc["2018-12-31 16:00":"2018-12-31","holiday"] = True
    halfhours.loc["2017-02-07 17:00":"2017-02-07","holiday"] = True
    halfhours.loc["2018-8-17":"2018-8-17 11:59","holiday"] = True

    mydateparser = lambda x: dt.datetime.strptime(x, "%d/%m/%y")
    daily = pd.read_csv("Appels entrants PFT par jour 2015 - 2018.csv", delimiter=";",
                        names=["date", "numcalls"], parse_dates=["date"], date_parser=mydateparser)

    idx = pd.date_range(daily.date.min(), daily.date.max())
    daily.index = pd.DatetimeIndex(daily.date)
    daily = daily.reindex(idx, fill_value=np.NaN)[["numcalls"]]

    # Remove bizarre Sunday which is thrown in
    daily.loc["2017-04-02"] = np.NaN

    pseudoHolidays = ["2015-12-26", "2016-01-02", "2016-12-24", "2016-12-31",
                      "2017-07-15", "2017-12-23", "2017-12-30"]

    dateAHoliday = np.array([d in holidays.France() or str(d) in pseudoHolidays for d in daily.index.date])
    # daily[dateAHoliday] = np.NaN
    daily["holiday"] = False
    daily.loc[dateAHoliday, "holiday"] = True
    daily["holiday"] = daily["holiday"].astype(np.bool)

    return halfhours, daily

def min_max_dates(df):
    return df.index[0].date(), df.index[-1].date()

def add_nans_to_days(halfhours):
    # Reindex so that we have a NaN entry at the start and end
    # of each day, and NaN days on Sundays.
    startDate, endDate = min_max_dates(halfhours)
    startTime = "07:00:00"; endTime = "19:00:00"
    startDT = pd.to_datetime(f"{startDate} {startTime}", dayfirst=True)
    endDT = pd.to_datetime(f"{endDate} {endTime}", dayfirst=True)

    idx = pd.date_range(startDT, endDT, freq="30min")
    idx = idx[idx.indexer_between_time('07:00:00', '19:00:00')]
    halfhours = halfhours.reindex(idx, fill_value=np.NaN)

    return halfhours

def by_week(daily):
    dailyMI = daily.copy()
    dailyMI.index = [dailyMI.index.weekday, dailyMI.index.to_period('W').rename('Week')]
    return(dailyMI.unstack())

def put_days_between_ticks(numletters=1, dateskip=7, skiplast=False, ax=None):
    # Putting the tick labels between the ticks, adapted the code from
    # https://matplotlib.org/3.1.3/gallery/ticks_and_spines/centered_ticklabels.html
    if not ax:
        ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=12))

    def day_letter(x, pos):
        date = mdates.num2date(x)
        if numletters == 1:
            return date.strftime("%a")[0]
        elif numletters == 3:
            return date.strftime("%a")
        else:
            return date.strftime("%A")

    formatter = mpl.ticker.FuncFormatter(day_letter)
    ax.xaxis.set_minor_formatter(formatter)

    for tick in ax.xaxis.get_minor_ticks():
        tick.tick1line.set_markersize(0)
        tick.tick2line.set_markersize(0)
        tick.label1.set_horizontalalignment('center')

    lastDate = mdates.num2date(ax.get_xlim()[1]).date()

    def infreq_date(x, pos):
        date = mdates.num2date(x)
        day_num = date.strftime("%d")

        if (skiplast and date.day == lastDate.day) or dateskip == 0 or pos % dateskip != 0:
            return ""
        else:
            return "\n" + day_num

    formatter = mpl.ticker.FuncFormatter(infreq_date)
    ax.xaxis.set_major_formatter(formatter)

    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_horizontalalignment('center')

    plt.xticks(rotation=0)

def put_months_between_ticks(numletters=3, dateskip=1, ax=None):
    # Putting the tick labels between the ticks, adapted the code from
    # https://matplotlib.org/3.1.3/gallery/ticks_and_spines/centered_ticklabels.html
    if not ax:
        ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonthday=16))

    def month_letter(x, pos):
        if pos % dateskip != 0:
            return ""

        date = mdates.num2date(x)
        if numletters == 1:
            return date.strftime("%b")[0]
        elif numletters == 3:
            return date.strftime("%b")
        else:
            return date.strftime("%B")

    formatter = mpl.ticker.FuncFormatter(month_letter)
    ax.xaxis.set_minor_formatter(formatter)

    for tick in ax.xaxis.get_minor_ticks():
        tick.tick1line.set_markersize(0)
        tick.tick2line.set_markersize(0)
        tick.label1.set_horizontalalignment('center')

    def infreq_year(x, pos):
        date = mdates.num2date(x)
        year = date.strftime("%Y")

        if date.month == 1:
            return "\n" + year
        else:
            return ""

    formatter = mpl.ticker.FuncFormatter(infreq_year)
    ax.xaxis.set_major_formatter(formatter)

    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_horizontalalignment('center')

    plt.xticks(rotation=0)

# Adapted the following from
# https://scentellegher.github.io/programming/2017/07/15/pandas-groupby-multiple-columns-plot.html
def plot_by_day(df, label="", halfhourly=False):
    dfMI = df.numcalls.copy()
    dfMI.index = [df.index.time,
                  df.index.to_period('D').rename('Day')]

    unstacked = dfMI.unstack()
    unstacked.index = df.index[:len(unstacked.index)]
    plt.plot(unstacked)

    mins = [0, 30] if halfhourly else [0]
    plt.gca().xaxis.set_major_locator(mdates.MinuteLocator(byminute = mins))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    sns.despine()

    d1, d2 = min_max_dates(df)
    if label:
        plt.title(f"{label} ({d1} to {d2})");
    return d1, d2

def time_to_datetime_index(index):
    return(pd.to_datetime([f"{dt.date.today()} {t}" for t in index]))

def plot_median(ss, hourly=True, ax=None, quantile=True):

    if not ax:
        ax = plt.gca()

    if hourly:
        groups = ss.index.time
    else:
        groups = ss.index.dayofweek

    med = ss.groupby(groups).quantile(0.5)
    if hourly:
        med.index = time_to_datetime_index(med.index)
    else:
        dates = [dt.date(2018, 1, d) for d in range(1, len(med)+1)]
        med.index = pd.DatetimeIndex(dates)

    l = ax.plot(med, lw=2)

    if quantile:
        low = ss.groupby(groups).quantile(0.05)
        up = ss.groupby(groups).quantile(0.95)

        low.index = med.index
        up.index = med.index

        ax.fill_between(low.index, low, up, alpha=0.25, color=l[0].get_color(), lw=0)

    if hourly:
        ax.xaxis.set_major_locator(mdates.HourLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    sns.despine()

    return l

def broken_axis(df, leftpad=pd.Timedelta("7H"), sunHours=0):
    d2n = mdates.date2num

    allDays = pd.date_range(df.index[0].date(), df.index[-1], freq="1D")
    sundays = allDays[allDays.day_name() == "Sunday"]

    if len(sundays) == 0:
        return brokenaxes(xlims=[(d2n(df.index[0]), d2n(df.index[-1]))])

    if sunHours == 0:
        sundayStarts = [sun.replace(hour=0) - pd.Timedelta(hours=1, minutes=2) for sun in sundays]
        sundayEnds = [sun.replace(hour=23, minute=59) + pd.Timedelta(minutes=2) for sun in sundays]
    else:
        sundayStarts = [sun.replace(hour=sunHours) for sun in sundays]
        sundayEnds = [sun.replace(hour=24-sunHours) for sun in sundays]



    xParts = [(d2n(df.index[0] - leftpad), d2n(sundayStarts[0]))]
    for i in range(len(sundayStarts)-1):
        xParts.append( (d2n(sundayEnds[i]), d2n(sundayStarts[i+1])) )

    xParts.append( (d2n(sundayEnds[len(sundayStarts)-1]), d2n(df.index[-1])) )

    bax = brokenaxes(xlims=xParts, wspace=.02, d=.005)

    return bax

def trim_nan(df):
    isnan = df.isna().all(axis=1)
    firstValid, lastValid = isnan[~isnan].index[[0, -1]]
    return df.loc[firstValid:lastValid]

def stretch_time(times, scale=2):
    dayStarts = [t.replace(hour=0, minute=0) for t in times]
    timesSecs = [(t - t0).total_seconds() for t, t0 in zip(times, dayStarts)]
    newTimeSecs = [(t-12*60*60)*scale + 12*60*60 for t in timesSecs]
    newTimes = pd.DatetimeIndex([t0 + pd.Timedelta(seconds=int(s)) for s, t0 in zip(newTimeSecs, dayStarts)])
    return newTimes

@njit()
def gen_lorden_criterion(ms, Ns, Ms):
    mLen = len(ms)
    tMax, R = Ms.shape
    M = Ms[-1]

    ExpN_T = np.empty(mLen)
    for i, m in enumerate(ms):
        alarmed = M > m

        N_T = np.empty(R, dtype=np.int64)
        for r in range(R):
            if alarmed[r]:
                T = np.searchsorted(Ms[:,r], m)
            else:
                T = tMax-1

            N_T[r] = Ns[T,r]

        ExpN_T[i] = N_T.mean()

    return ExpN_T

COLOUR = True

if not COLOUR:

    # To make all figures monochrome
    from cycler import cycler

    # Create cycler object. Use any styling from above you please
    monochrome = (cycler('color', ['k']) * \
                  cycler('marker', ['', '^','D', '.']) * \
                  cycler('linestyle', ['solid', 'dotted', 'dashed', 'dashdot', (0, (1, 10))]))
    plt.rcParams['axes.prop_cycle'] = monochrome