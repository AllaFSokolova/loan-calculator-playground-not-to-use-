# Нажмите Shift + Enter для выполнения этой клетки, ее запустите один раз, потом поставте знак "#" перед кодом pip install pyxirr ----->>>>>
# !pip install pyxirr

# Нажмите Shift + Enter для выполнения этой клетки ----->>>>>
import pandas as pd
from datetime import datetime, timedelta
from pyxirr import xirr


def payment_calendar_builder(start_date, payment_period, loan_period):

    if not isinstance(start_date, str):
        start_date = start_date.strftime('%Y-%m-%d')
        
    date = datetime.strptime(start_date, '%Y-%m-%d')
    payment_calendar = [date]

    end_date = date + timedelta(days=loan_period)

    while date < end_date:
        date = date + timedelta(days=payment_period)
        if date <= end_date:
            payment_calendar.append(date)

    df = pd.DataFrame(payment_calendar, columns=['payment_date'])
    return df


def building_payments_schema (comission, payment_period, loan_period):
    period = payment_period
    local_comission = comission
    percents_per_year = loan_period - 3

    intervals = []
    coeficients = []
    rates = []
    supports = []
    comissions = []
    sums = []

    total_sum = 0
    i = 1

    while total_sum < percents_per_year and i <= 30:
        # interval = i
        days = float(input(f"Введите к-во дней в интервале №{i}: "))
        coef = days // period
        rate = float(input(f"Введите ставку для интервала №{i}, %: "))
        support = float(input(f"Введите комиссию за обслуживание для интервала №{i}, %: "))
        if i == 1:
            local_comission = local_comission
        else:
            local_comission = 0

        coeficients.append(coef)
        rates.append(rate)
        supports.append(support)
        comissions.append(local_comission)

        if i == 1:
            sum_i = rate * coef * period + support * coef * period + local_comission
        else:
            sum_i = rate * coef * period + support * coef * period
        sums.append(sum_i)

        total_sum = sum(sums)

        i += 1

    base_df = pd.DataFrame({
        'Коэфициент': coeficients,
        'Ставка': rates,
        'Комиссия за обслуживание': supports,
        'Разовая комиссия' : comissions,
        'Проценты начисленые': sums
    })

    df = base_df

    if sum(df['Проценты начисленые']) > percents_per_year:
        df.loc[df.index[-1], 'Проценты начисленые'] = percents_per_year - sum(df.iloc[:-1]['Проценты начисленые'])
        df.loc[df.index[-1], 'Коэфициент'] = df.loc[df.index[-1], 'Проценты начисленые'] / (
            df.loc[df.index[-1], 'Ставка']
            + df.loc[df.index[-1], 'Комиссия за обслуживание']
            + df.loc[df.index[-1], 'Разовая комиссия']) / period // 1
        df.loc[df.index[-1], 'Проценты начисленые'] = df.loc[df.index[-1], 'Коэфициент'] * (
            df.loc[df.index[-1], 'Ставка']
            + df.loc[df.index[-1], 'Комиссия за обслуживание']
            + df.loc[df.index[-1], 'Разовая комиссия']
        ) * period
    else:
        df = df
    return df


# adding dates
def adding_dates (df, start_date, payment_period):
    period = payment_period
    date = datetime.strptime(start_date, '%Y-%m-%d')

    df.loc[0, 'date'] = date + timedelta(days=df.loc[0, 'Коэфициент'] * period)

    for i in range(1, len(df)):
        df.loc[i, 'date'] = df.loc[i-1, 'date'] + timedelta(days=df.loc[i, 'Коэфициент'] * period)

    date_ranges = []

    date_range = pd.date_range(start=date, end=df.loc[0, 'date']).tolist()
    date_ranges.append(date_range)

    for i in range(1, len(df)):
        start = df.loc[i-1, 'date'] + timedelta(days=1)
        end = df.loc[i, 'date']
        date_range = pd.date_range(start=start, end=end).tolist()
        date_ranges.append(date_range)

    df['date_ranges'] = date_ranges
    return df

def last_rate_calculation (df, payment_calendar):

    rest_percent = (loan_period - 3) - sum(df['Проценты начисленые'])

    max_date_df = max(df.date)
    max_date_payment_date = max(payment_calendar.payment_date)

    rest_days = (max_date_payment_date - max_date_df).days
    if rest_days == 0:
        last_percent = 0  # or set to a default value
    else:
        last_percent = rest_percent / rest_days

    return last_percent

def product_constructor(df, payment_calendar_df, amount, comission, payment_period, last_rate):
    for i in range(1, len(payment_calendar_df)):
        for j in range(len(df)):
            if payment_calendar_df.loc[i, 'payment_date'] in df.loc[j, 'date_ranges']:
                payment_calendar_df.loc[i, 'rate'] = df.loc[j, 'Ставка']
                payment_calendar_df.loc[i, 'support'] = df.loc[j, 'Комиссия за обслуживание']

    if last_rate > 0.01:
        payment_calendar_df.iloc[1:, payment_calendar_df.columns.get_loc('rate')] = payment_calendar_df.iloc[1:, payment_calendar_df.columns.get_loc('rate')].fillna(last_rate)
    else:
        payment_calendar_df.iloc[1:, payment_calendar_df.columns.get_loc('rate')] = payment_calendar_df.iloc[1:, payment_calendar_df.columns.get_loc('rate')].fillna(0.01)

    payment_calendar_df['comission'] = 0.0
    payment_calendar_df.loc[1, 'comission'] = comission

    payment_calendar_df['amount'] = 0.0
    payment_calendar_df.loc[0, 'amount'] = -amount
    payment_calendar_df.loc[payment_calendar_df.index[-1], 'amount'] = amount
    payment_calendar_df = payment_calendar_df.fillna(0)

    payment_calendar_df['rate_absolute'] = round(amount * payment_calendar_df['rate'] / 100 * payment_period, 2)
    payment_calendar_df['support_absolute'] = round(amount * payment_calendar_df['support'] / 100 * payment_period, 2)
    payment_calendar_df['comission_absolute'] = round(amount * payment_calendar_df['comission'] / 100, 2)

    payment_calendar_df['rate'] = round(payment_calendar_df['rate'], 4)
    payment_calendar_df['support'] = round(payment_calendar_df['support'], 4)
    payment_calendar_df['comission'] = round(payment_calendar_df['comission'], 4)

    payment_calendar_df['CF'] = payment_calendar_df['rate_absolute'] + payment_calendar_df['support_absolute'] + payment_calendar_df['comission_absolute'] + payment_calendar_df['amount']
    return payment_calendar_df


def result_descriptions (payment_calendar_df, schema_dates_df, amount, loan_period):
    # XIRR
    payment_calendar_df['payment_date'] = pd.to_datetime(payment_calendar_df['payment_date'])
    payment_calendar_df['CF'] = pd.to_numeric(payment_calendar_df['CF'])

    xirr_value = xirr(payment_calendar_df['payment_date'], payment_calendar_df['CF'])
    xirr_day = (1+xirr_value)**(1/365)-1

    # Value of loan
    value = sum(payment_calendar_df.CF) - amount
    # Cost of loan
    cost = sum(payment_calendar_df.CF)

    # print("-------------------------------- >>>>>>> Расчетные показатели <<<<<< -----------------------------------")
    # print(f"XIRR year: {round(xirr_value * 100, 2)}%" )
    # print(f"XIRR day: {round(xirr_day * 100, 2)}%" )
    # print(f"Общие расходы по кредиту: {round(cost, 2)} грн." )
    # print(f"Общая стоимость по кредита: {round(value, 2)} грн." )
    # print(f"Среднедневная номинальная ставка: {round(cost/amount/loan_period*100,4)}%")
   # print(f"Количество дней, когда достигается 1%/день при заданых условиях: {(schema_dates_df['date_ranges'].iloc[-1][-1] - min(payment_calendar_df.payment_date)).days} дней" )
   # print(f"Дата последнего начисления процентов: {schema_dates_df['date_ranges'].iloc[-1][-1]}" )
    # print("-------------------------------- >>>>>>> График платежей (Схема процентов) <<<<<< -----------------------------------")
    # display(payment_calendar_df[['payment_date', 'rate', 'support', 'comission']])


def result_function (start_date, payment_period, loan_period, amount, comission):
    df = building_payments_schema (comission, payment_period, loan_period)
    df_dates = adding_dates (df, start_date, payment_period)

    payment_calendar_df = payment_calendar_builder (start_date, payment_period, loan_period)

    last_rate = last_rate_calculation(df_dates, payment_calendar_df)

    final = product_constructor(df_dates, payment_calendar_df, amount, comission, payment_period, last_rate)

    result_descriptions (final, df_dates, amount, loan_period)
    return final[['payment_date', 'rate', 'support', 'comission']], df

# 1 - календарь платежей
def payment_calendar_builder(start_date, payment_period, loan_period):
    
    if not isinstance(start_date, str):
        start_date = start_date.strftime('%Y-%m-%d')
    
    date = datetime.strptime(start_date, '%Y-%m-%d')
    payment_calendar = [date]
    
    end_date = date + timedelta(days=loan_period)
    
    while date < end_date:
        date = date + timedelta(days=payment_period)
        if date <= end_date:
            payment_calendar.append(date)
    
    df = pd.DataFrame(payment_calendar, columns=['payment_date'])
    return df
    
# 3 собрать последние даты месяцев в периоде
def collect_end_of_month_dates(start_date, loan_period):

    date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = date + timedelta(days=loan_period)
    
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    
    # Generate a range of dates from start to end, with a frequency of 'M' (month-end)
    month_end_dates = pd.date_range(start=start_date, end=end_date, freq='ME')
    
    # Convert to list of date objects (without time)
    month_end_dates_list = [date.date() for date in month_end_dates]
    timestamp_list = [pd.Timestamp(date) for date in month_end_dates_list]

    end_dates_df = pd.DataFrame(timestamp_list, columns=['payment_date'])
    
    return end_dates_df

# xnpv
def xnpv(rate, values, dates):
    '''Equivalent of Excel's XNPV function.

    >>> from datetime import date
    >>> dates = [date(2010, 12, 29), date(2012, 1, 25), date(2012, 3, 8)]
    >>> values = [-10000, 20, 10100]
    >>> xnpv(0.1, values, dates)
    -966.4345...
    '''
    if rate <= -1.0:
        return float('inf')
    d0 = dates[0]    # or min(dates)
    return sum([ vi / (1.0 + rate)**((di - d0).days / 365.0) for vi, di in zip(values, dates)])

# adding data_ranges
def adding_date_ranges (df):
    date_ranges = []
    for i in range(1, len(df)):
        start_date = df.loc[i - 1, 'payment_date'] + pd.Timedelta(days=1)
        end_date = df.loc[i, 'payment_date']
        date_ranges.append(pd.date_range(start=start_date, end=end_date).tolist())
    
    date_ranges.insert(0, [])
    
    df['date_ranges'] = date_ranges
    return df
def rates_schema (product_4_5, start_date, payment_period, loan_period, amount, comission):
    if product_4_5:
        df, payment_schema = result_function (start_date, payment_period, loan_period, amount, comission)
        df.loc[df.index[-1], 'rate'] = 1.0
        print("Product 4-5")
    else:
        df, payment_schema = result_function (start_date, payment_period, loan_period, amount, comission)
# here we can replace on function adding_date_ranges (df)
    date_ranges = []
    for i in range(1, len(df)):
        start_date = df.loc[i - 1, 'payment_date'] + pd.Timedelta(days=1)
        end_date = df.loc[i, 'payment_date']
        date_ranges.append(pd.date_range(start=start_date, end=end_date).tolist())
    
    date_ranges.insert(0, [])
    
    df['date_ranges'] = date_ranges
    
    print("-------------------------------- >>>>>>> График платежей (Схема процентов) <<<<<< -----------------------------------")
    display(df[['payment_date', 'rate', 'support', 'comission']])
    return df, payment_schema

def payment_calendar_builder(start_date, payment_period, loan_period):

    if not isinstance(start_date, str):
        start_date = start_date.strftime('%Y-%m-%d')
    
    date = datetime.strptime(start_date, '%Y-%m-%d')
    payment_calendar = [date]
    
    end_date = date + timedelta(days=loan_period)
    
    while date < end_date:
        date = date + timedelta(days=payment_period)
        if date <= end_date:
            payment_calendar.append(date)
    
    df = pd.DataFrame(payment_calendar, columns=['payment_date'])
    return df
    
def collect_payments(num_body_payment):
    # Lists to store user inputs
    dates = []
    body_amounts = []

    # Collect user inputs
    for i in range(1, num_body_payment + 1):
        date = input(f"Введите дату в формате ГГГГ-ММ-ДД № {i}: ")
        body_amount = float(input(f"Введите сумму тела № {i}: "))  # Ensure this is a float for monetary values
        dates.append(date)
        body_amounts.append(body_amount)

    # Create a DataFrame from the collected data
    payments_df = pd.DataFrame({
        'payment_date': dates,
        'body_amount': body_amounts
    })

    # Convert the 'date' column to datetime
    payments_df['payment_date'] = pd.to_datetime(payments_df['payment_date'])

    return payments_df


def calendar_body (dynamic_body_paments, start_date, payment_period, loan_period, amount):
    payment_calendar = payment_calendar_builder(start_date, payment_period, loan_period)
    if dynamic_body_paments:
        num_body_payment = int(input("Введите количество дополнительных платежей: "))
        payments_df = collect_payments(num_body_payment)
        merged_df = pd.merge(payment_calendar, payments_df, on='payment_date', how='outer')
        merged_df.loc[0,'body_amount'] = -amount
        merged_df.loc[merged_df.index[-1],'body_amount'] = - merged_df['body_amount'].iloc[:-1].sum()
        merged_df = merged_df.fillna(0.0)
        calendar_body_df = merged_df
    else:
        payment_calendar['body_amount'] = 0.0
        payment_calendar.loc[0, 'body_amount'] = -amount
        payment_calendar.loc[payment_calendar.index[-1], 'body_amount'] = amount
        calendar_body_df = payment_calendar
    
    
    date_ranges = []
    for i in range(1, len(calendar_body_df)):
        start_date = calendar_body_df.loc[i - 1, 'payment_date'] + pd.Timedelta(days=1)
        end_date = calendar_body_df.loc[i, 'payment_date']
        date_ranges.append(pd.date_range(start=start_date, end=end_date).tolist())
    
    date_ranges.insert(0, [])
    
    calendar_body_df['date_ranges'] = date_ranges
    
    return calendar_body_df

def daily_rates_amount_schema (rates_df, body_df, accrued_period):
    one_day_df = payment_calendar_builder(start_date, accrued_period, loan_period)
    for i in range(1, len(one_day_df)):
        for j in range(len(rates_df)):
            if one_day_df.loc[i, 'payment_date'] in rates_df.loc[j, 'date_ranges']:
                one_day_df.loc[i, 'rate'] = rates_df.loc[j, 'rate']
                one_day_df.loc[i, 'support'] = rates_df.loc[j, 'support']
                    
    for i in range(len(one_day_df)):
        for j in range(len(rates_df)):
            if one_day_df.loc[i, 'payment_date'] == rates_df.loc[j, 'payment_date']:
                one_day_df.loc[i, 'comission'] = rates_df.loc[j, 'comission']
    
    for i in range(len(one_day_df)):
        for j in range(len(body_df)):
            if one_day_df.loc[i, 'payment_date'] == body_df.loc[j, 'payment_date']:
                one_day_df.loc[i, 'amount'] = body_df.loc[j, 'body_amount']
    
    one_day_df = one_day_df.fillna(0.0)
    
    return one_day_df

def shift_one_day_payment(one_day_df, amount, accrued_end):
    if accrued_end:
        for i in range(1, len(one_day_df)):
                one_day_df.loc[i, 'rate_absolute'] = one_day_df.loc[i, 'rate'] * (-one_day_df.loc[:i-1, 'amount'].sum()) / 100
                one_day_df.loc[i,'support_absolute'] = one_day_df.loc[i, 'support'] * (-one_day_df.loc[:i-1, 'amount'].sum()) / 100
                one_day_df['comission_absolute'] = one_day_df['comission'] * amount / 100
    else:
        for i in range(1, len(one_day_df)-1):
            one_day_df.loc[0, 'rate_absolute'] = one_day_df.loc[1, 'rate'] * amount / 100
            one_day_df.loc[0,'support_absolute'] = one_day_df.loc[1, 'support'] * amount / 100
            one_day_df.loc[i, 'rate_absolute'] = one_day_df.loc[i + 1, 'rate'] * (-one_day_df.loc[:i-1, 'amount'].sum()) / 100
            one_day_df.loc[i,'support_absolute'] = one_day_df.loc[i + 1, 'support'] * (-one_day_df.loc[:i-1, 'amount'].sum()) / 100
            one_day_df['comission_absolute'] = one_day_df['comission'] * amount / 100
        
    one_day_df = one_day_df.fillna(0.0)
    return one_day_df

def rates_payment_calendar (rates_absolute, rates_df):
    rates_absolute_ = rates_absolute[['payment_date',
           'rate_absolute', 'support_absolute', 'comission_absolute']].copy()
    
    for i in range(len(rates_absolute_)):
        for j in range(len(rates_df)):
            if rates_absolute_.loc[i, 'payment_date'] in rates_df.loc[j, 'date_ranges']:
                rates_absolute_.loc[i, 'calendar_date'] = rates_df.loc[j, 'payment_date']
    
    rates_absolute_.loc[0, 'calendar_date'] = rates_absolute_.loc[1, 'calendar_date']
    
    rates_absolute_ = rates_absolute_.drop(columns=['payment_date'])
    
    rates_absolute_grouped = rates_absolute_.groupby('calendar_date', as_index=False).sum()
    
    return rates_absolute_grouped

def rates_payment_calendar (rates_absolute, rates_df):
    rates_absolute_ = rates_absolute[['payment_date',
           'rate_absolute', 'support_absolute', 'comission_absolute']].copy()
    
    for i in range(len(rates_absolute_)):
        for j in range(len(rates_df)):
            if rates_absolute_.loc[i, 'payment_date'] in rates_df.loc[j, 'date_ranges']:
                rates_absolute_.loc[i, 'calendar_date'] = rates_df.loc[j, 'payment_date']
    
    rates_absolute_.loc[0, 'calendar_date'] = rates_absolute_.loc[1, 'calendar_date']
    
    rates_absolute_ = rates_absolute_.drop(columns=['payment_date'])
    
    rates_absolute_grouped = rates_absolute_.groupby('calendar_date', as_index=False).sum()
    
    return rates_absolute_grouped


def first_part_builder (start_date, 
                        payment_period,
                        loan_period,
                        amount,
                        comission,
                        product_4_5, 
                        dynamic_body_paments,
                        accrued_period,
                        accrued_end):
    rates_df, payment_schema = rates_schema(product_4_5, start_date, payment_period, loan_period, amount, comission)
    
    body_df = calendar_body (dynamic_body_paments, start_date, payment_period, loan_period, amount)
    
    daily_schema = daily_rates_amount_schema (rates_df, body_df, accrued_period)
    
    rates_absolute = shift_one_day_payment(daily_schema, amount, accrued_end)
    
    rates_payments = rates_payment_calendar(rates_absolute, rates_df)
    cf_df = calculating_cf(rates_payments, body_df)
    cf_df.rename(columns={'payment_date': 'payment_date', 
                        'CF': 'CF',
                         'body_amount': 'CF_main_loan',
                        'rate_absolute': 'CF_interest',
                        'support_absolute': 'CF_support',
                        'comission_absolute': 'CF_comission'}, inplace=True)
    return cf_df, rates_absolute

def collect_end_of_month_dates(start_date, loan_period):

    date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = date + timedelta(days=loan_period)
    
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    
    # Generate a range of dates from start to end, with a frequency of 'M' (month-end)
    month_end_dates = pd.date_range(start=start_date, end=end_date, freq='ME')
    
    # Convert to list of date objects (without time)
    month_end_dates_list = [date.date() for date in month_end_dates]
    timestamp_list = [pd.Timestamp(date) for date in month_end_dates_list]

    end_dates_df = pd.DataFrame(timestamp_list, columns=['payment_date'])
    
    return end_dates_df
    
def append_end_month (start_date, loan_period, payment_calendar_df):
    end_dates_df = collect_end_of_month_dates(start_date = start_date, loan_period = loan_period)
    
    end_dates_df['CF'] = 0.0
    end_dates_df['CF_main_loan'] = 0.0
    end_dates_df['CF_comission'] = 0.0
    end_dates_df['CF_support'] = 0.0
    end_dates_df['CF_interest'] = 0.0
    
    merged = pd.concat([payment_calendar_df, end_dates_df], ignore_index=True)
    sorted = merged.sort_values(by='payment_date', ascending=True).reset_index(drop=True)
    
    sorted.CF = sorted.CF_main_loan + sorted.CF_comission + sorted.CF_interest + sorted.CF_support
    idx = sorted.groupby('payment_date')['CF_interest'].idxmax()
    
    result_df = sorted.loc[idx].reset_index(drop=True)
    
    return result_df

def second_part_df_builder (main_part, daily_percents_sum, start_date, loan_period, comission, amount):
    main_part_full = append_end_month (start_date, loan_period, main_part)
    first_part_df = main_part_full
    end_of_month_dates_list = first_part_df['payment_date'].dt.date.tolist()
    
    end_m_dates = first_part_df[(first_part_df['CF_interest'] == 0) & (
        first_part_df['payment_date'].dt.date.isin(end_of_month_dates_list))]['payment_date'].tolist()[1:]
    
    first_part_df['XNPV'] = 0.0
    
    first_part_df.loc[0, 'XNPV_principal_loan_amount'] = amount
    for i in range(1,len(first_part_df)):
        first_part_df.loc[i, 'XNPV_principal_loan_amount'] = first_part_df.loc[i - 1, 'XNPV_principal_loan_amount'] - first_part_df.loc[i, 'CF_main_loan']
        
    first_part_df['XNPV_accrued_interest'] = 0.0
    
    first_part_df['XNPV_unamortized discount/premium'] = 0.0
    first_part_df.loc[0, 'XNPV_unamortized discount/premium'] = round(- (comission * amount / 100), 2)
    
    first_part_df['XNPV_reserve'] = 0.0
    
    first_part_df.loc[0, 'XNPV'] = (first_part_df.loc[0,'XNPV_principal_loan_amount'] + 
                                    first_part_df.loc[0, 'XNPV_accrued_interest'] +
                                    first_part_df.loc[0, 'XNPV_unamortized discount/premium'] + 
                                    first_part_df.loc[0, 'XNPV_reserve'] 
                                   )
    
    # xirr
    dates = first_part_df['payment_date'].to_list()
    date_list = [date_obj.date() for date_obj in dates]
    amounts = first_part_df['CF'].to_list()
        
    xirr_ = xirr(pd.DataFrame({"dates": date_list, "amounts": amounts}))
    xirr_day = (1+xirr_)**(1/365)-1   
    
    # xnpv
    for i in range(1, len(first_part_df)):
        dates = first_part_df['payment_date'][i:].to_list()
        amounts = first_part_df['CF'][i:].to_list()
        date_list = [date_obj.date() for date_obj in dates]
            
        first_part_df.loc[i, 'XNPV'] = xnpv(xirr_, amounts, date_list) - first_part_df.loc[i, 'CF']
        
    first_part_df['payment_diff'] = first_part_df['payment_date'].diff().dt.days.fillna(0).astype(int)
    
    first_part_df = adding_date_ranges (first_part_df)
    
    percents_accrued_df = rates_payment_calendar(daily_percents_sum, first_part_df)
    percents_accrued_df['sum_payments'] = (
        percents_accrued_df['rate_absolute'] + 
        percents_accrued_df['support_absolute']
        )
    
    merged_df = first_part_df.merge(
        percents_accrued_df[['calendar_date', 'sum_payments']],
        left_on='payment_date',
        right_on='calendar_date',
        how='left'
    )
    merged_df = merged_df.fillna(0.0)
    
    for i in range(1,len(merged_df)):
        merged_df.loc[i, 'XNPV_accrued_interest'] = (merged_df.loc[0:i, 'sum_payments'].sum() - 
                                                     merged_df.loc[0:i, 'CF_interest'].sum() - 
                                                     merged_df.loc[0:i, 'CF_support'].sum()
                                                    )
    
    
    merged_df.loc[1:, 'XNPV_unamortized discount/premium'] = (merged_df.loc[1:, 'XNPV'] - 
                                                     merged_df.loc[1:, 'XNPV_principal_loan_amount'] - 
                                                     merged_df.loc[1:, 'XNPV_accrued_interest']
                                                    )
    
    
    
    print(f"XIRR год: {round(xirr_ * 100, 2)}%" )
    print(f"XIRR день: {round(xirr_day * 100, 2)}%" )
    
    return merged_df

def third_part_df (merged_df):
    second_part_df = merged_df
    second_part_df['effective_interest_total'] = 0.0
    second_part_df['nominal_interest_total'] = second_part_df['sum_payments']
    second_part_df['discount_total'] = 0.0
    
    for i in range(1, len(second_part_df)):
        second_part_df.loc[i, 'discount_total'] = second_part_df.loc[i, 'XNPV_unamortized discount/premium'] - second_part_df.loc[i-1, 'XNPV_unamortized discount/premium']
    
        second_part_df.loc[i, 'effective_interest_total'] = second_part_df.loc[i, 'nominal_interest_total'] + second_part_df.loc[i, 'discount_total']
    
    final_df = second_part_df
    return final_df


def rename_cut (final_df):
    final_df.rename(columns={'payment_date': 'Дата', 
                         'CF': 'Денежные потоки - Всего',
                        'CF_main_loan': 'Сумма кредита',
                        'CF_comission': 'Первоначальная комиссия (дисконт)',
                        'CF_interest': 'Проценты',
                        'CF_support': 'Комиссия за обслуживание',
                        'XNPV': 'Балансовая стоимость (амортизированная себестоимость) - Всего',
                        'XNPV_principal_loan_amount': 'Тело',
                        'XNPV_accrued_interest': 'Начисленные проценты',
                        'XNPV_unamortized discount/premium': 'Неамортизированный дисконт/премия',
                        'XNPV_reserve': 'Резерв',
                        'payment_diff': 'Расчетный период',
                        # 'xirr_year': 'Эффективная ставка (год)',
                        # 'xirr_daily': 'Эффективная ставка (день)',
                        'effective_interest_total': 'Всего по эфективной ставке',
                        'nominal_interest_total': 'Всего по номинальной ставке',
                        'discount_total': 'Корректировка доходов (амортизация)'}, inplace=True)
    
    final_df = final_df[['Дата', 'Денежные потоки - Всего', 'Сумма кредита',
       'Первоначальная комиссия (дисконт)', 'Проценты', 'Комиссия за обслуживание',
       'Балансовая стоимость (амортизированная себестоимость) - Всего', 'Тело',
       'Начисленные проценты', 'Неамортизированный дисконт/премия', 'Резерв',
       'Всего по эфективной ставке',
       'Всего по номинальной ставке', 'Корректировка доходов (амортизация)'
       #                   , 'Эффективная ставка (год)',
       # 'Эффективная ставка (день)'
                        ]]
    return final_df

def function_builder (start_date, 
                        payment_period,
                        loan_period,
                        amount,
                        comission,
                        product_4_5, 
                        dynamic_body_paments,
                        accrued_period,
                        accrued_end):
    main_part, daily_percents_sum = first_part_builder (start_date, 
                        payment_period,
                        loan_period,
                        amount,
                        comission,
                        product_4_5, 
                        dynamic_body_paments,
                        accrued_period,
                        accrued_end)
    second_df = second_part_df_builder (main_part, daily_percents_sum, start_date, loan_period, comission, amount)
    third_df = third_part_df (second_df)
    final = rename_cut (third_df)
    return final

# Parameters
start_date = '2024-10-24'
# start_date = '2024-02-26' : Дата начала кредита или финансового расчета (26 февраля 2024 года)

payment_period = 8
# payment_period = 6 : Периодичность платежей в днях (6 дней между платежами)

loan_period = 360
# loan_period = 360 : Общая продолжительность кредитного периода в днях (360 дней)

amount = 10000
# amount = 1000: Сумма кредита или первоначальная сумма (10,000 грн.)

comission = 15
# comission = 15: Разовая комиссия за оформление кредита (15%)

product_4_5 = True
# product_4_5 = True: Это продукт 4 или 5
# product_4_5 = False: Это новый продукт

dynamic_body_paments = False
# dynamic_body_paments = True: Были или предусмотрены частичные погашения тела в определенные даты
# dynamic_body_paments = False: Погашение тела во время пользовования кредитом не предусмотрено

accrued_period = 1
#accrued_period = 1: Начисления процентов раз в 1 день (каждый день)

accrued_end = True
# accrued_end = True: Начисления на конец периода
# accrued_end = False: Начисления на конец начало периода


part_with_comission = function_builder (start_date, 
                        payment_period,
                        loan_period,
                        amount,
                        comission,
                        product_4_5, 
                        dynamic_body_paments,
                        accrued_period,
                        accrued_end)