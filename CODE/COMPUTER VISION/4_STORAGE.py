#"hour_data" is sent when this file is called.Format: hour_data =["2",1390] (No leading 0s).

import datetime
import json

file_path = r"DATA\FINAL.json"

# Current date and time.
now = datetime.datetime.now()
current_year = str(now.year)
current_month = str(now.month)
current_day = str(now.day)

# Reading from JSON.
with open(file_path, 'r') as json_read:
    data = json.load(json_read)

def find_data_years():
    return list(data.keys())

def find_data_months(data_year):
    return list(data[data_year].keys())

def find_data_days(data_year, data_month):
    return list(data[data_year][data_month].keys())

# Writing to JSON.
def write(hour_data):
    # Check year, month and day and then write accordingly.
    data_years = find_data_years()

    if current_year in data_years:
        data_months = find_data_months(current_year)

        if current_month in data_months:
            data_days = find_data_days(current_year, current_month)

            if current_day in data_days:
                # CASE 1- ADD TO EXISTING DAY.
                data[current_year][current_month][current_day][hour_data[0]] = hour_data[1]
            else:
                # CASE 2- CREATE NEW DAY.
                data[current_year][current_month][current_day] = {}
                data[current_year][current_month][current_day][hour_data[0]] = hour_data[1]
        else:
            # CASE 3-CREATE NEW MONTH->DAY.
            data[current_year][current_month] = {}
            data[current_year][current_month][current_day] = {}
            data[current_year][current_month][current_day][hour_data[0]] = hour_data[1]
    else:
        # CASE 4-CREATE NEW YEAR->MONTH->DAY.
        data[current_year] = {}
        data[current_year][current_month] = {}
        data[current_year][current_month][current_day] = {}
        data[current_year][current_month][current_day][hour_data[0]] = hour_data[1]

    with open(file_path, 'w') as json_write:
        json.dump(data, json_write, indent=4)

    # Calculating total every hour i.e every write.
    total()

# Total for days, months and years.
def total():

    #1-TOTAL FOR DAYS.
    data_years = find_data_years()
    for data_year in data_years:
        data_months = find_data_months(data_year)
        data_months = [x for x in data_months if x !='total']

        for data_month in data_months:
            data_days = find_data_days(data_year, data_month)
            data_days = [x for x in data_days if x !='total']

            for data_day in data_days:
                hours_total = sum(value for key, value in data[data_year][data_month][data_day].items() if key != 'total')
                data[data_year][data_month][data_day]['total'] = hours_total

                with open(file_path, 'w') as json_write:
                    json.dump(data, json_write, indent=4)
   
    #2-TOTAL FOR MONTHS.
    for data_year in data_years:
        data_months = find_data_months(data_year)
        data_months = [x for x in data_months if x !='total']

        for data_month in data_months:
            data_days = find_data_days(data_year, data_month)
            data_days = [x for x in data_days if x !='total']
            days_total=0

            for data_day in data_days:
                days_total += data[data_year][data_month][data_day]['total']

            data[data_year][data_month]['total'] = days_total

            with open(file_path, 'w') as json_write:
                json.dump(data, json_write, indent=4)

    #3-TOTAL FOR YEARS.
    for data_year in data_years:
        data_months = find_data_months(data_year)
        data_months = [x for x in data_months if x !='total']
        months_total=0

        for data_month in data_months:
            months_total += data[data_year][data_month]['total']

        data[data_year]['total'] = months_total

        with open(file_path, 'w') as json_write:
            json.dump(data, json_write, indent=4)