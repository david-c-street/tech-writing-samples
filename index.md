**A technical copywriting exercise** – *Intended for an audience of programming beginners who are interested in data science as a hobby or as a possible profession*

## Anatomy of a basic data science project

*David C Street*

Statistics educator Christoper Chatfield defined his field as "the science of making decisions in the face of uncertainty." Is data science any different? By Chatfield's definition, not at all.

At its core, data science involves using a computer to draw conclusions from large amounts of information. This fits perfectly what we now think of when we see the word *data* – computers and digitally stored information. The *science* part is the same as in Chatfield's definition. It means trying to make better and more informed decisions.

Advanced forms of data science do require a lot of math, mostly statistics and probability. We'll briefly discuss possibilities for advanced methods later, but the analysis presented here uses nothing more than sums and averages.

I've included all the code and outputs. We won't dive into the code in detail, it's there for context.  For reference, the basic Python used here would take a beginner 5-10 hours to learn.

Most data science work takes place in this kind of environment. There are programs that can simplify some tasks, showing data and results in a fancier graphical user interface. But data scientists spend much of their time with raw code and raw outputs, like what we'll see below. Many prefer it that way. 

There are many ways to shorten and improve the code below. Add-ons to Python, packages, could handle many of these tasks in a single line of code. However, those packages take extra time to learn and hide complex operations going on in the background. We'll stick to basic code that is easier to read and interpret.

### The assignment: Make a profitable free app for Android and iOS

We're looking for an idea to make a free mobile phone application. The app will be available in English for Google Android and Apple iOS.

We assume that all revenue comes from ads. So we need to maximize the number of users and user engagement with ads. We've been told to make two key assumptions, which will help us narrow down the possibilities:

1. High number of reviews = High engagement with apps of that kind. High engagement means more ad impressions and more revenue.

2. Low average review score = Users are currently unsatisfied with the selection of apps in the category. We have a better chance of standing out in such a category.

We want to find a kind of app, across both the Google Play Store and the Apple App Store, that gets a lot of reviews, but currently has a low average review score.

### Step 1: Get the data

This step can be complicated and time-consuming. In this case, someone has already done the hard work for us. We'll use these two data sets for our analysis:

- Play store data ([Kaggle](https://www.kaggle.com/lava18/google-play-store-apps))
- Apple App Store data ([Kaggle](https://www.kaggle.com/ramamet4/app-store-apple-data-set-10k-apps))

Without such data sets, we would have to write code to go to a database, mine the web or collect the data in some other way. For most projects, a significant amount of time has to be reserved in the beginning just for data collection and quality assurance. Reliable data is essential. As one saying goes, "Garbage in, garbage out." Without reliable data, even the best project is doomed from the start.

Luckily for beginners, there are a lot of free, ready-to-go datasets available to play around with. The above datasets are from Kaggle, which is a good place to look.

Let's start by reading the data into Python and seeing what it looks like.

**Import the data from file and convert it to a format that Python can work with**


```python
# import both data sets
from csv import reader
openfile = open("./data/app-store/apple_store.csv")
readfile = reader(openfile)
app_store_list = list(readfile)

openfile = open("./data/play-store/play_store.csv")
readfile = reader(openfile)
play_store_list = list(readfile)
```

**Write a function to display a summary of the data in a way we can more easily read**


```python
# function to display data and show data set properties
def explore_data(dataset, start, end, rows_and_columns=False, col_indices=False):
    dataset_slice = dataset[start:end]
    print("Data sample:")
    for row in dataset_slice:
        print(row)
        print('\n') # adds a new (empty) line after each row

    if rows_and_columns:
        print('Number of rows: ' + str(len(dataset)))
        print('Number of columns: ' + str(len(dataset[0])))
    if col_indices:
        print("Column indices:")
        col_index = 0
        for col in dataset[0]:
            print(str(col_index) + " : " + str(col))
            col_index += 1
    return None
```

**Use the diplay function to look at the data**

*Play Store data*


```python
explore_data(play_store_list,1,2,True, True)
```

    Data sample:
    ['Photo Editor & Candy Camera & Grid & ScrapBook', 'ART_AND_DESIGN', '4.1', '159', '19M', '10,000+', 'Free', '0', 'Everyone', 'Art & Design', 'January 7, 2018', '1.0.0', '4.0.3 and up']
    
    
    Number of rows: 10842
    Number of columns: 13
    Column indices:
    0 : App
    1 : Category
    2 : Rating
    3 : Reviews
    4 : Size
    5 : Installs
    6 : Type
    7 : Price
    8 : Content Rating
    9 : Genres
    10 : Last Updated
    11 : Current Ver
    12 : Android Ver


*App Store data*


```python
explore_data(app_store_list,1,2,True, True)
```

    Data sample:
    ['1', '281656475', 'PAC-MAN Premium', '100788224', 'USD', '3.99', '21292', '26', '4', '4.5', '6.3.5', '4+', 'Games', '38', '5', '10', '1']
    
    
    Number of rows: 7198
    Number of columns: 17
    Column indices:
    0 : 
    1 : id
    2 : track_name
    3 : size_bytes
    4 : currency
    5 : price
    6 : rating_count_tot
    7 : rating_count_ver
    8 : user_rating
    9 : user_rating_ver
    10 : ver
    11 : cont_rating
    12 : prime_genre
    13 : sup_devices.num
    14 : ipadSc_urls.num
    15 : lang.num
    16 : vpp_lic


The data sample we see above is one "row" of the data set. Each entry represents a single app. You can probably imagine a spreadsheet filled with this kind of data.

Data spreadsheet   | App name | App category | ...
---  | -------- | ------------ | ---
App 1 | Photobook maker  | Art and design | ...
App 2 | Instagram | Social media | ...
... | ... | ... | ...

The columns are listed starting from zero, because that's how Python counts entries in a list. We'll use those column index numbers throughout this project to refer to a particular piece of data. Even though Python represents the data in a slightly different way, we'll make functions to handle the data as if it were in a spreadsheet with rows and columns.

We already see what data will be most useful: categories (prime_genre for the App Store), ratings and reviews. In the next step, we'll take a closer look at what that data looks like.

### Step 2: Check and clean the data

In a perfect world, data would be error-free and consistent. Most data is not, so we need to check and clean our data.

As a warning, this is the longest part of the project. That's not uncommon. In most data science projects, dealing with data errors, outliers and quality takes a lot of time. Some data scientists joke that data cleaner would be a more appropriate job title.

We'll start by breaking the data down by category to check for anything suspicious.

**Write a function to count how many apps have a particular value in one column**


```python
# function to create frequency tables from list of lists data set
def freq_table_ll(ll_data, col_index, header_row=True, percent=False):
    freq_dict = {}
    if header_row:
        start_row = 1
        num_rows = len(ll_data) - 1
    else:
        start_row = 0
        num_rows = len(ll_data)
    for row in ll_data[start_row:]:
        value = row[col_index]
        if value in freq_dict:
            freq_dict[value] += 1
        else:
            freq_dict[value] = 1
    if percent:
        for key in freq_dict:
            freq_dict[key] = 100*freq_dict[key]/num_rows
    return freq_dict
```

**Check the categories for bad or missing data**


```python
freq_table_apple_categories = freq_table_ll(app_store_list,12)
print(freq_table_apple_categories)
```

    {'Productivity': 178, 'Photo & Video': 349, 'Entertainment': 535, 'Travel': 81, 'Sports': 114, 'Food & Drink': 63, 'Book': 112, 'Music': 138, 'Health & Fitness': 180, 'Finance': 104, 'Business': 57, 'Social Networking': 167, 'Utilities': 248, 'News': 75, 'Lifestyle': 144, 'Medical': 23, 'Games': 3862, 'Catalogs': 10, 'Shopping': 122, 'Navigation': 46, 'Reference': 64, 'Weather': 72, 'Education': 453}


The format is messy, but we'll improve that later. The important thing is the data looks reasonable. There are no mispelled categories or weird entries with only one app.


```python
freq_table_play_categories = freq_table_ll(play_store_list,1)
print(freq_table_play_categories)
```

    {'LIBRARIES_AND_DEMO': 85, 'AUTO_AND_VEHICLES': 85, 'BUSINESS': 460, 'ENTERTAINMENT': 149, 'MEDICAL': 463, 'MAPS_AND_NAVIGATION': 137, '1.9': 1, 'LIFESTYLE': 382, 'GAME': 1144, 'BOOKS_AND_REFERENCE': 231, 'SHOPPING': 260, 'HOUSE_AND_HOME': 88, 'FAMILY': 1972, 'COMICS': 60, 'PHOTOGRAPHY': 335, 'PARENTING': 60, 'WEATHER': 82, 'ART_AND_DESIGN': 65, 'PERSONALIZATION': 392, 'DATING': 234, 'EVENTS': 64, 'BEAUTY': 53, 'HEALTH_AND_FITNESS': 341, 'VIDEO_PLAYERS': 175, 'FINANCE': 366, 'PRODUCTIVITY': 424, 'COMMUNICATION': 387, 'TRAVEL_AND_LOCAL': 258, 'SPORTS': 384, 'SOCIAL': 295, 'EDUCATION': 156, 'TOOLS': 843, 'NEWS_AND_MAGAZINES': 283, 'FOOD_AND_DRINK': 127}


First of all, we see that the Play Store has different categories. This will require interpretation later. More importantly for right now, we see what looks like an error. There is a category called "1.9" with only one app. To look closer, we'll have to write a quick function to search the data.

**Write a function to search a single column for a value and give us the row number**


```python
# function to get row index (or indices) of a value in a list of list data set
def get_row_ll(ll_data, search_val, col_index, header_row=True):
    if header_row:
        start_row = 1
        row_index = 1
    else:
        start_row = 0
        row_index = 0
    row_ind_list = []
    for row in ll_data[start_row:]:
        val = row[col_index]
        if search_val == val:
            row_ind_list.append(row_index)
        row_index += 1
    return row_ind_list
```

**Use the function to find what row that the "1.9" category is in**


```python
print(get_row_ll(play_store_list, "1.9",1))
```

    [10473]


**Look at the contents of that entire row**


```python
print(play_store_list[10473])
```

    ['Life Made WI-Fi Touchscreen Photo Frame', '1.9', '19', '3.0M', '1,000+', 'Free', '0', 'Everyone', '', 'February 11, 2018', '1.0.19', '4.0 and up']


It looks like the category just got left out of this row. But let's quickly check the length of the row to see.

**Calculate the length of this row and of the header row for comparison**


```python
print(len(play_store_list[10473]))
print(len(play_store_list[0]))
```

    12
    13


Indeed, this row is too short by one element, it should have 13 elements. A quick [search](https://play.google.com/store/apps/details?id=com.lifemade.internetPhotoframe) shows that the category should be "LIFESTYLE".

**Let's copy the existing data and insert the missing data**


```python
play_store_list[10473] = ['Life Made WI-Fi Touchscreen Photo Frame', 'LIFESTYLE', '1.9', '19', '3.0M', '1,000+', 'Free', '0', 'Everyone', 'Lifestyle', 'February 11, 2018', '1.0.19', '4.0 and up']
```

The App Store data has just one column for categories (prime_genre). But the Play Store data has a "Category" and a "Genres" column. Let's see what's in the "Genres" column.

**Check the the second category column of the Play Store data**


```python
freq_table_play_genres = freq_table_ll(play_store_list,9)
print(freq_table_play_genres)
```

    {'Parenting;Music & Video': 6, 'Communication;Creativity': 1, 'Food & Drink': 127, 'Music;Music & Video': 3, 'Lifestyle;Pretend Play': 1, 'Art & Design;Pretend Play': 2, 'Books & Reference;Creativity': 1, 'Board;Brain Games': 15, 'Business': 460, 'Adventure;Action & Adventure': 13, 'Photography': 335, 'Simulation;Action & Adventure': 11, 'Education;Music & Video': 5, 'Educational;Creativity': 5, 'Puzzle;Brain Games': 19, 'Parenting;Brain Games': 1, 'Casual;Brain Games': 13, 'Simulation': 200, 'Educational;Pretend Play': 19, 'Social': 295, 'Card;Action & Adventure': 2, 'Entertainment;Education': 1, 'Tools;Education': 1, 'Video Players & Editors;Music & Video': 3, 'Entertainment;Creativity': 3, 'Casual;Music & Video': 2, 'Personalization': 392, 'Board;Pretend Play': 1, 'Educational': 37, 'Arcade': 220, 'Entertainment': 623, 'Sports': 398, 'Entertainment;Pretend Play': 2, 'Lifestyle': 382, 'Trivia': 38, 'Shopping': 260, 'Parenting;Education': 7, 'Books & Reference;Education': 2, 'Racing;Action & Adventure': 20, 'Entertainment;Music & Video': 27, 'Strategy;Creativity': 1, 'Word': 29, 'Role Playing': 109, 'Comics': 59, 'News & Magazines': 283, 'Role Playing;Pretend Play': 5, 'Education': 549, 'Education;Pretend Play': 23, 'Health & Fitness;Education': 1, 'Finance': 366, 'House & Home': 88, 'Communication': 387, 'Video Players & Editors': 173, 'Action;Action & Adventure': 17, 'Productivity': 424, 'Arcade;Action & Adventure': 16, 'Art & Design;Action & Adventure': 2, 'Libraries & Demo': 85, 'Music': 22, 'Board': 44, 'Dating': 234, 'Racing': 98, 'Events': 64, 'Health & Fitness': 341, 'Education;Education': 50, 'Beauty': 53, 'Racing;Pretend Play': 1, 'Simulation;Education': 3, 'Role Playing;Brain Games': 1, 'Puzzle;Education': 1, 'Auto & Vehicles': 85, 'Maps & Navigation': 137, 'Education;Creativity': 7, 'Education;Action & Adventure': 6, 'Travel & Local;Action & Adventure': 1, 'Strategy;Action & Adventure': 2, 'Card': 48, 'Travel & Local': 257, 'Educational;Action & Adventure': 4, 'Casino': 39, 'Sports;Action & Adventure': 4, 'Puzzle;Action & Adventure': 5, 'Action': 365, 'Strategy': 107, 'Entertainment;Action & Adventure': 3, 'Puzzle': 140, 'Art & Design;Creativity': 7, 'Casual;Action & Adventure': 21, 'Casual;Education': 3, 'Puzzle;Creativity': 2, 'Role Playing;Education': 1, 'Adventure;Education': 2, 'Video Players & Editors;Creativity': 2, 'Comics;Creativity': 1, 'Educational;Education': 41, 'Adventure;Brain Games': 1, 'Casual;Pretend Play': 31, 'Health & Fitness;Action & Adventure': 1, 'Strategy;Education': 1, 'Board;Action & Adventure': 3, 'Art & Design': 58, 'Role Playing;Action & Adventure': 7, 'Trivia;Education': 1, 'Parenting': 46, 'Adventure': 75, 'Educational;Brain Games': 6, 'Music & Audio;Music & Video': 1, 'Simulation;Pretend Play': 4, 'Medical': 463, 'Education;Brain Games': 5, 'Card;Brain Games': 1, 'Casual;Creativity': 7, 'Arcade;Pretend Play': 1, 'Entertainment;Brain Games': 8, 'Weather': 82, 'Lifestyle;Education': 1, 'Tools': 842, 'Casual': 193, 'Books & Reference': 231}


What a mess. If we look at each entry, we see that they are subcategories. This could be helpful, but we also lack something similar for the Apple Store data. It would be hard to compare between the two datasets, which we have to do in the end.

We would also have to clean up this column. Each entry is often several entries joined together by semicolons. Take "Travel & Local;Action & Adventure" for example.

We won't consider this column for now. We know it's there if we decide later we want to explore more details in the Play Store.

We still need to check the data in a couple more columns. The ratings and number of reviews data will both be useful.

The way we imported the data means that the raw data will be in "string" form. This means that Python treats all the data like words, even if there are numbers. We'll need to convert these columns to numbers to do sums and averages later. Let's check to see if that will work how we expect it to.

**Write a function to check if numerical columns can be converted properly**


```python
#function to count and display NaNs in a numerical column
from math import isnan
def count_nan_ll(ll_data, col_index, header_row=True):
    nan_entries = []
    nan_count = 0
    if header_row:
        start_row = 1
    else:
        start_row = 0
    for row in ll_data[start_row:]:
        num = float(row[col_index])
        if isnan(num):
            nan_count += 1
            nan_entries.append(row[col_index])
    print(nan_entries[-5:])
    return nan_count
```

**Check all four numerical columns we intend to use**


```python
count_nan_ll(play_store_list,2)
```

    ['NaN', 'NaN', 'NaN', 'NaN', 'NaN']
    1474




```python
count_nan_ll(play_store_list,3)
```

    []
    0




```python
count_nan_ll(app_store_list,6)
```

    []
    0




```python
count_nan_ll(app_store_list,8)
```

    []
    0



It looks like the rating column in the Play Store has over a thousand NaN (not a number) entries. Let's take a closer look.

**Show the values for a few rows with "NaN" ratings**


```python
nan_rows = get_row_ll(play_store_list,"NaN",2)
print(play_store_list[nan_rows[10]])
print(play_store_list[nan_rows[100]])
print(play_store_list[nan_rows[300]])
print(play_store_list[nan_rows[600]])
print(play_store_list[nan_rows[1000]])
```

    ['Y! Mobile menu', 'BUSINESS', 'NaN', '9', '1.2M', '100,000+', 'Free', '0', 'Everyone', 'Business', 'April 9, 2018', '1.0.5', '6.0 and up']
    ['Dare EMS Protocols', 'MEDICAL', 'NaN', '0', '20M', '10+', 'Free', '0', 'Everyone 10+', 'Medical', 'July 26, 2018', '1.8.3', '4.1 and up']
    ['Gun AK 47', 'FAMILY', 'NaN', '3', '6.2M', '1,000+', 'Free', '0', 'Everyone', 'Entertainment', 'April 18, 2017', '1.0', '2.3 and up']
    ['Bacterial Vaginosis Symptoms & Treatment', 'MEDICAL', 'NaN', '0', '8.7M', '500+', 'Free', '0', 'Everyone', 'Medical', 'January 18, 2018', '1.1', '4.0.3 and up']
    ['DN Driver', 'FAMILY', 'NaN', '0', '7.5M', '5+', 'Free', '0', 'Everyone', 'Education', 'June 9, 2018', '1.0.2', '4.1 and up']


These apps have a low number of reviews. For low numbers of reviews, an average rating isn't displayed in the Play Store. We'll keep this in mind later and just ignore these apps when calculating average ratings.

Next let's check to see if any of the entries are for the same app. For the Play Store, we'll use the app names to check for duplicates. For the App Store, we can do a little better and use a unique ID number.

**Write a function to check for duplicate entries**


```python
# function to check for duplicate entries in list of lists data set
def get_dups_ll(ll_data, name_col, header_row=True):
    if header_row:
        start_row = 1
    else:
        start_row = 0
    unq_list = []
    dup_names = []
    for row in ll_data[start_row:]:
        name = row[name_col]
        if name in unq_list and name not in dup_names:
            dup_names.append(name)
        else:
            unq_list.append(name)
    return dup_names
```

**Use that function to count duplicates**

*Play Store*


```python
play_store_dups = get_dups_ll(play_store_list,0)
print(len(play_store_dups))
```

    798


*App Store*


```python
app_store_dups = get_dups_ll(app_store_list,1)
print(len(app_store_dups))
```

    0


The App Store data is free of duplicates while the Play Store data has quite a few. Let's dig a little further.

**Display duplicate names**


```python
play_store_dups[0:10]
```




    ['Quick PDF Scanner + OCR FREE',
     'Box',
     'Google My Business',
     'ZOOM Cloud Meetings',
     'join.me - Simple Meetings',
     'Zenefits',
     'Google Ads',
     'Slack',
     'FreshBooks Classic',
     'Insightly CRM']



**Show the full entries for one of the duplicates**


```python
get_row_ll(play_store_list, "Slack", 0)
```




    [241, 270, 295]




```python
print(play_store_list[241])
print(play_store_list[270])
print(play_store_list[295])
```

    ['Slack', 'BUSINESS', '4.4', '51507', 'Varies with device', '5,000,000+', 'Free', '0', 'Everyone', 'Business', 'August 2, 2018', 'Varies with device', 'Varies with device']
    ['Slack', 'BUSINESS', '4.4', '51507', 'Varies with device', '5,000,000+', 'Free', '0', 'Everyone', 'Business', 'August 2, 2018', 'Varies with device', 'Varies with device']
    ['Slack', 'BUSINESS', '4.4', '51510', 'Varies with device', '5,000,000+', 'Free', '0', 'Everyone', 'Business', 'August 2, 2018', 'Varies with device', 'Varies with device']


Two entries are identical and one has a few more reviews. The repeat entries were likely collected at different times. Let's keep only the most recent entry, which means the entry with the most reviews.

**Write a function to keep the one duplicate entry with the most reviews**


```python
# function to remove duplicates based on maximizing a single criteria (a positive number in the data set)
def repl_dups_ll(ll_data, name_col, crit_col, header_row=True):
    dups = get_dups_ll(ll_data, name_col, header_row)
    # make a copy of the data, only the non-duplicates for now
    ll_data_out = []
    if header_row:
        ll_data_out.append(ll_data[0])
        row_start = 1
    else:
        row_start = 0
    for row in ll_data[row_start:]:
        if row[name_col] not in dups:
            ll_data_out.append(row)
    # select one row for each duplicate and add it
    for name in dups:
        dup_ind = get_row_ll(ll_data, name, name_col, header_row)
        max_crit_val = -1
        chosen_index = None
        for index in dup_ind:
            crit_val = float(ll_data[index][crit_col])
            if crit_val > max_crit_val:
                max_crit_val = crit_val
                chosen_index = index
        ll_data_out.append(ll_data[chosen_index])
    return ll_data_out
```

**Remove duplicates, check the new length and make sure no duplicates are left**


```python
play_store_list = repl_dups_ll(play_store_list, 0, 3)
print(len(play_store_list))
print(len(get_dups_ll(play_store_list, 0)))
```

    9661
    0


We originally had 10,842 rows, so we've removed over 1,000 rows with duplicate app entries.

As a final step, let's filter down to only free, English-language apps.

We'll use a simple function to check for a large proportion of ASCII characters with codes higher than 127. Codes up to 127 include the basic English letters, numbers and punctuation. It won't be perfect, but it should catch most of the apps in a language other than English.

**Write a function to check for possible non-English app names**


```python
# function to check for too high a proportion of non-basic English characters
def is_eng(string, ne_limit=0.15):
    ne_count = 0
    for ch in string:
        if ord(ch) > 127:
            ne_count += 1
    if ne_count/len(string) > ne_limit:
        return False
    else:
        return True
```

**Use that function and a simple check for non-zero prices to filter apps**


```python
play_store_list_en_free = [play_store_list[0]]
for row in play_store_list[1:]:
    if is_eng(row[0]) and row[7] == "0":
        play_store_list_en_free.append(row)
app_store_list_en_free = [app_store_list[0]]
for row in app_store_list[1:]:
    if is_eng(row[2]) and row[5] == "0":
        app_store_list_en_free.append(row)
```

**Check the number of apps we have left in each dataset**

*Play Store*


```python
print(len(play_store_list_en_free))
```

    8907


*App Store*


```python
print(len(app_store_list_en_free))
```

    3806


### Step 3: Inspect the data

Next, let's take a closer look at the distribution of the data we have left. We already know we want to use ratings and number of reviews as decision criteria, but maybe something else will jump out at us.

**Write a function to sort and summarize category data**


```python
# function to display data from frequency table, sorted and labeled
def display_table_ll(ll_data, col_index, header_row=True, percent=True):
    table = freq_table_ll(ll_data, col_index, header_row, percent)
    table_display = []
    for key in table:
        key_val_as_tuple = (table[key], key)
        table_display.append(key_val_as_tuple)

    table_sorted = sorted(table_display, reverse = True)
    for entry in table_sorted:
        if percent:
            print(str(entry[1]) + ': ' + str(round(entry[0],1)) + "%")
        else:
            print(str(entry[1]) + ': ' + str(entry[0]))
```

**Show the sorted data for the Play Store categories**


```python
display_table_ll(play_store_list_en_free,1)
```

    FAMILY: 18.0%
    GAME: 9.0%
    TOOLS: 8.0%
    BUSINESS: 4.0%
    SPORTS: 3.0%
    PRODUCTIVITY: 3.0%
    PERSONALIZATION: 3.0%
    MEDICAL: 3.0%
    LIFESTYLE: 3.0%
    HEALTH_AND_FITNESS: 3.0%
    FINANCE: 3.0%
    COMMUNICATION: 3.0%
    TRAVEL_AND_LOCAL: 2.0%
    SOCIAL: 2.0%
    SHOPPING: 2.0%
    PHOTOGRAPHY: 2.0%
    NEWS_AND_MAGAZINES: 2.0%
    BOOKS_AND_REFERENCE: 2.0%
    VIDEO_PLAYERS: 1.0%
    MAPS_AND_NAVIGATION: 1.0%
    FOOD_AND_DRINK: 1.0%
    EDUCATION: 1.0%
    DATING: 1.0%
    WEATHER: 0.0%
    PARENTING: 0.0%
    LIBRARIES_AND_DEMO: 0.0%
    HOUSE_AND_HOME: 0.0%
    EVENTS: 0.0%
    ENTERTAINMENT: 0.0%
    COMICS: 0.0%
    BEAUTY: 0.0%
    AUTO_AND_VEHICLES: 0.0%
    ART_AND_DESIGN: 0.0%


**Show the sorted data for the App Store categories**


```python
display_table_ll(app_store_list_en_free,12)
```

    Games: 55.0%
    Entertainment: 8.0%
    Photo & Video: 4.0%
    Social Networking: 3.0%
    Shopping: 3.0%
    Education: 3.0%
    Utilities: 2.0%
    Sports: 2.0%
    Lifestyle: 2.0%
    Travel: 1.0%
    Productivity: 1.0%
    News: 1.0%
    Music: 1.0%
    Health & Fitness: 1.0%
    Finance: 1.0%
    Book: 1.0%
    Weather: 0.0%
    Reference: 0.0%
    Navigation: 0.0%
    Medical: 0.0%
    Food & Drink: 0.0%
    Catalogs: 0.0%
    Business: 0.0%


The App Store has a ton of free games in it. If games shows up as a possibility later, we'll keep that in mind. There are very few apps in some categories, but we weren't told to take that into account. Nothing else looks too surprising, so let's keep looking through the data.

Let's see averages for rating and number of reviews by category.

**Write a function to display a column average by category**


```python
# function that displays the average of some number for each category
# function does not include NaN values in the calculation
def display_breakdown(ll_data, cat_col, num_col, header_row=True):
    freq_dict = {}
    sum_dict = {}
    if header_row:
        start_row = 1
    else:
        start_row = 0
    for row in ll_data[start_row:]:
        cat = row[cat_col]
        num = float(row[num_col])
        if isnan(num) == False:
            if cat in freq_dict:
                freq_dict[cat] += 1
                sum_dict[cat] += num
            else:
                freq_dict[cat] = 1
                sum_dict[cat] = num
    avg_dict = {}
    for key in freq_dict:
        avg_dict[key] = sum_dict[key]/freq_dict[key]
    table_display = []
    for key in avg_dict:
        key_val_as_tuple = (avg_dict[key], key)
        table_display.append(key_val_as_tuple)
    table_sorted = sorted(table_display, reverse = True)
    for entry in table_sorted:
            print(str(entry[1]) + ': ' + str(round(entry[0],2)))
    return None
```

**Run the function for the rating and number of reviews**

*Play Store, average rating by category*


```python
display_breakdown(play_store_list_en_free,1,2)
```

    EVENTS: 4.44
    BOOKS_AND_REFERENCE: 4.35
    EDUCATION: 4.34
    PARENTING: 4.34
    ART_AND_DESIGN: 4.34
    PERSONALIZATION: 4.3
    BEAUTY: 4.28
    SOCIAL: 4.25
    HEALTH_AND_FITNESS: 4.24
    GAME: 4.23
    WEATHER: 4.23
    SHOPPING: 4.23
    SPORTS: 4.21
    AUTO_AND_VEHICLES: 4.18
    PRODUCTIVITY: 4.18
    COMICS: 4.18
    LIBRARIES_AND_DEMO: 4.18
    FAMILY: 4.17
    FOOD_AND_DRINK: 4.17
    PHOTOGRAPHY: 4.16
    MEDICAL: 4.15
    HOUSE_AND_HOME: 4.14
    FINANCE: 4.13
    COMMUNICATION: 4.13
    ENTERTAINMENT: 4.12
    NEWS_AND_MAGAZINES: 4.11
    BUSINESS: 4.1
    LIFESTYLE: 4.08
    TRAVEL_AND_LOCAL: 4.07
    MAPS_AND_NAVIGATION: 4.04
    VIDEO_PLAYERS: 4.04
    TOOLS: 4.03
    DATING: 3.98


*Play Store, average number of reviews by category*


```python
display_breakdown(play_store_list_en_free,1,3)
```

    COMMUNICATION: 992152.32
    SOCIAL: 965830.99
    GAME: 681943.64
    VIDEO_PLAYERS: 422694.23
    PHOTOGRAPHY: 402539.09
    TOOLS: 305325.82
    ENTERTAINMENT: 301752.25
    SHOPPING: 222767.92
    PERSONALIZATION: 180508.35
    WEATHER: 171250.77
    PRODUCTIVITY: 160170.91
    MAPS_AND_NAVIGATION: 140650.48
    TRAVEL_AND_LOCAL: 129484.43
    SPORTS: 116938.61
    FAMILY: 112311.6
    NEWS_AND_MAGAZINES: 91787.62
    BOOKS_AND_REFERENCE: 86191.12
    HEALTH_AND_FITNESS: 78094.97
    FOOD_AND_DRINK: 57478.79
    EDUCATION: 56290.9
    COMICS: 41828.55
    FINANCE: 38535.9
    LIFESTYLE: 33576.32
    HOUSE_AND_HOME: 26435.47
    ART_AND_DESIGN: 24273.79
    BUSINESS: 24181.11
    DATING: 21953.27
    PARENTING: 16378.71
    AUTO_AND_VEHICLES: 14140.28
    LIBRARIES_AND_DEMO: 10925.81
    BEAUTY: 7476.23
    MEDICAL: 3730.15
    EVENTS: 2555.84


*App Store, average rating by category*


```python
display_breakdown(app_store_list_en_free,12,8)
```

    Productivity: 3.95
    Music: 3.94
    Photo & Video: 3.79
    Games: 3.7
    Shopping: 3.64
    Health & Fitness: 3.58
    Business: 3.55
    Education: 3.54
    Utilities: 3.49
    Food & Drink: 3.43
    Weather: 3.37
    Travel: 3.34
    Reference: 3.3
    Entertainment: 3.29
    Social Networking: 3.06
    Sports: 2.99
    Lifestyle: 2.93
    Medical: 2.88
    News: 2.85
    Finance: 2.29
    Navigation: 2.19
    Catalogs: 2.06
    Book: 2.06


*App Store, average number of reviews by category*


```python
display_breakdown(app_store_list_en_free,12,6)
```

    Reference: 67447.9
    Music: 56482.03
    Social Networking: 54995.49
    Weather: 48794.97
    Navigation: 28800.06
    Photo & Video: 27414.04
    Food & Drink: 22834.24
    Travel: 21336.13
    Sports: 20651.81
    Games: 20261.75
    Health & Fitness: 19952.32
    Shopping: 19724.24
    Productivity: 19326.34
    Finance: 16687.35
    News: 16164.82
    Utilities: 14269.65
    Book: 12427.58
    Entertainment: 11494.58
    Lifestyle: 10291.94
    Business: 6702.58
    Education: 6362.74
    Catalogs: 2002.0
    Medical: 459.75


The data looks useful, but it's still too much to digest. We'll go back to our initial assumptions to help us out:

1. High number of reviews = High engagement with apps of that kind

2. Low average review score = Users are currently unsatisfied with the selection apps in the category

Let's create a simple scoring system for each of these criteria. We'll rank the categories based on each and give them points based on place their in the ranking.

**Write two functions, one to rank and one to score the categories**


```python
# function to return a tuple that ranks categories based on a column average
# ignore NaN values in the calculations
def cat_rank_avg_ll(ll_data, cat_col, num_col, header_row=True, reverse_sort=True):
    freq_dict = {}
    sum_dict = {}
    if header_row:
        start_row = 1
    else:
        start_row = 0
    for row in ll_data[start_row:]:
        cat = row[cat_col]
        num = float(row[num_col])
        if isnan(num) == False:
            if cat in freq_dict:
                freq_dict[cat] += 1
                sum_dict[cat] += num
            else:
                freq_dict[cat] = 1
                sum_dict[cat] = num
    avg_dict = {}
    for key in freq_dict:
        avg_dict[key] = sum_dict[key]/freq_dict[key]
    avg_tup_list = []
    for key in avg_dict:
        key_val_as_tuple = (avg_dict[key], key)
        avg_tup_list.append(key_val_as_tuple)
    table_sorted = sorted(avg_tup_list, reverse=reverse_sort)
    return table_sorted

# function to rank categories for each data set based on two criteria:
# The lower the average app ranking for the category, the better
# The higher the average number of reviews for the category, the better
# Use a simple points system based on the category rank for each criteria
def score_categories(ll_data, cat_col, rating_col, review_col, header_row=True):
    rating_ranking = cat_rank_avg_ll(ll_data, cat_col, rating_col, reverse_sort=False)
    review_ranking = cat_rank_avg_ll(ll_data, cat_col, review_col)
    points_dict = {}
    points = len(rating_ranking)
    for ranking in rating_ranking:
        points_dict[ranking[1]] = points
        points -= 1
    points = len(review_ranking)
    for ranking in review_ranking:
        points_dict[ranking[1]] += points
        points -= 1
    sorted_ranking = []
    for key in points_dict:
        key_val_as_tuple = (points_dict[key], key)
        sorted_ranking.append(key_val_as_tuple)
    sorted_ranking = sorted(sorted_ranking, reverse=True)
    for entry in sorted_ranking:
            print(str(entry[1]) + ': ' + str(round(entry[0],1)))
    return None
```

Now lets run the function on both app store data sets

*Play store ranking*


```python
score_categories(play_store_list_en_free,1,2,3)
```

    VIDEO_PLAYERS: 61.0
    TOOLS: 60.0
    COMMUNICATION: 57.0
    MAPS_AND_NAVIGATION: 52.0
    ENTERTAINMENT: 52.0
    TRAVEL_AND_LOCAL: 50.0
    PHOTOGRAPHY: 49.0
    NEWS_AND_MAGAZINES: 44.0
    GAME: 41.0
    SOCIAL: 40.0
    DATING: 40.0
    LIFESTYLE: 39.0
    SHOPPING: 38.0
    PRODUCTIVITY: 38.0
    FAMILY: 37.0
    WEATHER: 35.0
    FINANCE: 35.0
    BUSINESS: 35.0
    FOOD_AND_DRINK: 34.0
    SPORTS: 33.0
    HOUSE_AND_HOME: 32.0
    PERSONALIZATION: 31.0
    COMICS: 29.0
    HEALTH_AND_FITNESS: 25.0
    MEDICAL: 23.0
    LIBRARIES_AND_DEMO: 21.0
    BOOKS_AND_REFERENCE: 19.0
    AUTO_AND_VEHICLES: 19.0
    EDUCATION: 17.0
    ART_AND_DESIGN: 14.0
    PARENTING: 10.0
    BEAUTY: 10.0
    EVENTS: 2.0


*App Store rankings*


```python
score_categories(app_store_list_en_free,12,8,6)
```

    Navigation: 40.0
    Social Networking: 36.0
    Reference: 36.0
    Weather: 31.0
    Sports: 31.0
    Finance: 30.0
    Book: 30.0
    Travel: 28.0
    News: 28.0
    Food & Drink: 27.0
    Music: 24.0
    Catalogs: 24.0
    Lifestyle: 22.0
    Photo & Video: 21.0
    Entertainment: 20.0
    Medical: 19.0
    Health & Fitness: 19.0
    Games: 18.0
    Utilities: 17.0
    Shopping: 17.0
    Productivity: 12.0
    Education: 11.0
    Business: 11.0


### Step 4: Make a recommendation

Let's look at the top half of both rankings, side by side. The categories aren't the same, so we'll just have to match them up the best we can.

Play Store top half | App Store top half
 ------------------ | ----------------
VIDEO_PLAYERS | Navigation
TOOLS | Social Networking
COMMUNICATION | Reference
MAPS_AND_NAVIGATION | Weather
ENTERTAINMENT | Sports
TRAVEL_AND_LOCAL | Finance
PHOTOGRAPHY | Book
NEWS_AND_MAGAZINES | Travel
GAME | News
SOCIAL | Food & Drink
DATING | Music
LIFESTYLE | NA
SHOPPING | NA
PRODUCTIVITY |NA
FAMILY | NA
WEATHER | NA

That gives us the following possibilities:

1. Navigation
2. Weather
3. Travel
4. Social
5. News, reference, books, magazines

Get the engineers working on it!

We'd need to do a deeper competitive analysis before actually starting such an app. But this narrows the field for further research, which is the best we can expect from a dataset with so few details.

This is a good start, though. We could report our findings and then think about what we could improve for a future version of the analysis.

### What's next?

What are some ways we could improve this analysis? The first things that come to my mind are:

- **Investigate the distinctions between categories.** Who decides what goes in what category? Google and Apple? The app makers? What are the common features of apps in each category? Looking at data with little to no context is usually a bad idea. I know I felt like the categories were unclear for me. More context might give us ideas for how to make a recommendation.
- **Come up with a a better recommendation metric.** This might be an ideal place for machine learning algorithms. These algorithms take in factors and return a kind of "line of best fit." That "line" would tell you what factors contribute more or less to a desired outcome. For example, we could see what combination of factors in the dataset most often led to a high number of reviews.
- **Add more details about each app.** The app descriptions would be a rich source of information. As a simple starting point, we could see what words (besides common words like "the") are repeated most for each app and category. This could give us an idea of the main features, proposed benefits and selling points for each app and category.

We'd have to check in and see what the expectations for the analysis are. Often, those expectations will be clearly spelled out from the start, but sometimes they aren't. As we've done here, we may just have to start with the basics, see what we find and ask for feedback as we gradually improve the quality of our analysis.

*Note: I got the idea for this article from the [Dataquest](https://www.dataquest.io/) course materials. A version of this analysis was the subject of one of Dataquest's guided projects. I changed a few details and adapted my version of the project to form the basis of this article.*
