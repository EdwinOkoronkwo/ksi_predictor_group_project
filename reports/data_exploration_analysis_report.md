# Analysis Report

## 1. Data Overview
The dataset contains **18,957 records** and **54 columns**, representing various attributes related to incidents, locations, environmental conditions, and involved entities.

### 1.1 Data Structure
- **Total Rows:** 18,957
- **Total Columns:** 54
- **Data Types:**
  - **Integer Columns:** 4
  - **Float Columns:** 6
  - **Object (String) Columns:** 44
- **Memory Usage:** Approximately 7.8 MB

### 1.2 Sample Data
The first five rows indicate structured incident records with OBJECTID, INDEX, ACCNUM, DATE, TIME, STREET1, STREET2, and other attributes.

## 2. Key Column Analysis

### 2.1 Unique and Missing Values
- **ACCNUM (Incident Number):**
  - Unique: 4,955
  - Missing: 4,930 records (~26% of data)
  - Wide range (25,301 to 4,008,024,139) with a high standard deviation, suggesting significant variance in numbering.

- **DATE & TIME:**
  - Unique Dates: 4,128
  - Time ranges from **00:00 to 23:59**, with a mean of **1364 (13:04 hours)** and a median of **1450 (14:50 hours)**.

- **Street Information:**
  - **STREET1:** 1,942 unique street names.
  - **STREET2:** 2,822 unique street names, with 1,706 missing values (~9%).

- **Location Coordinates:**
  - **Latitude:** Range (43.5897 - 43.8554), Mean = 43.7103.
  - **Longitude:** Range (-79.6384 to -79.1230), Mean = -79.3965.
  - **Unique Lat-Long Pairs:** 4,783 and 5,202 respectively, indicating potential duplicate records.

### 2.2 Incident Classification
- **ACCLASS (Incident Classification):**
  - Binary classification (0,1).
  - Mean = 0.14, suggesting most incidents belong to class 0.

- **IMPACTYPE (Impact Type):**
  - 10 unique types, e.g., 'Approaching', 'Pedestrian Collisions'.

- **INVTYPE (Involvement Type):**
  - 19 unique categories, including 'Passenger', 'Driver', 'Pedestrian'.

### 2.3 Environmental Factors
- **Visibility Conditions:** 8 unique values (e.g., Clear, Snow, Rain).
- **Light Conditions:** 9 unique values (e.g., Daylight, Dark, Dusk).
- **Road Surface Conditions:** 9 unique values (e.g., Dry, Wet, Ice, Loose Snow).

## 3. Observations & Recommendations
1. **Data Completeness Issues:**
   - High missing values in ACCNUM (~26%), OFFSET (80%), and certain categorical fields.
   - Fields like CYCLISTYPE, CYCACT, and EMERG_VEH have very sparse data.

2. **Possible Data Redundancy:**
   - Repeated Latitude-Longitude pairs suggest possible duplicate records.
   - The high variance in ACCNUM could indicate multiple data sources.

3. **Incident Pattern Analysis:**
   - Accidents are more frequent in daylight and clear visibility conditions.
   - The majority of recorded incidents involve **passengers and drivers**.
   - Some road classifications might be more prone to accidents.

4. **Next Steps:**
   - **Data Cleaning:** Remove duplicates and impute missing values.
   - **Normalization:** Standardize ACCNUM format and incident time.
   - **Further Analysis:** Investigate correlations between incident type and environmental factors.

This report provides an initial exploration of the dataset, highlighting key insights and areas for deeper investigation.

