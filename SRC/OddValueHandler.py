def handle_missing_and_outliers(df): 
    #removes values that are empty or duplicates
    before_drop = df.shape[0]
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    after_drop = df.shape[0]
    print(f"Removed {before_drop - after_drop} rows with missing values.")
    
    #Removes outliers
    numerical_cols = ['GPA', 'StudyTimeWeekly', 'Absences', 'Age']
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        LB = Q1 - 1.5 * IQR
        UB = Q3 + 1.5 * IQR
        
        before_outliers = df.shape[0]
        df = df[(df[col] >= LB) & (df[col] <= UB)]
        after_outliers = df.shape[0]
        print(f"Removed {before_outliers - after_outliers} outliers in {col}.")

    return df
    