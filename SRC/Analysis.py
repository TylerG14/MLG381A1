import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Univariate Analysis - simple plots for specific columns
def univariate_analysis(df):
    print("\n--- Univariate Analysis ---")

    # GPA
    plt.figure()
    plt.hist(df['GPA'], bins=20, color='skyblue', edgecolor='black')
    plt.xlabel('GPA')
    plt.ylabel('Frequency')
    plt.title('Distribution of GPA')
    plt.grid(True)
    plt.show()

    # Age
    plt.figure()
    plt.hist(df['Age'], bins=10, color='skyblue', edgecolor='black')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.title('Distribution of Age')
    plt.grid(True)
    plt.show()

    # Gender
    plt.figure()
    df['Gender'].value_counts().plot(kind='bar', color='lightgreen', edgecolor='black')
    plt.xlabel('Gender')
    plt.ylabel('Count')
    plt.title('Count of Gender')
    plt.grid(True)
    plt.show()

    # Ethnicity
    plt.figure()
    df['Ethnicity'].value_counts().plot(kind='bar', color='lightgreen', edgecolor='black')
    plt.xlabel('Ethnicity')
    plt.ylabel('Count')
    plt.title('Count of Ethnicity')
    plt.grid(True)
    plt.show()

    # ParentalEducation
    plt.figure()
    df['ParentalEducation'].value_counts().plot(kind='bar', color='lightgreen', edgecolor='black')
    plt.xlabel('Parental Education')
    plt.ylabel('Count')
    plt.title('Count of Parental Education')
    plt.grid(True)
    plt.show()

    # StudyTimeWeekly
    plt.figure()
    plt.hist(df['StudyTimeWeekly'], bins=20, color='skyblue', edgecolor='black')
    plt.xlabel('Study Time Weekly')
    plt.ylabel('Frequency')
    plt.title('Distribution of Study Time Weekly')
    plt.grid(True)
    plt.show()

    # Absences
    plt.figure()
    plt.hist(df['Absences'], bins=20, color='skyblue', edgecolor='black')
    plt.xlabel('Absences')
    plt.ylabel('Frequency')
    plt.title('Distribution of Absences')
    plt.grid(True)
    plt.show()

    # Tutoring
    plt.figure()
    df['Tutoring'].value_counts().plot(kind='bar', color='lightgreen', edgecolor='black')
    plt.xlabel('Tutoring')
    plt.ylabel('Count')
    plt.title('Count of Tutoring')
    plt.grid(True)
    plt.show()
    
    # Extracurricular
    plt.figure()
    df['Extracurricular'].value_counts().plot(kind='bar', color='lightcoral', edgecolor='black')
    plt.xlabel('Extracurricular')
    plt.ylabel('Count')
    plt.title('Count of Extracurricular Activities')
    plt.grid(True)
    plt.show()

    # Sports
    plt.figure()
    df['Sports'].value_counts().plot(kind='bar', color='lightcoral', edgecolor='black')
    plt.xlabel('Sports')
    plt.ylabel('Count')
    plt.title('Count of Sports Participation')
    plt.grid(True)
    plt.show()

    # Music
    plt.figure()
    df['Music'].value_counts().plot(kind='bar', color='lightcoral', edgecolor='black')
    plt.xlabel('Music')
    plt.ylabel('Count')
    plt.title('Count of Music Participation')
    plt.grid(True)
    plt.show()

    # Volunteering
    plt.figure()
    df['Volunteering'].value_counts().plot(kind='bar', color='lightcoral', edgecolor='black')
    plt.xlabel('Volunteering')
    plt.ylabel('Count')
    plt.title('Count of Volunteering')
    plt.grid(True)
    plt.show()

    # GradeClass
    plt.figure()
    df['GradeClass'].value_counts().sort_index().plot(kind='bar', color='lightblue', edgecolor='black')
    plt.xlabel('Grade Class')
    plt.ylabel('Count')
    plt.title('Distribution of Grade Class')
    plt.grid(True)
    plt.show()


    #Boxplot for GPA
    plt.figure()
    sns.boxplot(y=df['GPA'], color='skyblue')
    plt.title('Boxplot of GPA')
    plt.grid(True)
    plt.show()

    # 2. Boxplot for StudyTimeWeekly
    plt.figure()
    sns.boxplot(y=df['StudyTimeWeekly'], color='skyblue')
    plt.title('Boxplot of Study Time Weekly')
    plt.grid(True)
    plt.show()

    # 3. Bar Plot for Binned Absences
    plt.figure()
    # Bin absences into ranges (e.g., 0-5, 6-10, etc.)
    bins = [0, 5, 10, 15, 20, 25, 30]
    labels = ['0-5', '6-10', '11-15', '16-20', '21-25', '26-30']
    df['Absences_Binned'] = pd.cut(df['Absences'], bins=bins, labels=labels, include_lowest=True)
    df['Absences_Binned'].value_counts().sort_index().plot(kind='bar', color='lightgreen', edgecolor='black')
    plt.xlabel('Absences Range')
    plt.ylabel('Count')
    plt.title('Distribution of Absences (Binned)')
    plt.grid(True)
    plt.show()
    # Clean up temporary column
    df = df.drop(columns=['Absences_Binned'])
    
    # 4. Pie Chart for Gender
    plt.figure()
    df['Gender'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['lightblue', 'lightpink'])
    plt.title('Gender Distribution')
    plt.ylabel('')
    plt.show()

    # 5. KDE Plot for GPA
    plt.figure()
    sns.kdeplot(df['GPA'], color='purple', fill=True)
    plt.xlabel('GPA')
    plt.title('Kernel Density Estimation of GPA')
    plt.grid(True)
    plt.show()

# Bivariate Analysis - Focus on correlations
def bivariate_analysis(df):
    print("\n--- Bivariate Analysis ---")

    # Correlation matrix for numerical variables
    numerical_cols = ['GPA', 'Age', 'StudyTimeWeekly', 'Absences']
    corr_matrix = df[numerical_cols].corr()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title('Correlation Matrix of Numerical Variables')
    plt.show()

    # Scatter plots for key correlations
    # 1. GPA vs StudyTimeWeekly
    plt.figure()
    sns.scatterplot(x='StudyTimeWeekly', y='GPA', data=df, color='blue', alpha=0.5)
    plt.title('GPA vs Study Time Weekly')
    plt.grid(True)
    plt.show()

    # 2. GPA vs Absences
    plt.figure()
    sns.scatterplot(x='Absences', y='GPA', data=df, color='red', alpha=0.5)
    plt.title('GPA vs Absences')
    plt.grid(True)
    plt.show()

    # Boxplot for GPA across categorical variables (if relevant)
    # 3. GPA by Tutoring
    plt.figure()
    sns.boxplot(x='Tutoring', y='GPA', data=df)
    plt.title('GPA by Tutoring Status')
    plt.grid(True)
    plt.show()

# Example usage (you'd replace this with your actual dataset)
# df = pd.read_csv('your_dataset.csv')
# univariate_analysis(df)
# bivariate_analysis(df)