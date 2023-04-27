import streamlit as st 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Run To Inspire", page_icon=":snow_capped_mountain:")

header = st.container()
dataset = st.container()
features = st.container()
model = st.container()
predict = st.container()

st.set_option('deprecation.showPyplotGlobalUse', False)

with header:
    # Define title text
    title_text = "Welcome to my cool Data Science project about running!"
    # Display centered title
    st.markdown(f"<h1 style='text-align: center'>{title_text}</h1>", unsafe_allow_html=True)
    
    st.markdown(
                '''
                This is a very simple "just to try all parts together" progect with a little bit 
                of EDA and basic model training.  
                My project provides valuable insights for runners looking to improve their training strategies, 
                and demonstrates the potential of data science in sports analysis. (not yet) :smile:
                '''
                )

with dataset:
    st.header('Here is some information about my dataset')
    st.markdown(
                '''
                I decided to use my own running results over the past few years 
                to create a dataset for EDA and model training.  
                I chose only some of the many features that you can analyze to get some better
                insights into you performance.
                '''
                )
    
    st.write("<br>", unsafe_allow_html=True)
    st.subheader('Dataframe head:')
    df = pd.read_csv('data/running_data.csv')
    st.write(df.head())
    st.write("<br>", unsafe_allow_html=True)

    
    st.subheader('Let\'s see some summury statistics:')
    st.write(df.describe())
    st.write("<br>", unsafe_allow_html=True)

   
    st.subheader('Some visualizations:')
    st.markdown(
            '''
            Here you can see distribution of Average Heart Rate over the years with
            mean and median displayed as lines for clearer visualization.
            '''
            )
    # Create a figure and axes
    fig, ax = plt.subplots(figsize=(8,6))
    # Plot the distribution of data
    sns.histplot(df['Average Heart Rate'], 
                ax=ax,
                bins=15
                )
    # Create a more descriptive x axis label
    ax.set(xlabel="Average Heart Rate",
        ylabel='Counts',
        #xlim=(0,10),
        #ylim=(0,55),
        title="Average Heart Rate",
        )
    # Add vertical lines for the median and mean
    ax.axvline(x=df['Average Heart Rate'].median(), color='m', label='Median', linestyle='--', linewidth=2)
    ax.axvline(x=df['Average Heart Rate'].mean(), color='r', label='Mean', linestyle='-', linewidth=2)
    # Show the legend and plot the data
    ax.legend()
    st.pyplot()

    st.write("<br>", unsafe_allow_html=True)

    st.subheader('Correlation matrix:')
    st.markdown(
            '''
            Now let's see if there are any correlations in our data.
            '''
            )
    # plotting correlation heatmap
    corr_matrix = df.drop(['WatchPace'], axis=1).corr().round(2)
    dataplot = sns.heatmap(corr_matrix, cmap="YlGnBu", annot=True)
    # displaying heatmap
    st.pyplot()

    st.write("<br>", unsafe_allow_html=True)
    
    st.subheader('Regression plot:')
    st.markdown(
            '''
            From correlation matrix we can observe that there are some good  correlation coefficients 
            between variables in our data, let's try to explore one of them.
            '''
            )
    joint = sns.jointplot(
        y='Pace',
        x='Average Heart Rate', 
        data=df,
        kind="reg",
        )
    joint.set_axis_labels('Heart Rate', 'Pace')
    st.pyplot()

    st.write("<br>", unsafe_allow_html=True)

with features:
    st.subheader('This is some information about features of my dataset')
    st.markdown(
            '''
            List of features in my dataset to use for model training:
            '''
            )
    st.table(df.columns)

    st.write("<br>", unsafe_allow_html=True)

with model:
    st.header('Model training')
    st.markdown(
            '''
            In this part we will train Random Forest Regressor.
            In the menu below you can choose some options to tune model training parameters
            and see how model performance reacts on them.
            '''
            )
    
    sel_col, disp_col = st.columns(2)   

    max_depth = sel_col.slider('What should be the max depth of the model?', 
                                min_value=10, max_value=100, value=20, step=10)

    n_estimators = sel_col.selectbox('How many trees there will be?', options=[5, 10, 20, 30], index=0)

    input_feature = sel_col.text_input('Which feature should we use as an input?', 'Average Heart Rate')

    # Model
    X = df[input_feature].values.reshape(-1, 1)
    y = df['Pace'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    regr = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)

    regr.fit(X_train, y_train)

    y_pred = regr.predict(X_test)

    disp_col.subheader('Mean absolute error of the model is: ')
    disp_col.write(mean_absolute_error(y_test, y_pred))

    disp_col.subheader('Mean squared error of the model is: ')
    disp_col.write(mean_squared_error(y_test, y_pred))

    disp_col.subheader('R score of the model is: ')
    disp_col.write(r2_score(y_test, y_pred)) 

    st.write("<br>", unsafe_allow_html=True)

with predict:  
    st.header('Predicting pace for your training')
    st.markdown(
            '''
            Here you can predict Pace for your run depending on what feature you've
            traind the model.  
            For example: today you wish to run with low effort and don't go over your
            second heart rate zone, let's say 150 bpm.  
            As input feature for your model you choose - Average Heart Rate,
            and now you can enter your disied heart rate for the run.
            '''
            )  

    col1, col2 = st.columns(2)

    input_hr = col1.text_input('Enter preferable Heart Rate for your run', 160) 
    
    X_new = np.array([[input_hr]])
    
    y_new = regr.predict(X_new)
    #col2.subheader('Your preferable for training is:')
    
    def to_pace(pace):
        pace_minutes = int(pace)
        pace_seconds = round((pace - pace_minutes) * 60)
        return float('{:02d}'.format(pace_minutes) + '.' + '{:02d}'.format(pace_seconds))
    
    col2.write('Your preferable Pace for training is {} min/km'.format(to_pace(y_new[0])))
    
    



