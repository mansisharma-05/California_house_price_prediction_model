import streamlit as st
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler
import pickle
import time

# Title
col = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']
st.title('California Housing Price Prediction')
st.image('https://miro.medium.com/v2/resize:fit:1358/1*smf9XOCYX-7NuQSkE4m84g.jpeg')

st.header('Model of housing prices to predict median house values in California',divider=True)

# st.subheader('''User Must Enter Given Values To Predict Price:
# ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']''')

st.sidebar.title('Select House Features 🏠')

st.sidebar.image('https://wallpapercave.com/wp/wp4110657.jpg')

# read_data
temp_df = pd.read_csv('california.csv')

random.seed(42)

all_values = []

for i in temp_df[col]:
    min_value, max_value = temp_df[i].agg(['min','max'])

    var = st.sidebar.slider(f'Select {i} value',int(min_value),int(max_value),
                      random.randint(int(min_value), int(max_value)))

    all_values.append(var)


ss = StandardScaler()
ss.fit(temp_df[col])

final_value = ss.transform([all_values])

with open('house_price_pred_ridge_model.pkl','rb') as f:
    chatGPT = pickle.load(f)

price = chatGPT.predict(final_value)[0]

st.write(pd.DataFrame(dict(zip(col,all_values)),index = [1]))

progress_bar = st.progress(0)
placeholder = st.empty()
placeholder.subheader('Predicting Price...')
place = st.empty()
place.image('https://leapcodetech.com/wp-content/uploads/2024/10/seo_marketing_banner.gif',width=60)


if price>0:
    
    for i in range(100):
        time.sleep(0.05)
        progress_bar.progress(i+1)
    
    body = (f'Predicted Median House Price: ${round(price,2)} Thousand Dollars')
    placeholder.empty()
    place.empty()
    #st.subheader(body)
    st.success(body)
else:
    body = 'Invalid House features Values'
    st.warning(body)










