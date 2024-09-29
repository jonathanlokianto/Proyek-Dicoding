import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import streamlit as st
from matplotlib.ticker import FuncFormatter 

sns.set_theme(style='dark')

day_df = pd.read_csv('day.csv')
day_df['dteday'] = pd.to_datetime(day_df['dteday'])
day_df.sort_values(by='dteday', inplace=True)
day_df.reset_index(drop=True, inplace=True)


with st.sidebar:
    st.image("https://png.pngtree.com/png-vector/20240515/ourmid/pngtree-illustration-of-a-bicycle-logo-png-image_12473420.png")
    st.write("Nama: Jonathan Lokianto")
    st.write("E-Mail: jonathanlokianto")
    st.write("ID Dicoding: jonathanlokianto")
    st.write("Dataset yang dipakai: Bike Sharing Dataset")
st.header('Proyek Akhir Analisis Data')

casual_sum = day_df["casual"].sum()
regist_sum = day_df["registered"].sum()
cnt_sum = day_df["cnt"].sum()

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Jumlah user casual: ", f"{casual_sum:,.0f}")
with col2:
    st.metric("Jumlah user registered: ", f"{regist_sum:,.0f}")
with col3:
    st.metric("Total user casual dan registered: ", f"{cnt_sum:,.0f}")


st.subheader("Perbandingan jumlah user Casual dengan user Registered")
day_df.groupby(by="yr").agg(
    {
        "casual": "sum",
        "registered":"sum"
    }
)
sum_df = day_df[['casual', 'registered']].sum().reset_index()
sum_df.columns = ['type', 'sum']
colors = ['#4379F2', '#FFEB00']

plt.figure(figsize=(10, 5))
sns.barplot(
    data=sum_df.sort_values(by="sum", ascending=True), 
    x='type', 
    y='sum',
    hue='type',
    palette=colors
)

for index, value in enumerate(sum_df['sum']):
    formatted_value = f"{value:,.0f}"
    plt.text(index, value, formatted_value, ha='center', va='bottom')

plt.title('Total Casual and Registered Counts')
plt.ylabel(None)
plt.xlabel(None)
plt.show()

plt.tight_layout()
st.pyplot(plt)


st.subheader("Visualisasi performa penyewaan sepeda pada 10 bulan terakhir")
monthly_data_df = day_df.resample(rule="ME", on="dteday").agg({
    "casual": "sum",
    "registered": "sum",
    "cnt":"sum"
})

monthly_data_df = monthly_data_df.reset_index()
monthly_data_df.rename(columns={
    "casual": "Total Casual Cyclist",
    "registered": "Total Registered Cyclist",
    "cnt":"Total Casual & Registered Cyclist"
}, inplace=True)

last_ten_months = monthly_data_df.tail(10)
last_ten_months.loc[:, 'dteday'] = pd.to_datetime(last_ten_months['dteday'])
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(
    last_ten_months["dteday"],
    last_ten_months["Total Casual Cyclist"],
    marker='o', 
    linewidth=2,
    color="#72BCD4",
    label="Casual Cyclist"
)

ax.plot(
    last_ten_months["dteday"],
    last_ten_months["Total Registered Cyclist"],
    marker='o', 
    linewidth=2,
    color="#FF6347",
    label="Registered Cyclist"
)

plt.plot(
    last_ten_months["dteday"],
    last_ten_months["Total Casual & Registered Cyclist"],
    marker='o', 
    linewidth=2,
    color="#42ffbf",
    label="Total Casual & Registered Cyclist"
)

ax.set_title("Number of Cyclists Over Last 10 Months", loc="center", fontsize=20)
ax.set_xlabel("Months")
ax.set_ylabel("Number of Cyclists")

ax.legend()
formatter = FuncFormatter(lambda x, _: f"{int(x):,}")
ax.yaxis.set_major_formatter(formatter)
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%B'))
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid()
plt.tight_layout()
st.pyplot(fig)

st.subheader("Analisa Clustering dengan method: Binning")
binning_df = pd.DataFrame(day_df)
binning_df['temp']=pd.cut(binning_df['temp'], bins=3, labels=['Low', 'Medium', 'High'])
binning_df['humidity']=pd.cut(binning_df['hum'], bins=4, labels=[' Low', 'Moderate', 'High' , 'Very high'])
binning_df['windspeed']=pd.cut(binning_df['windspeed'], bins=3, labels=[' Fresh Breeze', 'Moderate Breeze','Strong Breeze'])

fig, axs = plt.subplots(1, 3, figsize=(15, 10))

axs[0].bar(binning_df['temp'].value_counts().sort_index().index, 
           binning_df['temp'].value_counts().sort_index(), 
           color='red')
axs[0].set_title('Temperature Bins')
axs[0].tick_params(axis='x', rotation=90)
axs[0].set_xlabel(None)

axs[1].bar(binning_df['humidity'].value_counts().sort_index().index, 
           binning_df['humidity'].value_counts().sort_index(), 
           color='blue')
axs[1].set_title('Humidity Bins')
axs[1].tick_params(axis='x', rotation=90)
axs[1].set_xlabel(None)

axs[2].bar(binning_df['windspeed'].value_counts().sort_index().index, 
           binning_df['windspeed'].value_counts().sort_index(), 
           color='green')
axs[2].set_title('Windspeed Bins')
axs[2].tick_params(axis='x', rotation=90)
axs[2].set_xlabel(None)

plt.tight_layout()
st.pyplot(fig)