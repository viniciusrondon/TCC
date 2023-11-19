import streamlit as st
import pandas as pd
from math import*
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
from scipy import signal
from scipy import integrate
import scipy.signal.signaltools as signaltools
from scipy.signal import find_peaks
import sys
from numpy import NaN, Inf, arange, isscalar, asarray
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statistics
import obspy
from obspy.signal.detrend import polynomial
from io import StringIO
from pandas import * 
import openai
import webbrowser
# ====== Page Config ========= # 
st.set_page_config(
    page_title="Vibration Level Analysis",
    layout="wide"
)



#### ============= Funções ============ ####
# ====== Leitura do TXT ========= # 
@st.cache_data(max_entries=5)
def readd(str):
#Gerando a série temporal de g a partir da leitura do txt. uma lista de contendo o delta t e a data(respectiva medição)

    f=open(str)             # variável f recebe o arquivo txt
    lines=f.readlines()                             # função readlines é todas as linhas de um arquivo txt em uma lista "lines". execute lines para entender 

    for i in lines:                                 
        contador=0                                 
        line=[]                                    
        for j in range(len(i)):
            if i[j]==' ':           
                line.remove(' ') 
                contador=j+1
                
    serie_data=[]                                   #variável onde será armazenada a serie temporal
    for i in lines:                                 # para cada i nesta lista lines faca:
        contador=0                                  #variável 0
        line=[]                                     # lista line SEM S NAO É A LISTA ANTERIOR
        for j in range(len(i)):                     #para cada j de 0 ao tamanho de i que é do tamanho de lines, logo todas as linhas
            if i[j]=='\t'or i[j]=='\n':             # se i[j] semelhante a tabular ou nova linha faca
                line.append((i[contador:j]))        # inclua na lista line como ponto flutuante, ou seja, ele inclui str por str ate o ultimo numero
                contador=j+1
        serie_data.append(line)
    f.close()

    for i in range(len(serie_data)):
        serie_data[i].pop(0)

    serie_data_1 = [[float(j) for j in i ] for i in serie_data ]
    return serie_data_1
# ====== Signal AXS ========= # 
@st.cache_data(max_entries=5)
def axs(serie_data_1,Delta):
    # Passar separa o sinal(data) em uma lista no tempo e em data

    lista_sample=[]
    lista_data_x=[]
    lista_data_y=[]
    lista_data_z=[]
    lista_data2=serie_data_1
    for i in range(len(lista_data2)):
        lista_sample.append(i*Delta)                                    # tempo 
        lista_data_x.append(lista_data2[i][0]*9.81)                     # medicao = lista_data
        lista_data_y.append(lista_data2[i][1]*9.81)                     # medicao = lista_data
        lista_data_z.append(lista_data2[i][2]*9.81)                     # medicao = lista_data
        
    med_x = np.mean(lista_data_x) 
    med_y = np.mean(lista_data_y)
    med_z = np.mean(lista_data_z)
    
    lista_data_x=[]
    lista_data_y=[]
    lista_data_z=[]
    for i in range(len(lista_data2)):
        lista_data_x.append(lista_data2[i][0]*9.81 - med_x)                     # medicao = lista_data
        lista_data_y.append(lista_data2[i][1]*9.81 - med_y)                     # medicao = lista_data
        lista_data_z.append(lista_data2[i][2]*9.81 - med_z)                     # medicao = lista_data
        
    n= len(lista_data_x)
    lista_data_min, lista_data_max = min(lista_data_x), max(lista_data_x)
    
    return lista_sample, lista_data_x, lista_data_y, lista_data_z
# ============ To read file as string ===========#
@st.cache_data(max_entries=5)
def read_file_str (uploaded_file):
    df = pd.read_csv(uploaded_file, sep=';')
    #df = pd.read_csv(uploaded_file, sep='/t')
    df = df.reset_index()
    columns =df.columns
    columns[1]
    lista_sample_1 =  df[columns[0]]
    lista_data_x_1  = df[columns[1]]
    lista_data_y_1 = df[columns[2]] 
    lista_data_z_1 = df[columns[3]]
    return lista_sample_1 ,lista_data_x_1,lista_data_y_1,lista_data_z_1,df
# ============ figure 1: raw signal ===========#
@st.cache_data
def raw_signal_fig(lista_sample, lista_data_x,lista_data_y,lista_data_z):
    fig = make_subplots(rows=3, cols=1)

    fig.add_trace(go.Scatter(x=lista_sample,y=lista_data_x, mode="lines", name="axs_x", line_width=0.4), row=1, col=1)
    fig.add_trace(go.Scatter(x=lista_sample,y=lista_data_y, mode="lines", name="axs_y", line_width=0.4), row=2, col=1)
    fig.add_trace(go.Scatter(x=lista_sample,y=lista_data_z, mode="lines", name="axs_z", line_width=0.4), row=3, col=1)


    fig.update_layout(title_text="Raw signal", title_x=0.5,
                    title_font_size=30,
                    xaxis_title="Tempo (s)",
                    yaxis_title="Amplitude [m/s²]")

    # Definir o título do eixo Y do subplot na linha 2, coluna 1
    fig.update_yaxes(title="Amplitude [m/s²]", row=2, col=1)
    fig.update_xaxes(title="Tempo (s)", row=2, col=1)

    # Definir o título do eixo Y do subplot na linha 2, coluna 1
    fig.update_yaxes(title="Amplitude [m/s²]", row=3, col=1)
    fig.update_xaxes(title="Tempo (s)", row=3, col=1)

    #fig.show()
    return fig

# ============ série de aceleração em velocidade ============= #
@st.cache_data
def cumtrap_vel(lista_data_x,lista_sample,fs):
    #serie_data_x = []

    #for i in range(len(lista_data_x)):
        
        #serie_data_x.append([lista_sample[i], lista_data_x[i]])

    lista_v = integrate.cumulative_trapezoid(lista_data_x, lista_sample, initial=0)
    sos_x=signal.butter(30,2,'highpass',analog=False,fs=fs,output='sos') # N (2**n), Wn(frequencia critica), btype=lowpass’, ‘highpass’, ‘bandpass’, ‘bandstop, analog=False, output='sos', fs=1/delta
    lista_v = signal.sosfilt(sos_x, lista_v)
    lista_samples = lista_sample
    #len(lista_v)
    lista_v = lista_v
    return lista_v, lista_samples

# ============ figure 2: raw signal veloc ===========#
@st.cache_data
def raw_vel_fig(lista_sample,lista_v_x,lista_v_y,lista_v_z):
    fig = make_subplots(rows=3, cols=1)

    fig.add_trace(go.Scatter(x=lista_sample,y=lista_v_x*1000, mode="lines", name="axs_x", line_width=0.4), row=1, col=1)
    fig.add_trace(go.Scatter(x=lista_sample,y=lista_v_y*1000, mode="lines", name="axs_y", line_width=0.4), row=2, col=1)
    fig.add_trace(go.Scatter(x=lista_sample,y=lista_v_z*1000, mode="lines", name="axs_z", line_width=0.4), row=3, col=1)


    fig.update_layout(title_text="Velocity", title_x=0.5,
                    title_font_size=30,
                    xaxis_title="Tempo (s)",
                    yaxis_title="Amplitude [mm/s]")

    # Definir o título do eixo Y do subplot na linha 2, coluna 1
    fig.update_yaxes(title="Amplitude [mm/s]", row=2, col=1)
    fig.update_xaxes(title="Tempo (s)", row=2, col=1)

    # Definir o título do eixo Y do subplot na linha 2, coluna 1
    fig.update_yaxes(title="Amplitude [mm/s]", row=3, col=1)
    fig.update_xaxes(title="Tempo (s)", row=3, col=1)
    return fig

# ============ Série Temporal da pos ============= #
@st.cache_data
def cumtrap_pos(lista_v, lista_sample):
# Função que integra a série de velocidade em posição "x"

    #serie_v = []

    #for i in range(len(lista_v)):
        
        #serie_v.append([lista_sample[i], lista_v[i]])

    lista_pos = integrate.cumulative_trapezoid(lista_v, lista_sample, initial=0)
    return lista_pos

# ============ figure 3: raw signal pos ============= #
@st.cache_data
def raw_pos_fig(lista_sample,lista_pos_x,lista_pos_y,lista_pos_z):
    fig = make_subplots(rows=3, cols=1)

    fig.add_trace(go.Scatter(x=lista_sample,y=lista_pos_x*1000, mode="lines", name="axs_x", line_width=0.4), row=1, col=1)
    fig.add_trace(go.Scatter(x=lista_sample,y=lista_pos_y*1000, mode="lines", name="axs_y", line_width=0.4), row=2, col=1)
    fig.add_trace(go.Scatter(x=lista_sample,y=lista_pos_z*1000, mode="lines", name="axs_z", line_width=0.4), row=3, col=1)


    fig.update_layout(title_text="Dispacement", title_x=0.5,
                    title_font_size=30,
                    xaxis_title="Tempo (s)",
                    yaxis_title="Amplitude [mm]")

    # Definir o título do eixo Y do subplot na linha 2, coluna 1
    fig.update_yaxes(title="Amplitude [mm]", row=2, col=1)
    fig.update_xaxes(title="Tempo (s)", row=2, col=1)

    # Definir o título do eixo Y do subplot na linha 2, coluna 1
    fig.update_yaxes(title="Amplitude [mm]", row=3, col=1)
    fig.update_xaxes(title="Tempo (s)", row=3, col=1)
    return fig

# ============ filter 1 ============= #
@st.cache_data
def fil_butter(order,serie,first_filter_freq, first_filter,fs):
    
    sos=signal.butter(order,first_filter_freq, first_filter,analog=False,fs=fs,output='sos') # N (2**n), Wn(frequencia critica), btype=lowpass’, ‘highpass’, ‘bandpass’, ‘bandstop, analog=False, output='sos', fs=1/delta
    lista_clean = signal.sosfilt(sos, serie)
    return lista_clean

# ============ filter 2 ============= #
@st.cache_data
def fil_butter_2(N,serie, second_filter_freq_1, second_filter_freq_2, Second_filter, fs_,T,sample):
    
    sos=signal.butter(N,[second_filter_freq_2, second_filter_freq_1], Second_filter,analog=False,fs=fs_,output='sos') # N (2**n), Wn(frequencia critica), btype=lowpass’, ‘highpass’, ‘bandpass’, ‘bandstop, analog=False, output='sos', fs=1/delta
    lista_clean = signal.sosfilt(sos, serie)
    
    fs_novo = second_filter_freq_1*2.5
    delta_novo = 1/fs_novo
    sample_novo = T/delta_novo
    q = int(sample/sample_novo)
    
    lista_clean = signal.decimate(lista_clean, q, n=None, ftype='iir', axis=-1, zero_phase=True)
    sample_time_novo = np.linspace(0, T, int(sample_novo))
    sample_time_novo = sample_time_novo
    return lista_clean,sample_time_novo, fs_novo

# ============ RMS  ============= #
@st.cache_data
def rms_vel(lista_v):
    rms_v = np.sqrt(sum((lista_v*1000)**2)/(len(lista_v)))
    rms_v = "{:.4f}".format(rms_v)
    rms_v = float(rms_v)
    return rms_v

# ============ FFT  ============= #
@st.cache_data
def fft(serie,fs):
    nin= len(serie)
    d = 1.00/(fs)
    fhat_x= np.fft.rfft(serie,nin, norm='ortho')            # variavel recebe a fft
    freq = np.fft.rfftfreq(nin, d)                #frequencia de aquisição
    Lin= np.arange(1,np.floor(nin/2),dtype='int')       #cria um array NumPy que contém uma sequência de números. 
    #A função floor() arredonda um número para baixo para o menor inteiro não negativo. cria um array que contém os índices de 1 a n/2.
    return fhat_x,freq,Lin,nin

# ============ 4th plot ============= #
@st.cache_data
def fft_fig(freq,fhat_x,fhat_y,fhat_z,L):
    fig = make_subplots(rows=3, cols=1)

    fig.add_trace(go.Scatter(x=freq[L],y=np.abs(fhat_x[L]), mode="lines", name="axs_x", line_width=0.4), row=1, col=1)
    fig.add_trace(go.Scatter(x=freq[L],y=np.abs(fhat_y[L]), mode="lines", name="axs_y", line_width=0.4), row=2, col=1)
    fig.add_trace(go.Scatter(x=freq[L],y=np.abs(fhat_z[L]), mode="lines", name="axs_z", line_width=0.4), row=3, col=1)


    fig.update_layout(title_text="Velocity Vibration Spectrum", title_x=0.5,
                    title_font_size=30,
                    xaxis_title="Frequencia [Hz]",
                    yaxis_title="Amplitude [mm/s]",
                    )
    #yaxis_type="log"
    # Definir o título do eixo Y do subplot na linha 2, coluna 1
    fig.update_yaxes(title="Amplitude [mm/s]", row=2, col=1)
    fig.update_xaxes(title="Frequencia [Hz]", row=2, col=1)

    # Definir o título do eixo Y do subplot na linha 2, coluna 1
    fig.update_yaxes(title="Amplitude [mm/s]", row=3, col=1)
    fig.update_xaxes(title="Frequencia [Hz]", row=3, col=1)
    return fig


# ============ anti-leakage windows ============= #
@st.cache_data
def hanning(fft):
    B = 1.5 # janela de hanning
    hanning = np.abs(fft)/np.sqrt((1/2*B))
    
    return hanning

@st.cache_data
def flattop(fft):
    B = 3.77 # janela flat top
    flattop = np.abs(fft)/np.sqrt((1/2*B))

    return flattop

@st.cache_data
def window(leakage,fft):
    if leakage == 'hanning':
        window = hanning(fft)
    elif leakage == 'flattop':
        window = flattop(fft)
    else:
        window = np.ones(len(fft))
    return window

# ============ inter all ============= #
@st.cache_data
def cumtrp_vel_all(lista_sample, lista_data_x, lista_data_y, lista_data_z,fs):
    lista_v_x, lista_samples = cumtrap_vel(lista_data_x,lista_sample,fs) 
    lista_v_y, lista_samples = cumtrap_vel(lista_data_y,lista_sample,fs)
    lista_v_z, lista_samples = cumtrap_vel(lista_data_z,lista_sample,fs)
    return lista_v_x,lista_v_y,lista_v_z, lista_samples
# ============ apll fil 1 all ============= #
@st.cache_data
def fill_butter_all(order_1,lista_v_x,lista_v_y, lista_v_z, first_filter_freq, first_filter, fs):
    lista_v_x = fil_butter(order_1,lista_v_x, first_filter_freq, first_filter, fs)
    lista_v_y = fil_butter(order_1,lista_v_y, first_filter_freq, first_filter, fs)
    lista_v_z = fil_butter(order_1,lista_v_z, first_filter_freq, first_filter, fs)
    return lista_v_x,lista_v_y, lista_v_z

# ============ apll fil 2 all ============= #
@st.cache_data
def fill_butter_2_all(order, lista_v_x, lista_v_y, lista_v_z, second_filter_freq_1, second_filter_freq_2, Second_filter, fs_,T,sample):
    lista_v_x1,sample_time_novo, fs_novo = fil_butter_2(order, lista_v_x, second_filter_freq_1, second_filter_freq_2, Second_filter, fs_,T,sample)
    lista_v_y1,sample_time_novo, fs_novo = fil_butter_2(order, lista_v_y, second_filter_freq_1, second_filter_freq_2, Second_filter, fs_,T,sample)
    lista_v_z1,sample_time_novo, fs_novo = fil_butter_2(order, lista_v_z, second_filter_freq_1, second_filter_freq_2, Second_filter, fs_,T,sample)
    return lista_v_x1,lista_v_y1, lista_v_z1,sample_time_novo, fs_novo

@st.cache_data
def fig_1(lista_sample, lista_data_x, lista_data_y, lista_data_z):
    with st.container():
        figure1 = raw_signal_fig(lista_sample, lista_data_x, lista_data_y, lista_data_z)
        st.write(figure1)



# ============ 2nd plot ============= #
@st.cache_data
def fig_2(lista_sample, lista_v_x, lista_v_y, lista_v_z):
    with st.container():
        figure2 = raw_vel_fig(lista_sample, lista_v_x, lista_v_y, lista_v_z)
        st.write(figure2)

# ============ Série Temporal da pos ============= #
@st.cache_data
def cumtrap_pos_all(lista_v_x,lista_v_y,lista_v_z, lista_sample):
    lista_pos_x = cumtrap_pos(lista_v_x, lista_sample)
    lista_pos_y = cumtrap_pos(lista_v_y, lista_sample)
    lista_pos_z = cumtrap_pos(lista_v_z, lista_sample)
    return lista_pos_x, lista_pos_y, lista_pos_z
# ============ 3rd plot ============= #
@st.cache_data
def fig_3(lista_sample,lista_pos_x,lista_pos_y,lista_pos_z):
    with st.container():
        figure3 = raw_pos_fig(lista_sample,lista_pos_x,lista_pos_y,lista_pos_z)
        st.write(figure3)

# ============ RMS Vel ============= #
@st.cache_data
def rms_all(lista_v_x,lista_v_y,lista_v_z):
    rms_v_x = rms_vel(lista_v_x)
    rms_v_y = rms_vel(lista_v_y)
    rms_v_z = rms_vel(lista_v_z)
    rms_med = np.sqrt(rms_v_x**2 + rms_v_y**2 + rms_v_z**2)/3
    return rms_v_x, rms_v_y, rms_v_z, rms_med


@st.cache_data
def metric_fig(rms_v_x, rms_v_y, rms_v_z, rms_med):
    col1, col2,col3, col4 = st.columns(4)
    st.divider()
    col1.metric(label = 'RMS Velocity X [mm/s]', value= rms_v_x, delta = dict2[Machinery]) 
    col2.metric(label = 'RMS Velocity Y [mm/s]', value= rms_v_y, delta = dict2[Machinery]) 
    col3.metric(label = 'RMS Velocity Z [mm/s]', value= rms_v_z, delta = dict2[Machinery]) 
    col4.metric(label = 'RMS Velocity mean [mm/s]', value= rms_med, delta = dict2[Machinery])

    st.divider()
    with st.sidebar.expander("Rms"):
        st.write('Rms no eixo x é :', rms_v_x, '[mm/s Rms]')
        st.write('Rms no eixo y é :', rms_v_y, '[mm/s Rms]')
        st.write('Rms no eixo z é :', rms_v_z, '[mm/s Rms]')

# ============ FFT - Vel  ============= #

@st.cache_data
def ftt_all(lista_v_x,lista_v_y,lista_v_z,fs):
    fft_v_x, freq, Lin, nin = fft(lista_v_x,fs)
    fft_v_y, freq, Lin, nin = fft(lista_v_y,fs)
    fft_v_z, freq, Lin, nin = fft(lista_v_z,fs)
    return fft_v_x, fft_v_y, fft_v_z, freq, Lin, nin
# ============ 4th plot ============= #

def fig_4(lista_v_x,lista_v_y,lista_v_z,fs, fft_v_x, fft_v_y, fft_v_z, freq, Lin, nin):
    with st.container():

        leakage = st.selectbox(
            'Selec your anti-leakage window',
            ('hanning', 'flattop', None),
            index=2,
            key="leakage_selectbox_1")

        st.write('You selected:', leakage)

        age = st.slider('Slide to zoom', 2, 200, 5)
        st.write("zoom ", age)



        col_leak1, col_leak2 = st.columns(2)
        with col_leak1:
            Reset_1 = st.button("Back", type="primary")
            if Reset_1 is True:
                fft_v_x, fft_v_y, fft_v_z, freq, Lin, nin = ftt_all(lista_v_x,lista_v_y,lista_v_z,fs)
                age = 2
        with col_leak1:
            Apply1 = st.button('See')
            if Apply1 is True:
                if leakage==None:
                    fft_v_x, fft_v_y, fft_v_z, freq, Lin, nin = ftt_all(lista_v_x,lista_v_y,lista_v_z,fs)
                else:
                    fft_v_x = window(leakage,fft_v_x)
                    fft_v_y = window(leakage,fft_v_y)
                    fft_v_z = window(leakage,fft_v_z)
                
                Lin= np.arange(1,np.floor(nin/age),dtype='int')
                
                
                
        figure4 = fft_fig(freq,fft_v_x,fft_v_y,fft_v_z,Lin)
        st.write(figure4)

    st.divider()

# ============ all grafs ============= #

def all_figs(lista_data_x, lista_data_y, lista_data_z, lista_sample,lista_pos_x,lista_pos_y,lista_pos_z,lista_v_x,lista_v_y,lista_v_z,fs, fft_v_x, fft_v_y, fft_v_z, freq, Lin, nin,rms_v_x, rms_v_y, rms_v_z, rms_med):
    fig_1(lista_sample, lista_data_x, lista_data_y, lista_data_z)
    fig_2(lista_sample, lista_v_x, lista_v_y, lista_v_z)
    fig_3(lista_sample,lista_pos_x,lista_pos_y,lista_pos_z)
    fig_4(lista_v_x,lista_v_y,lista_v_z,fs, fft_v_x, fft_v_y, fft_v_z, freq, Lin, nin)
    metric_fig(rms_v_x, rms_v_y, rms_v_z, rms_med)
    return


# ============ Tittle ============= #
"""
# Vibration Level Analysis.
"""
"""
**Developed by Rondon.**

This program aims to analyze vibration measurements on board a ship and compare them to the limits established by the classification society, 
with respect to comfort, equipment operation, and structural safety.
"""

# ============ Bottom link ============= #
btn = st.button("Press to access DNV Rules")
if btn:
    webbrowser.open_new_tab("https://www.dnv.com/news/dnv-gl-rules-for-classification-ships-july-2017--107749")

st.divider()
# ============ start debug ============= #
#lista_sample = [0]
# ========= Botão Analyze ===========#

rpm_op = st.number_input(
    "Insert a equipment RPM", value=750, placeholder="Type a number...")              
st.write('RPM is: ', rpm_op)

Structural = st.selectbox('Select a Structural Vibration',
    ('Steel', 'Aluminium'), 
    index=1)
st.write('You selected:', Structural)

Machinery = st.selectbox(
    'Select a Machinery',
    ('Shaft line bearings', 'Diesel engines < 200 RPM', 'Diesel engines > 200 RPM', 'Turbochargers',
    'Diesel driven generators, eletrical motors on thursters', 'Turbines', 'Turbines driven generators', 
    'Gears', 'Eletric Motors', 'Hydraulic pumps', 'fans', 'Compressors', 'reciprocating compressors and pumps',
    'Boilers', 'Pipes', 'Eletronic instruments and equipment','Compartment'),
    index=4)
st.write('You selected:', Machinery)

dict = {'Steel' : 45, 
        'Aluminium': 15}

dict2 = {'Shaft line bearings': 5,
        'Diesel engines < 200 RPM': 10,
        'Diesel engines > 200 RPM': 25, 
        'Turbochargers': 45,
    'Diesel driven generators, eletrical motors on thursters': 18,
    'Turbines': 7,
    'Turbines driven generators': 7,
    'Gears': 7,
    'Eletric Motors': 7,
    'Hydraulic pumps': 7,
    'fans': 7,
    'Compressors': 7,
    'reciprocating compressors and pumps': 30,
    'Boilers': 45,
    'Pipes': 45,
    'Eletronic instruments and equipment': 25,
    'Compartment': 2}


##### ++++++ SIDEBAR +++++++++ ########

st.sidebar.image("image/robot.png")

st.divider()

Delta = st.number_input("Insert a Delta", value=0.000488, placeholder="Type a number...")              #dt
st.write('dt is: ', Delta)

st.sidebar.divider()

# ============= Botão Upload =========== #


uploaded_file = st.file_uploader("Choose a .csv file, separeted by ';' ", type="csv") 


if uploaded_file:
    
    tab1, tab2 = st.tabs(["Tab 1", "Tab2"])
    tab1.write("Measure")
    tab2.write("AI")
    
    with tab1:
        lista_sample,lista_data_x,lista_data_y,lista_data_z,df = read_file_str (uploaded_file)
        
        # ======== Measure ========= #
        sample = len(lista_data_x)
        T = sample * Delta
        fs = 1 / Delta 
        
        col1, col2,col3, col4 = st.columns(4)
        st.divider()
        col1.metric(label = 'The Number of Sample is:', value= sample, delta = dict2[Machinery]) 
        col2.metric(label = 'The Period is:', value= T, delta = dict2[Machinery]) 
        col3.metric(label = 'dt is:', value= Delta, delta = dict2[Machinery]) 
        col4.metric(label = 'The sample rate is', value= fs, delta = dict2[Machinery])
        st.divider()
        
        lista_v_x,lista_v_y,lista_v_z, lista_samples = cumtrp_vel_all(lista_sample, lista_data_x, lista_data_y, lista_data_z,fs)
        lista_pos_x, lista_pos_y, lista_pos_z = cumtrap_pos_all(lista_v_x,lista_v_y,lista_v_z, lista_sample)
        # ============ filter 1 ============= #
        with st.sidebar.expander("Filter 1"):
            order_1 = st.number_input(
            "Insert The order of the filter", value=30, placeholder="Type a int number...")           
            st.write('The order is : ', order_1)
            
            first_filter = st.selectbox(
                'Select a filter',
                ('highpass', 'lowpass', None),
                index=0)
            st.write('You selected:', first_filter)

            first_filter_freq = st.number_input(
                "Insert a Frequency", value=0.5, placeholder="Type a number...")              
            st.write('Frequency is: ', first_filter_freq)
                
                    
            col_filt_1, col_filt_2 = st.columns(2)
            with col_filt_1:
                Reset_filt1 = st.button("Reset", type="primary")    #, on_click= cumtrp_vel_all, args=[lista_sample, lista_data_x, lista_data_y, lista_data_z]
                if Reset_filt1:
                    lista_sample, lista_data_x, lista_data_y, lista_data_z, sample, T, fs, Delta = read_file_str (uploaded_file)
            with col_filt_2: 
                Apply_filt1 = st.button('Apply') #, on_click= fill_butter_all, args=[order_1,lista_v_x,lista_v_y, lista_v_z, first_filter_freq, first_filter, fs]
                if Apply_filt1:
                    lista_v_x,lista_v_y, lista_v_z = fill_butter_all(order_1,lista_v_x,lista_v_y, lista_v_z, first_filter_freq, first_filter, fs)

        # ============ filter 2 ============= #

        with st.sidebar.expander("Filter 2"):
            Second_filter = st.selectbox(
                'Apply a bandpass filter',
                ('bandpass', None),
                index=0)

            st.write('You selected:', Second_filter)

            order = st.number_input(
                "Insert The order of the new filter", value=30, placeholder="Type a int number...")           

            st.write('The order is : ', order)

            second_filter_freq_1 = st.number_input(
                "Insert a Top Frequency", value=200, placeholder="Type a number...")   

            second_filter_freq_2 = st.number_input(
                "Insert a Low Frequency", value=4, placeholder="Type a number...")           

            st.write('Frequency : ', second_filter_freq_1, second_filter_freq_2)
            
            col_filt_3, col_filt_4 = st.columns(2)
            with col_filt_3:
                Reset_filt2 = st.button("Reset", type="secondary")    #, on_click= cumtrp_vel_all, args=[lista_sample, lista_data_x, lista_data_y, lista_data_z]
                if Reset_filt2:
                    lista_sample, lista_data_x, lista_data_y, lista_data_z, sample, T, fs, Delta = read_file_str (uploaded_file)
            with col_filt_4: 
                Apply_filt2 = st.button('Run', key='app_fil_2')  #, on_click= fill_butter_2_all, args=[order, lista_v_x, lista_v_y, lista_v_z, second_filter_freq_1, second_filter_freq_2, Second_filter, fs,T,sample]
                if Apply_filt2:
                    lista_v_x,lista_v_y, lista_v_z, lista_sample,fs = fill_butter_2_all(order, lista_v_x, lista_v_y, lista_v_z, second_filter_freq_1, second_filter_freq_2, Second_filter, fs,T,sample)
        fft_v_x, fft_v_y, fft_v_z, freq, Lin, nin = ftt_all(lista_v_x,lista_v_y,lista_v_z,fs)
        rms_v_x, rms_v_y, rms_v_z, rms_med = rms_all(lista_v_x,lista_v_y,lista_v_z)
        all_figs(lista_data_x, lista_data_y, lista_data_z, lista_sample,lista_pos_x,lista_pos_y,lista_pos_z,lista_v_x,lista_v_y,lista_v_z,fs, fft_v_x, fft_v_y, fft_v_z, freq, Lin, nin,rms_v_x, rms_v_y, rms_v_z, rms_med)
        

    with tab2:
        # ============ AI ============= #
        """
            # AI
        """
        with st.expander("Chat Bot"):
            txt = st.text_input(
                "openai api_key",
                value='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'
                )

            if txt is not None:
                openai.api_key = txt
                completion = openai.ChatCompletion.create(
                    model = 'gpt-3.5-turbo',
                    messages = [
                        {"role": "system", "content": "Você é um especialista em análise de vibração e leu a norma ISO_10816-7_2009 no link: https://drive.google.com/file/d/17egSeNv3ejGtRrfdQPJpH45yBOkFwMnp/view?usp=sharing ."},
                        {"role": "user", "content": f'Os datasets a seguir se referem a uma medição de vibração adquirida em um equipamento. consulte a tabela para o equipamento {Machinery}, na estrutura {Structural}, o equipamento está operando a {rpm_op} RPM foi adquirido atravez de um acelerometro triaxial cujos dados de aquisição os seguintes valores de RMS globais de velocidade {rms_v_x, rms_v_y,rms_v_z,rms_med} em mm/s na faixa de frequencia de {second_filter_freq_2} a {second_filter_freq_1} foi atendido todos os requisitos exigidos pela norma para a analise, a frequencia sugerida nas tabelas,  analisando a componete da velocidade. Com base na norma ISO_10816-7_2009 faça uma analise das datasets e da RMS.'}
                    ]
                    )

                st.markdown(completion['choices'][0]['message']['content'])

            #''

