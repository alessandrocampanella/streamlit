import streamlit as st
import pandas as pd
import plotly.express as px
from statsmodels.tsa.stattools import adfuller, acf, pacf
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from sklearn.decomposition import PCA
import tensorflow as tf
import pmdarima as pm

st.markdown("<h1>📊 Analisi di serie temporali</h1>",unsafe_allow_html=True)
st.markdown("<h3> 👀 Visualizzazione dei dati </h3>",unsafe_allow_html=True)
st.markdown("""Di seguito sono mostrati i dataset pre-trattati (sono state rimosse le colonne 
<i>Temperatura Intervallo low</i> e <i>high</i>). I dati considerati sono i seguenti:
""",unsafe_allow_html=True)

st.markdown("""
- Dati meteo storici (Cicalino 1)
- Grafico delle catture (Cicalino 1)
- Dati meteo storici (Cicalino 2)
- Grafico delle catture (Cicalino 2)
""",unsafe_allow_html=True)

# il decoratore @st.cache_data serve per memorizzare nella cache il risultato di una funzione, 
# in modo da evitare ricalcoli inutili e migliorare le prestazioni dell'applicazione.
@st.cache_data
# funzione per leggere file excel
def load_data(file):
    return pd.read_excel(file,engine="openpyxl")

df_1 = load_data("./datasets/dati-meteo-storici (Cicalino 1).xlsx")
df_2 = load_data("./datasets/grafico-delle-catture (Cicalino 1).xlsx")
df_3 = load_data("./datasets/dati-meteo-storici (Cicalino 2).xlsx")
df_4 = load_data("./datasets/grafico-delle-catture (Cicalino 2).xlsx")

# Si creano 2 colonne per i dati
col1, col2 = st.columns(2)

# si stampano le tabelle
with col1:
    st.markdown("<h6 style='text-align: center;'>Dati Meteo Storici (Cicalino 1) - df_1</h6>", unsafe_allow_html=True)
    st.write(df_1)
    st.markdown("<h6 style='text-align: center;'>Dati Meteo Storici (Cicalino 2) - df_3</h6>", unsafe_allow_html=True)
    st.write(df_3)

with col2:
    st.markdown("<h6 style='text-align: center;'>Grafico delle Catture (Cicalino 1) - df_2</h6>", unsafe_allow_html=True)
    st.write(df_2)
    st.markdown("<h6 style='text-align: center;'>Grafico delle Catture (Cicalino 2) - df_4</h6>", unsafe_allow_html=True)
    st.write(df_4)

st.markdown("<br><h3> 🛠️Preprocessing dei dati </h3>",unsafe_allow_html=True)
st.markdown("""
La colonna <i>DateTime</i> è stata impostata come indice dei quattro DF.<i>df_1 e df_3</i> 
contengono dati di <i>Temperatura </i> e <i>Umidità</i> per ogni giorno e ora:
""",unsafe_allow_html=True)

# Per ogni foglio del Dataset abbiamo convertito la colonna 'DateTime' in formato specifico
# come mostrato in output, se fallisce mostra Nat in output (errors = 'coerce')
for df in [df_1, df_2, df_3, df_4]:
    df["DateTime"] = pd.to_datetime(df["DateTime"], format="%d.%m.%Y %H:%M:%S", errors='coerce')

# Abbiamo impostato la colonna 'DateTime' di ogni foglio xlsx come indice
# dopo averla usata come indice la colonna 'DateTime' viene cancellata
df_1.index = df_1.DateTime
del df_1["DateTime"]

df_2.index = df_2.DateTime
del df_2["DateTime"]

df_3.index = df_3.DateTime
del df_3["DateTime"]

df_4.index = df_4.DateTime
del df_4["DateTime"]

# Si divide il foglio in due colonne
column1, column2 = st.columns(2)
with column1:
    st.write(df_1)
with column2:
    st.write(df_3)

# E' stao effettuato il resampling per aggregare i dati su base giornaliera -> DAILY RESAMPLING
df1_daily_resampling = df_1.resample('D').mean()
df3_daily_resampling = df_3.resample('D').mean()

# Concatenazione dei due fogli Excel, creando dei duplicati nelle date se hanno gli stessi giorni
df_merged = pd.concat([df1_daily_resampling,df3_daily_resampling])

st.markdown("""<i>df_2 e df_4</i> contengono dati relativi agli <i>insetti</i>. Come prima cosa è necessario 
<b>normalizzare</b> gli indici, si <b>rimuove</b> l'<i>orario</i> e si conserva la data (informazione di interesse).
""",unsafe_allow_html=True)
c1,c2 = st.columns(2)

for df, col in zip([df_2, df_4], [c1, c2]):
    df.index = pd.to_datetime(df.index).normalize()
    with col:
        st.dataframe(df)
    
st.markdown("""Sono poi state rimosse le colonne di <i>Recensito</i> ed <i>Evento</i>, poiché contenenti 
valori costanti o quasi totalmente nulli.
""",unsafe_allow_html=True)

# Sono state create delle nuove colonne per la stampa dei dataset
col_1,col_2 = st.columns(2)
for df, col in zip([df_2, df_4], [col_1, col_2]):
    df.drop(columns=['Recensito', 'Evento'], errors='ignore', inplace=True)
    df.fillna(0, inplace=True)

    with col:
        st.write(df) 

st.markdown("""Si concatenano i dataframe <i>df_2</i> e <i>df_4</i> raggruppandoli per data e si calcola la 
somma giornaliera del <b>Numero di insetti</b> e <b>Numero di catture (per evento)</b>:
""",unsafe_allow_html=True)

# Si concatena i due fogli
df_merged2 = pd.concat([df_2, df_4])

# Si calcola la somma del numero di insetti e catture dopo aver raggruppato per DateTime
df_merged2 = df_merged2.groupby(df_merged2.index).sum()
st.dataframe(df_merged2, use_container_width=True) # si stampa la tabella in modo da coprire il 100% della larghezza

st.markdown("""I dati risultano idonei all'analisi, dal momento che sia i <i>df_merged</i> (relativo ai fenomeni 
metereologici) che <i>df_merged2</i> (relativo allo studio degli insetti) contengono un'istanza per ogni giorno 
di osservazione dal <b>2024-07-05</b> al <b>2024-08-23</b>.
""",unsafe_allow_html=True)

st.markdown("<br><h3> 📈Analisi dei dati </h3>",unsafe_allow_html=True)

# Funzione per generare il riepilogo
def display_dataframe_summary(df,name):
    st.markdown(name,unsafe_allow_html=True)
    date_range = (df.index.max() - df.index.min()).days
    st.write(f"- **Indice**: DateTime")
    st.write(f"- **Numero di righe**: {df.shape[0]}")
    st.write(f"- **Numero di colonne**: {df.shape[1]}")
    
    for col in df.columns:
        st.write(f" - **{col}**: {df[col].dtype}")
    
    for col in df.columns:
        st.write(f"- **Valori nulli per {col}**: {df[col].isna().sum()}")
    
    st.markdown("<br><h6> Statistiche per ogni colonna numerica:</h6>",unsafe_allow_html=True)
    st.dataframe(df.describe(),use_container_width=True)

# Chiamata alla funzione per visualizzare il riepilogo
display_dataframe_summary(df_merged,"<h4> 📊 Dati Meteorologici</h4>")
display_dataframe_summary(df_merged2,"<br><h4> 🐛 Dati Insetti</h4>")

st.markdown("""Si concatenano i due dataframe ottenendone un unico. Le informazioni sono raggruppate 
su base giornaliera. Per ogni istanza saranno unite le colonne dei due dataframe:
""",unsafe_allow_html=True)

# concatenazione e stampa
merged_df = pd.concat([df_merged, df_merged2], axis=1)
st.dataframe(merged_df,use_container_width=True)

st.markdown("""Si riportano i grafici relativi alla distribuzione e all'andamento dei dati. 
Attraverso tale analisi, sarà inoltre possibile valutare gli outlier.
""",unsafe_allow_html=True)

st.markdown("<br><h3> 📉 Grafici</h3>",unsafe_allow_html=True)

# Si mostra la frequenza di occorrenza di 'Numero di Insetti'
st.markdown("<h4>#️⃣ Frequenza di occorrenza - Numero di insetti </h4>",unsafe_allow_html=True)
st.bar_chart(merged_df["Numero di insetti"].value_counts().sort_index()) 

# Si mostra la frequenza di occorrenza di 'Nuove catture (per evento)'
st.markdown("<h4>#️⃣ Frequenza di occorrenza - Nuove catture (per evento) </h4>",unsafe_allow_html=True)
st.bar_chart(merged_df["Nuove catture (per evento)"].value_counts().sort_index())

# Si mostra l'andamento nel tempo della 'Media Umidità'
st.markdown("<h4>➗ Media Umidità nel tempo</h4>",unsafe_allow_html=True)
st.line_chart(merged_df["Media Umidità"])

# Si mostra l'andamento nel tempo della 'Media Temperatura'
st.markdown("<h4>➗ Media Temperatura nel tempo</h4>",unsafe_allow_html=True)
st.line_chart(merged_df["Media Temperatura"])

st.markdown("""E' necessario a questo punto verificare eventali relazioni tra le variabili. Nei seguenti 
<i>Scatter Plot</i> sono mostrate le distribuzioni dei valori di <b>Numero di Insetti</b> e <b>Nuove Catture 
(per evento)</b> e la loro (eventuale) relazione con <b>Media Temperatura</b> e <b>Media Umidità.</b>
""",unsafe_allow_html=True)

# Funzione per la creazione dello Scatter Plot
def plot_interactive_scatter(df, x_col, y_col, target):

    st.markdown("<h4> 🫧 Scatter Plot - "+target+" - "+x_col+" - "+y_col+"</h4>",unsafe_allow_html=True)
    # Creazione di  ScatterPlot
    fig = px.scatter(df, x=x_col, y=y_col, hover_name=df.index, labels={x_col: x_col, y_col: y_col})

    # Visualizzazione del grafico
    st.plotly_chart(fig, key=target)

# Invocazione della funzione
plot_interactive_scatter(merged_df, "Media Temperatura", "Media Umidità", "Numero di insetti")
plot_interactive_scatter(merged_df, "Media Temperatura", "Media Umidità", "Nuove catture (per evento)")

st.markdown("""Si osserva (sia per numero di insetti che catture) che la temperatura è compresa tra i 
<b>26°C</b> e <b>28°C</b>, mentre l'umidità è compresa tra <b>50%</b> e <b>70%.</b>
""",unsafe_allow_html=True)

st.markdown("<h3>📍Analisi approfondita dei dati</h3>",unsafe_allow_html=True)
st.markdown("""Attraverso la funzione <i>dataset_statistics</i>, si calcolano le statistiche 
fondamentali utili per le decisioni future. Si possono valutare le seguenti metriche:
""",unsafe_allow_html=True)

st.markdown("""- <b>Media</b>: ottenuta dalla somma di campioni considerati rispetto al numero totale di essi;
""",unsafe_allow_html=True)
st.markdown(r"""
- <b>Percentile</b>: definito come il numero di elementi che in percentuale si trovano al di sotto dell'elemento 
che sto considerando. Se il dataset lo dividiamo in quattro parti, possiamo lavorare con:
    - il <b>primo quartile</b> ($$Q_1 - 25\%$$ delle osservazioni sono minori di esso); 
    - il <b>secondo quartile o mediana</b> ($$Q_2 - 50\%$$ delle osservazioni sono minori di esso);
    - il <b>terzo quartile</b> ($$Q_3 - 75\%$$ delle osservazioni sono minori di esso).
    <br><br>Si può quindi parlare di <b>range inter-quartile (IRQ)</b>, come la differenza tra il terzo quartile 
    e il primo quartile:
""",unsafe_allow_html=True)
st.latex(r"\text{IQR}={Q_3-Q_1}")

st.markdown("""
- <b>Skewness (Asimmetria)</b>: misura il grado di asimmetria della distribuzione rispetto a una normale:
    - un valore positivo indica una <i>coda</i> verso destra; 
    - un valore negativo indica una <i>coda</i> verso sinistra.
""",unsafe_allow_html=True)

st.markdown("""
- <b>Kurtosis (Curtosi)</b>: descrive la forma della distribuzione e la concentrazione dei dati intorno alla media:
    - un valore elevato indica un maggiore appiattimento della funzione (presenta più valori agli estremi); 
    - un valore elevato indica un minore appiattimento della funzione (presenta meno valori agli estremi).
""",unsafe_allow_html=True)

st.markdown("""
- <b>Varianza</b>: misura la dispersione dei dati intorno alla media. Se i dati si discostano maggiormente dalla 
media, il valore di varianza sarà più elevato.
""",unsafe_allow_html=True)

st.markdown(r"""
Si effettua un test di <b>Augmented Dickey-Fuller (ADF)</b>, che verifica la <i>stazionarietà dei dati</i>. Si usa 
per valutare se la serie temporale ha proprietà stabili nel tempo:
- $$H_0$$: la serie non è stazionaria; 
- $$H_1$$: la serie è stazionaria;
""",unsafe_allow_html=True)

st.markdown(r"se il $$\text{p-value} < \alpha$$, si rigetta l'ipotesi nulla.",unsafe_allow_html=True)

# Funzione che calcola le statistiche del dataset
def dataset_statistics(df):
    # Si selezionano solo le colonne con dati numerici
    numeric_df = df.select_dtypes(include=[np.number])

    # Si calcolano statistiche descrittive (inclusi percentili) e si traspongono in un formato leggibile (.T)
    stats = numeric_df.describe(percentiles=[.25, .5, .75]).T

    # Si calcolano metriche aggiuntive:
    # Skewness (asimmetria),
    # Kurtosis (misura la coda dei dati), 
    # varianza (dispersione dei dati),
    # Range interquartile (tra il 75° e il 25° percentile)
    stats['Skewness'] = numeric_df.skew()
    stats['Kurtosis'] = numeric_df.kurtosis()
    stats['Variance'] = numeric_df.var()
    stats['IQR'] = stats['75%'] - stats['25%']

    # Esegue l'ADF test per valutare la stazionarietà della serie temporale per ogni colonna numerica
    adf_results = {}
    for column in numeric_df.columns:
        adf_test = adfuller(numeric_df[column].dropna()) # si applica il test, eliminando valori nulli
        adf_results[column] = {
            'ADF Statistic': adf_test[0],   # statistica del test ADF
            'p-value': adf_test[1],         # p-value per verificare la stazionarietà (se <0.05 si rigetta l'ipotesi
                                            # nulla e si può concludere che la serie è stazionaria)
            'Critical Values': adf_test[4]  # valori critici del test 
        }

    # Si crea un dataframe dai rilsultati del test ADF e conserva solo statistiche ADF e p-value
    adf_df = pd.DataFrame(adf_results).T
    adf_df = adf_df[['ADF Statistic', 'p-value']]

    # Si combinano le statistiche descrittive con i risultati ADF
    stats = pd.concat([stats, adf_df], axis=1)

    return stats

# Invocazione della funzione
stats_total = dataset_statistics(merged_df)
st.write(stats_total)

st.markdown("""Osservando il p-value si può concludere che l'ipotesi nulla è rigettata: i dati 
risultano stazionari.
""",unsafe_allow_html=True)

st.markdown("""<p>La funzione <i>advanced_time_series_analysis</i>, è utilizzata per raccogliere le 
informazioni necessarie per <b>la scelta dei parametri migliori</b> per un modello <b>ARIMA</b>.
Prende in input il DF contente i dati temporali, la colonna da analizzare, il n° di lags (ritardi) 
per l'analisi dell'autocorrelazione e la periodicità della serie.
</p>""",unsafe_allow_html=True)

st.markdown(""" Si osservano i seguenti valori:
- <b>ACF (Autocorrelation Function)</b>: analizza la relazione tra i valori attuali della serie e i 
loro ritardi (lags). Attraverso questa funzione, è possibile valutare se i dati sono correlati nel tempo.
""",unsafe_allow_html=True)

st.markdown("""
- <b>PACF (Partial Autocorrelation Function)</b>: mostra la correlazione diretta tra i valori correnti e 
i ritardi, non considerando gli effetti delle correlazioni intermedie.
""",unsafe_allow_html=True)

st.markdown("""
- <b>Decomposizione della serie temporale</b>: separa la serie in tre componenti principali:
    <br>1. <b>Trend</b>: indica <i>movimento complessivo</i> della serie;
    <br>2. <b>Stagionalità</b>: indica i pattern <i>regolari</i> che si ripetono con frequenza specifica
    <br>3. <b>Residuo (o errore)</b>: le variazioni casuali o il rumore che non può essere spiegato dal 
    trend o la stagionalità
""",unsafe_allow_html=True)


# Funzione per analisi avanzata delle serie temporali con slider
def advanced_time_series_analysis(df, column, lags, period):
    time_series = df[column]

    # Calcolo e visualizzazione dell'Autocorrelation Function (ACF)
    acf_values, confint = acf(time_series.dropna(), nlags=lags, alpha=0.05)
    fig_acf = go.Figure()
    fig_acf.add_trace(go.Bar(x=np.arange(len(acf_values)), y=acf_values, name='ACF', marker_color='royalblue'))
    fig_acf.update_layout(
        title=f"Autocorrelation Function (ACF) - {column}", 
        xaxis_title='Lag', 
        yaxis_title="ACF", 
        template="plotly_white"
    )
    
    # Calcolo e visualizzazione della Partial Autocorrelation Function (PACF)
    pacf_values, confint = pacf(time_series.dropna(), nlags=lags, alpha=0.05)
    fig_pacf = go.Figure()
    fig_pacf.add_trace(go.Bar(x=np.arange(len(pacf_values)), y=pacf_values, name='PACF', marker_color='darkorange'))
    fig_pacf.update_layout(
        title=f"Partial Autocorrelation Function (PACF) - {column}", 
        xaxis_title='Lag', 
        yaxis_title="PACF", 
        template="plotly_white"
    )
    
    # Calcolo e visualizzazione della Decomposizione stagionale
    result = seasonal_decompose(time_series.dropna(), model='additive', period=period)
    fig_decomp = go.Figure()
    fig_decomp.add_trace(go.Scatter(x=result.trend.index, y=result.trend, mode='lines', name='Trend'))
    fig_decomp.add_trace(go.Scatter(x=result.seasonal.index, y=result.seasonal, mode='lines', name='Seasonal'))
    fig_decomp.add_trace(go.Scatter(x=result.resid.index, y=result.resid, mode='lines', name='Residual'))
    fig_decomp.update_layout(
        title=f"Seasonal Decomposition - {column}", 
        xaxis_title='Time', 
        yaxis_title="Value", 
        template="plotly_white")
    
    # Mostra i grafici
    st.plotly_chart(fig_acf)
    st.plotly_chart(fig_pacf)
    st.plotly_chart(fig_decomp)


# Slider per selezionare il numero di lags e il periodo stagionale
lags = st.slider("Seleziona il numero di lag", min_value=5, max_value=25, value=25)
period = st.slider("Seleziona il periodo stagionale", min_value=1, max_value=24, value=12)

# Se il bottone è premuto esegui l'analisi (in modo da evitare la visualizzazione non richiesta in caso di ricarica della pagina)
if st.button("Esegui analisi (ACF, PACF, Decomposizione)", key="btn1"):
    advanced_time_series_analysis(merged_df, "Numero di insetti", lags, period)
    advanced_time_series_analysis(merged_df, "Nuove catture (per evento)", lags, period)

st.markdown("<h3>🧪 ARIMA</h3>",unsafe_allow_html=True)
st.markdown("""
Il primo modello di regressione usato è <b>AUTO-ARIMA (AutoRegressive Integrated Moving Average)</b>. 
E' costituito di tre componenti principali:
""",unsafe_allow_html=True)

st.markdown("""
1. <b>AR</b> è la componente <b>auto-regressiva</b>, esprime la relazione tra la serie temporale e 
i suoi valori passati. L'ordine di AR dipende da <i>p</i>, ovvero il <b>numero di lag</b> da considerare.
2. <b>I</b> è la componente di <b>differenziazione</b>, mi permette di rendere la <b>rete stazionaria</b>. 
L'ordine di differenziazione dipende da <i>d</i>.
3. <b>MA</b> è il componente relativo alla <b>media mobile</b>, rappresenta la relazione tra un <b>
valore della serie temporale</b> e gli <b>errori precedenti</b>. Il numero di errori dipende da <i>q</i>
""",unsafe_allow_html=True)

st.markdown("""
Il modello complessivo è espresso come <b>ARIMA(p,d,q)</b>. Attraverso <b>AUTO-ARIMA</b> è possibile 
estendere il modello <b>ARIMA</b>, definendo una <b>versione automatizzata</b> con la quale è possibile 
calcolare i valori di <b>p,d,q</b>. Attraverso AUTO-ARIMA, è possibile usando criteri statistici quali 
<i>AIC (Akaike Information Criterion)</i> e <i>BIC (Bayesian Information Criterion)</i> determinare i 
<b>valori migliori</b> per la specifica serie temporale.
""",unsafe_allow_html=True)

st.markdown("""
<b>ARIMAX (AutoRegressive Integrated Moving Average with eXogenous variables)</b> 
è un'ulteriore estensione del modello <b>ARIMA</b>, si aggiungono <b>variabili indipendenti</b> 
(esogene) per <b>migliorare le previsioni della variabile target</b>. Combina componenti 
autoregressive, di differenziazione e di media mobile, ma considera anche i fattori esterni.
""",unsafe_allow_html=True)

st.markdown("""
Nel nostro caso le <i>variabili di risposta</i> da prevedere sono <b>Numero di Insetti 
e Nuove catture (per evento)</b>, invece <b>Media Temperatura e Media Umidità
</b> sono i <i>predittori</i>, ovvero le variabili indipendenti che influenzano quelle dipendenti.
""",unsafe_allow_html=True)

st.markdown("""
Si calcolano poi per il <b>training set</b> e per il <b>test set</b> i seguenti valori:
- <i>Residui</i>: rappresentano la differenza tra valori predretti e osservati, ovvero gli <b>errori 
di previsione</b>.
- <i>Root Mean Square (RMSE)</i>: <b>radice quadrata della media degli errori quadratici</b>.
- <i>Mean Absolute Error (MAE)</i>: media del valore assoluto degli errori, che misura l'entità (media) 
degli errori, non considerando il segno.
""",unsafe_allow_html=True)

st.markdown("""
Attraverso queste metriche, possiamo valutare la qualità delle previsioni sia sul <b>training</b> che sul <b>
test set</b>. Con queste metriche si valutano eventuali problemi di <b>overfitting o underfitting</b> e si 
verifica l'<b>accuracy</b> del modello.
""",unsafe_allow_html=True)

st.markdown("""
Si effettua un <b>partitioning 80%-20%</b>, l'80% dei dati è usato per il training e il restante 20% per il test.
""",unsafe_allow_html=True)

st.markdown("""
La funzione <i>auto_arima</i> ricerca i valori migliori per <i>p,d,q</i> variandoli da 0 a 15 e identificando 
la combinazione che meglio si adatti ai dati.
""",unsafe_allow_html=True)

st.markdown("""
<i>fitted_values_train</i> rappresenta le previsioni del modello per il <b>set di addestramento</b>, 
la funzione <i>forecast</i> lavora, invece, con l'insieme di <b>test</b>. I valori sono <b>limitati 
inferiormente a zero</b> (in modo da non avere valori negativi) e si arrotondano all'intero più vicino. 
Gli <b>intervalli di confidenza</b> relativi ai valori predetti sono memorizzati da <i>conf_int</i>.
""",unsafe_allow_html=True)

# Funzione per la creazione del modello ARIMAX
@st.cache_data
def arimax(col,col1,col2):
    st.markdown(col,unsafe_allow_html=True)
    
    # Selezione delle colonne
    x = col1
    y = col2

    # Divisione in training e test
    train_size = int(len(df_merged2) * 0.8)
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    X_train, X_test = x.iloc[:train_size], x.iloc[train_size:]

    # Creazione del modello ARIMAX
    with st.spinner("Addestramento del modello ARIMAX..."):
        arimax_model = sm.tsa.ARIMA(endog=y_train, exog=X_train, order=(10, 0, 11)).fit()

    # Predizioni
    fitted_values_train = arimax_model.fittedvalues.clip(lower=0).round()
    arimax_forecast = arimax_model.get_forecast(steps=len(y_test), exog=X_test)
    forecasted_values_test = arimax_forecast.predicted_mean.clip(lower=0).round()
    confidence_intervals_test = arimax_forecast.conf_int()
    confidence_intervals_test[confidence_intervals_test < 0] = 0
    confidence_intervals_test = confidence_intervals_test.round()

    # Metriche di valutazione
    rmse_train = np.sqrt(mean_squared_error(y_train, fitted_values_train))
    mae_train = mean_absolute_error(y_train, fitted_values_train)
    rmse_test = np.sqrt(mean_squared_error(y_test, forecasted_values_test))
    mae_test = mean_absolute_error(y_test, forecasted_values_test)

    # Grafico con Plotly
    fig = go.Figure()

    # Linea valori reali
    fig.add_trace(go.Scatter(x=y.index, y=y, mode='lines', name='Actual'))

    # Linea fitted training
    fig.add_trace(go.Scatter(x=fitted_values_train.index, y=fitted_values_train, 
                                mode='lines', name='Fitted (Train)', line=dict(dash='dash')))

    # Linea previsione test
    fig.add_trace(go.Scatter(x=forecasted_values_test.index, y=forecasted_values_test, 
                                mode='lines', name='Forecast (Test)'))

    # Area intervallo di confidenza
    fig.add_trace(go.Scatter(
        x=confidence_intervals_test.index, y=confidence_intervals_test.iloc[:, 0], fill=None, mode='lines',
        line=dict(color='green', width=0), showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=confidence_intervals_test.index, y=confidence_intervals_test.iloc[:, 1], fill='tonexty', mode='lines',
        line=dict(color='green', width=0), name='Confidence Interval'
    ))

    # Layout del grafico
    fig.update_layout(
        title=f"Model: Training and Forecasting | Training RMSE: {rmse_train:.2f}, MAE: {mae_train:.2f} | "
                f"Test RMSE: {rmse_test:.2f}, MAE: {mae_test:.2f}",
        xaxis_title="Time",
        yaxis_title="Value",
        legend_title="Type",
        template="plotly_dark"
    )

    # Mostra il grafico in Streamlit
    st.plotly_chart(fig)


# Al click del bottone si invocano le funzioni
if st.button("Esegui analisi (ARIMAX)"):
    arimax("""<h4>🐛 Numero di insetti</h4>""",df_merged2['Nuove catture (per evento)'],df_merged2['Numero di insetti'])
    arimax("""<h4>📸 Nuove catture (per evento)</h4>""",df_merged2['Numero di insetti'],df_merged2['Nuove catture (per evento)'])

st.markdown("""
Il modello <i>AUTO-ARIMA</i> confronta i diversi risultati di (p,d,q) partendo da (0,0,0) fino a (5,0,0). 
Seleziona la combinazione di valori che <b>minimizza il valore dell'AIC</b>. Tale indice confronta la 
qualità del training ai dati e alla sua complessità, prediligendo modelli semplici ma meno precisi.
""",unsafe_allow_html=True)

st.markdown("""<h3>💪🏽VARMAX</h3>""",unsafe_allow_html=True)
st.markdown("""
E' stato poi utilizzato <b>VARMAX</b> <i>(Vector AutoRegressive Moving Average with Exogenous Variables)</i>. 
Attraverso tale modello è possibile prevedere variabili temporali, sfruttando relazioni tra varaibili <b>endogene 
</b>(interne) ed <b>esogene</b> (esterne). Le prime sono spiegate dal modello; il loro comportamento dipende da 
altre variabili all'interno dello stesso modello, nel caso di esogene invece fattori esterni influenzano il modello, 
ma non sono spiegate dal modello stesso.
""",unsafe_allow_html=True)

st.markdown("""
Nel nostro caso le variabili <b>endogene</b> sono rappresentate da:
- <b>Variabile target</b>: Numero di insetti oppure Nuove catture (per evento)
- <b>Temperatura</b>
- <b>Umidità</b>
""",unsafe_allow_html=True)

st.markdown("""
Le variabili <b>esogene</b> sono invece:
- <b>Media_mobile_4gg</b>: la somma dei valori della variabile target, calcolata su una finestra di 4 giorni.
- <b>PCA_Component</b>: una variabile ottenuta attraverso la Principal Component Analisys (PCA) che combina
- <b>Nuove catture (per evento)</b> oppure <b>Numero di insetti</b>
""",unsafe_allow_html=True)


# Funzione per la creazione del modello VARMAX
@st.cache_data
def varmax(target, es, ar, ma):
    # Si fa una copia della funzione
    merged_df_copy = merged_df.copy()
    
    # Calcolo della media mobile a 4 giorni per la variabile target
    merged_df_copy['Media mobile 4 giorni'] = merged_df_copy[target].rolling(window=4).sum()

    # Riempimento dei valori NaN con 0
    merged_df_copy = merged_df_copy.fillna(0)

    # Selezione delle variabili per PCA
    features_for_pca = merged_df_copy[['Media Temperatura', 'Media Umidità', target]]

    # Standardizzazione delle variabili
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_for_pca)

    # Riduzione dimensionale a una componenti principale
    pca = PCA(n_components=1)
    pca_result = pca.fit_transform(features_scaled)

    # Aggiunta della componente principale al dataframe
    merged_df_copy['temp_humidity_pca'] = pca_result

    # Si fa una copia di merged_df_copy
    data_clean = merged_df_copy.copy()

    # Variabili endogene ed esogene
    endog = data_clean[['Media Temperatura', 'Media Umidità', target]] #Variabili endogene
    exog = data_clean[['Media mobile 4 giorni', 'temp_humidity_pca', es]] #variabili esogene

    # Partitioninig
    train_size = int(len(data_clean) * 0.8)
    endog_train, endog_test = endog.iloc[:train_size], endog.iloc[train_size:]
    exog_train, exog_test = exog.iloc[:train_size], exog.iloc[train_size:]

    p = ar
    q = ma

    # Definizione del modello VARMAX model sui dati di training
    varmax_model = sm.tsa.VARMAX(endog_train, exog=exog_train, order=(p, q), trend='n').fit(disp=False)

    # Predizioni sui dati di training
    fitted_values_train = varmax_model.fittedvalues.copy()
    fitted_values_train[target] = fitted_values_train[target].clip(lower=0).round()

    # Forecasting sui dati di test
    varmax_forecast = varmax_model.get_forecast(steps=len(endog_test), exog=exog_test)
    forecasted_values_test = varmax_forecast.predicted_mean.copy()
    forecasted_values_test[target] = forecasted_values_test[target].clip(lower=0).round()

    # Confidence Intervals sul Test Forecast
    confidence_intervals_test = varmax_forecast.conf_int()
    confidence_intervals_test = confidence_intervals_test.clip(lower=0)  # Ensure non-negative bounds

    # Calculo delle metriche: Training Metrics
    rmse_train = np.sqrt(mean_squared_error(endog_train[target], fitted_values_train[target]))
    mae_train = mean_absolute_error(endog_train[target], fitted_values_train[target])

    # Calculo delle metriche: Test Metrics
    rmse_test = np.sqrt(mean_squared_error(endog_test[target], forecasted_values_test[target]))
    mae_test = mean_absolute_error(endog_test[target], forecasted_values_test[target])

    # Visualizzazione dei risultati
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=endog.index, y=endog[target], mode='lines', name='Actual', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=fitted_values_train.index, y=fitted_values_train[target], mode='lines', name='Fitted', line=dict(color='red', dash='dash')))
    fig.add_trace(go.Scatter(x=forecasted_values_test.index, y=forecasted_values_test[target], mode='lines', name='Forecast', line=dict(color='pink')))
    fig.add_trace(go.Scatter(x=confidence_intervals_test.index, y=confidence_intervals_test['lower '+ target], fill=None, mode='lines', line=dict(color='green', width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=confidence_intervals_test.index, y=confidence_intervals_test['upper '+ target], fill='tonexty', mode='lines', line=dict(color='green', width=0), name='Confidence Interval'))
    
    fig.update_layout(title="VARMAX Model Forecast", xaxis_title="Date", yaxis_title="Lower CI, Upper CI, Value", legend_title="Series", template="plotly_dark")

    # Output in Streamlit
    st.plotly_chart(fig)
    st.markdown(f"**Training RMSE:** {rmse_train:.2f}, **MAE:** {mae_train:.2f}\n\n**Test RMSE:** {rmse_test:.2f}, **MAE:** {mae_test:.2f}")

# Slider per selezionare p e q
ar = st.slider("Seleziona AR (p)", min_value=0, max_value=5, value=2)
ma = st.slider("Seleziona MA (q)", min_value=0, max_value=5, value=2)

if st.button("Esegui analisi (VARMAX)"):
# Chiamata alla funzione ottimizzata
    varmax('Numero di insetti', 'Nuove catture (per evento)', ar, ma)
    varmax('Nuove catture (per evento)', 'Numero di insetti', ar, ma)

st.markdown("""<h3>🤗Ensamble</h3>""",unsafe_allow_html=True)

st.markdown("""
L'ensemble è una <b>combinazione di più modelli</b> di previsione che, lavorando insieme, cercano di 
migliorare le performance complessive. Si cerca di trovare i <b>punti di forza</b> di ogni modello, 
<b>riducendo/eliminando</b> gli errori individuali al fine di ottenere previsioni più robuste e accurate. 
L'approccio è utile quando i singoli modelli sono soggetti a <b>overfitting/underfitting</b>, combinando 
le previsioni, si possono attenuare gli errori. Sono stati usati i seguenti algoritmi:
- <b>Gradient Boosting Regression (GBM)</b>: attraverso tale algoritmo costruiamo iterativamente alberi 
decisionali, correggendo gli errori residui dei modelli precedenti. Facendo ciò, è possibile migliorare 
la previsione dando maggiore importanza agli errori che sono difficili da correggere.
- <b>Random Forest Regressor</b>: attraverso tale algoritmo si crea una <i>foresta</i> di alberi decisionali,
tutti forniranno una previsione. La previsione finale è la media delle singole previsioni, si <b>riduce la 
varianza</b> e <b>aumenta la stabilità e robustezza del modello</b>.
""",unsafe_allow_html=True)

st.markdown("""
Per entrambi i modelli sono stati usati <b>100 estimatori</b>, ovvero 100 alberi decisionali. Aumentare il
numero di alberi, garantirebbe una maggiore precisione, ma anche <b>overfitting</b>. Il <b>learning rate</b> 
è pari a <b>0.1</b>, ciò mi permette di dire che ogni albero contribuisce a un <b>moderato miglioramento 
delle previsioni</b>. La <i>max_depth</i> è pari a <b>3</b> in modo da non avere un modello troppo complesso 
e <b>facilitando la generalizzazione</b>.
""",unsafe_allow_html=True)

# E' stata usata la funzione fornita dalla professoressa adattata ai dati in nostro possesso
def create_lagged_features(df, n_lags=5, target_col='Numero di insetti'):
    """ Create lagged features for the target """
    # Create lagged features for the target column only
    for i in range(1, n_lags + 1):
        df[f'lag_{i}'] = df[target_col].shift(i)
    df.dropna(inplace=True) # Drop rows with NaN values
    return df

# E' stata usata la funzione fornita dalla professoressa adattata ai dati in nostro possesso
def train_and_visualize(df, n_lags, exog_cols, target_col):
    df = create_lagged_features(df, n_lags=n_lags, target_col=target_col)
    feature_cols = [f'lag_{i}' for i in range(1, n_lags + 1)] + exog_cols
    X = df[feature_cols]
    y = df[target_col]

    train_size = int(len(df) * 0.8)
    X_train, y_train = X.iloc[:train_size], y.iloc[:train_size]
    X_test, y_test = X.iloc[train_size:], y.iloc[train_size:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Gradient Boosting Machine": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=3)
    }

    st.markdown("<h3>"+target_col+"</h3>",unsafe_allow_html=True)

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)

        y_pred_train = np.clip(y_pred_train, 0, None).round()
        y_pred_test = np.clip(y_pred_test, 0, None).round()

        st.markdown("<b>" + str(name) + " - Training Data</b>",unsafe_allow_html=True)
        fig_train = go.Figure()
        fig_train.add_trace(go.Scatter(x=y_train.index, y=y_train, mode='lines+markers', name='Actual'))
        fig_train.add_trace(go.Scatter(x=y_train.index, y=y_pred_train, mode='lines+markers', name='Fitted'))
        st.plotly_chart(fig_train)

        st.markdown("<b>" +  str(name) + " - Test Data</b>",unsafe_allow_html=True)
        fig_test = go.Figure()
        fig_test.add_trace(go.Scatter(x=y_test.index, y=y_test, mode='lines+markers', name='Actual'))
        fig_test.add_trace(go.Scatter(x=y_test.index, y=y_pred_test, mode='lines+markers', name='Forecasted'))
                
        st.plotly_chart(fig_test)
    
    # Feature Importances
    for name, model in models.items():
        feature_importances = model.feature_importances_
        st.markdown("<b>"+ str(name) + " - Feature Importances</b>",unsafe_allow_html=True)
        fig_importance = go.Figure()
        fig_importance.add_trace(go.Bar(y=feature_cols, x=feature_importances, orientation='h'))
        st.plotly_chart(fig_importance)
    
    return models, df

# Slider per selezionare il numero di lag
n_lags = st.slider("Seleziona il numero di lag", min_value=1, max_value=10, value=3, key="n_lags")

# Colonne esogene
exog_cols = ['Media Temperatura', 'Media Umidità']

# Chiamata della funzione
if st.button("Esegui analisi (ENSEMBLE)"):
    models, real_data = train_and_visualize(merged_df, n_lags=4, exog_cols=exog_cols, target_col='Numero di insetti')
    models, real_data = train_and_visualize(merged_df, n_lags=4, exog_cols=exog_cols, target_col='Nuove catture (per evento)')

st.markdown("""
Il modello di <b>Multi-Layer Perceptron (MLP)</b> è una <i>rete neurale</i>. Esso usa come input un insieme 
di variabili <b>laggate ed esogene</b>. Le variabili di lag sono i valori passati della variabile target, 
le variabili esogene contengono informazioni aggiuntive come la media della temperatura e dell'umidità. 
Sono stati creati 3 lag della variabile target per visualizzare l'andamento temporale. <br><br>E' importante 
<b>normalizzare</b> i dati in modo da ottenere variabili con <b>media nulla e varianza unitaria</b>. Attraverso 
la normalizzazione nelle reti neurali è utile per aumentare la stabilità e l'efficienza del processo di 
apprendimento. Per la realizzazione del modello è stata usata l'<b>API di TensorFlow</b>. 
""",unsafe_allow_html=True)

st.markdown("""
La rete è composta da due <b>strati completamente connessi (dense layers)</b>:
- il <b>primo strato</b> ha <b>5 neuroni</b>, esso effettua una prima <b>trasformazione dei dati in ingresso</b>;
- il <b>secondo strato</b> ha <b>30 neuroni</b>, è usato per catturare <b>relazioni più complesse tra le variabili</b>.
""",unsafe_allow_html=True)

st.markdown("""
Per l'addestramento, è stata scelta la <b>Mean Squared Error</b> come funzione di perdita. Essa misura la differenza tra
i valori reali e predetti dal modello; tale metrica deve poter essere minimizzata in fase di training. Come ottimizzatore 
si è scelto <b>Adam</b> dati i suoi risultati nella convergenza delle reti neurali.
""",unsafe_allow_html=True)

st.markdown("""
Il modello è stato addestrato per <b>100 epoche</b> con un <b>batch size</b> di <b>32</b>. Le epoche sono il numero di 
passaggi completi sui dati di training, mentre il batch size ci permette di valutare il numero di osservazioni processate
alla volta. Si sceglie questa configurazione per bilanciare efficienza computazionale e capacità di apprendimento del modello.
""",unsafe_allow_html=True)

# E' stata usata la funzione fornita dalla professoressa adattata ai dati in nostro possesso
def create_lagged_features(df, n_lags=n_lags, target_col='Nuove catture (per evento)', exog_cols=[]):
    lagged_df = df.copy()
    for i in range(1, n_lags + 1):
        lagged_df[f'lag_{i}'] = lagged_df[target_col].shift(i)
    lagged_df = lagged_df.dropna()
    return lagged_df

# E' stata usata la funzione fornita dalla professoressa adattata ai dati in nostro possesso
def mlp(n,target):
    st.markdown("<b>MLP - "+target+"</b>",unsafe_allow_html=True)
    n_lags = n  # Number of lags
    exog_cols = ['Media Temperatura', 'Media Umidità']

    # Create lagged features
    lagged_df = create_lagged_features(merged_df, n_lags=n_lags, target_col=target, exog_cols=exog_cols)
    feature_cols = [f'lag_{i}' for i in range(1, n_lags + 1)] + exog_cols
    X = lagged_df[feature_cols]
    y = lagged_df[target]

    # Splitting Data
    train_size = len(lagged_df) - 20
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

    # Scaling Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define MLP Model
    inputs = tf.keras.Input(shape=(X_train_scaled.shape[1],))
    x = tf.keras.layers.Dense(5, activation='relu')(inputs)
    x = tf.keras.layers.Dense(30, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1)(x)
    model_mlp = tf.keras.Model(inputs=inputs, outputs=outputs)

    model_mlp.compile(optimizer='adam', loss='mean_squared_error')

    # Train Model
    history = model_mlp.fit(
        X_train_scaled, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_test_scaled, y_test),
        verbose=0
    )

    # Predictions
    y_train_pred = model_mlp.predict(X_train_scaled).flatten()
    y_test_pred = model_mlp.predict(X_test_scaled).flatten()

    # Post-processing predictions
    y_train_pred = np.clip(y_train_pred, 0, None).round()
    y_test_pred = np.clip(y_test_pred, 0, None).round()

    # Calculate Metrics
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    mae_train = mean_absolute_error(y_train, y_train_pred)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
    mae_test = mean_absolute_error(y_test, y_test_pred)

    # Plot Results
    fig_train = go.Figure()
    fig_train.add_trace(go.Scatter(x=y_train.index, y=y_train, mode='lines+markers', name='Reale', line=dict(color='blue')))
    fig_train.add_trace(go.Scatter(x=y_train.index, y=y_train_pred, mode='lines+markers', name='Previsto', line=dict(color='green', dash='dash')))
    fig_train.update_layout(title=f'Train Data: Reale vs Previsto (RMSE: {rmse_train:.2f}, MAE: {mae_train:.2f})')
    st.plotly_chart(fig_train)

    fig_test = go.Figure()
    fig_test.add_trace(go.Scatter(x=y_test.index, y=y_test, mode='lines+markers', name='Reale', line=dict(color='blue')))
    fig_test.add_trace(go.Scatter(x=y_test.index, y=y_test_pred, mode='lines+markers', name='Previsto', line=dict(color='green', dash='dash')))
    fig_test.update_layout(title=f'Test Data: Reale vs Previsto (RMSE: {rmse_test:.2f}, MAE: {mae_test:.2f})')
    st.plotly_chart(fig_test)

    st.write("Training and Validation Loss")
    fig_loss = go.Figure()
    fig_loss.add_trace(go.Scatter(y=history.history['loss'], mode='lines', name='Training Loss'))
    fig_loss.add_trace(go.Scatter(y=history.history['val_loss'], mode='lines', name='Validation Loss'))
    fig_loss.update_layout(title='Model Loss Over Epochs', xaxis_title='Epochs', yaxis_title='Loss')
    st.plotly_chart(fig_loss)

# Slider per la selezione del numero di lags
n_lags = st.slider("Seleziona il numero di lag", min_value=1, max_value=10, value=4, key="n_lags_mlp")

# Invocazione della funzione
if st.button("Esegui analisi (MLP)"):
    mlp(n_lags,'Numero di insetti')
    mlp(n_lags,'Nuove catture (per evento)')

st.markdown("""<h3>⚔️ Confronto tra modelli</h3""",unsafe_allow_html=True)

st.markdown("""
Sono stati usati dei grafici a barre per il confronto dei diversi modelli: <b>ARMAX, VARMAX, GBM, Random Forest ,MPL, LSTM</b>
considerando come variabili target sia il <i>Numero di insetti</i> che le <i>Nuove catture (per evento)</i>. Sono state 
confrontate le metriche di <b>RMSE</b> e <b>MAE</b> sia per i dati di training che test. Attraverso i grafici è stato possibile
valutare l'<b>accuracy</b> di ciascun modello, visualizzando le <b>differenze di performance tra i modelli</b>.
""",unsafe_allow_html=True)

# E' stata usata la funzione fornita dalla professoressa adattata ai dati in nostro possesso
def final_analisys(train_rmse,test_rmse,train_mae,test_mae,target):
    models = ["ARMAX", "VARMAX", "Random Forest", "GBM", "MLP", "LSTM"]

    # Define metrics
    train_rmse = train_rmse  
    test_rmse = test_rmse

    train_mae = train_mae
    test_mae = test_mae

    st.markdown("<b>Model Performance Comparison -" + target + "</b>",unsafe_allow_html=True)

    # Create RMSE plot
    fig_rmse = go.Figure()
    fig_rmse.add_trace(go.Bar(x=models, y=train_rmse, name='Train RMSE', marker_color='skyblue'))
    fig_rmse.add_trace(go.Bar(x=models, y=test_rmse, name='Test RMSE', marker_color='salmon'))
    fig_rmse.update_layout(title='RMSE for Train and Test Data by Model', xaxis_title='Models', yaxis_title='RMSE', barmode='group')
    st.plotly_chart(fig_rmse)

    # Create MAE plot
    fig_mae = go.Figure()
    fig_mae.add_trace(go.Bar(x=models, y=train_mae, name='Train MAE', marker_color='lightgreen'))
    fig_mae.add_trace(go.Bar(x=models, y=test_mae, name='Test MAE', marker_color='coral'))
    fig_mae.update_layout(title='MAE for Train and Test Data by Model', xaxis_title='Models', yaxis_title='MAE', barmode='group')
    st.plotly_chart(fig_mae)

train_rmse_ins = [0.39, 0.22, 0.53, 0.00, 1.15, 0.68]
test_rmse_ins = [1.48, 0.55, 0.45, 0.63, 1.16, 2.19]
train_mae_ins = [0.15, 0.05, 0.28, 0.00, 0.78, 0.34]
test_mae_ins = [1.40, 0.30, 0.20, 0.40, 0.75, 2.00]

train_rmse_catt = [0.32, 0.32, 0.41, 0.00, 0.71, 0.56]  
test_rmse_catt = [0.45, 0.00, 0.32, 0.45, 1.07, 0.89]
train_mae_catt = [0.10, 0.10, 0.41, 0.00, 0.39, 0.25]
test_mae_catt = [0.20, 0.00, 0.10, 0.20, 0.85, 0.80]

# Invocazione della funzione
if st.button("Confronta modelli"):
    final_analisys(train_rmse_ins,test_rmse_ins,train_mae_ins,test_mae_ins,"Numero di insetti")
    final_analisys(train_rmse_catt,test_rmse_catt,train_mae_catt,test_mae_catt,"Nuove catture (per evento)")