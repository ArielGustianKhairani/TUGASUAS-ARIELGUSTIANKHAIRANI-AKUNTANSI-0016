import pandas as pd

# Membaca data dari file CSV
data = pd.read_csv('data_penjualan_pancake_croffle.csv')

# Menampilkan 5 baris pertama data
print(data.head())
import matplotlib.pyplot as plt
import seaborn as sns

# Menghitung matriks korelasi
correlation_matrix = data[['jumlah_penjualan', 'harga_per_unit', 'total_pendapatan']].corr()

# Menampilkan matriks korelasi
print(correlation_matrix)

# Visualisasi matriks korelasi
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Matriks Korelasi')
plt.show()
# Mengubah variabel kategori 'jenis_produk' menjadi variabel dummy
data = pd.get_dummies(data, columns=['jenis_produk'], drop_first=True)

# Memilih fitur dan target
X = data[['harga_per_unit', 'jenis_produk_pancake']]
y = data['jumlah_penjualan']
from sklearn.model_selection import train_test_split

# Membagi data menjadi set pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Melatih model
model = LinearRegression()
model.fit(X_train, y_train)

# Memprediksi data pengujian
y_pred = model.predict(X_test)

# Mengevaluasi model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Melatih model
model = LinearRegression()
model.fit(X_train, y_train)

# Memprediksi data pengujian
y_pred = model.predict(X_test)

# Mengevaluasi model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')
# Menampilkan koefisien model
coef_df = pd.DataFrame(model.coef_, X.columns, columns=['Koefisien'])
print(coef_df)

# Plot hasil prediksi vs aktual
plt.scatter(y_test, y_pred)
plt.xlabel('Aktual')
plt.ylabel('Prediksi')
plt.title('Prediksi vs Aktual')
plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Membaca data dari file CSV
data = pd.read_csv('data_penjualan_pancake_croffle.csv')

# Menampilkan 5 baris pertama data
print(data.head())

# Menghitung matriks korelasi
correlation_matrix = data[['jumlah_penjualan', 'harga_per_unit', 'total_pendapatan']].corr()

# Menampilkan matriks korelasi
print(correlation_matrix)

# Visualisasi matriks korelasi
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Matriks Korelasi')
plt.show()

# Mengubah variabel kategori 'jenis_produk' menjadi variabel dummy
data = pd.get_dummies(data, columns=['jenis_produk'], drop_first=True)

# Memilih fitur dan target
X = data[['harga_per_unit', 'jenis_produk_pancake']]
y = data['jumlah_penjualan']

# Membagi data menjadi set pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Melatih model
model = LinearRegression()
model.fit(X_train, y_train)

# Memprediksi data pengujian
y_pred = model.predict(X_test)

# Mengevaluasi model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Menampilkan koefisien model
coef_df = pd.DataFrame(model.coef_, X.columns, columns=['Koefisien'])
print(coef_df)

# Plot hasil prediksi vs aktual
plt.scatter(y_test, y_pred)
plt.xlabel('Aktual')
plt.ylabel('Prediksi')
plt.title('Prediksi vs Aktual')
plt.show()

# Analisis tambahan
# Menampilkan statistik deskriptif untuk setiap jenis produk
print(data.groupby('jenis_produk_pancake').describe())

# Visualisasi distribusi jumlah penjualan berdasarkan jenis produk
plt.figure(figsize=(10, 6))
sns.boxplot(x='jenis_produk_pancake', y='jumlah_penjualan', data=data)
plt.title('Distribusi Jumlah Penjualan Berdasarkan Jenis Produk')
plt.xlabel('Jenis Produk (0: Croffle, 1: Pancake)')
plt.ylabel('Jumlah Penjualan')
plt.show()

# Visualisasi distribusi total pendapatan berdasarkan jenis produk
plt.figure(figsize=(10, 6))
sns.boxplot(x='jenis_produk_pancake', y='total_pendapatan', data=data)
plt.title('Distribusi Total Pendapatan Berdasarkan Jenis Produk')
plt.xlabel('Jenis Produk (0: Croffle, 1: Pancake)')
plt.ylabel('Total Pendapatan')
plt.show()

# Scatter plot harga per unit vs jumlah penjualan
plt.figure(figsize=(10, 6))
sns.scatterplot(x='harga_per_unit', y='jumlah_penjualan', hue='jenis_produk_pancake', data=data)
plt.title('Harga per Unit vs Jumlah Penjualan')
plt.xlabel('Harga per Unit')
plt.ylabel('Jumlah Penjualan')
plt.legend(title='Jenis Produk', labels=['Croffle', 'Pancake'])
plt.show()







