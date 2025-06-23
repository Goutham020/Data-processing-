## 📁 Dataset Used

- train_genetic_disorders.csv
- test_genetic_disorders.csv

---

## 🧪 Steps Performed

### 1. Import and Explore
- Loaded train and test CSV files
- Checked data types, null values, and basic structure

### 2. Handle Missing Values
- Filled missing numeric values with median
- Filled missing categorical values with mode

### 3. Encode Categorical Variables
- Applied one-hot encoding to all categorical features
- Ensured consistent encoding between train and test data

### 4. Normalize/Standardize Features
- Standardized numerical features using StandardScaler

### 5. Outlier Detection and Removal
- Visualized outliers using boxplots (Seaborn)
- Removed outliers using IQR method

---

## 🛠 Libraries Used

- Python 3
- Pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
