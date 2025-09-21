import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# HÀM XỬ LÝ CHO TỪNG BẢNG DỮ LIỆU
def process_bureau_and_balance(bureau, bureau_balance):
    print("Xử lý bureau và bureau_balance...")
    # Tổng hợp bureau_balance
    bb_agg = bureau_balance.groupby('SK_ID_BUREAU')['MONTHS_BALANCE'].agg(['count']).reset_index()
    bb_agg.columns = ['SK_ID_BUREAU', 'MONTHS_COUNT']
    bureau = bureau.merge(bb_agg, on='SK_ID_BUREAU', how='left')

    # Tổng hợp bureau
    bureau_agg = bureau.groupby('SK_ID_CURR').agg({
        'DAYS_CREDIT': ['mean', 'max', 'min', 'var'],
        'CREDIT_DAY_OVERDUE': ['mean', 'max'],
        'AMT_CREDIT_SUM': ['mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['mean', 'sum'],
    }).reset_index()
    bureau_agg.columns = pd.Index(['SK_ID_CURR'] + ['BUREAU_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()[1:]])

    # Tổng hợp các cột phân loại
    bureau_cat_agg = pd.get_dummies(bureau, columns=['CREDIT_ACTIVE', 'CREDIT_TYPE'], dummy_na=True)
    bureau_cat_agg = bureau_cat_agg.groupby('SK_ID_CURR').agg('sum').reset_index()
    dummy_cols = [col for col in bureau_cat_agg.columns if 'CREDIT_ACTIVE_' in col or 'CREDIT_TYPE_' in col]
    return bureau_agg.merge(bureau_cat_agg[['SK_ID_CURR'] + dummy_cols], on='SK_ID_CURR', how='left')

def process_previous_app(previous_app):
    print("Xử lý previous_application...")
    previous_app[['DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE', 'DAYS_LAST_DUE_1ST_VERSION', 'DAYS_LAST_DUE', 'DAYS_TERMINATION']] = \
        previous_app[['DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE', 'DAYS_LAST_DUE_1ST_VERSION', 'DAYS_LAST_DUE', 'DAYS_TERMINATION']].replace(365243, np.nan)
    previous_app['APP_CREDIT_PERCENT'] = previous_app['AMT_APPLICATION'] / (previous_app['AMT_CREDIT'] + 1e-6)

    prev_app_agg = previous_app.groupby('SK_ID_CURR').agg({
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'APP_CREDIT_PERCENT': ['min', 'max', 'mean', 'var'],
        'CNT_PAYMENT': ['mean', 'sum'],
    }).reset_index()
    prev_app_agg.columns = pd.Index(['SK_ID_CURR'] + ['PREV_APP_' + e[0] + "_" + e[1].upper() for e in prev_app_agg.columns.tolist()[1:]])

    prev_app_cat_agg = pd.get_dummies(previous_app, columns=['NAME_CONTRACT_STATUS'], dummy_na=True)
    prev_app_cat_agg = prev_app_cat_agg.groupby('SK_ID_CURR').agg('sum').reset_index()
    dummy_cols = [col for col in prev_app_cat_agg.columns if 'NAME_CONTRACT_STATUS_' in col]
    return prev_app_agg.merge(prev_app_cat_agg[['SK_ID_CURR'] + dummy_cols], on='SK_ID_CURR', how='left')

def process_installments(installments):
    print("Xử lý installments_payments...")
    installments['PAYMENT_PERC'] = installments['AMT_PAYMENT'] / (installments['AMT_INSTALMENT'] + 1e-6)
    installments['PAYMENT_DIFF'] = installments['AMT_INSTALMENT'] - installments['AMT_PAYMENT']
    installments['DAYS_PAST_DUE'] = installments['DAYS_ENTRY_PAYMENT'] - installments['DAYS_INSTALMENT']
    installments['DAYS_BEFORE_DUE'] = installments['DAYS_INSTALMENT'] - installments['DAYS_ENTRY_PAYMENT']

    installments_agg = installments.groupby('SK_ID_CURR').agg({
        'PAYMENT_PERC': ['mean', 'max', 'var'],
        'PAYMENT_DIFF': ['mean', 'max', 'sum'],
        'DAYS_PAST_DUE': ['mean', 'max', 'sum'],
        'DAYS_BEFORE_DUE': ['mean', 'min'],
        'NUM_INSTALMENT_VERSION': ['nunique']
    }).reset_index()
    installments_agg.columns = pd.Index(['SK_ID_CURR'] + ['INSTALL_' + e[0] + "_" + e[1].upper() for e in installments_agg.columns.tolist()[1:]])
    return installments_agg

# FEATURE ENGINEERING
# Tải dữ liệu
print("Tải dữ liệu...")
app_train = pd.read_csv(r'C:\Users\DatGo\OneDrive\Documents\Personal_Project\Home_Credit_Default_Risk\application_train_cleaned.csv')
app_test = pd.read_csv(r'C:\Users\DatGo\OneDrive\Documents\Personal_Project\Home_Credit_Default_Risk\application_test_cleaned.csv')
bureau = pd.read_csv(r'C:\Users\DatGo\OneDrive\Documents\Personal_Project\Home_Credit_Default_Risk\bureau.csv')
bureau_balance = pd.read_csv(r'C:\Users\DatGo\OneDrive\Documents\Personal_Project\Home_Credit_Default_Risk\bureau_balance.csv')
previous_app = pd.read_csv(r'C:\Users\DatGo\OneDrive\Documents\Personal_Project\Home_Credit_Default_Risk\previous_application.csv')
installments = pd.read_csv(r'C:\Users\DatGo\OneDrive\Documents\Personal_Project\Home_Credit_Default_Risk\installments_payments.csv')

# Chạy các hàm xử lý
bureau_features = process_bureau_and_balance(bureau, bureau_balance)
prev_app_features = process_previous_app(previous_app)
installments_features = process_installments(installments)

# Gộp tất cả features lại
print("\nGộp tất cả các đặc trưng...")
df_train = app_train.merge(bureau_features, on='SK_ID_CURR', how='left')
df_train = df_train.merge(prev_app_features, on='SK_ID_CURR', how='left')
df_train = df_train.merge(installments_features, on='SK_ID_CURR', how='left')

df_test = app_test.merge(bureau_features, on='SK_ID_CURR', how='left')
df_test = df_test.merge(prev_app_features, on='SK_ID_CURR', how='left')
df_test = df_test.merge(installments_features, on='SK_ID_CURR', how='left')

print("Gộp dữ liệu hoàn tất!")
print(f"Kích thước cuối cùng của tập train: {df_train.shape}")
print(f"Kích thước cuối cùng của tập test: {df_test.shape}")

# LƯU KẾT QUẢ CUỐI CÙNG
print("\nLưu các tệp dữ liệu đã được làm giàu...")

df_train.to_csv(r'C:\Users\DatGo\OneDrive\Documents\Personal_Project\Home_Credit_Default_Risk\application_train_features.csv', index=False)
df_test.to_csv(r'C:\Users\DatGo\OneDrive\Documents\Personal_Project\Home_Credit_Default_Risk\application_test_features.csv', index=False)

print("\nHoàn tất! Dữ liệu đã được làm giàu và lưu vào các tệp:")
print("- application_train_features.csv")
print("- application_test_features.csv")