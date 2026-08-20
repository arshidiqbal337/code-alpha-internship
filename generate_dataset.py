import pandas as pd
import numpy as np

def generate_credit_dataset(num_samples=1000, random_seed=42):
    """
    Generates a synthetic dataset for credit scoring.
    Features include income, debt, payment history, savings, etc.
    """
    np.random.seed(random_seed)
    
    # 1. Base Demographic & Financial Attributes
    age = np.random.randint(21, 68, size=num_samples)
    income = np.round(np.random.normal(65000, 25000, size=num_samples), 2)
    income = np.clip(income, 18000, 250000) # Enforce realistic bounds
    
    debt = np.round(np.random.exponential(20000, size=num_samples), 2)
    debt = np.clip(debt, 500, 150000)
    
    credit_limit = np.round(income * np.random.uniform(0.3, 1.2, size=num_samples), 2)
    credit_limit = np.clip(credit_limit, 2000, 100000)
    
    savings_balance = np.round(np.random.exponential(12000, size=num_samples), 2)
    
    delinquencies = np.random.choice([0, 1, 2, 3, 4, 5], size=num_samples, p=[0.65, 0.18, 0.09, 0.04, 0.02, 0.02])
    credit_inquiries = np.random.choice([0, 1, 2, 3, 4, 5], size=num_samples, p=[0.50, 0.25, 0.13, 0.07, 0.03, 0.02])
    credit_age_years = np.random.randint(1, 35, size=num_samples)
    num_open_accounts = np.random.randint(1, 12, size=num_samples)
    
    # 2. Derive Default Probability based on financial rules
    dti_ratio = debt / income
    utilization = np.minimum(debt / credit_limit, 1.5)
    
    # Risk Score baseline formula
    risk_score = (
        0.35 * (delinquencies / 5.0) +
        0.25 * np.clip(dti_ratio / 1.5, 0, 1) +
        0.20 * np.clip(utilization / 1.0, 0, 1) +
        0.10 * (credit_inquiries / 5.0) -
        0.10 * (credit_age_years / 35.0)
    )
    
    # Add random noise for realism
    risk_prob = 1 / (1 + np.exp(-(risk_score * 6 - 2.5) + np.random.normal(0, 0.5, num_samples)))
    
    # Target: 1 = High Risk / Default, 0 = Low Risk / Good Credit
    credit_risk = (risk_prob > 0.40).astype(int)
    
    df = pd.DataFrame({
        'Age': age,
        'Income': income,
        'Debt': debt,
        'Credit_Limit': credit_limit,
        'Savings_Balance': savings_balance,
        'Delinquencies': delinquencies,
        'Credit_Inquiries': credit_inquiries,
        'Credit_Age_Years': credit_age_years,
        'Num_Open_Accounts': num_open_accounts,
        'Credit_Risk': credit_risk
    })
    
    return df

if __name__ == '__main__':
    df = generate_credit_dataset(1500)
    df.to_csv('credit_data.csv', index=False)
    print("Dataset generated successfully and saved as 'credit_data.csv'.")
    print(df.head())
    print("\nTarget Distribution (0 = Good, 1 = Bad/High Risk):")
    print(df['Credit_Risk'].value_counts())
