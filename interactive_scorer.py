"""
Interactive Credit Risk & Score Assessor
Allows users to evaluate creditworthiness for custom applicants using the trained models.
"""

from generate_dataset import generate_credit_dataset
from credit_scoring_model import train_and_evaluate, score_applicant

def main():
    print("=" * 60)
    print("      INTERACTIVE CREDIT SCORING & RISK EVALUATOR")
    print("=" * 60)
    
    # 1. Prepare Model
    print("Training Credit Scoring models on historical financial data...")
    df = generate_credit_dataset(num_samples=1500)
    metrics_summary, trained_models, scaler, feature_names = train_and_evaluate(df)
    
    best_model_name = metrics_summary.iloc[0]['Model']
    best_model = trained_models[best_model_name]
    is_scaled = (best_model_name == 'Logistic Regression')
    
    print(f"\nModel Ready: Using {best_model_name} (ROC-AUC: {metrics_summary.iloc[0]['ROC-AUC']:.4f})")
    
    while True:
        print("\n" + "-" * 50)
        print("Choose an option:")
        print("1. Evaluate Preset Applicant (Low Risk / Good Credit)")
        print("2. Evaluate Preset Applicant (High Risk / Vulnerable)")
        print("3. Enter Custom Applicant Details")
        print("4. Exit")
        
        choice = input("Enter choice (1-4): ").strip()
        
        if choice == '1':
            applicant = {
                'Age': 42,
                'Income': 110000.0,
                'Debt': 12000.0,
                'Credit_Limit': 50000.0,
                'Savings_Balance': 45000.0,
                'Delinquencies': 0,
                'Credit_Inquiries': 0,
                'Credit_Age_Years': 15,
                'Num_Open_Accounts': 5
            }
        elif choice == '2':
            applicant = {
                'Age': 24,
                'Income': 24000.0,
                'Debt': 19000.0,
                'Credit_Limit': 8000.0,
                'Savings_Balance': 400.0,
                'Delinquencies': 4,
                'Credit_Inquiries': 5,
                'Credit_Age_Years': 1,
                'Num_Open_Accounts': 9
            }
        elif choice == '3':
            try:
                age = float(input("Age: "))
                income = float(input("Annual Income ($): "))
                debt = float(input("Total Current Debt ($): "))
                credit_limit = float(input("Total Credit Limit ($): "))
                savings = float(input("Savings Balance ($): "))
                delinquencies = float(input("Late/Missed Payments (past 2 yrs): "))
                inquiries = float(input("Credit Inquiries (past 6 mos): "))
                credit_age = float(input("Oldest Account Age (years): "))
                accounts = float(input("Number of Open Accounts: "))
                
                applicant = {
                    'Age': age,
                    'Income': income,
                    'Debt': debt,
                    'Credit_Limit': credit_limit,
                    'Savings_Balance': savings,
                    'Delinquencies': delinquencies,
                    'Credit_Inquiries': inquiries,
                    'Credit_Age_Years': credit_age,
                    'Num_Open_Accounts': accounts
                }
            except ValueError:
                print("Invalid input! Please enter numeric values.")
                continue
        elif choice == '4':
            print("Exiting Credit Assessor. Good luck!")
            break
        else:
            print("Invalid choice! Try again.")
            continue
            
        result = score_applicant(applicant, best_model, scaler, feature_names, is_scaled)
        
        print("\n" + "=" * 40)
        print("          CREDIT SCORING RESULT")
        print("=" * 40)
        for k, v in result.items():
            print(f"  {k:<24}: {v}")
        print("=" * 40)

if __name__ == '__main__':
    main()
