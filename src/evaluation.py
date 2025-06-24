from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import pandas as pd

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    correctly_classified = (comparison_df['Actual'] == comparison_df['Predicted']).sum()
    incorrectly_classified = (comparison_df['Actual'] != comparison_df['Predicted']).sum()

    plt.figure(figsize=(8, 8))
    plt.pie(
        [correctly_classified, incorrectly_classified],
        labels=['Correctly Classified', 'Misclassified'],
        autopct='%1.1f%%',
        startangle=140,
        colors=['#4CAF50', '#FF5733']
    )
    plt.title('Classification Accuracy')
    plt.show()
