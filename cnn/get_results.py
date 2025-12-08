import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import argparse

def show_result(path, model_name="rgb_apps-all", label_class=3, bin_number=1, report=True):
    # read test result
    true_labels = np.load(f"{path}/{model_name}_label{label_class}_bin{bin_number}_true.npy")
    predictions = np.load(f"{path}/{model_name}_label{label_class}_bin{bin_number}_pred.npy")
    
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(true_labels, axis=1)  # one-hot
    
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    if report:
        report = classification_report(true_labels, predicted_labels)
        print("\nClassification Report:\n", report)
    return accuracy, precision, recall, f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", type=str, default=False, help="path of result")
    args = parser.parse_args()

    show_result(args.result_path)