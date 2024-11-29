import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Load data with updated caching
@st.cache_data
def load_data():
    data = pd.read_csv('mushrooms.csv')
    label = LabelEncoder()
    for col in data.columns:
        data[col] = label.fit_transform(data[col])
    return data

@st.cache_data
def split_data(df):
    y = df['type']
    x = df.drop(columns=['type'])
    return train_test_split(x, y, test_size=0.3, random_state=0)

def plot_metrics(metrics_list, model, x_test, y_test, class_names):
    if 'Confusion Matrix' in metrics_list:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, model.predict(x_test))
        ConfusionMatrixDisplay(cm, display_labels=class_names).plot()
        st.pyplot()

    if 'ROC Curve' in metrics_list:
        st.subheader("ROC Curve")
        RocCurveDisplay.from_estimator(model, x_test, y_test)
        st.pyplot()

    if 'Precision-Recall Curve' in metrics_list:
        st.subheader("Precision-Recall Curve")
        PrecisionRecallDisplay.from_estimator(model, x_test, y_test)
        st.pyplot()

def main():
    st.title("Binary Classification WebApp")
    st.markdown("Are your mushrooms edible or poisonous? 🍄")

    st.sidebar.title("Binary Classification")
    st.sidebar.markdown("Choose your classifier and parameters:")

    df = load_data()
    x_train, x_test, y_train, y_test = split_data(df)
    class_names = ['edible', 'poisonous']

    classifier = st.sidebar.selectbox("Classifier", ("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest"))

    if classifier == "Support Vector Machine (SVM)":
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01)
        kernel = st.sidebar.radio("Kernel", ("rbf", "linear"))
        gamma = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"))
        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Classify"):
            st.subheader("Support Vector Machine (SVM) Results")
            model = SVC(C=C, kernel=kernel, gamma=gamma)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy:", round(accuracy, 2))
            st.write("Precision:", round(precision_score(y_test, y_pred), 2))
            st.write("Recall:", round(recall_score(y_test, y_pred), 2))
            plot_metrics(metrics, model, x_test, y_test, class_names)

    elif classifier == "Logistic Regression":
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01)
        max_iter = st.sidebar.slider("Maximum number of iterations", 100, 500)
        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Classify"):
            st.subheader("Logistic Regression Results")
            model = LogisticRegression(C=C, max_iter=max_iter)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy:", round(accuracy, 2))
            st.write("Precision:", round(precision_score(y_test, y_pred), 2))
            st.write("Recall:", round(recall_score(y_test, y_pred), 2))
            plot_metrics(metrics, model, x_test, y_test, class_names)

    elif classifier == "Random Forest":
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators = st.sidebar.number_input("Number of trees", 100, 5000, step=10)
        max_depth = st.sidebar.number_input("Maximum depth of the tree", 1, 20, step=1)
        bootstrap = st.sidebar.radio("Bootstrap samples?", [True, False])
        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Classify"):
            st.subheader("Random Forest Results")
            model = RandomForestClassifier(n_estimators=int(n_estimators), max_depth=int(max_depth), bootstrap=bootstrap, n_jobs=-1)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy:", round(accuracy, 2))
            st.write("Precision:", round(precision_score(y_test, y_pred), 2))
            st.write("Recall:", round(recall_score(y_test, y_pred), 2))
            plot_metrics(metrics, model, x_test, y_test, class_names)

    if st.sidebar.checkbox("Show raw data"):
        st.subheader("Mushroom Data Set (Classification)")
        st.write(df)

if __name__ == '__main__':
    main()
