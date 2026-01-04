import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI NIDS", layout="wide")
st.title("AI-Based Network Intrusion Detection System")

# 1. Create sample (fake) network data
np.random.seed(42)
data = {
    "Flow_Duration": np.random.randint(1, 10000, 1000),
    "Total_Packets": np.random.randint(1, 500, 1000),
    "Packet_Length_Mean": np.random.randint(50, 1500, 1000),
    "Label": np.random.choice([0, 1], 1000)  # 0 = Normal, 1 = Attack
}
df = pd.DataFrame(data)

# 2. Split data
X = df.drop("Label", axis=1)
y = df["Label"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 4. Test model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.subheader("Model Accuracy")
st.write(f"Accuracy: {accuracy * 100:.2f}%")

# 5. Confusion Matrix
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
st.pyplot(fig)

# 6. Live Test
st.subheader("Live Traffic Test")

fd = st.number_input("Flow Duration", 1, 10000, 100)
tp = st.number_input("Total Packets", 1, 500, 50)
pl = st.number_input("Packet Length Mean", 50, 1500, 500)

if st.button("Check Traffic"):
    result = model.predict([[fd, tp, pl]])
    if result[0] == 1:
        st.error("⚠ Malicious Traffic Detected")
    else:
        st.success("✔ Normal Traffic")


