import pickle
import streamlit as st
import pandas as pd

st.header("Web Attack Detection Prediction")

# Load the trained model
model = pickle.load(open("xgb_model.pkl", 'rb'))

# Input fields
protocol = st.selectbox('Protocol', ['TCP', 'UDP', 'ICMP'])
flow_duration = st.number_input('Flow Duration', min_value=0.0)
total_fwd_pkts = st.number_input('Total Forward Packets', min_value=0.0)
total_bwd_pkts = st.number_input('Total Backward Packets', min_value=0.0)
totlen_fwd_pkts = st.number_input('Total Length of Forward Packets', min_value=0.0)
totlen_bwd_pkts = st.number_input('Total Length of Backward Packets', min_value=0.0)
fwd_pkt_len_max = st.number_input('Forward Packet Length Max', min_value=0.0)
fwd_pkt_len_min = st.number_input('Forward Packet Length Min', min_value=0.0)
fwd_pkt_len_mean = st.number_input('Forward Packet Length Mean', min_value=0.0)
fwd_pkt_len_std = st.number_input('Forward Packet Length Std', min_value=0.0)
bwd_pkt_len_max = st.number_input('Backward Packet Length Max', min_value=0.0)
bwd_pkt_len_min = st.number_input('Backward Packet Length Min', min_value=0.0)
bwd_pkt_len_mean = st.number_input('Backward Packet Length Mean', min_value=0.0)
bwd_pkt_len_std = st.number_input('Backward Packet Length Std', min_value=0.0)
flow_byts_per_sec = st.number_input('Flow Bytes per Second', min_value=0.0)
flow_pkts_per_sec = st.number_input('Flow Packets per Second', min_value=0.0)
flags = st.multiselect('Flags', ['FIN', 'SYN', 'RST', 'PSH', 'ACK', 'URG'])
down_up_ratio = st.number_input('Down/Up Ratio', min_value=0.0)
fwd_seg_size_avg = st.number_input('Forward Segment Size Avg', min_value=0.0)
bwd_seg_size_avg = st.number_input('Backward Segment Size Avg', min_value=0.0)
fwd_byts_per_b_avg = st.number_input('Forward Bytes per Byte Avg', min_value=0.0)
bwd_byts_per_b_avg = st.number_input('Backward Bytes per Byte Avg', min_value=0.0)
fwd_pks_per_b_avg = st.number_input('Forward Packets per Byte Avg', min_value=0.0)
bwd_pks_per_b_avg = st.number_input('Backward Packets per Byte Avg', min_value=0.0)
Timestap = st.number_input('Timestamp')

# Map protocol to numeric value
protocol_map = {'TCP': 1, 'UDP': 2, 'ICMP': 3}
protocol_num = protocol_map.get(protocol, 0)

# Create a DataFrame for the input data
input_data = pd.DataFrame({'Timestamp' : [Timestap]})

# Predict
if st.button("Predict"):
    # Ensure the input data matches the model's expected format
    prediction = model.predict(input_data)

    if prediction == 0:
        st.text("The model predicts that the traffic is not a web attack.")
    else:
        st.text("The model predicts that the traffic is a web attack.")
