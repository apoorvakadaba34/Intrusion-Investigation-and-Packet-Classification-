import os
import subprocess
import pandas as pd
from numpy import matlib
from COA import COA
from Global_Vars import Global_Vars
from MOA import MOA
from Model_CNN import Model_CNN
from Model_LSTM import Model_LSTM
from Model_Res_LSTM import Model_Res_LSTM
from Model_TCN import Model_TCN
from Model_TCN_ResLSTM import Model_TCN_ResLSTM
from NGO import NGO
from Plot_Results import *
from Proposed import Proposed
from RKOA import RKOA
from objfun_feat import objfun
from scapy.all import rdpcap


def run_zeek(pcap_path, zeek_output_dir="zeek_output"):
    """Run Zeek on the provided pcap file."""
    os.makedirs(zeek_output_dir, exist_ok=True)

    try:
        subprocess.run(
            ["zeek", "-C", "-r", pcap_path],
            cwd=zeek_output_dir,
            check=True
        )
        print(f"[+] Zeek analysis completed. Logs in: {zeek_output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"[-] Zeek failed: {e}")


def extract_flow_features(zeek_output_dir="zeek_output"):
    conn_log_path = os.path.join(zeek_output_dir, "conn.log")

    if not os.path.exists(conn_log_path):
        return []

    flow_features = []
    with open(conn_log_path, 'r') as f:
        headers = f.readline().strip().split("\t")  # Read headers to map columns
        for line in f:
            fields = line.strip().split("\t")
            flow_dict = {headers[i]: fields[i] for i in range(len(headers))}
            flow_features.append(flow_dict)
    return flow_features


def store_in_db(flow_features):
    flow_meter_db = []
    for flow in flow_features:
        duration = float(flow.get("duration", 0))
        orig_bytes = int(flow.get("orig_bytes", 0))
        resp_bytes = int(flow.get("resp_bytes", 0))
        fwd_pkts = int(flow.get("orig_pkts", 0))
        bwd_pkts = int(flow.get("resp_pkts", 0))

        fwd_pkts_per_sec = fwd_pkts / duration if duration > 0 else 0
        bwd_pkts_per_sec = bwd_pkts / duration if duration > 0 else 0

        flow_meter_db.append({
            "uid": flow.get("uid"),
            "originh": flow.get("id.orig_h"),
            "originp": flow.get("id.orig_p"),
            "responh": flow.get("id.resp_h"),
            "responp": flow.get("id.resp_p"),
            "flow_duration": duration,
            "fwd_pkts_tot": fwd_pkts,
            "bwd_pkts_tot": bwd_pkts,
            "fwd_pkts_per_sec": fwd_pkts_per_sec,
            "bwd_pkts_per_sec": bwd_pkts_per_sec,
            "flow_pkts_per_sec": (fwd_pkts + bwd_pkts) / duration if duration > 0 else 0,
            "payload_bytes_per_second": (orig_bytes + resp_bytes) / duration if duration > 0 else 0
        })
    return flow_meter_db


def export_to_csv(flow_meter_db, output_csv="HIKARI-2021.csv"):
    df = pd.DataFrame(flow_meter_db)
    df.to_csv(output_csv, index=False)


# Read Pcap Files
an = 0
if an == 1:
    path = './archive/anonymized/full_anonymized'
    in_dir = os.listdir(path)
    for i in range(len(in_dir)):
        fileName = path + '/' + in_dir[i]
        pcap_file = rdpcap(fileName)
        zeek_output_dir = "zeek_output"
        output_csv = "HIKARI-2021.csv"
        run_zeek(pcap_file, zeek_output_dir)
        flow_features = extract_flow_features(zeek_output_dir)
        flow_meter_db = store_in_db(flow_features)
        export_to_csv(flow_meter_db, output_csv)

# Read Investigation Data
an = 0
if an == 1:
    dataFrame = pd.read_csv('ALLFLOWMETER_HIKARI2021.csv')
    data_arr = np.asarray(dataFrame)
    zeroind = np.where(data_arr[:, -1] == 0)
    data_0 = data_arr[zeroind[0][:3000], :-2]
    tar_0 = data_arr[zeroind[0][:3000], -1]
    oneind = np.where(data_arr[:, -1] == 1)
    data_1 = data_arr[oneind[0][:3000], :-2]
    tar_1 = data_arr[oneind[0][:3000], -1]
    Data = np.append(data_0, data_1, axis=0)
    for i in range(Data.shape[1]):
        if Data[:, i][0] == str(Data[:, i][0]):
            uniq = np.unique(Data[:, i])
            value = Data[:, i]
            final_data = np.zeros((value.shape[0]))  # create within rage zero values
            for uni in range(len(uniq)):
                index = np.where(value == uniq[uni])
                final_data[index[0]] = uni + 1
            Data[:, i] = final_data

    Target = np.append(tar_0, tar_1, axis=0)
    Target = np.reshape(Target, (-1, 1))
    index = np.arange(len(Data))
    np.random.shuffle(index)
    Org_Img = np.asarray(Data)
    Shuffled_Datas = Org_Img[index]
    Shuffled_Target = Target[index]
    np.save('Index.npy', index)
    np.save('Investigation_Data.npy', Shuffled_Datas)
    np.save('Investigation_Target.npy', Shuffled_Target)

# Read Packet Data
an = 0
if an == 1:
    dataFrame = pd.read_csv('ALLFLOWMETER_HIKARI2021.csv')
    Data = dataFrame.values[:, :-2]
    for i in range(Data.shape[1]):
        if Data[:, i][0] == str(Data[:, i][0]):
            uniq = np.unique(Data[:, i])
            value = Data[:, i]
            final_data = np.zeros((value.shape[0]))  # create within rage zero values
            for uni in range(len(uniq)):
                index = np.where(value == uniq[uni])
                final_data[index[0]] = uni + 1
            Data[:, i] = final_data

    Target = dataFrame.values[:, -2]
    df = pd.DataFrame(Target)  # unique code
    uniq = df[0].unique()
    Tar = np.asarray(df[0])
    target = np.zeros((Tar.shape[0], len(uniq)))  # create within rage zero values
    for uni in range(len(uniq)):
        index = np.where(Tar == uniq[uni])
        target[index[0], uni] = 1

    index = np.arange(len(Data))
    np.random.shuffle(index)
    Org_Img = np.asarray(Data)
    Shuffled_Datas = Org_Img[index]
    Shuffled_Target = target[index]
    np.save('Packet_Data.npy', Shuffled_Datas)
    np.save('Packet_Target.npy', Shuffled_Target)

# Optimization for Feature Selection (Investigation Data)
an = 0
if an == 1:
    data = np.load('Investigation_Data.npy', allow_pickle=True)  # Load the Dataset
    Target = np.load('Investigation_Target.npy', allow_pickle=True)  # Load the Target
    Best_sol = []
    Global_Vars.Data = data
    Global_Vars.Target = Target
    Npop = 10
    Chlen = 25
    xmin = matlib.repmat((0 * np.ones((1, Chlen))), Npop, 1)
    xmax = matlib.repmat(((data.shape[1] - 1) * np.ones((1, Chlen))), Npop, 1)
    initsol = np.zeros(xmin.shape)
    for i in range(xmin.shape[0]):
        for j in range(xmin.shape[1]):
            initsol[i, j] = np.random.uniform(xmin[i, j], xmax[i, j])
    fname = objfun
    max_iter = 50

    print('COA....')
    [bestfit1, fitness1, bestsol1, Time1] = COA(initsol, fname, xmin, xmax, max_iter)  # COA

    print('RKOA....')
    [bestfit2, fitness2, bestsol2, Time2] = RKOA(initsol, fname, xmin, xmax, max_iter)  # RKOA

    print('NGO....')
    [bestfit3, fitness3, bestsol3, Time3] = NGO(initsol, fname, xmin, xmax, max_iter)  # NGO

    print('MOA....')
    [bestfit4, fitness4, bestsol4, Time4] = MOA(initsol, fname, xmin, xmax, max_iter)  # MOA

    print('PROPOSED....')
    [bestfit5, fitness5, bestsol5, Time5] = Proposed(initsol, fname, xmin, xmax, max_iter)  # Proposed

    BestSol = [bestsol1, bestsol2, bestsol3, bestsol4, bestsol5]
    np.save('Best_Sol.npy', BestSol)

# Feature Selection (Investigation Data)
an = 0
if an == 1:
    Data = np.load('Investigation_Data.npy', allow_pickle=True)
    best = np.load('Best_Sol.npy', allow_pickle=True)
    Feature = []
    for n in range(best.shape[0]):
        sol = np.round(best[n, :]).astype(np.int16)
        Feat = Data[:, sol]
        Feature.append(Feat)
    np.save('Selected_Feature.npy', Feature)

# Classification (Investigation Data)
an = 0
if an == 1:
    Feature = np.load('Selected_Feature.npy', allow_pickle=True)  # loading step
    Target = np.load('Investigation_Target.npy', allow_pickle=True)  # loading step
    Feat = Feature[4, :]
    EVAL = []
    Epochs = [20, 40, 60, 80, 100]
    for learn in range(len(Epochs)):
        learnperc = round(Feat.shape[0] * 0.75)
        Train_Data = Feat[:learnperc, :]
        Train_Target = Target[:learnperc, :]
        Test_Data = Feat[learnperc:, :]
        Test_Target = Target[learnperc:, :]
        Eval = np.zeros((5, 25))
        Eval[0, :], pred = Model_LSTM(Train_Data, Train_Target, Test_Data,
                                      Test_Target, Epochs[learn])  # Model LSTM
        Eval[1, :], pred1 = Model_CNN(Train_Data, Train_Target, Test_Data, Test_Target, Epochs[learn])  # Model CNN
        Eval[2, :], pred2 = Model_TCN(Train_Data, Train_Target, Test_Data, Test_Target, Epochs[learn])  # Model TCN
        Eval[3, :], pred3 = Model_Res_LSTM(Train_Data, Train_Target, Test_Data, Test_Target, Epochs[learn])  # Model ResLSTM
        Eval[4, :], pred4 = Model_TCN_ResLSTM(Feat, Target, Epochs[learn])  # TCN + ResLSTM
        EVAL.append(Eval)
    np.save('Evaluate_all.npy', EVAL)  # Save Eval

# Optimization for Feature Selection (Packet Data)
an = 0
if an == 1:
    data = np.load('Packet_Data.npy', allow_pickle=True)  # Load the Data
    Target = np.load('Packet_Target.npy', allow_pickle=True)  # Load the Target
    Best_sol = []
    Global_Vars.Data = data
    Global_Vars.Target = Target
    Npop = 10
    Chlen = 25
    xmin = matlib.repmat((0 * np.ones((1, Chlen))), Npop, 1)
    xmax = matlib.repmat(((data.shape[1] - 1) * np.ones((1, Chlen))), Npop, 1)
    initsol = np.zeros(xmin.shape)
    for i in range(xmin.shape[0]):
        for j in range(xmin.shape[1]):
            initsol[i, j] = np.random.uniform(xmin[i, j], xmax[i, j])
    fname = objfun
    max_iter = 50

    print('COA....')
    [bestfit1, fitness1, bestsol1, Time1] = COA(initsol, fname, xmin, xmax, max_iter)  # COA

    print('RKOA....')
    [bestfit2, fitness2, bestsol2, Time2] = RKOA(initsol, fname, xmin, xmax, max_iter)  # RKOA

    print('NGO....')
    [bestfit3, fitness3, bestsol3, Time3] = NGO(initsol, fname, xmin, xmax, max_iter)  # NGO

    print('MOA....')
    [bestfit4, fitness4, bestsol4, Time4] = MOA(initsol, fname, xmin, xmax, max_iter)  # MOA

    print('PROPOSED....')
    [bestfit5, fitness5, bestsol5, Time5] = Proposed(initsol, fname, xmin, xmax, max_iter)  # Proposed

    BestSol = [bestsol1, bestsol2, bestsol3, bestsol4, bestsol5]
    np.save('Sol.npy', BestSol)

# Feature Selection (Packet Data)
an = 0
if an == 1:
    Data = np.load('Packet_Data.npy', allow_pickle=True)
    best = np.load('Sol.npy', allow_pickle=True)
    Feature = []
    for n in range(best.shape[0]):
        sol = np.round(best[n, :]).astype(np.int16)
        Feat = Data[:, sol]
        Feature.append(Feat)
    np.save('Packet_Selected_Feature.npy', Feature)

# Classification (Packet Data)
an = 0
if an == 1:
    Feature = np.load('Packet_Selected_Feature.npy', allow_pickle=True)  # loading step
    Target = np.load('Packet_Target.npy', allow_pickle=True)  # loading step
    Feat = Feature[4, :]
    EVAL = []
    Learning_Rate = [0.11, 0.22, 0.33, 0.44, 0.55]
    for learn in range(len(Learning_Rate)):
        learnperc = round(Feat.shape[0] * 0.75)
        Train_Data = Feat[:learnperc, :]
        Train_Target = Target[:learnperc, :]
        Test_Data = Feat[learnperc:, :]
        Test_Target = Target[learnperc:, :]
        Eval = np.zeros((5, 25))
        Eval[0, :], pred = Model_LSTM(Train_Data, Train_Target, Test_Data,
                                      Test_Target, Learning_Rate[learn])  # Model LSTM
        Eval[1, :], pred1 = Model_CNN(Train_Data, Train_Target, Test_Data, Test_Target, Learning_Rate[learn])  # Model CNN
        Eval[2, :], pred2 = Model_TCN(Train_Data, Train_Target, Test_Data, Test_Target, Learning_Rate[learn])  # Model TCN
        Eval[3, :], pred3 = Model_Res_LSTM(Train_Data, Train_Target, Test_Data, Test_Target, Learning_Rate[learn])  # Model ResLSTM
        Eval[4, :], pred4 = Model_TCN_ResLSTM(Feat, Target, Learning_Rate[learn])  # TCN + ResLSTM
        EVAL.append(Eval)
    np.save('Eval_all.npy', EVAL)  # Save Eval

# Classification for Steps_per_epoch Variation
an = 0
if an == 1:
    Feature = np.load('Packet_Selected_Feature.npy', allow_pickle=True)  # loading step
    Target = np.load('Packet_Target.npy', allow_pickle=True)  # loading step
    Feat = Feature[4, :]
    EVAL = []
    steps_per_epoch = [100, 200, 300, 400, 500]
    for learn in range(len(steps_per_epoch)):
        learnperc = round(Feat.shape[0] * 0.75)
        Train_Data = Feat[:learnperc, :]
        Train_Target = Target[:learnperc, :]
        Test_Data = Feat[learnperc:, :]
        Test_Target = Target[learnperc:, :]
        Eval = np.zeros((5, 25))
        Eval[0, :], pred = Model_LSTM(Train_Data, Train_Target, Test_Data,
                                      Test_Target, steps_per_epoch[learn])  # Model LSTM
        Eval[1, :], pred1 = Model_CNN(Train_Data, Train_Target, Test_Data, Test_Target, steps_per_epoch[learn])  # Model CNN
        Eval[2, :], pred2 = Model_TCN(Train_Data, Train_Target, Test_Data, Test_Target, steps_per_epoch[learn])  # Model TCN
        Eval[3, :], pred3 = Model_Res_LSTM(Train_Data, Train_Target, Test_Data, Test_Target, steps_per_epoch[learn])  # Model ResLSTM
        Eval[4, :], pred4 = Model_TCN_ResLSTM(Feat, Target, steps_per_epoch[learn])  # TCN + ResLSTM
        EVAL.append(Eval)
    np.save('Packet_Evaluate.npy', EVAL)  # Save Eval


plotConvResults()
Plots_Results()
Plot_ROC_Curve()
Packet_PlotResults()
Packet_ROCCurve()
Table()
Proposed_PlotResults()
