import numpy as np
from matplotlib import pyplot as plt
from prettytable import PrettyTable
from itertools import cycle
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn import metrics
import warnings

warnings.filterwarnings('ignore')


def Statistical(data):
    Min = np.min(data)
    Max = np.max(data)
    Mean = np.mean(data)
    Median = np.median(data)
    Std = np.std(data)
    return np.asarray([Min, Max, Mean, Median, Std])


def plotConvResults():
    # matplotlib.use('TkAgg')
    Fitness = np.load('Fitness.npy', allow_pickle=True)[0]
    Algorithm = ['TERMS', 'COA', 'RKOA', 'NGO', 'MOA', 'MR-MOA']
    Terms = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
    Conv_Graph = np.zeros((len(Algorithm) - 1, len(Terms)))
    for j in range(len(Algorithm) - 1):  # for 5 algms
        Conv_Graph[j, :] = Statistical(Fitness[j, :])

    Table = PrettyTable()
    Table.add_column(Algorithm[0], Terms)
    for j in range(len(Algorithm) - 1):
        Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
    print('-------------------------------------------------- Statistical Analysis  ',
          '--------------------------------------------------')
    print(Table)

    length = np.arange(Fitness.shape[1])
    fig = plt.figure()
    fig.canvas.manager.set_window_title('Convergence Curve')
    plt.plot(length, Fitness[0, :], color='r', linewidth=3, marker='*', markerfacecolor='red',
             markersize=12, label=Algorithm[1])
    plt.plot(length, Fitness[1, :], color='g', linewidth=3, marker='*', markerfacecolor='green',
             markersize=12, label=Algorithm[2])
    plt.plot(length, Fitness[2, :], color='b', linewidth=3, marker='*', markerfacecolor='blue',
             markersize=12, label=Algorithm[3])
    plt.plot(length, Fitness[3, :], color='m', linewidth=3, marker='*', markerfacecolor='magenta',
             markersize=12, label=Algorithm[4])
    plt.plot(length, Fitness[4, :], color='k', linewidth=3, marker='*', markerfacecolor='black',
             markersize=12, label=Algorithm[5])
    plt.xlabel('No. of Iteration', fontname="Arial", fontsize=14, fontweight='bold', color='k')
    plt.ylabel('Cost Function', fontname="Arial", fontsize=14, fontweight='bold', color='k')
    plt.yticks(fontname="Arial", fontsize=15, fontweight='bold', color='k')
    plt.xticks(fontname="Arial", fontsize=15, fontweight='bold', color='k')
    plt.legend(loc=1, prop={'weight':'bold', 'size':12})
    plt.savefig("./Results/Conv.png")
    plt.show()


def Plot_ROC_Curve():
    cls = ['LSTM-DNN', 'CNN', 'CA-TCN', 'DResLSTM', 'HDeepNet']
    Actual = np.load('Investigation_Target.npy', allow_pickle=True).astype(np.int32)
    lenper = round(Actual.shape[0] * 0.75)
    Actual = Actual[lenper:, :]
    fig = plt.figure()
    fig.canvas.manager.set_window_title('ROC Curve')
    colors = cycle(["blue", "darkorange", "limegreen", "deeppink", "black"])
    for i, color in zip(range(len(cls)), colors):  # For all classifiers
        Predicted = np.load('Y_Score.npy', allow_pickle=True)[i]
        false_positive_rate, true_positive_rate, _ = roc_curve(Actual.ravel(), Predicted.ravel())
        roc_auc = roc_auc_score(Actual.ravel(), Predicted.ravel())
        roc_auc = roc_auc * 100

        plt.plot(
            false_positive_rate,
            true_positive_rate,
            color=color,
            lw=2,
            label=f'{cls[i]} (AUC = {roc_auc:.2f} %)')

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontname="Arial", fontsize=14, fontweight='bold', color='k')
    plt.ylabel("True Positive Rate", fontname="Arial", fontsize=14, fontweight='bold', color='k')
    plt.yticks(fontname="Arial", fontsize=14, fontweight='bold', color='k')
    plt.xticks(fontname="Arial", fontsize=14, fontweight='bold', color='k')
    plt.title("ROC Curve")
    plt.legend(loc="lower right", prop={'weight':'bold', 'size':12})
    path = "./Results/Investigation_ROC.png"
    plt.savefig(path)
    plt.show()


def Table():
    eval = np.load('Evaluates.npy', allow_pickle=True)
    Classifier = ['Activation_Function', 'LSTM-DNN', 'CNN', 'CA-TCN', 'DResLSTM', 'HDeepNet']
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1 Score',
             'MCC', 'FOR', 'PT', 'CSI', 'BA', 'FM', 'BM', 'MK', 'LR+', 'LR-', 'DOR', 'Prevalence']
    Graph_Terms = np.array([0, 2, 4, 6, 9, 15]).astype(int)
    Table_Terms = [0, 2, 4, 6, 9, 15]
    table_terms = [Terms[i] for i in Table_Terms]
    Act = ['Linear', 'Tanh', 'Relu', 'Softmax', 'Sigmoid']
    for k in range(len(Table_Terms)):
        value = eval[:, :, 4:]

        Table = PrettyTable()
        Table.add_column(Classifier[0], Act)
        for j in range(len(Classifier) - 1):
            Table.add_column(Classifier[j + 1], value[:, j, Graph_Terms[k]])
        print('-------------------------------------------', table_terms[k], '  Classifier Comparison',
              '-------------------------------------------')
        print(Table)


def Plots_Results():
    eval = np.load('Evaluate_all.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1 Score',
             'MCC', 'FOR', 'PT', 'CSI', 'BA', 'FM', 'BM', 'MK', 'LR+', 'LR-', 'DOR', 'Prevalence']
    Graph_Terms = [0, 1, 2, 5, 8, 12, 15]
    bar_width = 0.15
    Classifier = ['LSTM-DNN', 'CNN', 'CA-TCN', 'DResLSTM', 'HDeepNet']
    epochs = [20, 40, 60, 80, 100]
    for i in range(eval.shape[0]):
        for j in range(len(Graph_Terms)):
            Graph = np.zeros(eval.shape[1:3])
            for k in range(eval.shape[1]):
                for l in range(eval.shape[2]):
                    Graph[k, l] = eval[i, k, l, Graph_Terms[j] + 4]

            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_axes([0.12, 0.25, 0.8, 0.65])
            fig.canvas.manager.set_window_title('Method Comparison of No. of Epochs')
            X = np.arange(len(epochs))

            bars1 = ax.bar(X + 0.00, Graph[:, 0], color='m', edgecolor='w', width=0.15, label="LSTM-DNN")
            ax.bar_label(container=bars1, size=10, label_type='edge', labels=[f'{x:.0f}' for x in Graph[:, 0]],
                         fontweight='bold', padding=5)

            bars2 = ax.bar(X + 0.15, Graph[:, 1], color='y', edgecolor='w', width=0.15, label="CNN")
            ax.bar_label(container=bars2, size=10, label_type='edge', labels=[f'{x:.0f}' for x in Graph[:, 1]],
                         fontweight='bold', padding=5)

            bars3 = ax.bar(X + 0.30, Graph[:, 2], color='#9b5de5', edgecolor='w', width=0.15, label="CA-TCN")
            ax.bar_label(container=bars3, size=10, label_type='edge', labels=[f'{x:.0f}' for x in Graph[:, 2]],
                         fontweight='bold', padding=5)
            bars4 = ax.bar(X + 0.45, Graph[:, 3], color='#218380', edgecolor='w', width=0.15, label="DResLSTM")
            ax.bar_label(container=bars4, size=10, label_type='edge', labels=[f'{x:.0f}' for x in Graph[:, 3]],
                         fontweight='bold', padding=5)
            bars5 = ax.bar(X + 0.60, Graph[:, 4], color='#ef233c', edgecolor='w', width=0.15,
                           label="CA-TCN-DRLSTM")
            ax.bar_label(container=bars5, size=10, label_type='edge', labels=[f'{x:.0f}' for x in Graph[:, 4]],
                         fontweight='bold', padding=5)

            # Customizations
            plt.xticks(X + bar_width * 1.5, ['20', '40', '60', '80', '100'], fontname="Arial", fontsize=15,
                       fontweight='bold', color='k')
            plt.xlabel('No. of Epochs', fontname="Arial", fontsize=15, fontweight='bold', color='k')
            plt.ylabel(Terms[Graph_Terms[j]], fontname="Arial", fontsize=15, fontweight='bold', color='k')
            plt.yticks(fontname="Arial", fontsize=15, fontweight='bold', color='k')

            # Remove axes outline
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['left'].set_visible(False)
            plt.gca().spines['bottom'].set_visible(True)

            # # Adjusted custom circular legend placement (with space from figure edges)
            plt.figtext(0.04, 0.08, '01', fontsize=22, ha='center', color='white',
                        bbox=dict(boxstyle='circle', facecolor='m', edgecolor='white'))
            plt.figtext(0.069, 0.12, Classifier[0], fontsize=10, fontweight='bold', color='k', ha='left')
            plt.figtext(0.069, 0.08, 'Long Short-Term', fontsize=9, fontweight='bold',ha='left')
            plt.figtext(0.069, 0.06, 'Memory', fontsize=9, fontweight='bold',ha='left')

            plt.figtext(0.22, 0.08, '02', fontsize=22, ha='center', color='white',
                        bbox=dict(boxstyle='circle', facecolor='y', edgecolor='white'))
            plt.figtext(0.250, 0.12, Classifier[1], fontsize=10, fontweight='bold', color='k', ha='left')
            plt.figtext(0.250, 0.08, 'Convolutional ', fontsize=9, fontweight='bold',ha='left')
            plt.figtext(0.250, 0.06, 'Neural Network', fontsize=9, fontweight='bold',ha='left')

            plt.figtext(0.38, 0.08, '03', fontsize=22, ha='center', color='white',
                        bbox=dict(boxstyle='circle', facecolor='#9b5de5', edgecolor='white'))
            plt.figtext(0.408, 0.12, Classifier[2], fontsize=10, fontweight='bold', color='k', ha='left')
            plt.figtext(0.410, 0.08, 'Temporal Convolutional', fontsize=9, fontweight='bold',ha='left')
            plt.figtext(0.410, 0.06, 'Network', fontsize=9, fontweight='bold',ha='left')

            # Circle 4 (ResUNet)
            plt.figtext(0.585, 0.08, '04', fontsize=22, ha='center', color='white',
                        bbox=dict(boxstyle='circle', facecolor='#218380', edgecolor='white'))
            plt.figtext(0.615, 0.12, Classifier[3], fontsize=10, fontweight='bold', color='k', ha='left')
            plt.figtext(0.615, 0.08, 'Residual Long Short', fontsize=9, fontweight='bold',ha='left')
            plt.figtext(0.615, 0.06, 'Term Memory', fontsize=9, fontweight='bold',ha='left')

            # Circle 5 (Proposed)
            plt.figtext(0.77, 0.08, '05', fontsize=22, ha='center', color='white',
                        bbox=dict(boxstyle='circle', facecolor='#ef233c', edgecolor='white'))
            plt.figtext(0.800, 0.12, Classifier[4], fontsize=10, fontweight='bold', color='k', ha='left')
            plt.figtext(0.800, 0.08, 'Attention Based TCN ', fontsize=9, fontweight='bold',ha='left')
            plt.figtext(0.800, 0.06, 'with Dilated Res-LSTM', fontsize=9, fontweight='bold',ha='left')

            # Adjust layout to ensure the custom legend fits and leaves space around the plot
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.3)  # Add space at the bottom for the legend
            path = "./Results/%s_mod_bar.png" % (Terms[Graph_Terms[j]])
            plt.savefig(path)
            plt.show()


def Packet_PlotResults():
    eval = np.load('Eval_all.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1 Score',
             'MCC', 'FOR', 'PT', 'CSI', 'BA', 'FM', 'BM', 'MK', 'LR+', 'LR-', 'DOR', 'Prevalence']
    Graph_Terms = [0, 3, 4, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    bar_width = 0.15
    Classifier = ['LSTM-DNN', 'CNN', 'CA-TCN', 'DResLSTM', 'HDeepNet']
    Learning_rate = [0.11, 0.22, 0.33, 0.44, 0.55]
    for i in range(eval.shape[0]):
        for j in range(len(Graph_Terms)):
            Graph = np.zeros(eval.shape[1:3])
            for k in range(eval.shape[1]):
                for l in range(eval.shape[2]):
                    Graph[k, l] = eval[i, k, l, Graph_Terms[j] + 4]

            fig = plt.figure()
            ax = fig.add_axes([0.12, 0.12, 0.7, 0.7])
            ax.set_facecolor("#f0f0f0")
            fig.canvas.manager.set_window_title('Learninf Rate vs ' + Terms[Graph_Terms[j]])
            plt.plot(Learning_rate, Graph[:, 5], color='#8A2BE2', linewidth=3, marker='o', markersize=12,
                     label=Classifier[0])
            plt.plot(Learning_rate, Graph[:, 6], color='#DC143C', linewidth=3, marker='o', markersize=12,
                     label=Classifier[1])
            plt.plot(Learning_rate, Graph[:, 7], color='#FF00FF', linewidth=3, marker='o', markersize=12,
                     label=Classifier[2])
            plt.plot(Learning_rate, Graph[:, 8], color='#ff6700', linewidth=3, marker='o', markersize=12,
                     label=Classifier[3])
            plt.plot(Learning_rate, Graph[:, 4], color='k', linewidth=3, marker='o', markersize=12,
                     label=Classifier[4])
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['left'].set_visible(False)
            plt.gca().spines['bottom'].set_visible(False)
            # Custom Legend with Dot Markers, positioned at the top
            dot_markers = [plt.Line2D([2], [2], marker='o', color='w', markerfacecolor=color, markersize=13) for color
                           in ['#8A2BE2', '#DC143C', '#FF00FF', '#ff6700', 'k']]
            plt.legend(dot_markers, Classifier, loc='upper center', bbox_to_anchor=(0.5, 1.22), fontsize=10,
                       frameon=False, ncol=3, prop={'weight':'bold', 'size':12})
            plt.tight_layout()

            plt.xticks(Learning_rate, ('0.11', '0.22', '0.33', '0.44', '0.55'), fontname="Arial", fontsize=15,
                       fontweight='bold',
                       color='k')
            plt.yticks(fontname="Arial", fontsize=15, fontweight='bold',
                       color='k')
            plt.xlabel('Learning Rate', fontname="Arial", fontsize=15, fontweight='bold', color='k')
            plt.ylabel(Terms[Graph_Terms[j]], fontname="Arial", fontsize=15, fontweight='bold', color='k')
            plt.grid(color='c', linestyle='-', linewidth=2)
            path = "./Results/%s_Packet_line.png" % (Terms[Graph_Terms[j]])
            plt.savefig(path)
            plt.show()


def Packet_ROCCurve():
    cls = ['LSTM-DNN', 'CNN', 'CA-TCN', 'DResLSTM', 'HDeepNet']
    Actual = np.load('Packet_Target.npy', allow_pickle=True)
    lenper = round(Actual.shape[0] * 0.75)
    Actual = Actual[lenper:, :]
    fig = plt.figure()
    fig.canvas.manager.set_window_title('ROC Curve')
    colors = cycle(["r", "y", "blue", "m", "black"])
    for i, color in zip(range(len(cls)), colors):  # For all classifiers
        Predicted = np.load('Packet_Score.npy', allow_pickle=True)[i]
        false_positive_rate, true_positive_rate, _ = roc_curve(Actual.ravel(), Predicted.ravel())
        roc_auc = roc_auc_score(Actual.ravel(), Predicted.ravel())
        roc_auc = roc_auc * 100

        plt.plot(
            false_positive_rate,
            true_positive_rate,
            color=color,
            lw=2,
            label=f'{cls[i]} (AUC = {roc_auc:.2f} %)')

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontname="Arial", fontsize=15, fontweight='bold', color='k')
    plt.ylabel("True Positive Rate", fontname="Arial", fontsize=15, fontweight='bold', color='k')
    plt.xticks(fontname="Arial", fontsize=15, fontweight='bold', color='k')
    plt.yticks(fontname="Arial", fontsize=15, fontweight='bold', color='k')
    plt.title("ROC Curve")
    plt.legend(loc="lower right", prop={'weight':'bold', 'size':12})
    path = "./Results/packet_ROC.png"
    plt.savefig(path)
    plt.show()


def Proposed_PlotResults():
    eval = np.load('Packet_Evaluate.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1 Score',
             'MCC', 'FOR', 'PT', 'CSI', 'BA', 'FM', 'BM', 'MK', 'LR+', 'LR-', 'DOR', 'Prevalence']
    Graph_Terms = [0, 3, 4, 8, 9, 12, 16]
    Classifier = ['Ref 3', 'Ref 7', 'CA-TCN', 'CA-TCN- \n DRLSTM', 'MR-MOA-\n HDeepNet']
    step_per_Epoch = [100, 200, 300, 400, 500]
    for i in range(eval.shape[0]):
        for j in range(len(Graph_Terms)):
            Graph = np.zeros(eval.shape[1:3])
            for k in range(eval.shape[1]):
                for l in range(eval.shape[2]):
                    Graph[k, l] = eval[i, k, l, Graph_Terms[j] + 4]

            fig = plt.figure()
            ax = fig.add_axes([0.12, 0.25, 0.6, 0.6])
            ax.set_facecolor("#f0f0f0")
            fig.canvas.manager.set_window_title('Epoch vs ' + Terms[Graph_Terms[j]])
            plt.plot(step_per_Epoch, Graph[:, 3], color='k', linewidth=5, marker='p', markersize=15,
                     label=Classifier[3])
            plt.plot(step_per_Epoch, Graph[:, 4], color='k', linewidth=5, marker='<', markersize=15,
                     label=Classifier[4])
            plt.xticks(step_per_Epoch, ('100', '200', '300', '400', '500'), fontname="Arial", fontsize=14,
                       fontweight='bold',
                       color='#35530a')
            plt.yticks(fontname="Arial", fontsize=14, fontweight='bold',
                       color='#35530a')
            plt.grid(color='k', linestyle='-', linewidth=1, axis='y')
            plt.xlabel('Steps Per Epoch', fontname="Arial", fontsize=14, fontweight='bold', color='k')
            plt.ylabel(Terms[Graph_Terms[j]], fontname="Arial", fontsize=14, fontweight='bold', color='k')
            plt.legend(loc='upper center', bbox_to_anchor=(1.23, 0.68), fontsize=9, ncol=1, fancybox=True, shadow=True, prop={'weight':'bold', 'size':12})
            path = "./Results/%s_line.png" % (Terms[Graph_Terms[j]])
            plt.savefig(path)
            plt.show()


if __name__ == '__main__':
    plotConvResults()
    Plots_Results()
    Plot_ROC_Curve()
    Packet_PlotResults()
    Packet_ROCCurve()
    Table()
    Proposed_PlotResults()
