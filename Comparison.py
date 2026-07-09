import csv
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
os.makedirs("Diagrams", exist_ok=True)

#read csv into dictionary tuple:value
def read_csv(file_path):
        with open(file_path, mode='r', newline='', encoding='utf-8') as file:
                reader = csv.reader(file)
                labels = next(reader)[1:]
                data = {}
                print("Labels:", labels)
                for row in reader:
                        i = 0
                        for value in row[1:]:
                                if row[0] != labels[i]: # skip if it is the same language pair
                                        data[(row[0], labels[i])] = float(value)
                                        i += 1
        return data

accuracy = read_csv('accuracy.csv')
similarity = read_csv('similarity.csv')
# print("Accuracy:", accuracy.values())
# print("Similarity:", similarity)

#---------------Create scatterplot between accuracy and similarity---------------#
# data for the scatterplot
scatter_data = { 
        "Accuracy": accuracy.values(),
        "Similarity": similarity.values()
}
scatter_df = pd.DataFrame(scatter_data)
#calculate the slope and intercept of the best fit line
slope, intercept = np.polyfit(scatter_df["Accuracy"], scatter_df["Similarity"], 1)

#base scatterplot
scatter_df.plot.scatter(x='Accuracy', y='Similarity', title='Scatterplot of Accuracy vs Similarity')

# generate y values for the trend line
trend_y = slope * scatter_df["Accuracy"] + intercept

# add trend line to the scatterplot
plt.plot(scatter_df["Accuracy"], trend_y, color='red', label='Trend Line')
plt.legend()
plt.savefig("./Diagrams/accuracy vs similarity.png", dpi=300, bbox_inches="tight")
plt.show()

#---------------Normalize the data for comparison---------------#
def normalize_data(data):
        mean = sum(data.values()) / len(data)
        std = (sum((x - mean) ** 2 for x in data.values()) / len(data)) ** 0.5
        z_data = {}
        for key in data.keys():
                z = (data[key] - mean) / std if std else 0
                z_data[key] = round(z, 2)
        return z_data

z_accuracy = normalize_data(accuracy)
z_similarity = normalize_data(similarity)
z_similarity = {key: -value for key, value in normalize_data(similarity).items()} #flip z scores since lower similarity should mean better accuracy
# print("Normalized Accuracy:", z_accuracy)
# print("Normalized Similarity:", z_similarity)

#---------------Compare the normalized data---------------#
def compare_data(z_accuracy, z_similarity):
        comparison = {}
        for key in z_accuracy.keys():
                comparison[key] = round(z_accuracy[key] - z_similarity[key], 2)
        return comparison

comparison = compare_data(z_accuracy, z_similarity)

# Convert dictionary to matrix for heatmap visualization
def create_heatmap_matrix(data):
        labels = ['Taiwanese', 'Greek', 'Gujurati', 'Gishu', 'Runyankore', 'French', 'Indonesian', 'Singaporean', 'Hindi', 'Br. Portuguese', 'Cantonese', 'Russian', 'Korean', 'Japanese', 'Turkish', 'Mandarin', 'Spanish', 'Vietnamese', 'Hebrew', 'German']
        matrix = pd.DataFrame(np.nan, index=labels, columns=labels)

        for (a1, a2), value in data.items():
                matrix.loc[a1, a2] = value
                matrix.loc[a2, a1] = value  # mirror the value

        for label in labels:
                matrix.loc[label, label] = np.nan

        return matrix
#Accuracy Heatmap
sns.heatmap(
    create_heatmap_matrix(z_accuracy),
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    square=True,
    linewidths=0.3,
    linecolor="white",
    annot_kws={"size":8},
    cbar_kws={"label":"Standard Deviation"}
)

plt.title("Accuracy normalized Z score", fontsize=9)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)

plt.tight_layout()
plt.savefig("./Diagrams/z_accuracy.png", dpi=300, bbox_inches="tight")
plt.show()

#Similarity Heatmap
sns.heatmap(
    create_heatmap_matrix(z_similarity),
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    square=True,
    linewidths=0.3,
    linecolor="white",
    annot_kws={"size":8},
    cbar_kws={"label":"Standard Deviation"}
)

plt.title("Similarity normalized Z score", fontsize=9)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)

plt.tight_layout()
plt.savefig("./Diagrams/z_similarity.png", dpi=300, bbox_inches="tight")
plt.show()

#Comparison heatmap
sns.heatmap(
    create_heatmap_matrix(comparison),
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    square=True,
    linewidths=0.3,
    linecolor="white",
    annot_kws={"size":8},
    cbar_kws={"label":"Accuracy - Similarity Z-score Difference"}
)

plt.title("Z-score differences between Accuracy and Similarity", fontsize=9)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)

plt.tight_layout()
plt.savefig("./Diagrams/accuracy-similarity.png", dpi=300, bbox_inches="tight")
plt.show()