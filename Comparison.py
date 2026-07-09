import csv
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import islice
os.makedirs("Diagrams", exist_ok=True)
os.makedirs("Diagrams/Heatmaps", exist_ok=True)
os.makedirs("Diagrams/Distributions", exist_ok=True)
os.makedirs("Diagrams/Rankings", exist_ok=True)

#read csv into dictionary tuple:value
def read_csv(file_path):
        with open(file_path, mode='r', newline='', encoding='utf-8') as file:
                reader = csv.reader(file)
                labels = next(reader)[1:]
                data = {}

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
plt.close()

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

#---------------Heatmap Visualization---------------------#
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
    fmt=".1f",
    cmap="coolwarm",
    square=True,
    linewidths=0.3,
    linecolor="white",
    annot_kws={"size":5},
    cbar_kws={"label":"Standard Deviation"},
    vmin = -3, 
    vmax = 3
)

plt.title("Accuracy normalized Z score", fontsize=8)
plt.xticks(rotation=45, ha="right", fontsize=5)
plt.yticks(rotation=0, fontsize=5)

plt.tight_layout()
plt.savefig("./Diagrams/Heatmaps/z_accuracy.png", dpi=600, bbox_inches="tight")
plt.close()

#Similarity Heatmap
sns.heatmap(
    create_heatmap_matrix(z_similarity),
    annot=True,
    fmt=".1f",
    cmap="coolwarm",
    square=True,
    linewidths=0.3,
    linecolor="white",
    annot_kws={"size":5},
    cbar_kws={"label":"Standard Deviation"},
    vmin = -3, 
    vmax = 3
)

plt.title("Similarity normalized Z score", fontsize=8)
plt.xticks(rotation=45, ha="right", fontsize=5)
plt.yticks(rotation=0, fontsize=5)

plt.tight_layout()
plt.savefig("./Diagrams/Heatmaps/z_similarity.png", dpi=600, bbox_inches="tight")
plt.close()

#Comparison heatmap
sns.heatmap(
    create_heatmap_matrix(comparison),
    annot=True,
    fmt=".1f",
    cmap="coolwarm",
    square=True,
    linewidths=0.3,
    linecolor="white",
    annot_kws={"size":5},
    cbar_kws={"label":"Accuracy - Similarity Z-score Difference"},
    vmin = -4, 
    vmax = 4
)

plt.title("Z-score differences between Accuracy and Similarity", fontsize=8)
plt.xticks(rotation=45, ha="right", fontsize=5)
plt.yticks(rotation=0, fontsize=5)

plt.tight_layout()
plt.savefig("./Diagrams/Heatmaps/accuracy-similarity.png", dpi=600, bbox_inches="tight")
plt.close()

#----------------------------------------Distribution Visualization----------------------------------------#
sns.histplot(accuracy, x = list(accuracy.values()), bins=20, kde=True, line_kws={"label": "Trend"})
plt.title("Accuracy Distribution", fontsize=8)
plt.xlabel("Accuracy", fontsize=5)
plt.ylabel("Frequency", fontsize=5)
plt.xticks(fontsize=5)
plt.yticks(fontsize=5)
plt.tight_layout()
#Add the vertical mean line
plt.axvline(x=sum(accuracy.values()) / len(accuracy), color="red", linestyle="-", linewidth=1, label="Mean")
plt.legend(fontsize=5)
plt.savefig("./Diagrams/Distributions/accuracy_distribution.png", dpi=600, bbox_inches="tight")
plt.close()

sns.histplot(z_accuracy, x = list(z_accuracy.values()), bins=[-4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], kde=True, line_kws={"label": "Trend"})
plt.title("Accuracy Distribution", fontsize=8)
plt.xlabel("Z-Score", fontsize=5)
plt.ylabel("Frequency", fontsize=5)
plt.xticks(fontsize=5)
plt.yticks(fontsize=5)
plt.tight_layout()
#Add the vertical mean line
plt.axvline(x=sum(z_accuracy.values()) / len(z_accuracy), color="red", linestyle="-", linewidth=1, label="Mean")
plt.legend(fontsize=5)
plt.savefig("./Diagrams/Distributions/z_accuracy_distribution.png", dpi=600, bbox_inches="tight")
plt.close()

sns.histplot(similarity, x = list(similarity.values()), bins=10, kde=True, line_kws={"label": "Trend"})
plt.title("Distance Distribution", fontsize=8)
plt.xlabel("Distance", fontsize=5)
plt.ylabel("Frequency", fontsize=5)
plt.xticks(fontsize=5)
plt.yticks(fontsize=5)
plt.tight_layout()
plt.axvline(x=sum(similarity.values()) / len(similarity), color="red", linestyle="-", linewidth=1, label="Mean")
plt.legend(fontsize=5)
plt.savefig("./Diagrams/Distributions/distance_distribution.png", dpi=600, bbox_inches="tight")
plt.close()

sns.histplot(z_similarity, x = list(z_similarity.values()), bins=[-4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], kde=True, line_kws={"label": "Trend"})
plt.title("Similarity Distribution", fontsize=8)
plt.xlabel("Z-score", fontsize=5)
plt.ylabel("Frequency", fontsize=5)
plt.xticks(fontsize=5)
plt.yticks(fontsize=5)
plt.tight_layout()
plt.axvline(x=sum(z_similarity.values()) / len(z_similarity), color="red", linestyle="-", linewidth=1, label="Mean")
plt.legend(fontsize=5)
plt.savefig("./Diagrams/Distributions/z_similarity_distribution.png", dpi=600, bbox_inches="tight")
plt.close()

sns.histplot(comparison, x = list(comparison.values()), bins=[-4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], kde=True, line_kws={"label": "Trend"})
plt.title("Distribution of Z-score differences between Accuracy and Similarity", fontsize=8)
plt.xlabel("Z-score Accuracy - Similarity", fontsize=5)
plt.ylabel("Frequency", fontsize=5)
plt.xticks(fontsize=5)
plt.yticks(fontsize=5)
plt.tight_layout()
plt.axvline(x=sum(comparison.values()) / len(comparison), color="red", linestyle="-", linewidth=1, label="Mean")
plt.legend(fontsize=5)
plt.savefig("./Diagrams/Distributions/comparison_distribution.png", dpi=600, bbox_inches="tight")
plt.close()

#---------------Ranking Visualization---------------------#
comparison_sorted = { }
for key, value in comparison.items():
        comparison_sorted[key[0]+"-"+key[1]] = value
comparison_sorted = dict(sorted(comparison_sorted.items(), key=lambda item: item[1], reverse=True))
# print("Sorted comparison:", comparison_sorted)
comparison_top_half = dict(islice(comparison_sorted.items(), 0, int(len(comparison_sorted)/2)))    
comparison_bottom_half = dict(islice(comparison_sorted.items(), int(len(comparison_sorted)/2), len(comparison_sorted)))
comparison_bottom_half = dict(reversed(comparison_bottom_half.items()))
for key in comparison_bottom_half:
    comparison_bottom_half[key] = -comparison_bottom_half[key]

# print("Top half of the comparison:", comparison_top_half)
# print("Bottom half of the comparison:", comparison_bottom_half)
sns.barplot(data=dict(islice(comparison_top_half.items(), 0, 20)), color="b", orient="h")
plt.xlim(0, 4.5)
plt.title("Top 20 Accent Pairs for Diffference in Z-score (Accuracy - Similarity)", fontsize=8)
plt.xlabel("Difference")
plt.ylabel("Accents")
plt.xticks(ha="right", fontsize=5)
plt.yticks(fontsize=5)
plt.tight_layout()
plt.savefig("./Diagrams/Rankings/accuracy-similarity.png", dpi=600, bbox_inches="tight")
plt.close()

sns.barplot(data=dict(islice(comparison_bottom_half.items(), 0, 20)), color="b", orient="h")
plt.xlim(0, 4.5)
plt.title("Top 20 Accent Pairs for Diffference in Z-score (Similarity - Accuracy)", fontsize=8)
plt.xlabel("Difference")
plt.ylabel("Accents")
plt.xticks(ha="right", fontsize=5)
plt.yticks(fontsize=5)
plt.tight_layout()
plt.savefig("./Diagrams/Rankings/similarity-accuracy.png", dpi=600, bbox_inches="tight")
plt.close()

comparison_abs = {key: abs(value) for key, value in comparison_sorted.items()}
comparison_abs = dict(sorted(comparison_abs.items(), key=lambda item: item[1], reverse=True))
sns.barplot(data=dict(islice(comparison_abs.items(), 0, 20)), color="b", orient="h")
plt.xlim(0, 4.5)
plt.title("Top 20 Accent Pairs for Absolute Diffference in Z-score", fontsize=8)
plt.xlabel("Absolute Difference")
plt.ylabel("Accents")
plt.xticks(ha="right", fontsize=5)
plt.yticks(fontsize=5)
plt.tight_layout()
plt.savefig("./Diagrams/Rankings/absolute-difference.png", dpi=600, bbox_inches="tight")
plt.close()