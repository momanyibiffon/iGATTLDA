#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


LS = pd.read_csv('dataset/processed_data/lncrna_functional_similarity.csv', index_col=0)
DS = pd.read_csv('dataset/processed_data/disease_similarity_matrix.csv', index_col=0)
AM = pd.read_csv('dataset/processed_data/lncrna_disease_similarity.csv', index_col=0)
association_df = pd.read_csv('dataset/processed_data/association_df.csv', index_col=0)


# In[3]:


# data shapes
LS.shape, DS.shape, AM.shape, association_df.shape


# In[4]:


# LS visualizations
LS


# In[5]:


# DS visualizations
DS


# In[6]:


# Adjacency matrix visualization
AM


# In[7]:


# lncRNA-disease association visualization
association_df


# In[8]:


# creating a dataframe of diseases and lncrnas using index
lncrnas = []
diseases = []

for i in LS.index:
    lncrnas.append(i)
    
for i in DS.index:
    diseases.append(i)

# converting diseases and lncrnas lists to dataframe with a unique index (0 to n-1)
lncrnas_df = pd.DataFrame(lncrnas, index=range(len(lncrnas)), columns=['lncrnas'])
diseases_df = pd.DataFrame(diseases, index=range(len(diseases)), columns=['disease'])
len(diseases), len(lncrnas)


# In[9]:


lncrnas_df


# In[10]:


diseases_df


# ### GAT model

# In[11]:


import torch


# In[12]:


# mapping a unique disease ID to the disease ID
unique_disease_id = association_df['disease'].unique()
unique_disease_id = pd.DataFrame(data={
    'disease': unique_disease_id,
    'mappedID': pd.RangeIndex(len(unique_disease_id)),
})
print("Mapping of disease IDs to consecutive values:")
print("*********************************************")
print(unique_disease_id.head())

# mapping a unique lncrna ID to the lncrna ID
unique_lncrna_id = association_df['lncrna'].unique()
unique_lncrna_id = pd.DataFrame(data={
    'lncrna': unique_lncrna_id,
    'mappedID': pd.RangeIndex(len(unique_lncrna_id)),
})
print("Mapping of lncrna IDs to consecutive values:")
print("*********************************************")
print(unique_lncrna_id.head())

# Perform merge to obtain the edges from lncrna and diseases:
association_disease_id = pd.merge(association_df["disease"], unique_disease_id,
                            left_on='disease', right_on='disease', how='left')
association_disease_id = torch.from_numpy(association_disease_id['mappedID'].values)


association_lncrna_id = pd.merge(association_df['lncrna'], unique_lncrna_id,
                            left_on='lncrna', right_on='lncrna', how='left')
association_lncrna_id = torch.from_numpy(association_lncrna_id['mappedID'].values)

# construct `edge_index` in COO format
edge_index_disease_to_lncrna = torch.stack([association_disease_id, association_lncrna_id], dim=0)
print("Final edge indices from diseases to lncrnas")
print("*********************************************")
print(edge_index_disease_to_lncrna)
print(edge_index_disease_to_lncrna.shape)


# In[13]:


# disease and lncrna features
disease_feat = torch.from_numpy(DS.values).to(torch.float) # disease features in total
lncrna_feat = torch.from_numpy(LS.values).to(torch.float) # lncrna features in total
disease_feat.size(), lncrna_feat.size()


# In[14]:


# # plotting feature similarity heatmaps
# plt.figure(figsize=(8, 6))
# sns.heatmap(LS, annot=True, fmt='.2f', cmap='YlGnBu', square=True)
# plt.title('LncRNA Similarity Heatmap')
# plt.show()


# In[15]:


# len(unique_lncrna_id), len(LS)


# In[16]:


# HeteroData object initialization and passing necessary info
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T

data = HeteroData()
# Saving node indices
data["disease"].node_id = torch.arange(len(unique_disease_id))
data["lncrna"].node_id = torch.arange(len(LS))
# Adding node features and edge indices
data["disease"].x = disease_feat
data["lncrna"].x = lncrna_feat

data["disease", "associates_with", "lncrna"].edge_index = edge_index_disease_to_lncrna
# Adding reverse edges(GNN used this to pass messages in both directions)
data = T.ToUndirected()(data)
print(data)


# In[17]:


data.edge_index_dict, data.x_dict, data.edge_types


# In[18]:


# plotting the graph
import networkx as nx
from matplotlib import pyplot as plt
import torch_geometric

snrs = []
for i in unique_disease_id['mappedID']:
    snrs.append(i)

G = torch_geometric.utils.to_networkx(data.to_homogeneous())
pos = nx.kamada_kawai_layout(G)
nx.draw(G, pos=pos, with_labels=False, node_color="#3d7336", edgecolors="#96bd2b", linewidths=0.5, node_size=200)  # Draw the original graph
# Draw a subgraph, reusing the same node positions
nx.draw(G.subgraph(snrs), pos=pos, node_color="#96bd2b", edgecolors="#3d7336", linewidths=0.5, node_size=300)
plt.axis('off')
plt.savefig('case_study/case_study_HetGraph500dpi.png', dpi=500)
plt.show()


# In[19]:


G.number_of_edges(), G.number_of_nodes()


# ### Selecting a disease for the case study

# In[20]:


def select_case_study_disease(disease_id):
    selected_disease = {
        'disease': torch.tensor([disease_id]), # selecting a disease using mapped_IDs
    }
    return selected_disease

# diseases tested (colon cancer id = 276, lung cancer id = 83, stomach cancer id = 82)
disease_id = 82 # case study disease
selected_disease = select_case_study_disease(disease_id)
# retrieving lncRNAs connected to the selected disease
selected_disease_subgraph = data.subgraph(selected_disease)
selected_disease_associated_lncrnas = selected_disease_subgraph['disease', 'associates_with', 'lncrna'].edge_index[1]
selected_disease_associated_lncrnas


# In[21]:


# creating a new graph used to train the model for case study
# Note that selected disease and associated lncrnas were disconnected thus serving as
# unknown associations

# selected disease and associated lncrnas
selected_nodes = {
        'disease': selected_disease['disease'],
        'lncrna': selected_disease_associated_lncrnas
    }

print(selected_nodes)

subgraph = data.subgraph(selected_nodes)
print("==================================================")
print("Selected disease and its associated lncrnas")
print("==================================================")
print(subgraph)

# The copy data has been generated for case study purpose only

import copy
main_graph = data # Creating a copy of main graph
testing_data = copy.deepcopy(main_graph)
selected_disease = selected_nodes['disease']

# Identify edges corresponding to selected disease and associated lncrnas
disease_lncrna_edges = main_graph['disease', 'associates_with', 'lncrna'].edge_index
selected_disease_edges = disease_lncrna_edges[:, disease_lncrna_edges[0, :] == selected_disease]

# Remove edges corresponding to selected disease and associated lncrna from the test data
testing_data['disease', 'associates_with', 'lncrna'].edge_index = np.delete(
    testing_data['disease', 'associates_with', 'lncrna'].edge_index, selected_disease_edges[1, :], axis=1)
testing_data['lncrna', 'rev_associates_with', 'disease'].edge_index = np.delete(
    testing_data['lncrna', 'rev_associates_with', 'disease'].edge_index, selected_disease_edges[1, :], axis=1)

print("*********************************************")
print("The final graph object after removing known associations between selected disease and associated lncrnas")
print("*********************************************")
print(testing_data)

# therefore, testing data is the new dataset used for case study
# Next we need to train the model on remaining edges in main graph
# and use the trained model to predict associations between disconnected lncrnas
# and the selected disease


# In[22]:


# split data into training, validation, and test splits
transform = T.RandomLinkSplit(
    num_val=0.1,
    num_test=0.1,
    disjoint_train_ratio=0.3,
    neg_sampling_ratio=1.0,
    add_negative_train_samples=False,
    is_undirected = True, # added
    edge_types=("disease", "associates_with", "lncrna"),
    rev_edge_types=("lncrna", "rev_associates_with", "disease"),
)
case_study_data = testing_data
train_data, val_data, test_data = transform(case_study_data)
print(train_data)


# In[23]:


import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Visualizing train, validation, and test splits

def visualize_edges(data, title, color):
    edge_index = data["disease", "associates_with", "lncrna"].edge_index
    G = nx.Graph()
    for i in range(edge_index.shape[1]):
        source, target = edge_index[:, i]
        G.add_edge(source.item(), target.item())

    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=False, node_color=color, font_weight='bold', node_size=200)
    plt.title(title)

# Count the edges in each split
num_train_edges = train_data["disease", "associates_with", "lncrna"].edge_index.shape[1]
num_val_edges = val_data["disease", "associates_with", "lncrna"].edge_index.shape[1]
num_test_edges = test_data["disease", "associates_with", "lncrna"].edge_index.shape[1]

train_legend = Line2D([0], [0], marker='o', color='w', label=f'Training ({num_train_edges} edges)', markersize=10, markerfacecolor='orange')
val_legend = Line2D([0], [0], marker='o', color='w', label=f'Validation ({num_val_edges} edges)', markersize=10, markerfacecolor='green')
test_legend = Line2D([0], [0], marker='o', color='w', label=f'Test ({num_test_edges} edges)', markersize=10, markerfacecolor='red')

plt.figure(figsize=(12, 8))
visualize_edges(train_data, "Training Data Edges", color='orange') # Visualize training edges in orange
visualize_edges(val_data, "Validation Data Edges", color='green') # Visualize validation edges in green
visualize_edges(test_data, "Test Data Edges", color='red') # Visualize test edges in red
plt.legend(handles=[train_legend, val_legend, test_legend])
plt.savefig('case_study/case_study_train_val_test_graph500dpi.png', dpi=500)
plt.show()


# In[24]:


# creating a mini-batch loader for generating subgraphs used as input into our GNN
import torch_sparse
from torch_geometric.loader import LinkNeighborLoader

# Defining seed edges
edge_label_index = train_data["disease", "associates_with", "lncrna"].edge_label_index
edge_label = train_data["disease", "associates_with", "lncrna"].edge_label
train_loader = LinkNeighborLoader(
    data=train_data,
    num_neighbors=[20] * 2,
    neg_sampling_ratio=2.0,
    edge_label_index=(("disease", "associates_with", "lncrna"), edge_label_index),
    edge_label=edge_label,
    batch_size=4 * 32,
    shuffle=True,
)  

# Inspecting a sample
sampled_data = next(iter(train_loader))

print("Sampled mini-batch:")
print("===================")
print(sampled_data)

G = torch_geometric.utils.to_networkx(sampled_data.to_homogeneous())
# Plot the graph
nx.draw(G, with_labels=False)
plt.show()


# In[ ]:





# In[25]:


sampled_data['disease'].x


# ### Heterogeneous GNN

# In[26]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, to_hetero
from torch import Tensor
from torch.nn import Linear
from torch.nn import Transformer

class MLPClassifier(torch.nn.Module):
    def __init__(self, mlp_hidden_channels, mlp_out_channels):
        super(MLPClassifier, self).__init__()
        self.fc1 = torch.nn.Linear(mlp_hidden_channels, mlp_hidden_channels)
        self.fc2 = torch.nn.Linear(mlp_hidden_channels, mlp_out_channels)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels=32, num_heads=4, transformer_dim=64, num_layers=2):
        super(GNN, self).__init__()
        self.attn1 = GATConv((-1, -1), hidden_channels, heads=num_heads, add_self_loops=False)
        self.attn2 = GATConv((-1, -1), hidden_channels, heads=num_heads, add_self_loops=False)
        self.attn3 = GATConv((-1, -1), out_channels, heads=num_heads, add_self_loops=False)
        self.num_heads = num_heads
        self.penalty_linear = nn.Linear(out_channels, 1)

        # Linear projections for input
        self.linear_disease = nn.Linear(hidden_channels, transformer_dim)
        self.linear_lncrna = nn.Linear(hidden_channels, transformer_dim)

        # Transformer encoder for capturing global dependencies
        self.transformer = nn.Transformer(d_model=transformer_dim, nhead=num_heads, num_encoder_layers=num_layers)
        
    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.attn1(x, edge_index)
        x = x.relu()
        x = x.view(-1, self.num_heads, x.shape[1] // self.num_heads)
        x = x.mean(dim=1)
        x = self.attn2(x, edge_index)
        x = x.relu()
        x = x.view(-1, self.num_heads, x.shape[1] // self.num_heads)
        x = x.mean(dim=1)
        x = self.attn3(x, edge_index)
        x = x.relu()
        x = x.view(-1, self.num_heads, x.shape[1] // self.num_heads)
        x = x.mean(dim=1)
        penalty = self.penalty_linear(x)
        x = x * torch.exp(penalty)
        
        # Applying linear projections to ensure consistent feature dimensions
        x_disease = self.linear_disease(x)
        x_lncrna = self.linear_lncrna(x)

        # The Transformer encoder that captures global dependencies
        transformer_output_disease = self.transformer(x_disease, x_disease)
        transformer_output_lncrna = self.transformer(x_lncrna, x_lncrna)

        return x  # You need to return a final output


class Classifier(torch.nn.Module):
    def __init__(self, mlp_hidden_channels, mlp_out_channels):
        super(Classifier, self).__init__()
        self.mlp = MLPClassifier(mlp_hidden_channels, mlp_out_channels)

    def forward(self, x_disease: Tensor, x_lncrna: Tensor, edge_label_index: Tensor) -> Tensor:
        edge_feat_disease = x_disease[edge_label_index[0]]
        edge_feat_lncrna = x_lncrna[edge_label_index[1]]
        concat_edge_feats = torch.cat((edge_feat_disease, edge_feat_lncrna), dim=-1)
        return self.mlp(concat_edge_feats)


class Model(torch.nn.Module):
    def __init__(self, hidden_channels=128, num_graphs=3, mlp_hidden_channels=64, mlp_out_channels=1):
        super(Model, self).__init__()
        self.num_graphs = num_graphs
        self.graphs = torch.nn.ModuleList()
        for i in range(num_graphs):
            self.graphs.append(GNN(hidden_channels))
        self.disease_lin = torch.nn.Linear(412, hidden_channels)
        self.lncrna_lin = torch.nn.Linear(240, hidden_channels)
        self.disease_emb = torch.nn.Embedding(data["disease"].num_nodes, hidden_channels)
        self.lncrna_emb = torch.nn.Embedding(data["lncrna"].num_nodes, hidden_channels)
        self.gnn = GNN(hidden_channels)
        self.gnn = to_hetero(self.gnn, metadata=data.metadata())
        self.classifier = Classifier(mlp_hidden_channels, mlp_out_channels)

    def forward(self, data: HeteroData) -> Tensor:
        x_dict = {
            "disease": self.disease_lin(data["disease"].x) + self.disease_emb(data["disease"].node_id),
            "lncrna": self.lncrna_lin(data["lncrna"].x) + self.lncrna_emb(data["lncrna"].node_id),
        }
        x_dict = self.gnn(x_dict, data.edge_index_dict)
        pred = self.classifier(
            x_dict["disease"],
            x_dict["lncrna"],
            data["disease", "associates_with", "lncrna"].edge_label_index,
        )
        return pred

# Instantiate the model
model = Model(hidden_channels=128, mlp_hidden_channels=64, mlp_out_channels=1)
print(model)


# In[27]:


train_loader.data['disease', 'associates_with', 'lncrna'].edge_label.size()


# In[28]:


# validation loader
edge_label_index = val_data["disease", "associates_with", "lncrna"].edge_label_index
edge_label = val_data["disease", "associates_with", "lncrna"].edge_label

val_loader = LinkNeighborLoader(
    data=val_data,
    num_neighbors=[30] * 3,
    edge_label_index=(("disease", "associates_with", "lncrna"), edge_label_index),
    edge_label=edge_label,
    batch_size=128,
    shuffle=True,
)
sampled_data = next(iter(val_loader))
print(sampled_data)


# In[29]:


import torch_geometric
import networkx as nx

# evaluate the GNN model
# The new LinkNeighborLoader iterates over edges in the validation set
# obtaining predictions on validation edges
# then evaluate the performance by computing the AUC

# Define the validation seed edges:
edge_label_index = test_data["disease", "associates_with", "lncrna"].edge_label_index
edge_label = test_data["disease", "associates_with", "lncrna"].edge_label

test_loader = LinkNeighborLoader(
    data=test_data,
    num_neighbors=[30] * 3,
    edge_label_index=(("disease", "associates_with", "lncrna"), edge_label_index),
    edge_label=edge_label,
    batch_size=128,
    shuffle=True,
)
sampled_data = next(iter(test_loader))
print(sampled_data)

G = torch_geometric.utils.to_networkx(sampled_data.to_homogeneous())
# Networkx seems to create extra nodes from our heterogeneous graph, so we remove them
isolated_nodes = [node for node in G.nodes() if G.out_degree(node) == 0]
[G.remove_node(i_n) for i_n in isolated_nodes]
# Plot the graph
nx.draw(G, with_labels=False)
plt.show()


# In[30]:


import torch
import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: {}".format(device))

model = model.to(device)

# Create a learning rate scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)  # scheduler

num_folds = 10
num_epochs = 300

# Lists to store training loss values for each fold, learning curve values, and learning rates
fold_loss_values = []
learning_curve = []
learning_rates = []
true_labels = []
predicted_probs = []

for fold in range(num_folds):
    # Create a progress bar for the current fold
    fold_bar = tqdm.tqdm(total=num_epochs, position=0, leave=True)
    fold_loss = []  # Training loss values for the current fold
    for epoch in range(1, num_epochs + 1):
        total_loss = total_examples = 0
        model.train()  # Set the model to training mode
        
        # Get the current learning rate from the optimizer
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)

        for sampled_data in train_loader:
            optimizer.zero_grad()
            sampled_data = sampled_data.to(device)  # Move the data to the device

            # Forward pass
            pred = model(sampled_data)
            ground_truth = sampled_data["disease", "associates_with", "lncrna"].edge_label
            pred_prob = torch.sigmoid(pred)
            loss = F.binary_cross_entropy(pred_prob.squeeze(), ground_truth)

            loss.backward()
            optimizer.step()

            total_loss += float(loss) * pred_prob.numel()
            total_examples += pred_prob.numel()

            # Store true labels and predicted probabilities for calibration curve
            true_labels.extend(ground_truth.tolist())
            predicted_probs.extend(pred_prob.squeeze().tolist())

        # Append the current training loss to the list
        fold_loss.append(total_loss / total_examples)
        fold_bar.update(1)
        fold_bar.set_description(f"Fold {fold + 1}/{num_folds}, Epoch {epoch}/{num_epochs}, Loss: {total_loss / total_examples:.4f}")
        
        # Calculate and store the performance metric for learning curve (you should replace this with your metric)
        performance_metric = 1.0 - (total_loss / total_examples)
        learning_curve.append(performance_metric)
        
        scheduler.step()  # Adjust the learning rate based on the scheduler

    fold_bar.close()  # Close the progress bar for the current fold
    fold_loss_values.append(fold_loss)

# Save the final model
torch.save(model.state_dict(), 'cv_trained_model.pth')

# Generate the calibration curve data
prob_true, prob_pred = calibration_curve(true_labels, predicted_probs, n_bins=10, strategy='uniform')

# Plotting the training loss, learning, learning rate, and calibration curves in a 2x2 grid
fig, axes = plt.subplots(2, 2, figsize=(10, 7))

# Plot training loss curves
for fold in range(num_folds):
    axes[0, 0].plot(range(1, num_epochs + 1), fold_loss_values[fold], label=f'Fold {fold + 1}')

axes[0, 0].set_title('Training Loss Curves')
axes[0, 0].set_xlabel('Epochs')
axes[0, 0].set_ylabel('Training Loss')
axes[0, 0].legend()

# Plot learning curves (you should replace this with your actual metric values)
axes[0, 1].plot(range(1, num_epochs * num_folds + 1), learning_curve, label='Learning Curve', color='green')

axes[0, 1].set_title('Learning Curve')
axes[0, 1].set_xlabel('Epochs')
axes[0, 1].set_ylabel('Performance Metric')
axes[0, 1].legend()

# Plot learning rates
axes[1, 0].plot(range(1, (num_epochs * num_folds) + 1), learning_rates, label='Learning Rate', color='blue')

axes[1, 0].set_title('Learning Rate Schedule')
axes[1, 0].set_xlabel('Epochs')
axes[1, 0].set_ylabel('Learning Rate')
axes[1, 0].legend()

# Plot calibration curve
axes[1, 1].plot(prob_pred, prob_true, marker='o', label='Calibration Curve', color='b')
axes[1, 1].plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
axes[1, 1].set_xlabel('Mean Predicted Probability')
axes[1, 1].set_ylabel('Observed Probability')
axes[1, 1].set_title('Calibration Curve')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('2x2_Plots_500dpi.png', dpi=500)
plt.show()


# In[31]:


# # After 10-fold CV, the model is trained with entire training data and evaluate and tested on val and test data

# # Reassigning data loaders for the final training, validation and testing data
# final_train_loader = train_loader
# final_val_loader = val_loader
# final_test_loader = test_loader

# # Loading the saved model
# cv_trained_model = Model(hidden_channels=128, mlp_hidden_channels=64, mlp_out_channels=1) # new model instance
# cv_trained_model.load_state_dict(torch.load('cv_trained_model.pth'))
# cv_trained_model = cv_trained_model.to(device)  # Move the model to the device

# # final training optimizer
# final_optimizer = torch.optim.Adam(cv_trained_model.parameters(), lr=0.001)

# # Training the final model on the entire training data
# final_loss_values = []
# final_epochs = []

# total_epochs = 1500  # Total number of training epochs

# # Create a single progress bar for all epochs
# final_training_bar = tqdm.tqdm(total=total_epochs, position=0, leave=True)

# for epoch in range(1, total_epochs + 1):
#     total_loss = total_examples = 0
#     cv_trained_model.train()  # Set the model to training mode
#     for sampled_data in final_train_loader:
#         final_optimizer.zero_grad()
#         sampled_data = sampled_data.to(device)  # Move the data to the device

#         # Forward pass
#         pred = cv_trained_model(sampled_data)
#         ground_truth = sampled_data["disease", "associates_with", "lncrna"].edge_label

#         pred_prob = torch.sigmoid(pred)
#         loss = F.binary_cross_entropy(pred_prob.squeeze(), ground_truth)

#         loss.backward()
#         final_optimizer.step()

#         total_loss += float(loss) * pred_prob.numel()
#         total_examples += pred_prob.numel()

#     final_loss_values.append(total_loss / total_examples)
#     final_epochs.append(epoch)

#     final_training_bar.update(1)
#     final_training_bar.set_description(f"Epoch {epoch}/{total_epochs}, Loss: {total_loss / total_examples:.4f}")

# final_training_bar.close()  # Close the progress bar for the final training

# # Creating a Pandas DataFrame with training loss data for the final training
# final_training_data = {'Epoch': final_epochs, 'Final Training Loss': final_loss_values}
# final_df = pd.DataFrame(final_training_data)

# # Plot the final training loss curve
# plt.figure(figsize=(6, 4))
# sns.set(style="white")
# ax = sns.lineplot(data=final_df, x='Epoch', y='Final Training Loss', label='Final Training Loss')
# ax.set(title='Final Training Loss Curve', xlabel='Epoch', ylabel='Loss')
# plt.savefig('final_training_loss_curve.png', dpi=500)
# plt.show()


# We retrain the model with new case study data set by fine-tuning the saved model trained with 10-fold CV

# In[32]:


import tqdm

# After 10-fold CV, the model is trained with entire training data and evaluate and tested on val and test data
# Reassigning data loaders for the final training, validation, and testing data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: {}".format(device))

final_train_loader = train_loader
final_val_loader = val_loader
final_test_loader = test_loader

# Load the saved model
cv_trained_model = Model(hidden_channels=128, mlp_hidden_channels=64, mlp_out_channels=1)  # Create a new model instance
cv_trained_model.load_state_dict(torch.load('cv_trained_model.pth'))
cv_trained_model = cv_trained_model.to(device)  # Move the model to the device

# Define the optimizer for final training
final_optimizer = torch.optim.Adam(cv_trained_model.parameters(), lr=0.001)

# Training parameters
total_epochs = 1000
best_validation_loss = float('inf')
best_model_path = 'best_model.pth'

# Define lists to store training and validation loss values
final_loss_values = []
validation_loss_values = []
final_epochs = []

# Create a progress bar for all epochs
final_training_bar = tqdm.tqdm(total=total_epochs, position=0, leave=True)

for epoch in range(1, total_epochs + 1):
    total_loss = total_examples = 0
    total_val_loss = total_val_examples = 0

    cv_trained_model.train()  # Set the model to training mode

    for sampled_data in final_train_loader:
        final_optimizer.zero_grad()
        sampled_data = sampled_data.to(device)

        # Forward pass
        pred = cv_trained_model(sampled_data)
        ground_truth = sampled_data["disease", "associates_with", "lncrna"].edge_label

        pred_prob = torch.sigmoid(pred)
        loss = F.binary_cross_entropy(pred_prob.squeeze(), ground_truth)

        loss.backward()
        final_optimizer.step()

        total_loss += float(loss) * pred_prob.numel()
        total_examples += pred_prob.numel()

    final_loss_values.append(total_loss / total_examples)
    final_epochs.append(epoch)

    cv_trained_model.eval()  # Set the model to evaluation mode

    with torch.no_grad():
        for sampled_data in final_val_loader:
            sampled_data = sampled_data.to(device)
            pred = cv_trained_model(sampled_data)
            ground_truth = sampled_data["disease", "associates_with", "lncrna"].edge_label

            pred_prob = torch.sigmoid(pred)
            loss = F.binary_cross_entropy(pred_prob.squeeze(), ground_truth)

            total_val_loss += float(loss) * pred_prob.numel()
            total_val_examples += pred_prob.numel()

    validation_loss = total_val_loss / total_val_examples
    validation_loss_values.append(validation_loss)

    final_training_bar.update(1)
    final_training_bar.set_description(f"Epoch {epoch}/{total_epochs}, Loss: {total_loss / total_examples:.4f}, Val Loss: {validation_loss:.4f}")

    if validation_loss < best_validation_loss:
        best_validation_loss = validation_loss
        torch.save(cv_trained_model.state_dict(), best_model_path)

final_training_bar.close()  # Close the progress bar for the final training

# Create a Pandas DataFrame with training and validation loss data for the final training
final_training_data = {'Epoch': final_epochs, 'Final Training Loss': final_loss_values, 'Validation Loss': validation_loss_values}
final_df = pd.DataFrame(final_training_data)

# Plot the final training and validation loss curves
plt.figure(figsize=(6, 4))
sns.set(style="white")
ax = sns.lineplot(data=final_df, x='Epoch', y='Final Training Loss', label='Final Training Loss')
ax = sns.lineplot(data=final_df, x='Epoch', y='Validation Loss', label='Validation Loss')
ax.set(title='Final Training and Validation Loss Curve', xlabel='Epoch', ylabel='Loss')
plt.legend()
plt.savefig('case_study/case_study_training_and_validation_loss.png', dpi=500)
plt.show()


# In[33]:


# testing accuracy
from sklearn.metrics import roc_auc_score, classification_report, RocCurveDisplay, roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix, f1_score
from scipy.interpolate import interp1d
import seaborn as sns

# this function converts predictions from continous to binary specifically for 
# use in the classification report which doesn't accept continuous labels
def binary_predictions(threshold, x):
    predictions_binary = (x > threshold).astype(int)
    return predictions_binary
    
# main model for training and testing
def test_val_accuracy(loader):
    tv_preds = []
    tv_ground_truths = []
    for sampled_data in tqdm.tqdm(loader):
        with torch.no_grad():
            sampled_data.to(device)
            preds = model(sampled_data).view(-1)
            tv_preds.append(preds)
            tv_ground_truths.append(sampled_data["disease", "associates_with", "lncrna"].edge_label)
    tv_preds = torch.cat(tv_preds, dim=0).cpu().numpy()
    tv_ground_truths = torch.cat(tv_ground_truths, dim=0).cpu().numpy()
    # tv_auc = roc_auc_score(tv_ground_truths, tv_preds)
    
    # plotting AUC Curve
    binary_ground_truths = np.array([1 if label == 2.0 or label == 1.0 else 0 for label in tv_ground_truths]) # converting ground truth values to {0, 1}
    
    # Check if there are any positive samples in y_true
    if np.sum(binary_ground_truths) == 0:
        # There are no positive samples, so set AUC to a default value of 0.5
        roc_auc = 0.5
    else:
        # plotting the AUC using seaborn
        sns.set_style('white')
        sfpr, stpr, _ = roc_curve(binary_ground_truths, tv_preds)
        roc_auc = round(auc(sfpr, stpr), 2)
        sns.lineplot(x=sfpr, y=stpr, label=f'iGATTLDA (AUC = {roc_auc})', errorbar=('ci', 99))
        sns.lineplot(x=[0,1], y=[0,1], color='black', linestyle='dashed')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('AUC')
        plt.legend(loc='lower right')
        plt.savefig('case_study/case_study_AUC.png', dpi=500)
        plt.show()

        # Calculate AUPRC
        auprc = average_precision_score(binary_ground_truths, tv_preds)

        # Plotting AUPRC Curve
        sns.set_style('white')
        precision, recall, _ = precision_recall_curve(binary_ground_truths, tv_preds)
        auprc = round(auprc, 2)
        sns.lineplot(x=recall, y=precision, label=f'iGATTLDA (AUPRC = {auprc})', errorbar=('ci', 99))
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='lower left')
        plt.savefig('case_study/case_study_AUPRC.png', dpi=500)
        plt.show()

        # converting predictions to binary so as to print a classification report
        binary_preds = binary_predictions(0.5, tv_preds)
        # classification report
        clf_report = classification_report(binary_ground_truths, binary_preds)
        print("Classification report")
        print(clf_report)
    #     print(binary_preds)
    #     print(binary_ground_truths)
    #     print(tv_preds)

        # plotting confusion matrix
        cm = confusion_matrix(binary_ground_truths, binary_preds)    
        class_labels = ["Negative", "Positive"] # Define class labels for the confusion matrix
        sns.set(font_scale=1.2)  # Adjusting font size
        plt.figure(figsize=(6, 4))  # Adjusting figure size
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=class_labels, yticklabels=class_labels)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()

        # f1-score
        f1 = f1_score(binary_ground_truths, binary_preds)
        print("F1-score:", f1)
    print(binary_ground_truths)
    print(tv_preds)
    
    return roc_auc, tv_preds


# In[34]:


# Evaluate the final model on evaluation data
# print("Validation performance")
# evaluation_roc_auc = train_val_accuracy(final_val_loader)

# Evaluate the final model on test data
print("Testing performance")
test_roc_auc = test_val_accuracy(final_test_loader)


# ### Case study prediction scores between selected disease and associated lncrnas

# In[35]:


# selected disease and its associated lncrnas used for case study after removing their known associations

# subgraph = case_study_data.subgraph(selected_nodes)
# print(subgraph)

# # plotting the subgraph for disease 0 and its associated lncrnas

# sub_G = torch_geometric.utils.to_networkx(subgraph.to_homogeneous())
# # Networkx seems to create extra nodes from our heterogeneous graph, so we remove them
# isolated_nodes = [node for node in sub_G.nodes() if sub_G.out_degree(node) == 0]
# [sub_G.remove_node(i_n) for i_n in isolated_nodes]
# # Plot the graph
# nx.draw(sub_G, with_labels=True)
# plt.show()


# In[36]:


case_study_preds = test_val_accuracy(final_test_loader)


# In[37]:


type(case_study_preds)


# In[38]:


# testing_data is the new graph with lncRNAs disconnected from the selected disease
case_study_predictions = case_study_preds[1]
case_study_predictions


# In[39]:


# getting node ids of the lncRNAs
lncrna_node_ids = testing_data['lncrna'].node_id
lncrna_node_ids


# In[40]:


# predicted values for the lncRNAs disconnected from selected disease
predictions = []
for i, j in sorted(zip(lncrna_node_ids, case_study_predictions), key=lambda x: x[1], reverse=True):
    for x in unique_lncrna_id.index:
        if x == i.item():
            predictions.append([i.item(), unique_lncrna_id['lncrna'][x], j])

predictions = pd.DataFrame(predictions, columns =['mappedID', 'lncRNA id', 'Prediction score' ])

disease = ''
for i in unique_disease_id.index:
    if unique_disease_id['mappedID'][i] == selected_disease.item():
        print("Disease: {}, mappedID: {}".format(unique_disease_id['disease'][i], unique_disease_id['mappedID'][i]))
        disease = unique_disease_id['disease'][i]

predictions.to_csv('case_study/disease_{}_case_study_predictions.csv'.format(disease))# saving the predictions as csv
predictions.head(10) # top 10 predictions


# In[41]:


# how to check for specific disease/lncrna from the mapped dataframe
for i in unique_disease_id.index:
    print(unique_disease_id['disease'][i], unique_disease_id['mappedID'][i])

