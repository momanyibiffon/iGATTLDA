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


# In[7]:


# plotting lncrna and disease similarity figures
fig, (fig1, fig2) = plt.subplots(1, 2, figsize=(13, 13))

# Set titles for the subplots
fig1.title.set_text('lncRNA similarity')
fig2.title.set_text('Diseases similarity')

# Set x and y labels for the first subplot
fig1.set_xlabel('lncRNA Index')
fig1.set_ylabel('lncRNA Index')

# Set x and y labels for the second subplot
fig2.set_xlabel('Disease Index')
fig2.set_ylabel('Disease Index')

# Adjust vmin and vmax to control the color intensity
fig1.imshow(LS, cmap="Blues", interpolation="none", vmin=0.1, vmax=0.9)
fig2.imshow(DS, cmap="Blues", interpolation="none", vmin=0.1, vmax=0.9)

# Save the figure with high dpi
plt.savefig('lncrna_disease_similarities.png', dpi=500)


# In[7]:


# Adjacency matrix visualization
AM


# In[8]:


# Plot the adjacency matrix
plt.figure(figsize=(20,9))
plt.imshow(AM, cmap='Blues', interpolation='none')
plt.colorbar()
plt.title('Adjacency Matrix')
plt.savefig('Adjacency_matrix.png', dpi=500)
plt.show()


# In[9]:


# lncRNA-disease association visualization
association_df


# In[10]:


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


# In[11]:


lncrnas_df


# In[12]:


diseases_df


# ### GAT model

# In[13]:


import torch


# In[14]:


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


# In[15]:


# disease and lncrna features
disease_feat = torch.from_numpy(DS.values).to(torch.float) # disease features in total
lncrna_feat = torch.from_numpy(LS.values).to(torch.float) # lncrna features in total
disease_feat.size(), lncrna_feat.size()


# In[16]:


# # plotting feature similarity heatmaps
# plt.figure(figsize=(8, 6))
# sns.heatmap(LS, annot=True, fmt='.2f', cmap='YlGnBu', square=True)
# plt.title('LncRNA Similarity Heatmap')
# plt.show()


# In[17]:


# len(unique_lncrna_id), len(LS)


# In[18]:


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


# In[ ]:





# In[19]:


data.edge_index_dict, data.x_dict, data.edge_types


# In[20]:


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
plt.savefig('HetGraph500dpi.png', dpi=500)
plt.show()


# In[21]:


import networkx as nx
import torch_geometric
import matplotlib.pyplot as plt

# Extracting the necessary data
edge_index = data["disease", "associates_with", "lncrna"].edge_index.numpy()
disease_nodes = data["disease"].node_id.numpy()
drug_nodes = data["lncrna"].node_id.numpy()

G = torch_geometric.utils.to_networkx(data.to_homogeneous()) # Convert HeteroData object to a NetworkX graph
pos = nx.kamada_kawai_layout(G) # Get positions of nodes using Kamada-Kawai layout

# Create figure and subplots
fig, axs = plt.subplots(1, 3, figsize=(18, 6))
# Draw the original graph
nx.draw(G, pos=pos, with_labels=False, node_color="#77AADD", edgecolors="#000000", linewidths=0.5, node_size=200, ax=axs[0])
# Draw a subgraph for diseases, reusing the same node positions
nx.draw(G.subgraph(disease_nodes), pos=pos, node_color="#FFA500", edgecolors="#000000", linewidths=0.5, node_size=300, ax=axs[0])
axs[0].set_title("Heterogeneous Graph")
axs[0].axis('off')

# Degree rank plot
degree_sequence = sorted((d for n, d in G.degree()), reverse=True)
axs[1].plot(degree_sequence, "b-", marker="o")
axs[1].set_title("Degree Rank Plot")
axs[1].set_ylabel("Degree")
axs[1].set_xlabel("Rank")

# Degree histogram
axs[2].hist(degree_sequence, bins='auto', color='b')
axs[2].set_title("Degree Histogram")
axs[2].set_xlabel("Degree")
axs[2].set_ylabel("# of Nodes")

# Adjust layout
plt.tight_layout()

# Save or display the figure
plt.savefig('HetGraph_Degree_Plot.png', dpi=300)
plt.show()


# In[22]:


# import plotly.graph_objects as go

# # Assuming 'data' is your torch_geometric data object
# G = torch_geometric.utils.to_networkx(data.to_homogeneous())
# snrs = list(unique_disease_id['mappedID'])

# edge_trace = go.Scatter(
#     x=[],
#     y=[],
#     line=dict(width=0.5, color='#888'),
#     hoverinfo='none',
#     mode='lines')

# for edge in G.edges():
#     x0, y0 = pos[edge[0]]
#     x1, y1 = pos[edge[1]]
#     edge_trace['x'] += tuple([x0, x1, None])
#     edge_trace['y'] += tuple([y0, y1, None])

# node_trace = go.Scatter(
#     x=[],
#     y=[],
#     text=[],
#     mode='markers',
#     hoverinfo='text',
#     marker=dict(
#         showscale=True,
#         colorscale='YlGnBu',
#         size=10,
#         colorbar=dict(
#             thickness=15,
#             title='Node Connections',
#             xanchor='left',
#             titleside='right'
#         )
#     )
# )

# for node in G.nodes():
#     x, y = pos[node]
#     node_trace['x'] += tuple([x])
#     node_trace['y'] += tuple([y])

# fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(showlegend=False, hovermode='closest'))
# fig.show()


# In[23]:


# G.number_of_edges(), G.number_of_nodes()


# In[24]:


# split associations into training, validation, and test splits
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
train_data, val_data, test_data = transform(data)
print(train_data)


# In[25]:


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
plt.savefig('train_val_test_graph500dpi.png', dpi=500)
plt.show()


# In[26]:


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

# G = torch_geometric.utils.to_networkx(sampled_data.to_homogeneous())
# # Plot the graph
# nx.draw(G, with_labels=False)
# plt.show()


# In[27]:


sampled_data['disease'].x


# ### Heterogeneous GNN

# In[28]:


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
    def __init__(self, hidden_channels, out_channels=32, num_heads=16, transformer_dim=64, num_layers=2):
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
        x1 = self.attn1(x, edge_index)
        x1 = x1.relu()
        x1 = x1.view(-1, self.num_heads, x1.shape[1] // self.num_heads)
        x1 = x1.mean(dim=1)

        x2 = self.attn2(x1, edge_index)
        x2 = x2.relu()
        x2 = x2.view(-1, self.num_heads, x2.shape[1] // self.num_heads)
        x2 = x2.mean(dim=1)

        x3 = self.attn3(x2, edge_index)
        x3 = x3.relu()
        x3 = x3.view(-1, self.num_heads, x3.shape[1] // self.num_heads)
        x3 = x3.mean(dim=1)

        penalty = self.penalty_linear(x3)
        x3 = x3 * torch.exp(penalty)
        
        # Applying linear projections to ensure consistent feature dimensions
        x_disease = self.linear_disease(x3)
        x_lncrna = self.linear_lncrna(x3)

        # The Transformer encoder that captures global dependencies
        transformer_output_disease = self.transformer(x_disease, x_disease)
        transformer_output_lncrna = self.transformer(x_lncrna, x_lncrna)

        return x3  # return a final output


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


# In[29]:


train_loader.data['disease', 'associates_with', 'lncrna'].edge_label.size()


# In[30]:


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


# In[31]:


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


# In[32]:


# import torch
# import tqdm
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# import seaborn as sns

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print("Device: {}".format(device))

# model = model.to(device)

# # Create a learning rate scheduler
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)  # Example scheduler

# num_folds = 5
# num_epochs = 20

# # Lists to store training loss values for each fold, learning curve values, and learning rates
# fold_loss_values = []
# learning_curve = []
# learning_rates = []

# for fold in range(num_folds):
#     # Create a progress bar for the current fold
#     fold_bar = tqdm.tqdm(total=num_epochs, position=0, leave=True)
#     fold_loss = []  # Training loss values for the current fold
#     for epoch in range(1, num_epochs + 1):
#         total_loss = total_examples = 0
#         model.train()  # Set the model to training mode
        
#         # Get the current learning rate from the optimizer
#         current_lr = optimizer.param_groups[0]['lr']
#         learning_rates.append(current_lr)

#         for sampled_data in train_loader:
#             optimizer.zero_grad()
#             sampled_data = sampled_data.to(device)  # Move the data to the device

#             # Forward pass
#             pred = model(sampled_data)
#             ground_truth = sampled_data["disease", "associates_with", "lncrna"].edge_label
#             pred_prob = torch.sigmoid(pred)
#             loss = F.binary_cross_entropy(pred_prob.squeeze(), ground_truth)

#             loss.backward()
#             optimizer.step()

#             total_loss += float(loss) * pred_prob.numel()
#             total_examples += pred_prob.numel()

#         # Append the current training loss to the list
#         fold_loss.append(total_loss / total_examples)
#         fold_bar.update(1)
#         fold_bar.set_description(f"Fold {fold + 1}/{num_folds}, Epoch {epoch}/{num_epochs}, Loss: {total_loss / total_examples:.4f}")
        
#         # Calculate and store the performance metric for learning curve (you should replace this with your metric)
#         performance_metric = 1.0 - (total_loss / total_examples)
#         learning_curve.append(performance_metric)
        
#         scheduler.step()  # Adjust the learning rate based on the scheduler

#     fold_bar.close()  # Close the progress bar for the current fold
#     fold_loss_values.append(fold_loss)

# # Save the final model
# torch.save(model.state_dict(), 'cv_trained_model.pth')

# # Plotting the training loss, learning, and learning rate curves for each fold
# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

# # Plot training loss curves
# for fold in range(num_folds):
#     ax1.plot(range(1, num_epochs + 1), fold_loss_values[fold], label=f'Fold {fold + 1}')

# ax1.set_title('Training Loss Curves for 10 Folds')
# ax1.set_xlabel('Epochs')
# ax1.set_ylabel('Training Loss')
# ax1.legend()

# # Plot learning curves (you should replace this with your actual metric values)
# ax2.plot(range(1, num_epochs * num_folds + 1), learning_curve, label='Learning Curve', color='green')

# ax2.set_title('Learning Curve')
# ax2.set_xlabel('Epochs')
# ax2.set_ylabel('Performance Metric')
# ax2.legend()

# # Plot learning rates
# ax3.plot(range(1, (num_epochs * num_folds) + 1), learning_rates, label='Learning Rate', color='blue')

# ax3.set_title('Learning Rate Schedule')
# ax3.set_xlabel('Epochs')
# ax3.set_ylabel('Learning Rate')
# ax3.legend()
# plt.savefig('Training_LC_LC_LRS500dpi.png', dpi=500)

# plt.show()


# In[33]:


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

num_folds = 5
num_epochs = 500

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


# In[ ]:





# In[34]:


# Set Seaborn style without gridlines
sns.set(style="white")

# Generate the calibration curve data
prob_true, prob_pred = calibration_curve(true_labels, predicted_probs, n_bins=10, strategy='uniform')

# Create a figure with three horizontally arranged subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot training loss curves
for fold in range(num_folds):
    axes[0].plot(range(1, num_epochs + 1), fold_loss_values[fold], label=f'Fold {fold + 1}')

axes[0].set_title('Training Loss Curves')
axes[0].set_xlabel('Epochs')
axes[0].set_ylabel('Training Loss')
axes[0].legend()

# Plot learning curves and learning rates
axes[1].plot(range(1, num_epochs * num_folds + 1), learning_curve, label='Learning Curve', color='green')
axes[1].plot(range(1, (num_epochs * num_folds) + 1), learning_rates, label='Learning Rate', color='blue')

axes[1].set_title('Learning Curve and Learning Rate Schedule')
axes[1].set_xlabel('Epochs')
axes[1].set_ylabel('Performance Metric / Learning Rate')
axes[1].legend()

# Plot calibration curve
axes[2].plot(prob_pred, prob_true, marker='o', label='Calibration Curve', color='b')
axes[2].plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
axes[2].set_xlabel('Mean Predicted Probability')
axes[2].set_ylabel('Observed Probability')
axes[2].set_title('Calibration Curve')
axes[2].legend()

# Adjust the spacing between subplots
plt.subplots_adjust(wspace=0.3)

# Save or display the figure
plt.savefig('3in1_Plots_Horizontal_500dpi.png', dpi=500)
plt.show()


# In[35]:


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


# In[36]:


# After 10-fold CV, the model is trained with entire training data and tested on val and test data
# Reassigning data loaders for the final training, validation, and testing data
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
plt.savefig('final_training_and_validation_loss_curve.png', dpi=500)
plt.show()


# In[37]:


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
            tv_preds.append(model(sampled_data))
            tv_ground_truths.append(sampled_data["disease", "associates_with", "lncrna"].edge_label)
    tv_preds = torch.cat(tv_preds, dim=0).cpu().numpy()
    tv_ground_truths = torch.cat(tv_ground_truths, dim=0).cpu().numpy()
    # tv_auc = roc_auc_score(tv_ground_truths, tv_preds)
    
    # plotting AUC Curve
    binary_ground_truths = np.array([1 if label == 2.0 or label == 1.0 else 0 for label in tv_ground_truths]) # converting ground truth values to {0, 1}
    
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
    plt.savefig('AUC.png', dpi=500)
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
    plt.savefig('AUPRC.png', dpi=500)
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
    
    return roc_auc


# In[38]:


# Evaluate the final model on evaluation data
# print("Validation performance")
# evaluation_roc_auc = train_val_accuracy(final_val_loader)

# Evaluate the final model on test data
print("Testing performance")
test_roc_auc = test_val_accuracy(final_test_loader)


# In[ ]:





# In[39]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, confusion_matrix
import numpy as np

# Assuming 'final_test_loader' contains the test data
test_predictions = []
true_labels = []

cv_trained_model.eval()  # Set the model to evaluation mode

with torch.no_grad():
    for sampled_data in final_test_loader:
        sampled_data = sampled_data.to(device)
        pred = cv_trained_model(sampled_data)
        pred_prob = torch.sigmoid(pred).squeeze().cpu().numpy()
        test_predictions.extend(pred_prob)
        true_labels.extend(sampled_data["disease", "associates_with", "lncrna"].edge_label.cpu().numpy())

# ROC Curve
fpr, tpr, _ = roc_curve(true_labels, test_predictions)
plt.figure(figsize=(12, 4))

# Subplot for ROC Curve
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Curve')
plt.legend()

# Subplot for Histogram of Predicted Probabilities
plt.subplot(1, 2, 2)
plt.hist(test_predictions, bins=20, color='lightblue', edgecolor='black')
plt.xlabel('Predicted Probabilities')
plt.ylabel('Frequency')
plt.title('Histogram of Predicted Probabilities')

plt.tight_layout()  # Adjust layout to prevent overlapping
plt.show()

# Confusion Matrix
conf_matrix = confusion_matrix(true_labels, np.round(test_predictions))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()


# In[40]:


import seaborn as sns
import matplotlib.pyplot as plt
import torch
import numpy as np

# Set a professional style
sns.set(style="white", font_scale=1.2)

# Assuming 'final_test_loader' contains the test data
test_predictions = []
true_labels = []

cv_trained_model.eval()  # Set the model to evaluation mode

with torch.no_grad():
    for sampled_data in final_test_loader:
        sampled_data = sampled_data.to(device)
        pred = cv_trained_model(sampled_data)
        pred_prob = torch.sigmoid(pred).squeeze().cpu().numpy()
        test_predictions.extend(pred_prob)
        true_labels.extend(sampled_data["disease", "associates_with", "lncrna"].edge_label.cpu().numpy())

# Calculate Cumulative Gain manually
sorted_indices = np.argsort(test_predictions)[::-1]
sorted_labels = np.array(true_labels)[sorted_indices]

cumulative_positives = np.cumsum(sorted_labels)
total_positives = np.sum(true_labels)

percentages = np.arange(1, len(true_labels) + 1) / len(true_labels)
cumulative_gains = cumulative_positives / total_positives

# Calculate Lift manually
lift = cumulative_gains / percentages

# Plotting Cumulative Gain Curve, Lift Curve, and Histogram of Predicted Probabilities in a single row
plt.figure(figsize=(18, 6))

# Subplot for Cumulative Gain Curve
plt.subplot(1, 3, 1)
sns.lineplot(x=percentages, y=cumulative_gains, color='darkgreen', lw=2, label='Cumulative Gain Curve')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('Percentage of Samples')
plt.ylabel('Cumulative Gain')
plt.title('Cumulative Gain Curve')
plt.legend()

# Subplot for Lift Curve
plt.subplot(1, 3, 2)
sns.lineplot(x=percentages, y=lift, color='darkred', lw=2, label='Lift Curve')
plt.xlabel('Percentage of Samples')
plt.ylabel('Lift')
plt.title('Lift Curve')
plt.legend()

# Subplot for Histogram of Predicted Probabilities
plt.subplot(1, 3, 3)
sns.histplot(test_predictions, bins=20, color='#3498db', edgecolor='black', kde=True)
plt.xlabel('Predicted Probabilities')
plt.ylabel('Frequency')
plt.title('Histogram of Predicted Probabilities')

plt.tight_layout()  # Adjust layout to prevent overlapping
plt.suptitle('Model Evaluation Metrics', y=1.02, fontsize=16)
plt.savefig('gain_lift_histogram_seaborn_500dpi.png', dpi=500)  # Save or display the figure
plt.show()


# In[ ]:




