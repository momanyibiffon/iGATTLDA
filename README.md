# iGATTLDA
Prediction of lncRNA-disease associations based on graph attention network (GAT) and Transformer.
lncRNA, disease similarity data, and the adjacency matrix are utilized to generate a heterogeneous network.
The heterogeneous network facilitates feature aggregation for both the GAT and transformer.
GAT and transformer are responsible for feature aggregation within local and global feature spaces, respectively.
An MLP classifier was then employed for prediction utilizing the rich node embeddings.
The result was a higher prediction performance than prior methods, especially based on the AUC.
A case study was conducted on the top predictions on selected diseases to verify the model prediction performance.
