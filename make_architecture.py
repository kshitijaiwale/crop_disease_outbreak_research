from graphviz import Digraph

g = Digraph(format='png')
g.attr(rankdir='LR', fontsize='12')

# Inputs
g.node('W', 'Weather Sequence\n(28 days)\nRH2M, T2M, Rain')
g.node('A', 'Agronomic Inputs\nVariety, Age,\nRatoon, NDVI')

# Weather branch (TCN)
g.node('T1', 'TCN Layer 1\n(dilation=1)')
g.node('T2', 'TCN Layer 2\n(dilation=2)')
g.node('T3', 'TCN Layer 3\n(dilation=4)')
g.node('Ew', 'Temporal Embedding')

# Agronomic branch (MLP)
g.node('M1', 'MLP Layer 1')
g.node('M2', 'MLP Layer 2')
g.node('Ea', 'Agro Embedding')

# Knowledge modulation
g.node('KG', 'Knowledge Graph\n(Modulation Weights)')

# Fusion & output
g.node('F', 'Fusion + Attention')
g.node('O', 'Risk Score\n(0–1)')

# Edges
g.edge('W', 'T1'); g.edge('T1', 'T2'); g.edge('T2', 'T3'); g.edge('T3', 'Ew')
g.edge('A', 'M1'); g.edge('M1', 'M2'); g.edge('M2', 'Ea')
g.edge('KG', 'F')
g.edge('Ew', 'F'); g.edge('Ea', 'F')
g.edge('F', 'O')

g.render('figure2_architecture', cleanup=True)