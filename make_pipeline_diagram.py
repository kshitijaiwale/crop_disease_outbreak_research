from graphviz import Digraph

g = Digraph(format='png')
g.attr(rankdir='TB', fontsize='12')

# Modules
g.node('In', 'Data Ingestion\n(NASA POWER API)')
g.node('FP', 'Feature Pipeline\n(Lag, Rolling, Z-Score)')
g.node('PE', 'Prediction Engine\n(KG-CTCN Inference)')
g.node('DB', 'Persistence Layer\n(SQLite DB)')
g.node('BF', 'Backfill System\n(Batch Inference)')
g.node('VI', 'Visualization\n(Matplotlib / Plotly)')

# Flow
g.edge('In', 'FP')
g.edge('FP', 'PE')
g.edge('PE', 'DB')
g.edge('DB', 'VI')
g.edge('In', 'BF')
g.edge('BF', 'DB')

g.render('paper/figure3_pipeline', cleanup=True)
