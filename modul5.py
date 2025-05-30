import networkx as nx
import matplotlib.pyplot as plt

# Buat graf kosong
G = nx.Graph()

# Tambahkan simpul (nodes) dan sisi (edges)
G.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'C'), ('C', 'D'), ('D', 'A')])

# Gambar graf
plt.figure(figsize=(6, 4))
nx.draw(G, with_labels=True, node_color='aqua', edge_color='mediumpurple', node_size=800)
plt.title("Representasi Graf")
plt.show()

# Tampilkan adjacency list
print("Adjacency List:")
for node in G.adj:
  print(f"{node}: {list(G.adj[node])}")
