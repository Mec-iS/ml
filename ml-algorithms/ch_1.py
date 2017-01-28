import scipy as sp

data = sp.genfromtxt("web_traffic.tsv", delimiter="\t")

print(data[:10])