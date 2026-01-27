import pandas as pd
import seaborn as sns

yarn = pd.read_csv("https://raw.githubusercontent.com/gwynniehayes/CS200B/refs/heads/main/yarn.csv")

yarn = pd.DataFrame(yarn)

yarn.describe(include = "all")

filtered_yarn = yarn[yarn['discontinued'] != "FALSE"]

filtered_yarn.describe(include = "all")

filter_yarn = yarn.query("discontinued == 'FALSE'")
