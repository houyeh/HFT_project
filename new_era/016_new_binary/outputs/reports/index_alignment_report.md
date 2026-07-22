# Index Alignment Report

Pipeline alignment:
1. raw LOB/trade -> feature frame
2. dropna on feature frame
3. rolling zscore (causal)
4. windowing with window_size=100, label index=i=end-1
5. horizon lookahead for markout (future i+H)
6. fill proxy conditioning

- NEW raw rows aggregated: 5,323,560
- NEW rows after feature/dropna: 5,323,560
- NEW3 raw rows aggregated: 5,658,022
- NEW3 rows after feature/dropna: 5,658,022

## bid primary dataset counts
- train: before=718,556, after=15,643, fill_proxy_rate=0.021770
- val: before=563,484, after=12,603, fill_proxy_rate=0.022366
- test: before=1,166,323, after=24,329, fill_proxy_rate=0.020860
- new3: before=5,657,368, after=119,809, fill_proxy_rate=0.021178

## ask primary dataset counts
- train: before=718,556, after=15,856, fill_proxy_rate=0.022066
- val: before=563,484, after=13,314, fill_proxy_rate=0.023628
- test: before=1,166,323, after=25,037, fill_proxy_rate=0.021467
- new3: before=5,657,368, after=119,183, fill_proxy_rate=0.021067
