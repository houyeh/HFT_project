# FI-2010 Data Manifest

Status: official FI-2010 archive was already available locally; this phase started from extraction and verification, with no download step.

## Official Dataset

- Dataset: Benchmark Dataset for Mid-Price Forecasting of Limit Order Book Data with Machine Learning Methods.
- Dataset ID: `73eb48d7-4dbc-4a10-a52a-da745b47a649`.
- Repository: Fairdata/Etsin.
- Dataset page: <https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649>.
- Paper: Ntakaris, Magris, Kanniainen, Gabbouj, and Iosifidis (2018), Journal of Forecasting.
- License reported by Fairdata metadata: CC BY 4.0.

## Local Archive Used

- Local path: `data\raw\BenchmarkDatasets.zip`.
- Acquisition note: the zip was already in hand before this Stage 1 run; the script does not download data and raises if the archive is missing.
- Size bytes: `1864361899`.
- SHA256: `cea93692a270724fa91e8f124da641db727d757e5e0f0bb85067709e9932f664`.
- Expected size bytes from Fairdata metadata: `1864361899`.
- Expected SHA256 from Fairdata metadata: `cea93692a270724fa91e8f124da641db727d757e5e0f0bb85067709e9932f664`.

## Extracted Files Used In Stage 1

| cache | source | size bytes | sha256 | events | features |
|---|---|---:|---|---:|---:|
| `train_cf7` | `data\extracted\BenchmarkDatasets\NoAuction\1.NoAuction_Zscore\NoAuction_Zscore_Training\Train_Dst_NoAuction_ZScore_CF_7.txt` | 607324298 | `0d2839ff7575f0910a03fd782edc6b3a561d544b07cf2329ebe7f33c8affe6c6` | 254750 | 40 |
| `test_cf7` | `data\extracted\BenchmarkDatasets\NoAuction\1.NoAuction_Zscore\NoAuction_Zscore_Testing\Test_Dst_NoAuction_ZScore_CF_7.txt` | 132259850 | `96ee1a73a581edfa46ffcbdbd41104746664cb4fc112ac841344f5c78ebab492` | 55478 | 40 |
| `test_cf8` | `data\extracted\BenchmarkDatasets\NoAuction\1.NoAuction_Zscore\NoAuction_Zscore_Testing\Test_Dst_NoAuction_ZScore_CF_8.txt` | 124378346 | `7a4b787876a667aeaf50dae37f144eff3b42c9307f2f5af8d7cdbfd630c4c8a0` | 52172 | 40 |
| `test_cf9` | `data\extracted\BenchmarkDatasets\NoAuction\1.NoAuction_Zscore\NoAuction_Zscore_Testing\Test_Dst_NoAuction_ZScore_CF_9.txt` | 76138106 | `b90a4aa63010662c89003134f0dbaa08c7ee14d36433073eb335d5478fb0a4e9` | 31937 | 40 |

## NoAuction/Zscore Members In Official Zip

Only `NoAuction/1.NoAuction_Zscore` is used. Other FI-2010 variants remain untouched.

| member | uncompressed bytes | compressed bytes | crc |
|---|---:|---:|---|
| `BenchmarkDatasets/NoAuction/1.NoAuction_Zscore/` | 0 | 0 | `00000000` |
| `BenchmarkDatasets/NoAuction/1.NoAuction_Zscore/NoAuction_Zscore_Testing/` | 0 | 0 | `00000000` |
| `BenchmarkDatasets/NoAuction/1.NoAuction_Zscore/NoAuction_Zscore_Testing/Test_Dst_NoAuction_ZScore_CF_1.txt` | 91538746 | 6130114 | `911f34fa` |
| `BenchmarkDatasets/NoAuction/1.NoAuction_Zscore/NoAuction_Zscore_Testing/Test_Dst_NoAuction_ZScore_CF_2.txt` | 68027738 | 4303060 | `fb3ab467` |
| `BenchmarkDatasets/NoAuction/1.NoAuction_Zscore/NoAuction_Zscore_Testing/Test_Dst_NoAuction_ZScore_CF_3.txt` | 88263130 | 5891672 | `223f391b` |
| `BenchmarkDatasets/NoAuction/1.NoAuction_Zscore/NoAuction_Zscore_Testing/Test_Dst_NoAuction_ZScore_CF_4.txt` | 82927738 | 5599340 | `f2d90419` |
| `BenchmarkDatasets/NoAuction/1.NoAuction_Zscore/NoAuction_Zscore_Testing/Test_Dst_NoAuction_ZScore_CF_5.txt` | 93338666 | 5875382 | `1b6682b8` |
| `BenchmarkDatasets/NoAuction/1.NoAuction_Zscore/NoAuction_Zscore_Testing/Test_Dst_NoAuction_ZScore_CF_6.txt` | 89033162 | 5666570 | `840bac2e` |
| `BenchmarkDatasets/NoAuction/1.NoAuction_Zscore/NoAuction_Zscore_Testing/Test_Dst_NoAuction_ZScore_CF_7.txt` | 132259850 | 7442026 | `81a4386d` |
| `BenchmarkDatasets/NoAuction/1.NoAuction_Zscore/NoAuction_Zscore_Testing/Test_Dst_NoAuction_ZScore_CF_8.txt` | 124378346 | 7055435 | `33411095` |
| `BenchmarkDatasets/NoAuction/1.NoAuction_Zscore/NoAuction_Zscore_Testing/Test_Dst_NoAuction_ZScore_CF_9.txt` | 76138106 | 4587693 | `f77066df` |
| `BenchmarkDatasets/NoAuction/1.NoAuction_Zscore/NoAuction_Zscore_Training/` | 0 | 0 | `00000000` |
| `BenchmarkDatasets/NoAuction/1.NoAuction_Zscore/NoAuction_Zscore_Training/Train_Dst_NoAuction_ZScore_CF_1.txt` | 94196906 | 5986185 | `a5aa8bc2` |
| `BenchmarkDatasets/NoAuction/1.NoAuction_Zscore/NoAuction_Zscore_Training/Train_Dst_NoAuction_ZScore_CF_2.txt` | 185735354 | 12064573 | `3570f6de` |
| `BenchmarkDatasets/NoAuction/1.NoAuction_Zscore/NoAuction_Zscore_Training/Train_Dst_NoAuction_ZScore_CF_3.txt` | 253762794 | 16323034 | `46d75b0d` |
| `BenchmarkDatasets/NoAuction/1.NoAuction_Zscore/NoAuction_Zscore_Training/Train_Dst_NoAuction_ZScore_CF_4.txt` | 342025626 | 22191727 | `c0dcb7a1` |
| `BenchmarkDatasets/NoAuction/1.NoAuction_Zscore/NoAuction_Zscore_Training/Train_Dst_NoAuction_ZScore_CF_5.txt` | 424953066 | 27735287 | `ac7d3f16` |
| `BenchmarkDatasets/NoAuction/1.NoAuction_Zscore/NoAuction_Zscore_Training/Train_Dst_NoAuction_ZScore_CF_6.txt` | 518291434 | 33549111 | `b0108750` |
| `BenchmarkDatasets/NoAuction/1.NoAuction_Zscore/NoAuction_Zscore_Training/Train_Dst_NoAuction_ZScore_CF_7.txt` | 607324298 | 38900173 | `80fca29c` |
| `BenchmarkDatasets/NoAuction/1.NoAuction_Zscore/NoAuction_Zscore_Training/Train_Dst_NoAuction_ZScore_CF_8.txt` | 739583850 | 46589931 | `bb8635b6` |
| `BenchmarkDatasets/NoAuction/1.NoAuction_Zscore/NoAuction_Zscore_Training/Train_Dst_NoAuction_ZScore_CF_9.txt` | 863961898 | 53620625 | `84f6f457` |

## Protocol Notes

- Setup 2 uses `Train_Dst_NoAuction_ZScore_CF_7.txt` for the first 7 days.
- The last 3 days are represented by the concatenated testing files `CF_7`, `CF_8`, and `CF_9`.
- The first 40 rows are used as raw LOB features; the last five rows are official labels.
- No project-internal data and no NEW1-NEW5 split were read or used.

Generated by `01_prepare_fi2010_stage1.py`.
