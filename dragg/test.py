
f = "data/nsrdb.csv"

# df_all = pd.read_excel(f, sheet_name=None)
# k1 = list(df_all.keys())[0]
# df = df_all[k1]
# for k, v in df_all.items():
#     if k == k1:
#         pass
#     else:
#         df = df.append(v, ignore_index=True)
#
# df = df[df["Settlement Point"] == "LZ_HOUSTON"]
# df = df.rename(columns={"Settlement Point Price": "SPP"})
# df["Hour Ending"] = df["Hour Ending"].str.replace(':00', '')
# df["Hour Ending"] = df["Hour Ending"].apply(pd.to_numeric)
# df["Hour Ending"] = df["Hour Ending"].apply(lambda x: x - 1)
# df["Hour Ending"] = df["Hour Ending"].astype(str)
# df["SPP"] = df["SPP"].apply(lambda x: x / 1000)
# df['ts'] = df[["Delivery Date", "Hour Ending"]].apply(lambda x: ' '.join(x), axis=1)
# df = df.drop(columns=['Delivery Date', 'Hour Ending', 'Repeated Hour Flag', 'Settlement Point'])
# col_order=["ts", "SPP"]
# df = df[col_order]
# df[["ts"]] = df.loc[:, "ts"].apply(lambda x: datetime.strptime(x, '%m/%d/%Y %H'))
# df = df.reset_index(drop=True)

# df = pd.read_csv(f, skiprows=2)
# df = df[df["Minute"] == 0]
# df = df.astype(str)
# df['ts'] = df[["Year", "Month", "Day", "Hour", "Minute"]].apply(lambda x: ' '.join(x), axis=1)
# df = df.rename(columns={"Temperature": "OAT"})
# df[["ts"]] = df.loc[:, "ts"].apply(lambda x: datetime.strptime(x, '%Y %m %d %H %M'))
# df = df.filter(["ts", "GHI", "OAT"])
# df[["GHI", "OAT"]] = df[["GHI", "OAT"]].astype(int)

# h = {
#         "name": "Kathleen",
#         "type": "pv_battery",
#         "hvac": {
#             "r": 7.169990821711213,
#             "c": 4.827825470364611,
#             "p_c": 3.5,
#             "p_h": 3.5
#         },
#         "wh": {
#             "r": 23.13817396069954,
#             "c": 4.714463178116934,
#             "p": 2.5
#         },
#         "battery": {
#             "max_rate": 5,
#             "capacity": 13.5,
#             "capacity_lower": 2.025,
#             "capacity_upper": 11.475,
#             "ch_eff": 0.95,
#             "disch_eff": 0.99
#         },
#         "pv": {
#             "area": 32,
#             "eff": 0.2
#         }
#     }

from redis import Redis

r = Redis()
