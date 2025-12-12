# ===============================
# 설정 (PYUPBIT)
# ===============================
markets = ["KRW-ETH", "USDT-ETH", "BTC-ETH"]
interval = "day"
start_date = "2021-01-01"
today = datetime.now().strftime("%Y-%m-%d")

# ===============================
# PyUpbit 데이터 수집 함수
# ===============================
def fetch_upbit_data(ticker, interval, count=200, to=None):
    """Upbit에서 OHLCV 데이터 가져오기"""
    df = pyupbit.get_ohlcv(ticker, interval=interval, count=count, to=to)
    return df

# ===============================
# 마켓별 CSV 저장 및 업데이트
# ===============================
for market in markets:
    csv_file = f"{market.replace('-', '_')}_ohlcv.csv"
    
    if not os.path.exists(csv_file):
        print(f"[{market}] CSV가 없어 새로 생성합니다. ({start_date}부터)")
        df_list = []
        current_end = None
        
        while True:
            df_temp = fetch_upbit_data(market, interval, count=200, to=current_end)
            if df_temp is None or df_temp.empty:
                break
            
            df_list.append(df_temp)
            current_end = df_temp.index[0].strftime("%Y-%m-%d")
            print(f"[{market}] 수집 완료: {len(df_temp)}행")
            
            if current_end <= start_date:
                break
        
        df = pd.concat(df_list).sort_index()
        df = df[df.index >= pd.to_datetime(start_date)]
        df = df[~df.index.duplicated(keep='first')]
        df.to_csv(csv_file)
        print(f"[{market}] CSV 저장 완료, 총 {len(df)}행")
    
    else:
        print(f"[{market}] 기존 CSV가 존재합니다. 마지막 이후 데이터만 추가")
        df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
        last_time = df.index[-1]
        df_new_list = []
        current_end = last_time
        
        while True:
            df_temp = fetch_upbit_data(market, interval, count=200, to=current_end)
            if df_temp is None or df_temp.empty:
                break
            
            df_temp = df_temp[df_temp.index > last_time]
            if df_temp.empty:
                break
            
            df_new_list.append(df_temp)
            current_end = df_temp.index[0].strftime("%Y-%m-%d")
            print(f"[{market}] 새로운 데이터 수집: {len(df_temp)}행")
            
            if len(df_temp) < 200:
                break
        
        if df_new_list:
            df_new = pd.concat(df_new_list).sort_index()
            df = pd.concat([df, df_new]).sort_index()
            df = df[~df.index.duplicated(keep='first')]
            df.to_csv(csv_file)
            print(f"[{market}] CSV 업데이트 완료: {len(df_new)}행 추가, 총 {len(df)}행")
        else:
            print(f"[{market}] 새로운 데이터 없음")

print("전체 마켓 데이터 수집 완료")
