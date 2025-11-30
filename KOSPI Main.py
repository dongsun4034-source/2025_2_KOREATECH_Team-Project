import os
import math
import warnings
import threading
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, Callback

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import tkinter as tk
from tkinter import ttk, messagebox

# matplotlib (Tkinter에 그래프 임베드)
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
from matplotlib import font_manager, rc
from matplotlib.dates import DateFormatter, DayLocator

warnings.filterwarnings('ignore')
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False  # 마이너스 깨짐 방지


# ---------------------------------------------------------
# 1. 데이터 불러오기 및 전처리
# ---------------------------------------------------------

# 1-1. 데이터 가공 - 데이터셋 날짜 형식 통일
# 날짜 데이터에 포함된 시간(예: 오후 4:45:00) 제거 및 문자열 변환
def clean_date_series(series):
    # 년월일만 추출, 공백 기준으로 3개 자른 뒤, 다시 결합
    return series.astype(str).apply(lambda x: ' '.join(x.split(' ')[:3]))


# 1-2. CSV 파일 읽기 + 날짜 인덱스 설정
# 여러 인코딩 시도 후 Date를 인덱스로 설정하고, 필요한 컬럼 이름을 통일
def load_data(filename, col_name=None):
    if not os.path.exists(filename):
        return None

    try:
        df = pd.read_csv(filename, encoding='utf-8')
    except:
        try:
            df = pd.read_csv(filename, encoding='cp949')
        except:
            df = pd.read_csv(filename, encoding='euc-kr')

    # 날짜 포맷 통일 후 인덱스 설정
    df['Date'] = pd.to_datetime(clean_date_series(df['Date']), format='%Y. %m. %d')
    df.set_index('Date', inplace=True)

    # 종가/환율 컬럼 이름 통일
    if col_name:
        if 'Close' in df.columns:
            df = df[['Close']].rename(columns={'Close': col_name})
        elif 'Rate' in df.columns:
            df = df[['Rate']].rename(columns={'Rate': col_name})
    return df


# 1-3. Sliding Window 데이터셋 구축
# 과거 time_step일의 입력으로 다음 날(target_col_idx)을 예측하는 학습 데이터 생성
def create_dataset(dataset, target_col_idx, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), :]       # 과거 time_step일간의 모든 입력 변수(X)
        dataX.append(a)
        dataY.append(dataset[i + time_step, target_col_idx])  # 다음날(31일째)의 KOSPI 종가(즉, 등락률 y)를 타겟으로 설정
    return np.array(dataX), np.array(dataY)


# 1-4. 예측된 수익률을 실제 지수로 복원
# 전일 지수 * (1 + 예측 수익률)을 이용하여 KOSPI 지수로 변환
def reconstruct_price(start_idx, predicted_returns, real_prices, time_step):
    reconstructed = []
    for i in range(len(predicted_returns)):
        prev_price = real_prices[start_idx + time_step + i]
        pred_return = predicted_returns[i][0]
        pred_price = prev_price * (1 + pred_return)
        reconstructed.append(pred_price)
    return np.array(reconstructed)


# ---------------------------------------------------------
# 2. 학습 진행률 표시 콜백 (Epoch Progress)
# ---------------------------------------------------------

class EpochProgressCallback(Callback):
    def __init__(self, progress_fn, max_epochs):
        super().__init__()
        self.progress_fn = progress_fn
        self.max_epochs = max_epochs

    def on_train_begin(self, logs=None):
        if self.progress_fn is not None:
            self.progress_fn(0, self.max_epochs)

    def on_epoch_end(self, epoch, logs=None):
        if self.progress_fn is not None:
            self.progress_fn(epoch + 1, self.max_epochs)


# ---------------------------------------------------------
# 3. LSTM 학습 파이프라인 (데이터 전처리 ~ 예측/시뮬레이션)
# ---------------------------------------------------------

def run_full_demo(log_fn=None, progress_fn=None, max_epochs=100):
    """
    GUI와 분리된 핵심 처리 함수
    - log_fn : 로그 출력 함수 (텍스트 박스로 전달)
    - progress_fn : Epoch 진행률 업데이트 함수
    """

    # 3-0. 로그 출력용 내부 함수
    def log(msg):
        if log_fn is not None:
            log_fn(msg)

    log("[알림] 데모 실행을 시작합니다. (시간이 다소 걸릴 수 있습니다)")
    log("[정보] GPU 장치 확인 중...")
    log(str(tf.config.list_physical_devices('GPU')))

    # -----------------------------------------------------
    # 3-1. 데이터 불러오기
    # -----------------------------------------------------
    base_dir = os.path.dirname(os.path.abspath(__file__))
    kospi_path = os.path.join(base_dir, "Kospi.csv")
    nasdaq_path = os.path.join(base_dir, "NASDAQ.csv")
    snp_path = os.path.join(base_dir, "SnP500.csv")
    exch_path = os.path.join(base_dir, "Exchange.csv")

    df_kospi = load_data(kospi_path, 'Kospi')
    df_nasdaq = load_data(nasdaq_path, 'Nasdaq')
    df_snp = load_data(snp_path, 'SnP')

    # 환율 데이터는 형식이 달라 별도 로드
    try:
        df_exch = pd.read_csv(exch_path, header=None, names=['Date', 'Rate'], encoding='cp949')
    except:
        df_exch = pd.DataFrame()
    if not df_exch.empty:
        df_exch['Date'] = pd.to_datetime(clean_date_series(df_exch['Date']), format='%Y. %m. %d')
        df_exch.set_index('Date', inplace=True)
        df_exch = df_exch.rename(columns={'Rate': 'Exch'})

    # -----------------------------------------------------
    # 3-2. 결측치 처리 및 데이터 병합
    # -----------------------------------------------------
    # 미국 휴장일 등으로 비어있는 데이터는 가장 최근 데이터로 채움 (ffill)
    dfs = [df_nasdaq, df_snp, df_exch]
    df_merged = df_kospi.copy()

    for df in dfs:
        if df is not None and not df.empty:
            df_merged = df_merged.join(df, how='left')
    # Forward Fill 적용 (직전 거래일 데이터로 채우기)
    df_merged = df_merged.ffill()
    df_merged = df_merged.dropna()

    log(f"[1단계] 전처리 완료된 데이터 크기: {df_merged.shape}")
    log(f"[1단계] 누락된 데이터(NaN) 개수: {df_merged.isnull().sum().sum()}")

    # -----------------------------------------------------
    # 3-3. 변화율(Return) 계산
    # -----------------------------------------------------
    # 변화율(Return)로 변환 (학습 안정성 확보)
    df_returns = df_merged.pct_change().dropna()

    # -----------------------------------------------------
    # 3-4. 데이터 정규화 (Min-Max Scaling 0~1)
    # -----------------------------------------------------
    # Min-Max Scaling: 0~1
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df_returns.values)

    scaler_y = MinMaxScaler(feature_range=(0, 1))  # 타겟(KOSPI)용 스케일러 별도 저장 (가격 복원용)
    scaler_y.fit(df_returns[['Kospi']].values)

    # 학습/테스트 데이터 분할 (80:20)
    training_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[0:training_size, :]
    test_data = scaled_data[training_size:len(scaled_data), :]

    # Sliding Window 길이 (30일)
    time_step = 30
    # 학습용 데이터셋 생성(3차원 배열로 변환)
    X_train, y_train = create_dataset(train_data, 0, time_step)
    X_test, y_test = create_dataset(test_data, 0, time_step)

    log(f"[2단계] 학습용 데이터셋 형태: {X_train.shape}")  # (샘플수, 30, 4)

    # -----------------------------------------------------
    # 3-5. 모델 구조 정의 및 학습 설정
    # -----------------------------------------------------
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(time_step, 4)))
    model.add(Dropout(0.2))
    model.add(LSTM(32, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))  # 다음날 수익률 1개

    model.compile(optimizer='adam', loss='mean_squared_error')

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    progress_cb = EpochProgressCallback(progress_fn, max_epochs=max_epochs)

    log(f"[3단계] 모델 학습 시작 (최대 {max_epochs} epoch, EarlyStopping 적용)...")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=max_epochs,
        batch_size=32,
        callbacks=[early_stop, progress_cb],
        verbose=1
    )

    log("[3단계] 모델 학습 완료!")

    # -----------------------------------------------------
    # 3-6. 예측 결과 복원(역정규화) 및 RMSE 평가
    # -----------------------------------------------------
    real_prices = df_merged['Kospi'].values

    pred_return_train = scaler_y.inverse_transform(model.predict(X_train))
    pred_return_test = scaler_y.inverse_transform(model.predict(X_test))

    train_reconstructed = reconstruct_price(0, pred_return_train, real_prices, time_step)
    test_reconstructed = reconstruct_price(training_size, pred_return_test, real_prices, time_step)

    train_Y_real = real_prices[time_step + 1: time_step + 1 + len(train_reconstructed)]
    test_Y_real = real_prices[training_size + time_step + 1:
                              training_size + time_step + 1 + len(test_reconstructed)]

    train_rmse = math.sqrt(mean_squared_error(train_Y_real, train_reconstructed))
    test_rmse = math.sqrt(mean_squared_error(test_Y_real, test_reconstructed))

    log("\n[최종 성능 평가]")
    log(f"Train RMSE: {train_rmse:.2f}")
    log(f"Test  RMSE: {test_rmse:.2f}")

    # -----------------------------------------------------
    # 3-8. 변동성 반영 미래 예측 시나리오 (Dec 1~5) - Simulation A
    # -----------------------------------------------------
    daily_volatility = np.std(df_returns['Kospi'])

    # B = Business Day (평일만)
    target_dates_mc = pd.date_range(start='2025-12-01', end='2025-12-05', freq='B')
    last_real_price_mc = df_merged['Kospi'].iloc[-1]

    def run_simulation(noise_level=0):
        prices = []
        curr_batch = scaled_data[-time_step:].reshape(1, time_step, 4)
        curr_price = last_real_price_mc

        for date in target_dates_mc:
            pred_scaled = model.predict(curr_batch, verbose=0)
            base_return = scaler_y.inverse_transform(pred_scaled)[0][0]

            shock = np.random.normal(0, daily_volatility) * noise_level
            final_return = base_return + shock

            pred_price = curr_price * (1 + final_return)
            prices.append(pred_price)
            curr_price = pred_price

            final_return_scaled = scaler_y.transform([[final_return]])
            last_exog = curr_batch[0, -1, 1:]
            new_row = np.concatenate([final_return_scaled[0], last_exog]).reshape(1, 1, 4)
            curr_batch = np.append(curr_batch[:, 1:, :], new_row, axis=1)

        return prices

    np.random.seed(42)
    simA_prices = run_simulation(noise_level=1.0)
    scenarios = {'Simulation A': simA_prices}

    df_combined_mc = df_merged[['Kospi']].copy()
    zoom_start_mc = '2025-11-01'
    actual_zoomed_mc = df_combined_mc.loc[zoom_start_mc:]

    # -----------------------------------------------------
    # 3-9. 그래프/표에 필요한 데이터 정리 후 반환
    # -----------------------------------------------------
    plot_data = {
        "dates_all": df_merged.index,
        "real_prices": real_prices,
        "train_reconstructed": train_reconstructed,
        "test_reconstructed": test_reconstructed,
        "training_size": training_size,
        "time_step": time_step,
        "mc_actual_dates": actual_zoomed_mc.index,
        "mc_actual_prices": actual_zoomed_mc['Kospi'].values,
        "mc_scenarios": scenarios,          # Simulation A만 포함
        "mc_target_dates": target_dates_mc
    }

    # 2025년 12월 1~5일 예측 결과 (표 출력용) - Simulation A 기준
    result_rows = []
    prev_price = last_real_price_mc
    for d, p in zip(target_dates_mc, simA_prices):
        ret_pct = (p / prev_price - 1.0) * 100.0
        result_rows.append((d.date().isoformat(), p, ret_pct))
        prev_price = p

    return result_rows, plot_data


# ---------------------------------------------------------
# 4. Tkinter GUI 구성
# ---------------------------------------------------------

def main():
    root = tk.Tk()
    root.title("KOSPI 예측 데모 프로그램")
    root.geometry("1200x820")

    MAX_EPOCHS = 100

    # 4-0. 그래프/플롯 상태 저장 변수
    state = {
        "plot_data": None,
        "fig": None,
        "ax": None,
        "canvas": None
    }

    # -----------------------------------------------------
    # 4-1. 상단 타이틀 / 실행 버튼
    # -----------------------------------------------------
    lbl_title = ttk.Label(
        root,
        text="KOSPI 12월 1주차 지수 예측",
        font=("맑은 고딕", 16, "bold")
    )
    lbl_title.pack(pady=10)

    frame_btn = ttk.Frame(root)
    frame_btn.pack(pady=10)

    style = ttk.Style()
    
    style.configure(
        "Big.TButton",
        font=("맑은 고딕", 15, "bold"),
        padding=(10, 10)
    )
    style.configure("Treeview", font=("맑은 고딕", 11))              # 셀 내용
    style.configure("Treeview.Heading", font=("맑은 고딕", 11, "bold"))  # 헤더
    style.configure("Treeview", rowheight=32)
    

    btn_run = ttk.Button(
        frame_btn,
        text="학습 및 예측 시작",
        width=20,
        style="Big.TButton"
    )
    btn_run.grid(row=0, column=0, padx=5)

    # -----------------------------------------------------
    # 4-2. 로그 출력 영역(Text + Scrollbar) - 생성만, pack은 나중에
    # -----------------------------------------------------
    txt_log = tk.Text(root, height=10)
    scroll = ttk.Scrollbar(txt_log, command=txt_log.yview)
    txt_log.configure(yscrollcommand=scroll.set)
    scroll.pack(side="right", fill="y")

    # -----------------------------------------------------
    # 4-3. Epoch 진행률 Progressbar - 생성만, pack은 나중에
    # -----------------------------------------------------
    frame_progress = ttk.Frame(root)

    lbl_progress = ttk.Label(frame_progress, text="Epoch 진행률:")
    lbl_progress.pack(side="left")

    progress = ttk.Progressbar(frame_progress, orient="horizontal",
                               mode="determinate", length=400)
    progress.pack(side="left", padx=10, fill="x", expand=True)

    # -----------------------------------------------------
    # 4-4. 하단 레이아웃 (그래프 + 표)
    # -----------------------------------------------------
    frame_bottom = ttk.Frame(root)

    # 왼쪽: 그래프 영역
    frame_graph = ttk.Frame(frame_bottom)
    frame_graph.pack(side="left", fill="both", expand=True, padx=(0, 5))

    frame_graph_btn = ttk.Frame(frame_graph)
    frame_graph_btn.pack(fill="x", pady=(0, 5))

    btn_10y = ttk.Button(frame_graph_btn, text="10년", width=10)
    btn_10y.pack(side="left", padx=(0, 5))

    btn_1m = ttk.Button(frame_graph_btn, text="1개월", width=10)
    btn_1m.pack(side="left")

    frame_graph_canvas = ttk.Frame(frame_graph)
    frame_graph_canvas.pack(fill="both", expand=True)

    # 오른쪽: 12월 1~5일 예측 결과 표
    frame_table = ttk.Frame(frame_bottom)
    frame_table.pack(side="right", fill="y", padx=(5, 0))

    lbl_table_title = ttk.Label(
        frame_table,
        text="2025년 12월 1~5일 예측 결과",
        font=("맑은 고딕", 11, "bold")
    )
    lbl_table_title.pack(pady=(0, 5))

    columns = ("date", "index", "change")
    tree = ttk.Treeview(frame_table, columns=columns, show="headings", height=8)

    tree.heading("date", text="날짜")
    tree.heading("index", text="지수 예측")
    tree.heading("change", text="등락률")

    tree.column("date", width=140, anchor="center")
    tree.column("index", width=90, anchor="e")
    tree.column("change", width=70, anchor="e")

    tree.pack(fill="y")

    # 🔁 여기서 최종 배치 순서 지정 (버튼 아래 순서)
    frame_progress.pack(fill="x", padx=10, pady=(0, 5))
    frame_bottom.pack(fill="both", expand=True, padx=10, pady=(0, 10))
    txt_log.pack(fill="x", padx=10, pady=(0, 10))

    # -----------------------------------------------------
    # 4-5. GUI 업데이트용 헬퍼 함수
    # -----------------------------------------------------
    def append_log(msg):
        def _():
            txt_log.insert(tk.END, msg + "\n")
            txt_log.see(tk.END)
        root.after(0, _)

    def update_progress(current, total):
        def _():
            progress['maximum'] = total
            progress['value'] = current
            progress.update_idletasks()
        root.after(0, _)

    # -----------------------------------------------------
    # 4-6. 그래프 그리기 함수들 (10년 / 1개월)
    # -----------------------------------------------------
    def ensure_figure():
        if state["fig"] is None:
            fig = Figure(figsize=(6, 4), dpi=100)
            ax = fig.add_subplot(111)
            canvas = FigureCanvasTkAgg(fig, master=frame_graph_canvas)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)

            state["fig"] = fig
            state["ax"] = ax
            state["canvas"] = canvas

    def draw_full_range():
        if state["plot_data"] is None:
            return
        ensure_figure()
        ax = state["ax"]
        canvas = state["canvas"]
        d = state["plot_data"]

        ax.clear()

        dates_all = d["dates_all"]
        real_prices = d["real_prices"]
        train_reconstructed = d["train_reconstructed"]
        test_reconstructed = d["test_reconstructed"]
        training_size = d["training_size"]
        time_step = d["time_step"]

        plot_train = np.empty(len(real_prices))
        plot_train[:] = np.nan
        plot_train[time_step + 1: time_step + 1 + len(train_reconstructed)] = train_reconstructed

        plot_test = np.empty(len(real_prices))
        plot_test[:] = np.nan
        plot_test[training_size + time_step + 1:
                  training_size + time_step + 1 + len(test_reconstructed)] = test_reconstructed

        ax.plot(dates_all, real_prices, label='Actual Price', alpha=0.6)
        ax.plot(dates_all, plot_train, label='Train Predict', alpha=0.8)
        ax.plot(dates_all, plot_test, label='Test Predict', color='red')

        ax.set_title('KOSPI Prediction (Full Range)')
        ax.set_xlabel('Date')
        ax.set_ylabel('KOSPI Index')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

        canvas.draw()

        # 1개월 버튼: 변동성 반영 예측 그래프 (Simulation A만)
        # 1개월 버튼: 변동성 반영 예측 그래프 (Simulation A만)
    def draw_monte_carlo():
        if state["plot_data"] is None:
            return
        ensure_figure()
        ax = state["ax"]
        canvas = state["canvas"]
        d = state["plot_data"]

        ax.clear()

        dates_hist = d["mc_actual_dates"]      # 11월 실제 거래일들
        prices_hist = d["mc_actual_prices"]
        scenarios = d["mc_scenarios"]
        target_dates_mc = d["mc_target_dates"] # 12/1~5 (평일만, B)

        # ---- 1) "거래일 인덱스" 축 만들기 ----
        # 과거 + 미래 날짜를 한 리스트로 묶어서
        # 0,1,2,... 순번을 부여 (주말은 애초에 없음)
        all_dates = list(dates_hist) + list(target_dates_mc)

        x_hist = np.arange(len(dates_hist))                    # 과거 구간 x
        x_future = np.arange(len(dates_hist)-1, len(all_dates))  # 마지막 과거 + 미래 구간 x

        # ---- 2) 과거 실제 데이터 (검은색 선) ----
        ax.plot(x_hist, prices_hist,
                label='실제 지수(~11/28)',
                color='black',
                linewidth=2)

        # ---- 3) Simulation A 예측 경로 (파란색 선) ----
        sim_prices = scenarios['Simulation A']
        last_price = prices_hist[-1]
        plot_prices = [last_price] + sim_prices  # 마지막 실제 + 12/1~5 예측

        ax.plot(
            x_future,
            plot_prices,
            label='지수 예측치(~12/5)',
            linestyle='-',
            linewidth=2
        )

        # ---- 4) 주 단위 세로선 그리기 (5일마다 / 혹은 월요일마다) ----
        # 방법 A: 단순히 5일마다
        # for i in range(4, len(all_dates), 5):   # 0부터 시작해서 5개마다
        #     ax.axvline(x=i, color='red', linestyle='--', linewidth=1, alpha=0.6)

        # 방법 B: 실제 '월요일'에 세로선 (주간 경계가 딱 맞음)
        week_bound_indices = [i for i, dt in enumerate(all_dates) if dt.weekday() == 0]  # 0=월요일
        for idx in week_bound_indices:
            ax.axvline(x=idx, color='red', linestyle='--', linewidth=1, alpha=0.6)

        # ---- 5) X축 눈금: 거래일 인덱스에 날짜 라벨만 붙이기 ----
        ax.set_xticks(np.arange(len(all_dates)))
        ax.set_xticklabels(
            [dt.strftime('%m-%d') for dt in all_dates],
            rotation=45,
            ha='right'
        )

        ax.set_title('KOSPI 12월 1주차 예측')
        ax.set_xlabel('Date')
        ax.set_ylabel('KOSPI Index')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

        canvas.draw()



    btn_10y.config(command=draw_full_range)
    btn_1m.config(command=draw_monte_carlo)

    # -----------------------------------------------------
    # 4-7. [학습 및 예측 시작] 버튼 동작
    # -----------------------------------------------------
    def on_run_clicked():
        btn_run.config(state="disabled")
        progress['value'] = 0
        progress.update_idletasks()

        txt_log.delete("1.0", tk.END)
        append_log("[알림] 데모 실행을 시작합니다. (시간이 다소 걸릴 수 있습니다)")

        if state["canvas"] is not None:
            state["canvas"].get_tk_widget().destroy()
            state["fig"] = None
            state["ax"] = None
            state["canvas"] = None

        for row in tree.get_children():
            tree.delete(row)

        def worker():
            try:
                result_rows, plot_data = run_full_demo(
                    log_fn=append_log,
                    progress_fn=update_progress,
                    max_epochs=MAX_EPOCHS
                )

                def on_done():
                    state["plot_data"] = plot_data

                    draw_monte_carlo()  # 기본 그래프는 10년 뷰

                    weekday_kor = ['월', '화', '수', '목', '금', '토', '일']
                    for date_str, price, ret_pct in result_rows:
                        dt = datetime.fromisoformat(date_str)
                        label_date = f"{dt.month:02d}월 {dt.day:02d}일({weekday_kor[dt.weekday()]})"
                        label_index = f"{price:,.2f}"
                        label_change = f"{ret_pct:+.2f}"
                        tree.insert("", tk.END, values=(label_date, label_index, label_change))

                    messagebox.showinfo("완료", "데모 실행이 완료되었습니다!\n그래프와 표를 확인하세요.")
                    btn_run.config(state="normal")

                root.after(0, on_done)

            except Exception as e:
                def on_error():
                    messagebox.showerror("에러 발생", f"실행 중 오류가 발생했습니다:\n{e}")
                    btn_run.config(state="normal")
                root.after(0, on_error)

        threading.Thread(target=worker, daemon=True).start()

    btn_run.config(command=on_run_clicked)

    root.mainloop()


if __name__ == "__main__":
    main()
