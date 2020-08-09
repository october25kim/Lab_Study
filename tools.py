import numpy as np
import pandas as pd
from scipy.stats import mode

def get_matched_alarm_record(matched_file, mach_id, alarm_record, start_year=2020, end_year=2020):
    """
    mach_id와 시간(일자)정보로 타겟파일에 해당하는 알람 기록 찾기

    :param list matched_file: 타겟파일 list
    :param string mach_id: 설비id
    :param pd.DataFrame alarm_record: 전체 알람기록파일
    :param int start_year: 타겟파일의 기록 시작 연도
    :param int start_year: 타겟파일의 기록 종료 연도
    :return: alarm
    """
    periods_start = [file[3] for file in matched_file]
    periods_end = [file[4] for file in matched_file]
    start_year, end_year = str(start_year), str(end_year)
    periods_start_dt = pd.Series([pd.to_datetime(start_year + period) for period in periods_start])
    periods_end_dt = pd.Series([pd.to_datetime(end_year + period) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1) for period in periods_end])
    period_dt = [periods_start_dt.min(), periods_end_dt.max()]
    period_mask = (alarm_record['START_TIME'] > period_dt[0]) & (alarm_record['START_TIME'] <= period_dt[1])
    alarm = alarm_record[(alarm_record["MACH_ID"] == int(mach_id)) & period_mask]
    period_dt = [str(periods_start_dt.min()), str(periods_end_dt.max())]
    return alarm, period_dt

def read_file(path_x, filename_list):
    """
    read data files and concat.
    set "COLLECT_TIME" as pd.datatime

    :param string path_x: 데이터폴더 주소
    :param string filename_list: 파일명의 list
    :return: pd.DataFrame
    """
    dat = pd.DataFrame()
    for filename in filename_list:
        dat = pd.concat([dat, pd.read_csv(path_x + filename + ".csv")])
    # 중간에 칼럼명 껴있는것 제거
    dat = dat.drop(dat[dat["COLLECT_TIME"] == 'COLLECT_TIME'].index).reset_index(drop=True)
    dat["COLLECT_TIME"] = pd.to_datetime(dat["COLLECT_TIME"])
    dat.iloc[:, 1:] = dat.iloc[:, 1:].astype('float')
    return dat

def alarm_labeling(dat_t, alarm, hour):
    alarm_start_time = alarm["START_TIME"].values
    alarm_num = len(alarm_start_time)
    dat_alarm = [pd.DataFrame(0, index=range(0, len(dat_t)), columns=["ALARM"]) for n in range(alarm_num)]
    alarm_ids = alarm["ALARM_ID"].values

    # 한 타겟파일에 알람 여러번 생겼을 수 있는거 고려하였음
    for idx_start_time, alarm_id in enumerate(alarm_ids):
        tmp_start_time = alarm_start_time[idx_start_time]
        alarm_time = pd.to_datetime(tmp_start_time)  # .strftime("%Y-%m-%d %H:%M:%S")
        dat_t_time = dat_t["COLLECT_TIME"].copy()  # dat_t_time = dat_t_time.apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))
        print("number of matching cases: ", sum(dat_t_time == alarm_time))
        if sum(dat_t_time == alarm_time) == 0:
            dat_t_time = pd.DataFrame(dat_t_time)
            dat_t_time = dat_t_time.astype("datetime64")
            print("dat_t_time.info(): ", dat_t_time.info())
            alarm_time_df = pd.DataFrame([alarm_time], columns=["COLLECT_TIME"], dtype="datetime64")
            dat_t_time_alarm = pd.concat((dat_t_time, alarm_time_df), axis=0).sort_values(by="COLLECT_TIME")
            idx_alarm = np.where(dat_t_time_alarm == alarm_time)[0][0]
        else:
            idx_alarm = np.where(dat_t_time == alarm_time)[0][0]
        time_normal = alarm_time - pd.Timedelta(hours=hour)
        before_alarm_times = dat_t["COLLECT_TIME"][:idx_alarm]
        idxs_alarm = (before_alarm_times > time_normal)
        dat_alarm[idx_start_time]["ALARM"][:idx_alarm][idxs_alarm] = alarm_id
    for i in range(alarm_num):
        dat_t = pd.concat((dat_t, dat_alarm[i]), axis=1)

    dat_t['ALARM_ID'] = dat_t['ALARM'].sum(axis=1)
    dat_t = dat_t.drop(['ALARM'], axis=1)
    return dat_t

def unify_time_unit(dat, unify_sec, idx_logging=False, verbose=False):
    """
    동일 시간단위로 통합 (시간 단위 내 값들을 평균취함)
    :param dat:
    :param unify_sec: 몇초로 통합할 것인지
    :param idx_logging: 옵션
    :param verbose: 옵션
    :return:
    """
    def to_micsec(time_gap):
        return (time_gap.days * 24 * 60 * 60 + time_gap.seconds) * 1000000 + time_gap.microseconds

    from_, to_, time_gap_micsec, unified_dat_x_idx = 0, 0, 0, 0
    from_time = dat["COLLECT_TIME"][from_]
    idx_log = []
    jump_idx = []
    # unified_dat_x = pd.DataFrame()
    # unified_dat_y = pd.DataFrame()
    unified_dat_x = []
    unified_dat_y = []
    num_of_alarm = 5

    finished = False
    while to_ < len(dat) - 1:
        to_ += 1

        if to_ == len(dat) - 1:
            finished = True
            time_gap_micsec = 0

        if not finished:
            time_gap = dat["COLLECT_TIME"][to_] - from_time
            time_gap_micsec = to_micsec(time_gap)

        if time_gap_micsec < unify_sec * 1000000:
            pass
        else:
            if idx_logging: idx_log.append([from_, to_])
            if verbose and unified_dat_x_idx % 100 == 0: print(from_ / len(dat))

            dat_range = dat.iloc[from_:to_, :]
            unified_time = dat_range.iloc[:, :2].min()
            unified_mach_info = dat_range.iloc[:, 2:10].median()
            unified_target = dat_range.iloc[:, 10:-1].mean()
            unified = pd.concat((unified_time, unified_mach_info, unified_target))
            unified_dat_x.append(unified)

            unify_target_y = dat.iloc[from_:to_, -1:].mode()
            unified_dat_y.append(unify_target_y)

            # check time-jump
            if not finished:
                unified_dat_x_idx += 1
                next_unified_gap = dat["COLLECT_TIME"][to_] - dat["COLLECT_TIME"][to_ - 1]
                next_unified_gap_micsec = to_micsec(next_unified_gap)
                if next_unified_gap_micsec > unify_sec * 1000000:
                    jump_idx.append(unified_dat_x_idx)  # 추후에 jump_idx를 포함하는 윈도우는 이를 첫 인덱스로 갖는 경우를 제외하고 제거하면 됨.
                from_ = to_
                from_time = dat["COLLECT_TIME"][from_]

    unified_dat_x = pd.DataFrame(unified_dat_x)
    unified_dat_x.iloc[:, 2:] = unified_dat_x.iloc[:, 2:].astype('float')
    unified_dat_y = [data.iloc[0].values for data in unified_dat_y]
    return unified_dat_x, unified_dat_y, jump_idx, idx_log

def windowing(dat_t_x, dat_t_y, jump_idx, window_size, shift_size, threshold=None):
    """
    윈도윙함. 수집간격이 통합단위보다 큰 경우 윈도윙을 중단하고 그 다음 시점부터 다시 윈도윙 수행
    threshold==None일 경우 윈도우 내 과반수의 레이블을 윈도우의 레이블로 지정

    :param pd.DataFrame dat_t_x:
    :param pd.Series dat_t_y:
    :param list jump_idx: 수집간격 큰 시점
    :param int window_size:
    :param int shift_size:
    :param int threshold: 한 윈도우 내에 타임스텝이 몇개 이상 전조 또는 이상일 때 윈도우를 해당 레이블로 레이블링할 지 임계값. Default:None 일땐 가장 많은 레이블로 결정.
    :return: X: (n_window, n_sensor, window_size) , y_label: (n_window, )
    """
    dat_t_x = np.array(dat_t_x)
    dat_t_y = np.array(dat_t_y)

    # create windows
    X = []
    y = []

    from_, to_ = 0, 0
    while to_ < len(dat_t_x) - 1:
        to_ = from_ + window_size
        if to_ > len(dat_t_x): break;

        # 수집간격 큰 시점 고려
        window_range = set(range(from_ + 1, to_))
        if len(window_range & set(jump_idx)) != 0:
            from_ = list(window_range & set(jump_idx))[-1]
        else:
            X.append(np.array(dat_t_x[from_:to_].transpose()))
            y.append(np.array(dat_t_y[from_:to_].transpose()))
            from_ = from_ + shift_size

    # to 3D array
    X = np.array(X)
    # set window label
    y_label = []
    for s in y:
        label = mode(s[0]).mode
        y_label.append(label)
    return X, y_label