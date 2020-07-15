import pickle
import gzip
from tools import *
from config import *

data_path = "//163.152.174.30/NewShare/1. 프로젝트/2020_현대자동차/data/raw/공장_세타(4,5월)/"
alarm_path = "//163.152.174.30/NewShare/1. 프로젝트/2020_현대자동차/data/alarm/"
save_path = "D:/Openset_signal/data/"

alarm_record = pd.read_excel(alarm_path + "alarm_sh_labmeeting" + ".xlsx")
line = 'BLOCK'
matched_file_name = ['108_세타_m37_0401_0430', '108_세타_m37_0501_0531']
matched_file = [['108','세타','m37','0401','0430'],['108','세타','m37','0501','0531']]
mach_id = 37

alarm, period = get_matched_alarm_record(matched_file, mach_id, alarm_record)
dat_t = read_file(data_path, matched_file_name)
dat_t = alarm_labeling(dat_t, alarm, 24)

u_dat_t_x, u_dat_t_y, jump_idx_t, idx_log_t = unify_time_unit(dat_t, unify_sec=unify_sec, idx_logging=False, verbose=True)
with gzip.open(save_path + "x_data_0.5sec", 'wb') as f:
    pickle.dump(u_dat_t_x, f, pickle.HIGHEST_PROTOCOL)
with gzip.open(save_path + "y_data_0.5sec", 'wb') as f:
    pickle.dump(u_dat_t_y, f, pickle.HIGHEST_PROTOCOL)

X_t, y_t = windowing(u_dat_t_x, u_dat_t_y, jump_idx_t, window_size=window_size, shift_size=shift_size, threshold=threshold)
with gzip.open(save_path + matched_file_name[0][:-10]+ '_' + period[0][:10] + '_' + period[1][:10]+ "_x", 'wb') as f:
    pickle.dump(X_t, f, pickle.HIGHEST_PROTOCOL)
with gzip.open(save_path + matched_file_name[0][:-10]+ '_' + period[0][:10] + '_' + period[1][:10]+ "_y", 'wb') as f:
    pickle.dump(y_t, f, pickle.HIGHEST_PROTOCOL)