from ultralytics.utils.plotting import plot_results

plot_results(
    file='runs_yolo/train_elnet12n_base/results_worse.csv',
    dir='runs_yolo/train_elnet12n_base',     # thư mục lưu lại
    plots=['loss', 'metrics'], # có thể chọn loss, P, R, mAP
)
