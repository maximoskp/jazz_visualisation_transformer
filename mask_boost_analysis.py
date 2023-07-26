import pandas as pd

figs_path = 'figs/analysis/'

m = pd.read_csv('data/model_transposition_train_history_log.csv')
n = pd.read_csv('data/model_no_mask_transposition_train_history_log.csv')

s = pd.DataFrame()

s['epoch'] = m['epoch']

s['mask_year_rolling_100'] = m['year_predictor_loss'].rolling(100).mean()
s['no_mask_year_rolling_100'] = n['year_predictor_loss'].rolling(100).mean()
s['mask_composer_rolling_100'] = m['composer_predictor_loss'].rolling(100).mean()
s['no_mask_composer_rolling_100'] = n['composer_predictor_loss'].rolling(100).mean()

s['year_difference'] = s['no_mask_year_rolling_100'] - s['mask_year_rolling_100']
s['composer_difference'] = s['no_mask_composer_rolling_100'] - s['mask_composer_rolling_100']


# after_1000 = pd.DataFrame()
# after_1000['epoch'] = s['epoch']
# after_1000['year_difference'] = s['year_difference'][1000:]

s.dropna()

# ax = s.plot(x='epoch', y=['mask_year_rolling_100', 'no_mask_year_rolling_100'])
ax = s.plot(x='epoch', y=['year_difference', 'composer_difference'])
ax.figure.savefig(figs_path + 'year_rolling_comparison.png', dpi=300)

# ax = after_1000.plot(x='epoch', y='year_difference')
# ax.figure.savefig(figs_path + 'year_rolling_comparison.png', dpi=300)