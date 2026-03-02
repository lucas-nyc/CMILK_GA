import time
import sys
sys.path.append('[enter GAIN repo path]')  # GAIN repo path

from gain import gain
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import torch
import tensorflow as tf
from miceforest import ImputationKernel
import re
import random

tf.random.set_seed(42)
tf.compat.v1.set_random_seed(42)
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

gain_parameters = {
    'batch_size': 64,
    'hint_rate': 0.9,
    'alpha': 100,
    'iterations': 1000,
}

vae_parameters = {
    'epochs': 150,
    'batch_size': 64,
    'alpha': 1e-2,
    'learning_rate': 1e-3,
    'noise_high_limit': 1e-1,
    'noise_zero': True,
    'test_iteration': 20,
    'train_with_complete': False,
    'loss_mode': 'log_masked',
    'kl_loss_mode': 'complex',
}

missing_datasets_root = 'missing_datasets/'
imputed_datasets_root = 'imputed_datasets/'
os.makedirs(imputed_datasets_root, exist_ok=True)

original_dataset_path = 'all_tbl_manual_annotated_v2_cleaned.csv'
original_dataset = pd.read_csv(original_dataset_path)

categories = ['MNAR', 'MCAR']
results_list = []
per_set_results = []
per_row_results = []

def return_mask_of_data(data):
    return 1.0 - np.isnan(data)


def return_layer(layer_input, output_size, norm=False, dropout=False, activation='relu'):
    output = tf.keras.layers.Dense(
        output_size,
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
    )(layer_input)
    if norm:
        output = tf.keras.layers.LayerNormalization()(output)
    if dropout:
        output = tf.keras.layers.Dropout(rate=dropout)(output)
    if activation:
        output = tf.keras.layers.Activation(activation=activation)(output)
    return output


class data_shuffle_noise:
    def __init__(self, mode, noise_zero, high, seed=None):
        self.mode = mode
        self.noise_zero = noise_zero
        self.high = high
        self.low = 0.0 if self.mode == 'one_hot' else -self.high
        self.rng = np.random.RandomState(seed) if seed is not None else np.random

    def _add_noise_(self, train_data, train_mask):
        train_data = train_data.copy()
        if self.noise_zero:
            train_data[np.isnan(train_data)] = 0.0
            return train_data
        noise = self.rng.uniform(low=self.low, high=self.high, size=train_data.shape)
        train_data[np.isnan(train_data)] = 0.0
        return train_data * train_mask + noise * (1.0 - train_mask)

    def dataNonenan(self, data, mask):
        index = []
        for i in range(data.shape[0]):
            if str(np.sum(data[i, :])) != 'nan':
                index.append(i)
        if len(index) == 0:
            return data, mask
        return data[index, :], mask[index, :]

    def data_shuffle(self, train_data, train_mask):
        idx = np.arange(train_data.shape[0])
        self.rng.shuffle(idx)
        train_d = train_data[idx]
        train_m = train_mask[idx]
        train_d = self._add_noise_(train_data=train_d, train_mask=train_m)
        return train_d, train_m


def vae_impute(data_missing_np, params, seed=42):
    data = np.asarray(data_missing_np, dtype=np.float32)
    mask = return_mask_of_data(data).astype(np.float32)

    if not np.any(mask):
        return np.nan_to_num(data, nan=0.0)

    col_min = np.nanmin(data, axis=0)
    col_max = np.nanmax(data, axis=0)
    col_min = np.where(np.isfinite(col_min), col_min, 0.0)
    col_max = np.where(np.isfinite(col_max), col_max, col_min + 1.0)
    col_range = col_max - col_min
    col_range = np.where(col_range < 1e-6, 1.0, col_range)
    data_scaled = (data - col_min) / col_range
    data_scaled = np.where(np.isnan(data), np.nan, data_scaled)

    num_samples, data_shape = data.shape
    batch_size = int(max(1, min(params.get('batch_size', 64), num_samples)))
    epochs = int(max(1, params.get('epochs', 150)))
    alpha = float(params.get('alpha', 1e-2))
    learning_rate = float(params.get('learning_rate', 1e-3))
    noise_high_limit = float(params.get('noise_high_limit', 1e-1))
    noise_zero = bool(params.get('noise_zero', True))
    test_iteration = params.get('test_iteration', 20)
    train_with_complete = bool(params.get('train_with_complete', False))
    loss_mode = params.get('loss_mode', 'log_masked')
    kl_loss_mode = params.get('kl_loss_mode', 'complex')

    if int(data_shape / 4.0) < 4:
        latent_size = 4
        mid_layer = int((latent_size + data_shape) / 2.0)
    else:
        latent_size = int(data_shape / 4.0)
        mid_layer = int(data_shape / 2.0)

    network_layer_G = [data_shape, mid_layer, latent_size]
    network_latent_layer_G = [latent_size]
    network_layer_D = [latent_size, mid_layer, data_shape]
    network_latent_layer_D = [data_shape]

    dsn = data_shuffle_noise(mode='one_hot', noise_zero=noise_zero, high=noise_high_limit, seed=seed)

    def return_encode_network_divide(x, output_mode='mean'):
        inputs = x
        if len(network_latent_layer_G) == 1:
            output = return_layer(layer_input=inputs, output_size=network_latent_layer_G[0], activation='tanh')
        else:
            output = inputs
            for i in range(len(network_latent_layer_G)):
                output = return_layer(layer_input=output, output_size=network_latent_layer_G[i], activation='tanh')
        return output

    def return_encode_network():
        inputs = tf.keras.Input(shape=(network_layer_G[0],))
        for i in range(len(network_layer_G)):
            if i == 0:
                output = return_layer(layer_input=inputs, output_size=network_layer_G[i + 1], activation='tanh')
            elif i == len(network_layer_G) - 1:
                output_p1 = return_encode_network_divide(x=output, output_mode='mean')
                output_p2 = return_encode_network_divide(x=output, output_mode='var')
                output = tf.keras.layers.Concatenate(axis=1)([output_p1, output_p2])
            else:
                output = return_layer(layer_input=output, output_size=network_layer_G[i + 1], activation='tanh')
        return tf.keras.Model(inputs=inputs, outputs=output)

    def return_decode_network():
        inputs = tf.keras.Input(shape=(network_layer_D[0],))
        for i in range(len(network_layer_D) - 1):
            if i == 0:
                output = return_layer(layer_input=inputs, output_size=network_layer_D[i + 1], activation='tanh')
            elif i == len(network_layer_D) - 2:
                output = return_layer(layer_input=output, output_size=network_layer_D[i + 1], activation='tanh')
            else:
                output = return_layer(layer_input=output, output_size=network_layer_D[i + 1], activation='tanh')
        return tf.keras.Model(inputs=inputs, outputs=output)

    def return_decode_network_divide(output_mode='mean'):
        inputs = tf.keras.Input(shape=(network_layer_D[-1],))
        if len(network_latent_layer_D) == 1:
            activation = 'sigmoid' if output_mode == 'mean' else 'tanh'
            output = return_layer(layer_input=inputs, output_size=network_latent_layer_D[0], activation=activation)
        else:
            output = inputs
            for i in range(len(network_latent_layer_D)):
                if i == len(network_latent_layer_D) - 1 and output_mode == 'mean':
                    activation = 'sigmoid'
                else:
                    activation = 'tanh' if output_mode != 'mean' else 'relu'
                output = return_layer(layer_input=output, output_size=network_latent_layer_D[i], activation=activation)
        return tf.keras.Model(inputs=inputs, outputs=output)

    def loss(x, m, mean, log_std):
        log_two_pi = tf.constant(np.log(2.0 * np.pi), dtype=tf.float32)
        log_prob = -0.5 * (log_two_pi + 2.0 * log_std + tf.square(x - mean) / tf.exp(2.0 * log_std))
        if loss_mode == 'log_masked':
            return -tf.reduce_sum(log_prob * m, axis=None) / tf.reduce_sum(m, axis=None)
        if loss_mode == 'log':
            return -tf.reduce_mean(log_prob, axis=None)
        return -tf.reduce_sum(log_prob * m, axis=None) / tf.reduce_sum(m, axis=None)

    def kl_loss(mu, log_std):
        if kl_loss_mode == 'sample':
            log_var = 2.0 * log_std
            return -0.5 * tf.reduce_mean(
                1.0 + log_var - tf.square(mu) + tf.exp(log_var)
            )
        return 0.5 * tf.reduce_mean(
            tf.reduce_sum(tf.exp(2.0 * log_std) + tf.square(mu) - 1.0 - 2.0 * log_std, axis=1)
        )

    graph = tf.Graph()
    with graph.as_default():
        tf.compat.v1.set_random_seed(seed)
        tf.random.set_seed(seed)
        x_ph = tf.compat.v1.placeholder(tf.float32, shape=[None, data_shape])
        m_ph = tf.compat.v1.placeholder(tf.float32, shape=[None, data_shape])

        model_e = return_encode_network()
        gen_p = model_e(x_ph)
        z_mean = gen_p[:, :latent_size]
        z_log_std = gen_p[:, latent_size:]
        eps = tf.random.normal(tf.shape(z_mean), seed=seed)
        z = z_mean + tf.exp(z_log_std) * eps
        model_d = return_decode_network()
        gen_x = model_d(z)
        model_ddm = return_decode_network_divide(output_mode='mean')
        model_ddv = return_decode_network_divide(output_mode='var')
        gen_x_mean = model_ddm(gen_x)
        gen_x_var = model_ddv(gen_x)
        loss_p2 = kl_loss(mu=z_mean, log_std=z_log_std)
        loss_p1 = loss(x=x_ph, m=m_ph, mean=gen_x_mean, log_std=gen_x_var)
        total_loss = tf.reduce_sum(loss_p1 + alpha * loss_p2)
        solver = tf.compat.v1.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(
            total_loss,
            var_list=(model_e.trainable_variables + model_d.trainable_variables +
                      model_ddm.trainable_variables + model_ddv.trainable_variables),
        )
        init = tf.compat.v1.global_variables_initializer()

    config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    with tf.compat.v1.Session(graph=graph, config=config) as sess:
        sess.run(init)
        if train_with_complete:
            train_data, train_mask = dsn.dataNonenan(data=data_scaled.copy(), mask=mask.copy())
        else:
            train_data, train_mask = data_scaled.copy(), mask.copy()

        for _ in range(epochs):
            train_d, train_m = dsn.data_shuffle(train_data, train_mask)
            for iteration in range(int(train_d.shape[0] / batch_size)):
                batch_d = train_d[iteration * batch_size:(iteration + 1) * batch_size]
                batch_m = train_m[iteration * batch_size:(iteration + 1) * batch_size]
                sess.run(solver, feed_dict={x_ph: batch_d, m_ph: batch_m})

        data_noised = dsn._add_noise_(train_data=data_scaled.copy(), train_mask=mask.copy())
        if test_iteration is None:
            data_imputed = sess.run(gen_x_mean, feed_dict={x_ph: data_noised, m_ph: mask})
        else:
            data_imputed = data_noised
            for _ in range(int(test_iteration)):
                data_imputed = sess.run(gen_x_mean, feed_dict={x_ph: data_imputed, m_ph: mask})

        data_generate = data_imputed * (1.0 - mask) + data_noised * mask

    tf.keras.backend.clear_session()
    data_generate = np.clip(data_generate, 0.0, 1.0)
    return data_generate * col_range + col_min


for category in categories:
    print(f"\nProcessing category: {category}")
    missing_datasets_dir = os.path.join(missing_datasets_root, category)
    imputed_datasets_dir = os.path.join(imputed_datasets_root, category)
    os.makedirs(imputed_datasets_dir, exist_ok=True)

    dataset_files = sorted([f for f in os.listdir(missing_datasets_dir) if f.endswith('.csv')])
    num_combination = len(dataset_files)
    print(f"Number of datasets in {category}: {num_combination}")

    for idx, file in enumerate(dataset_files, 1):  # Start from 1
        m = re.search(r'missing_dataset_set_(\d+)(?:_([A-Za-z0-9_]+))?\.csv$', file)
        if m:
            set_number = int(m.group(1))
            scenario = m.group(2) if m.group(2) is not None else category  # fallback to category if no scenario token
        else:
            nums = re.findall(r'(\d+)', file)
            set_number = int(nums[-1]) if nums else idx
            scenario = category
            print(f"Warning: filename '{file}' didn't match expected pattern. Using set_number={set_number}, scenario={scenario}.")

        print(f'Processing missing dataset file {idx}/{num_combination} -> set {set_number} (scenario={scenario}) in {category}')

        missing_dataset_path = os.path.join(missing_datasets_dir, file)
        data_test_missing = pd.read_csv(missing_dataset_path, header=None)
        data_test_missing = data_test_missing.apply(pd.to_numeric, errors='coerce')
        print(data_test_missing.shape)

        num_columns = data_test_missing.shape[1]
        column_names = [f"col_{i}" for i in range(num_columns)]
        data_test_missing.columns = column_names

        data_test_missing_np = data_test_missing.to_numpy()
        num_samples = data_test_missing_np.shape[0]

        gain_params_local = gain_parameters.copy()
        if num_samples < gain_params_local['batch_size']:
            gain_params_local['batch_size'] = num_samples

        try:
            vae_start_time = time.time()
            vae_params_local = vae_parameters.copy()
            vae_params_local['batch_size'] = min(vae_params_local['batch_size'], max(1, num_samples))
            imputed_vae = vae_impute(data_test_missing_np, vae_params_local, seed=42)
            vae_time = time.time() - vae_start_time
            print(f"VAE imputation completed in {vae_time:.2f} seconds.")

            if np.isnan(imputed_vae).all():
                print(f"VAE imputed dataset for set {set_number} is all NaN. Reverting to original.")
                imputed_vae = data_test_missing_np
        except Exception as e:
            print(f"VAE imputation failed for set {set_number}: {e}")
            imputed_vae = data_test_missing_np
            vae_time = float('inf')

        try:
            gain_start_time = time.time()
            imputed_gain = gain(data_test_missing_np, gain_params_local)
            gain_time = time.time() - gain_start_time
            print(f"GAIN imputation completed in {gain_time:.2f} seconds.")

            if np.isnan(imputed_gain).all():
                print(f"GAIN imputed dataset for set {set_number} is all NaN. Reverting to original.")
                imputed_gain = data_test_missing_np
        except Exception as e:
            print(f"GAIN imputation failed for set {set_number}: {e}")
            imputed_gain = data_test_missing_np
            gain_time = float('inf')

        try:
            mice_start_time = time.time()
            mice_kernel = ImputationKernel(data_test_missing, random_state=42)
            mice_kernel.mice(10)
            imputed_mice = mice_kernel.complete_data(0).to_numpy()
            mice_time = time.time() - mice_start_time
            print(f"miceforest imputation completed in {mice_time:.2f} seconds.")

            if np.isnan(imputed_mice).all():
                print(f"MiceForest imputed dataset for set {set_number} is all NaN. Reverting to original.")
                imputed_mice = data_test_missing_np
        except Exception as e:
            print(f"miceforest imputation failed for set {set_number}: {e}")
            imputed_mice = data_test_missing_np
            mice_time = float('inf')


        gain_dir = os.path.join(imputed_datasets_dir, 'GAIN')
        mice_dir = os.path.join(imputed_datasets_dir, 'MiceForest')
        vae_dir = os.path.join(imputed_datasets_dir, 'VAE')
        os.makedirs(gain_dir, exist_ok=True)
        os.makedirs(mice_dir, exist_ok=True)
        os.makedirs(vae_dir, exist_ok=True)

        if category == 'MNAR':
            base_filename = f'imputed_dataset_set_{set_number}_{scenario}.csv'
        else:
            base_filename = f'imputed_dataset_set_{set_number}.csv'
        gain_path = os.path.join(gain_dir, base_filename)
        mice_path = os.path.join(mice_dir, base_filename)
        vae_path = os.path.join(vae_dir, base_filename)
        pd.DataFrame(imputed_gain).to_csv(gain_path, index=False, header=False)
        pd.DataFrame(imputed_mice).to_csv(mice_path, index=False, header=False)
        pd.DataFrame(imputed_vae).to_csv(vae_path, index=False, header=False)
        print(f"Imputed datasets saved to: {gain_path}, {mice_path}, and {vae_path}")
        missing_mask = data_test_missing.isna().to_numpy()
        original_values = original_dataset.to_numpy()
        metric_rows = min(original_values.shape[0], data_test_missing_np.shape[0])
        metric_cols = min(original_values.shape[1], data_test_missing_np.shape[1])
        if metric_rows != data_test_missing_np.shape[0] or metric_cols != data_test_missing_np.shape[1]:
            print(
                f"Warning: original dataset shape {original_values.shape} does not match missing dataset shape "
                f"{data_test_missing_np.shape}. Metrics will use overlap {metric_rows}x{metric_cols}."
            )
        original_values = original_values[:metric_rows, :metric_cols]
        missing_mask_metric = missing_mask[:metric_rows, :metric_cols]
        metric_num_columns = missing_mask_metric.shape[1]

        for method_name, imputed_data, time_taken in [
            ("GAIN", imputed_gain, gain_time),
            ("MiceForest", imputed_mice, mice_time),
            ("VAE", imputed_vae, vae_time),
        ]:
            maes, rmses = [], []
            imputed_data_metric = imputed_data[:metric_rows, :metric_cols]

            for row_idx in range(metric_rows):
                row_mask = missing_mask_metric[row_idx]
                n_missing = np.sum(row_mask)
                if n_missing == 0:
                    continue 

                original_row = original_values[row_idx]
                imputed_row = imputed_data_metric[row_idx]

                try:
                    valid_mask = row_mask & np.isfinite(original_row) & np.isfinite(imputed_row)
                    if not np.any(valid_mask):
                        continue
                    mae = mean_absolute_error(original_row[valid_mask], imputed_row[valid_mask])
                    rmse = np.sqrt(mean_squared_error(original_row[valid_mask], imputed_row[valid_mask]))
                    maes.append(mae)
                    rmses.append(rmse)
                except Exception as e:
                    print(f"Error in row {row_idx} for {method_name}, set {set_number}: {e}")
                    continue

                missing_pct = n_missing / metric_num_columns * 100

                results_list.append({
                    'Method': method_name,
                    'Set_idx': set_number,
                    'Row_idx': row_idx,
                    'MAE': mae,
                    'RMSE': rmse,
                    'MissingPercentage': missing_pct,
                    'Category': category
                })

            if len(maes) > 0:
                per_row_results.append({
                    'Category': category,
                    'Method': method_name,
                    'Set_idx': set_number,
                    'MAE': np.mean(maes),
                    'RMSE': np.mean(rmses),
                    'Time': time_taken
                })


            missing_positions = missing_mask_metric.flatten()
            original_missing_values = original_values.flatten()[missing_positions]
            imputed_missing_values = imputed_data_metric.flatten()[missing_positions]

            if len(original_missing_values) > 0:
                valid_mask = np.isfinite(original_missing_values) & np.isfinite(imputed_missing_values)
                if not np.any(valid_mask):
                    continue
                set_mae = mean_absolute_error(original_missing_values[valid_mask], imputed_missing_values[valid_mask])
                set_rmse = np.sqrt(mean_squared_error(original_missing_values[valid_mask], imputed_missing_values[valid_mask]))

                per_set_results.append({
                    'Category': category,
                    'Method': method_name,
                    'Set_idx': set_number,
                    'MAE': set_mae,
                    'RMSE': set_rmse,
                    'Time': time_taken,
                    'Scenario': scenario
                })


results_df = pd.DataFrame(results_list)
results_df.to_csv('per_row_imputation_results_v5.csv', index=False)
print("Per-row imputation results saved to per_row_imputation_results_v5.csv")

per_row_df = pd.DataFrame(per_row_results)

mnar_df = per_row_df[per_row_df['Category'] == 'MNAR']
mcar_df = per_row_df[per_row_df['Category'] == 'MCAR']

mnar_df.to_csv('imputation_results_mnar_row_v5.csv', index=False)
mcar_df.to_csv('imputation_results_mcar_row_v5.csv', index=False)

print("Per-set MNAR results saved to imputation_results_mnar_row_v5.csv")
print("Per-set MCAR results saved to imputation_results_mcar_row_v5.csv")

per_set_df = pd.DataFrame(per_set_results)

mnar_df = per_set_df[per_set_df['Category'] == 'MNAR']
mcar_df = per_set_df[per_set_df['Category'] == 'MCAR']

mnar_df.to_csv('imputation_results_mnar_set_v5.csv', index=False)
mcar_df.to_csv('imputation_results_mcar_set_v5.csv', index=False)

print("Per-set MNAR results saved to imputation_results_mnar_set_v5.csv")
print("Per-set MCAR results saved to imputation_results_mcar_set_v5.csv")

summary_df = (
    per_set_df
    .groupby(['Category'], as_index=False)
    .agg(
        Mean_RMSE=('RMSE', 'mean'),
        Mean_MAE=('MAE', 'mean'),
        SD_RMSE=('RMSE', 'std'),
        SD_MAE=('MAE', 'std'),
        Mean_Time=('Time', 'mean'),
    )
)
summary_df = summary_df[['Category', 'Mean_RMSE', 'Mean_MAE', 'SD_RMSE', 'SD_MAE', 'Mean_Time']]
summary_df.to_csv('imputation_results_set_mean_vae.csv', index=False)
print("Per-set mean/std results saved to imputation_results_set_mean_vae.csv")
